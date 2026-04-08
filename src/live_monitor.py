"""
Live stream monitoring for Zhiyun Classroom.

Polls the catalogue API for a live HLS stream, segments it into 30-second
WAV chunks, transcribes each chunk with faster-whisper, performs pinyin-based
fuzzy keyword matching, and sends DingTalk alerts when keywords are confirmed
by an LLM.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import time
from collections import deque
from datetime import date, datetime
from typing import Generator

import requests

from src.crawler import CATALOGUE_API
from src.session_utils import mount_legacy_ssl

GET_SUB_INFO_API = (
    "https://classroom.zju.edu.cn/courseapi/v3/portal-home-setting/get-sub-info"
)
SCHEDULE_API = (
    "https://yjapi.cmc.zju.edu.cn/courseapi/v2/schedule/get-week-schedules"
)

logger = logging.getLogger(__name__)


def _make_schedule_session() -> requests.Session:
    """Return a session with DHFix adapter for yjapi.cmc.zju.edu.cn."""
    s = requests.Session()
    mount_legacy_ssl(s)
    return s


class TokenExpiredError(Exception):
    """Raised when the ZJU_TOKEN has expired (server returns auth failure)."""
    pass


# ---------------------------------------------------------------------------
# Schedule API: auto-discover live courses
# ---------------------------------------------------------------------------


def fetch_live_courses(token: str) -> list[dict]:
    """
    Return all courses currently live for the authenticated user.

    Decodes user_id and tenant_id from the JWT token, then queries the
    weekly schedule API for today, filtering for live items (sub_status='1').

    Returns a list of dicts: [{"course_id": "...", "title": "..."}]
    """
    # Decode JWT payload (no signature verification needed)
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.b64decode(payload_b64))
        user_id = payload["sub"]
        tenant_id = payload["tenant_id"]
    except Exception as exc:
        raise RuntimeError(f"Failed to decode ZJU_TOKEN JWT: {exc}") from exc

    today = date.today().isoformat()
    try:
        _sched_session = _make_schedule_session()
        resp = _sched_session.get(
            SCHEDULE_API,
            params={
                "user_id": user_id,
                "tenant_id": tenant_id,
                "start_at": today,
                "end_at": today,
                "token": token,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Schedule API request failed: {exc}") from exc

    # Flatten all course items across all days
    raw_result = data.get("result", {})
    # Server returns error string in data field when token is expired
    if isinstance(raw_result.get("data"), str) and "认证失败" in raw_result["data"]:
        raise TokenExpiredError(raw_result["data"])

    live = []
    for day_entry in raw_result.get("list", []):
        for item in day_entry.get("course", []):
            if str(item.get("status", "")) == "1":
                course_id = str(item.get("course_id", ""))
                title = item.get("course_title") or course_id
                if course_id:
                    live.append({"course_id": course_id, "title": title})

    if not live:
        print(f"[debug] Raw schedule response: {json.dumps(data, ensure_ascii=False)[:3000]}")

    return live


# ---------------------------------------------------------------------------
# Phase 1: live URL discovery
# ---------------------------------------------------------------------------


def fetch_live_url(session: requests.Session, course_id: str) -> tuple[str, str] | None:
    """
    Return (m3u8_url, live_sub_id) for the currently live session, or None.

    Two-step process:
    1. Catalogue API  → find the item with status='1' (live), get its sub_id
    2. get-sub-info API → extract data.live_url.output.m3u8
    """
    # Step 1: find live sub_id
    try:
        resp = session.get(CATALOGUE_API, params={"course_id": course_id})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("catalogue API request failed: %s", exc)
        return None

    if not data.get("success") or not data.get("result", {}).get("data"):
        return None

    live_sub_id = None
    for item in data["result"]["data"]:
        if str(item.get("status", "")) == "1":
            live_sub_id = str(item.get("sub_id", item.get("id", "")))
            print(
                f"[monitor] Live item found: sub_id={live_sub_id} title={item.get('title')!r}"
            )
            break

    if not live_sub_id:
        return None

    # Step 2: get live stream URL
    try:
        resp2 = session.get(
            GET_SUB_INFO_API,
            params={"course_id": course_id, "sub_id": live_sub_id},
        )
        resp2.raise_for_status()
        info = resp2.json()
    except Exception as exc:
        logger.error("get-sub-info API request failed: %s", exc)
        return None

    # Detect auth failure from get-sub-info API
    if info.get("code") == 500 and "认证失败" in str(info.get("msg", "")):
        raise TokenExpiredError(info.get("msg", "用户认证失败"))

    try:
        m3u8_url = info["data"]["live_url"]["output"]["m3u8"]
        if m3u8_url:
            return m3u8_url, live_sub_id
    except (KeyError, TypeError):
        pass

    logger.error("live_url.output.m3u8 not found in get-sub-info response: %s", info)
    return None


# ---------------------------------------------------------------------------
# Phase 2: HLS → WAV chunk generator
# ---------------------------------------------------------------------------


def stream_audio_chunks(
    m3u8_url: str,
    output_dir: str,
    chunk_seconds: int = 30,
) -> Generator[str, None, None]:
    """
    Run ffmpeg in the background to segment an HLS live stream into WAV files,
    and yield each completed chunk path as it becomes ready.

    A chunk is considered complete once the *next* numbered chunk file appears
    on disk, or when ffmpeg exits (for the final chunk).

    Cleans up the ffmpeg process on generator close/exception.
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "chunk_%05d.wav")
    ffmpeg_log = os.path.join(output_dir, "ffmpeg.log")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        m3u8_url,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-segment_format",
        "wav",
        pattern,
    ]

    print(f"[monitor] Starting ffmpeg (log → {ffmpeg_log})")
    with open(ffmpeg_log, "wb") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=log_f,
        )

    def _chunk_path(n: int) -> str:
        return os.path.join(output_dir, f"chunk_{n:05d}.wav")

    try:
        n = 0
        wait_ticks = 0
        while True:
            next_path = _chunk_path(n + 1)
            current_path = _chunk_path(n)

            # Wait until next chunk appears (meaning current is fully written)
            while not os.path.exists(next_path):
                if proc.poll() is not None:
                    # ffmpeg exited — show last lines of log for diagnosis
                    try:
                        with open(ffmpeg_log, "r", encoding="utf-8", errors="replace") as lf:
                            tail = lf.read()[-800:]
                        print(f"[monitor] ffmpeg exited (code={proc.returncode}), last log:\n{tail}")
                    except OSError:
                        pass
                    if os.path.exists(current_path):
                        yield current_path
                    return
                wait_ticks += 1
                if wait_ticks % 5 == 0:  # every 10s
                    print(f"[monitor] Waiting for chunk {n} to complete...")
                time.sleep(2)

            wait_ticks = 0
            print(f"[monitor] Chunk {n} ready: {current_path}")
            if os.path.exists(current_path):
                yield current_path

            n += 1

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# Phase 3: keyword detection (pinyin fuzzy match)
# ---------------------------------------------------------------------------


def check_keywords_pinyin(
    text: str,
    keywords: list[str],
    threshold: int = 80,
) -> tuple[str, float] | None:
    """
    Check whether any keyword appears (phonetically) in the transcribed text.

    Uses pypinyin to convert both the text and each keyword to pinyin, then
    rapidfuzz.fuzz.partial_ratio for sub-string matching. This tolerates
    Whisper mis-recognitions caused by regional accents (e.g. "小策" ≈ "小测").

    Returns:
        (keyword, score) for the first match, or None if no match.
    """
    from pypinyin import lazy_pinyin
    from rapidfuzz import fuzz

    text_py = " ".join(lazy_pinyin(text))
    for kw in keywords:
        kw_py = " ".join(lazy_pinyin(kw))
        score = fuzz.partial_ratio(kw_py, text_py)
        if score >= threshold:
            return (kw, score)
    return None


# ---------------------------------------------------------------------------
# Phase 4: LLM semantic confirmation
# ---------------------------------------------------------------------------


def confirm_with_llm(
    text: str,
    api_base: str,
    api_key: str,
    model: str,
    keywords: list[str] | None = None,
    fail_open: bool = True,
    debug: bool = False,
) -> bool:
    """
    Ask an LLM whether the transcription indicates a roll-call or quiz event.

    Args:
        text:      Transcription fragment to analyse
        api_base:  OpenAI-compatible API base URL
        api_key:   API key
        model:     Model name / ID
        keywords:  All configured keywords; LLM checks each one explicitly.
        fail_open: If True, return True (alert) when the API is unavailable.
                   Prefer not to miss an event over a false positive.

    Returns:
        True if the LLM believes an alertable event is occurring.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_base)
        if keywords:
            kw_str = "、".join(keywords)
            prompt = (
                f"以下是课堂录音的转录文字片段：\n\n{text}\n\n"
                f"请判断文字中是否出现或提及了以下任意一项内容：{kw_str}。\n"
                '只要有任意一项被提及（无论老师是否正在执行），就回答"是"；全部未提及才回答"否"。只回答"是"或"否"，不要解释。'
            )
        else:
            prompt = (
                f"以下是课堂录音的转录文字片段：\n\n{text}\n\n"
                '请判断文字中是否出现或提及了点名、考勤或小测相关内容。只回答"是"或"否"，不要解释。'
            )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            timeout=15,
        )
        answer = resp.choices[0].message.content.strip()
        if debug:
            print(f"[debug] LLM response: {answer}")
        return answer.startswith("是")

    except Exception as exc:
        logger.error("LLM confirmation failed: %s", exc)
        if fail_open:
            logger.warning(
                "fail_open=True — treating as confirmed to avoid missing event"
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Playback-generating status check
# ---------------------------------------------------------------------------


def is_stream_ended(session: requests.Session, course_id: str, live_sub_id: str) -> bool:
    """
    Return True if the monitored sub_id has ended.

    Only checks the specific sub_id that was live when monitoring started,
    so other already-finished items in the same course don't cause false exits.
    Status values: '1' = live, '2' = playback ready, '3' = playback generating.
    """
    try:
        resp = session.get(CATALOGUE_API, params={"course_id": course_id})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("catalogue API request failed during status check: %s", exc)
        return False

    if not data.get("success") or not data.get("result", {}).get("data"):
        return False

    for item in data["result"]["data"]:
        if str(item.get("sub_id", item.get("id", ""))) != live_sub_id:
            continue
        status = str(item.get("status", ""))
        title = item.get("title", "")
        if status == "3":
            print(f"[monitor] Item '{title}' status=3 (回放生成中)")
            return True
        if status == "2":
            print(f"[monitor] Item '{title}' status=2 (回放已就绪)")
            return True
        for field in ("title", "description", "status_text"):
            if "回放" in str(item.get(field, "")):
                print(f"[monitor] Item '{title}' contains '回放' in {field!r}")
                return True

    return False





def _build_message(
    keyword: str,
    course_id: str,
    course_title: str,
    recent_entries: list[str],
    llm_analysis: str,
) -> str:
    now = datetime.now().strftime("%H:%M:%S")
    title_str = f"{course_title}（{course_id}）" if course_title else course_id
    recent_str = "\n".join(recent_entries[-3:])
    return (
        f"[智云直播监控] 触发关键词：{keyword}\n"
        f"课程：{title_str}\n"
        f"时间：{now}\n"
        f"\n分析：{llm_analysis}\n"
        f"\n最近转录：\n{recent_str}"
    )


# ---------------------------------------------------------------------------
# LLM context analysis
# ---------------------------------------------------------------------------


def analyze_context_with_llm(
    recent_entries: list[str],
    keywords: list[str],
    api_base: str,
    api_key: str,
    model: str,
    debug: bool = False,
) -> str:
    """
    Given the last few transcript chunks, ask the LLM for a brief description
    of what alertable event is occurring and how it relates to the keywords.

    Returns a short Chinese summary string (1-2 sentences).
    Falls back to a plain string on API failure.
    """
    context = "\n".join(recent_entries[-3:])
    kw_str = "、".join(keywords)
    prompt = (
        f"以下是课堂录音的最近几段转录文字（每段前有时间戳）：\n\n{context}\n\n"
        f"请判断老师是否在宣布以下任一内容：{kw_str}。\n"
        '如果是，用1-2句话简要说明检测到的具体内容。如果否，回答"未检测到相关内容"。'
    )
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_base)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0,
            timeout=15,
        )
        answer = resp.choices[0].message.content.strip()
        if debug:
            print(f"[debug] LLM analysis: {answer}")
        return answer
    except Exception as exc:
        logger.error("LLM analysis failed: %s", exc)
        return "（LLM分析失败）"


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------


def monitor_loop(
    session: requests.Session,
    course_id: str,
    keywords: list[str],
    chunk_seconds: int,
    model_size: str,
    notifier_config: dict,
    llm_config: dict,
    poll_interval: int = 5,
    chunks_dir: str = "chunks",
    log_dir: str = "logs",
    course_title: str = "",
    debug: bool = False,
    credentials: tuple[str, str] | None = None,
) -> None:
    """
    Full monitoring pipeline:

    1. Poll catalogue API until a live HLS URL is found.
    2. Load the Whisper model once.
    3. For each 30-second audio chunk:
       a. Transcribe with the pre-loaded model.
       b. Run pinyin fuzzy keyword match.
       c. On match, confirm with LLM (with 120-second cooldown between alerts).
       d. On confirmation, send DingTalk notification.
       e. Delete the chunk to save disk space.
    """
    from faster_whisper import WhisperModel
    from src.transcriber import transcribe_with_model
    from src.notifier import send_dingtalk

    # --- Phase 1: wait for live stream ---
    print(f"[monitor] Waiting for live stream (course_id={course_id})...")
    live_url = None
    refresh_attempts = 0
    MAX_REFRESH_ATTEMPTS = 3
    while live_url is None:
        try:
            result = fetch_live_url(session, course_id)
        except TokenExpiredError as exc:
            if credentials and refresh_attempts < MAX_REFRESH_ATTEMPTS:
                refresh_attempts += 1
                print(f"[monitor] Token expired in Phase 1, refreshing... (attempt {refresh_attempts}/{MAX_REFRESH_ATTEMPTS})")
                from src.auth import refresh_token
                session, _ = refresh_token(*credentials)
                continue
            else:
                raise
        if result is None:
            print(f"[monitor] No live stream found, retrying in {poll_interval}s...")
            time.sleep(poll_interval)
        else:
            live_url, live_sub_id = result

    print(f"[monitor] Live stream detected: {live_url} (sub_id={live_sub_id})")

    # --- Phase 2: load model once ---
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[monitor] Loading Whisper model: {model_size} (device={device})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("[monitor] Model ready. Starting chunk processing...")

    # --- Phase 3: process chunks ---
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{course_id}_{date.today().isoformat()}.txt")
    print(f"[monitor] Transcript log → {log_path}")

    recent_entries: deque[str] = deque(maxlen=5)  # rolling last-5-chunks buffer
    last_alert_time = 0.0
    consecutive_empty = 0
    EMPTY_THRESHOLD = 5  # consecutive silent chunks before checking live status
    last_end_check_time = 0.0
    END_CHECK_INTERVAL = 60.0  # poll is_stream_ended every 60 seconds

    while True:
        # Check if stream is still live
        result = fetch_live_url(session, course_id)
        if result is None:
            print("[monitor] Stream is no longer live (status changed). Exiting.")
            break

        current_url, current_sub_id = result
        if current_sub_id != live_sub_id:
            print(f"[monitor] Stream is no longer live (status changed). Exiting.")
            break
        if current_url != live_url:
            print(f"[monitor] Live URL refreshed (auth_key updated)")
            live_url = current_url

        print(f"[monitor] Processing stream chunks...")
        try:
            for chunk_path in stream_audio_chunks(live_url, chunks_dir, chunk_seconds):
                try:
                    segments = transcribe_with_model(model, chunk_path, language="zh")
                    full_text = " ".join(seg.text for seg in segments)

                    # Periodic end-of-stream check every 60s regardless of content
                    now = time.time()
                    if now - last_end_check_time >= END_CHECK_INTERVAL:
                        last_end_check_time = now
                        if is_stream_ended(session, course_id, live_sub_id):
                            print("[monitor] Periodic check: stream ended. Stopping monitor.")
                            return

                    if not full_text.strip():
                        consecutive_empty += 1
                        if consecutive_empty >= EMPTY_THRESHOLD:
                            print(
                                f"[monitor] {consecutive_empty} consecutive empty chunks — "
                                "checking if stream ended..."
                            )
                            if is_stream_ended(session, course_id, live_sub_id):
                                print("[monitor] Stream ended. Stopping monitor.")
                                return
                            consecutive_empty = 0  # reset after check
                        continue

                    consecutive_empty = 0

                    # Log every chunk permanently
                    ts = datetime.now().strftime("%H:%M:%S")
                    entry = f"[{ts}] {full_text}"
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(entry + "\n")
                    recent_entries.append(entry)

                    if debug:
                        print(f"[debug] {full_text}")

                    result = check_keywords_pinyin(full_text, keywords)
                    if result is None:
                        continue

                    kw, score = result
                    now = time.time()

                    if now - last_alert_time < 120:
                        print(
                            f"[monitor] Keyword '{kw}' matched (score={score:.0f}) but in cooldown, skipping"
                        )
                        continue

                    print(
                        f"[monitor] Keyword '{kw}' matched (score={score:.0f}), confirming with LLM..."
                    )
                    # Skip LLM if keyword appears verbatim in the transcription
                    confirmed = kw in full_text or confirm_with_llm(
                        full_text, **llm_config, keywords=keywords, debug=debug
                    )
                    if kw in full_text:
                        print(f"[monitor] Keyword '{kw}' found verbatim, skipping LLM")
                    if confirmed:
                        analysis = analyze_context_with_llm(
                            list(recent_entries), keywords, debug=debug, **llm_config
                        )
                        message = _build_message(
                            keyword=kw,
                            course_id=course_id,
                            course_title=course_title,
                            recent_entries=list(recent_entries),
                            llm_analysis=analysis,
                        )
                        at_mobiles = notifier_config.get("at_mobiles") or []
                        ok = send_dingtalk(
                            webhook=notifier_config["webhook"],
                            secret=notifier_config["secret"],
                            message=message,
                            at_mobiles=at_mobiles,
                        )
                        if ok:
                            print(f"[monitor] Alert sent for keyword '{kw}'")
                            last_alert_time = now
                        else:
                            print(f"[monitor] Alert delivery failed for keyword '{kw}'")
                    else:
                        print(f"[monitor] LLM did not confirm keyword '{kw}', skipping alert")

                except Exception as exc:
                    logger.error("Error processing chunk %s: %s", chunk_path, exc)

                finally:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

        except Exception as exc:
            logger.error("Stream processing error: %s", exc)

        # ffmpeg exited — check if stream is still live before restarting
        print("[monitor] ffmpeg stopped, checking if stream is still active...")
        time.sleep(5)  # brief pause before retry
