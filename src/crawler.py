"""
Video URL extraction and audio download from Zhiyun Classroom.

Uses the course catalogue API to discover video playback URLs,
then downloads audio via ffmpeg.
"""

import json
import os
import re
import subprocess
import shutil
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

import requests

CATALOGUE_API = "https://classroom.zju.edu.cn/courseapi/v2/course/catalogue"


@dataclass
class Lesson:
    """Represents a single lesson/recording in a course."""

    sub_id: str
    title: str
    video_url: str | None


def parse_url(url: str) -> dict[str, str]:
    """
    Extract course_id, sub_id, tenant_code from a livingroom or coursedetail URL.

    Supports:
      - https://classroom.zju.edu.cn/livingroom?course_id=81771&sub_id=1892675&tenant_code=112
      - https://classroom.zju.edu.cn/coursedetail?course_id=81771&tenant_code=112
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    result = {}
    if "course_id" in params:
        result["course_id"] = params["course_id"][0]
    if "sub_id" in params:
        result["sub_id"] = params["sub_id"][0]
    if "tenant_code" in params:
        result["tenant_code"] = params["tenant_code"][0]

    if "course_id" not in result:
        raise ValueError(f"Cannot extract course_id from URL: {url}")

    return result


def fetch_lessons(session: requests.Session, course_id: str) -> list[Lesson]:
    """
    Fetch all lessons for a course from the catalogue API.

    Returns a list of Lesson objects (video_url may be None if no playback).
    """
    resp = session.get(CATALOGUE_API, params={"course_id": course_id})
    resp.raise_for_status()
    data = resp.json()

    if not data.get("success") or not data.get("result", {}).get("data"):
        raise RuntimeError(
            f"Failed to fetch course catalogue (course_id={course_id}): {data}"
        )

    lessons = []
    for item in data["result"]["data"]:
        title = item.get("title", "untitled")
        sub_id = str(item.get("sub_id", item.get("id", "")))
        video_url = None

        content_str = item.get("content", "")
        if content_str:
            try:
                content = json.loads(content_str)
                if content.get("playback", {}).get("url"):
                    video_url = content["playback"]["url"]
                elif content.get("url"):
                    video_url = content["url"]
            except (json.JSONDecodeError, TypeError):
                pass

        lessons.append(Lesson(sub_id=sub_id, title=title, video_url=video_url))

    return lessons


def _sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in file paths."""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def _check_ffmpeg():
    """Verify ffmpeg is available on PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. " "Install it: https://ffmpeg.org/download.html"
        )


def _download_video(
    video_url: str,
    output_path: str,
    max_retries: int = 10,
    timeout: int = 30,
) -> None:
    """
    Stream-download a video file with resume support and automatic retry.

    Uses HTTP Range headers to resume from where a previous attempt left off,
    so partial downloads from connection drops are not wasted.
    """
    # Check how much we already have (supports resuming across runs too)
    downloaded = 0
    if os.path.exists(output_path):
        downloaded = os.path.getsize(output_path)

    # Get total file size
    head = requests.head(video_url, allow_redirects=True, timeout=timeout)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    total_mb = total / (1024 * 1024) if total else 0

    if downloaded >= total and total > 0:
        print(f"  Video already fully downloaded ({total_mb:.0f} MB)")
        return

    if downloaded > 0:
        print(
            f"  Resuming download from {downloaded / (1024*1024):.0f}/{total_mb:.0f} MB"
        )

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            resp = requests.get(
                video_url,
                stream=True,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()

            mode = "ab" if downloaded > 0 else "wb"
            with open(output_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        dl_mb = downloaded / (1024 * 1024)
                        print(
                            f"\r  Progress: {dl_mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )

            print()
            return  # success

        except (requests.exceptions.RequestException, IOError) as e:
            # Update downloaded to actual file size on disk
            if os.path.exists(output_path):
                downloaded = os.path.getsize(output_path)
            dl_mb = downloaded / (1024 * 1024)

            if attempt < max_retries:
                wait = min(2**attempt, 30)
                print(
                    f"\n  Connection lost at {dl_mb:.0f} MB. "
                    f"Retry {attempt}/{max_retries} in {wait}s..."
                )
                import time

                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Download failed after {max_retries} retries "
                    f"({dl_mb:.0f}/{total_mb:.0f} MB downloaded): {e}"
                ) from e


def _extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from a local video file as 16kHz mono WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    # Use Popen to stream stderr so progress is visible and doesn't buffer
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr_lines = []
    for line in proc.stderr:
        stderr_lines.append(line)
        print(f"\r  {line.rstrip()}", end="", flush=True)
    proc.wait()
    print()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit code {proc.returncode}):\n{''.join(stderr_lines[-20:])}"
        )


def download_audio(
    video_url: str,
    title: str,
    output_dir: str = "output",
    cookies: dict | None = None,
) -> str:
    """
    Download video and extract audio as WAV.

    Strategy: stream-download the full MP4 first (with progress),
    then use ffmpeg locally to extract audio. This avoids ffmpeg's
    issues with seeking in large remote MP4 files.

    Args:
        video_url: Direct URL to the video (mp4)
        title: Lesson title (used for the output filename)
        output_dir: Directory to save the audio file
        cookies: Optional cookies dict (currently unused, video URLs are public)

    Returns:
        Path to the downloaded audio WAV file
    """
    _check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    safe_name = _sanitize_filename(title)
    audio_path = os.path.join(output_dir, f"{safe_name}.wav")
    video_path = os.path.join(output_dir, f"{safe_name}.mp4")

    if os.path.exists(audio_path):
        print(f"  Audio already exists: {audio_path}")
        return audio_path

    # Step 1: Download video (resume-aware, checks completeness)
    print(f"  Downloading video: {title}")
    _download_video(video_url, video_path)
    print(f"  Video ready: {video_path}")

    # Step 2: Extract audio
    print(f"  Extracting audio...")
    _extract_audio(video_path, audio_path)
    print(f"  Audio saved: {audio_path}")

    return audio_path
