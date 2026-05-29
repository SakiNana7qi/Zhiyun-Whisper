"""
Lecture summarization utilities.

Passes the complete transcript to an OpenAI-compatible LLM to synthesize a concise course summary.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SummaryPoint:
    """One important point from the lecture."""
    point: str


@dataclass(frozen=True)
class SummaryResult:
    """Structured lecture summary output."""
    source_path: str
    line_count: int
    title: str
    one_sentence_summary: str
    key_points: list[SummaryPoint]
    fallback_note: str | None = None


def load_transcript_text(path: str) -> str:
    """Load transcript as a single string."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def resolve_log_path(course_id: str, log_dir: str = "logs", date_str: str | None = None) -> str:
    """Resolve the default live-monitor log path for a course and date."""
    if date_str is None:
        from datetime import date

        date_str = date.today().isoformat()
    return os.path.join(log_dir, f"{course_id}_{date_str}.txt")


def _clean_llm_text(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _parse_plaintext_summary(payload: str) -> SummaryResult:
    text = _clean_llm_text(payload)
    if not text:
        raise ValueError("LLM returned empty content")

    lines = [line.rstrip() for line in text.splitlines()]

    title = "课程总结"
    one_sentence_summary = ""
    key_points: list[SummaryPoint] = []
    current_point: str | None = None
    in_key_points = False

    def flush_current_point() -> None:
        nonlocal current_point, key_points
        if current_point and current_point.strip():
            key_points.append(SummaryPoint(point=current_point))
        current_point = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("课程主题："):
            title = line.split("：", 1)[1].strip() or title
            continue
        if line.startswith("一句话总结："):
            one_sentence_summary = line.split("：", 1)[1].strip()
            continue
        if line.startswith("这节课讲了什么"):
            flush_current_point()
            in_key_points = True
            continue

        if in_key_points:
            point_match = re.match(r"^(?:\d+[\.、]|[-•])\s*(.+)$", line)
            if point_match:
                flush_current_point()
                current_point = point_match.group(1).strip()
            continue

    flush_current_point()

    if not key_points:
        key_points = [SummaryPoint(point="未能从模型输出中提取稳定的要点")]

    if not one_sentence_summary:
        one_sentence_summary = "未能生成一句话总结。"

    return SummaryResult(
        source_path="",
        line_count=0,
        title=title,
        one_sentence_summary=one_sentence_summary,
        key_points=key_points,
        fallback_note=None,
    )


def summarize_with_llm(
    transcript_text: str,
    api_base: str,
    api_key: str,
    model: str,
    max_points: int = 5,
) -> SummaryResult:
    """Summarize the entire transcript using an LLM."""
    from openai import OpenAI

    if not transcript_text:
        return SummaryResult(
            source_path="",
            line_count=0,
            title="未能提取到足够内容",
            one_sentence_summary="未能从转录中提取到足够可靠的课程内容。",
            key_points=[],
            fallback_note="输入里没有可用的转录内容。",
        )

    prompt = (
        "你是课堂纪要助手。请根据下面的转录内容理解课程重点并生成总结。\n"
        "要求：\n"
        "1. 只输出纯文本，不要引用原文句子。\n"
        "2. 结构固定为：\n"
        "课程主题：...\n"
        "一句话总结：...\n"
        "这节课讲了什么\n"
        "1. ...\n"
        "2. ...\n"
        "3. ...\n"
        "4. ...\n"
        "5. ...\n"
        "3. 不确定的地方写“不确定”。\n"
        "4. 重点放在知识点、步骤、结论、例子、作业要求。\n\n"
        "转录内容：\n"
        f"{transcript_text}"
    )

    client = OpenAI(api_key=api_key, base_url=api_base)
    request_kwargs = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8192,
        "timeout": 120,
    }

    resp = client.chat.completions.create(
        **request_kwargs,
        messages=[
            {"role": "system", "content": "你是一个认真理解转录内容的课堂总结助手。"},
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content or ""
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    finish_reason = getattr(choice, "finish_reason", None)
    refusal = getattr(getattr(choice, "message", None), "refusal", None)
    if not raw.strip():
        raise RuntimeError(
            f"LLM returned empty content. finish_reason={finish_reason!r}, refusal={refusal!r}."
        )

    summary = _parse_plaintext_summary(raw)
    
    key_points = summary.key_points[:max_points]

    return SummaryResult(
        source_path="",
        line_count=len(transcript_text.splitlines()),
        title=summary.title,
        one_sentence_summary=summary.one_sentence_summary,
        key_points=key_points,
        fallback_note=summary.fallback_note,
    )


def summarize_file(
    path: str,
    api_base: str,
    api_key: str,
    model: str,
    max_candidates: int = 18,  # Kept for CLI compatibility in main.py
    max_points: int = 5,
) -> tuple[SummaryResult, str]:
    """Load a transcript file and return both structured and rendered summaries."""
    transcript_text = load_transcript_text(path)
    summary = summarize_with_llm(
        transcript_text,
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_points=max_points,
    )
    
    summary = SummaryResult(
        source_path=path,
        line_count=summary.line_count,
        title=summary.title,
        one_sentence_summary=summary.one_sentence_summary,
        key_points=summary.key_points,
        fallback_note=summary.fallback_note,
    )
    
    rendered = render_summary_markdown(summary, path)
    return summary, rendered


def render_summary_markdown(summary: SummaryResult, source_path: str) -> str:
    """Render a summary as markdown for easy copy/paste or file output."""
    lines: list[str] = []
    lines.append(f"# {summary.title}")
    lines.append("")
    lines.append(f"- 来源：{source_path}")
    lines.append(f"- 参与总结的转录行数：{summary.line_count}")
    if summary.one_sentence_summary:
        lines.append(f"- 一句话概括：{summary.one_sentence_summary}")

    lines.append("")
    lines.append("## 这节课讲了什么")
    if summary.key_points:
        for point in summary.key_points:
            lines.append(f"- {point.point}")
    elif summary.fallback_note:
        lines.append(f"- {summary.fallback_note}")
    else:
        lines.append("- 未能生成稳定要点")

    lines.append("")
    lines.append("## 说明")
    lines.append("- 这个总结是基于转录内容的理解与归纳，不直接引用原文。")
    lines.append("- 如果某些内容看不清，会保留为不确定，而不是猜测。")

    return "\n".join(lines).rstrip() + "\n"


def write_summary(path: str, content: str) -> None:
    """Write summary content to a file, creating parent directories if needed."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
