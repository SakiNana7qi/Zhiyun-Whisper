"""
Evidence-grounded lecture summarization utilities.

The pipeline is intentionally two-stage:
1. Select the most information-rich transcript lines as candidate evidence.
2. Ask an OpenAI-compatible LLM to synthesize a concise course summary
   strictly from those lines, with timestamps and verbatim evidence.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_TIMESTAMPED_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")

_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "就",
    "都",
    "而",
    "及",
    "与",
    "或",
    "一个",
    "这个",
    "那个",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "然后",
    "但是",
    "所以",
    "因为",
    "如果",
    "就是",
    "不是",
    "没有",
    "可以",
    "这样",
    "进行",
    "一下",
    "还有",
    "以及",
    "或者",
    "大家",
    "同学",
    "老师",
    "现在",
}

_ASCII_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "be",
    "it",
    "this",
    "that",
    "hello",
    "world",
}


@dataclass(frozen=True)
class TranscriptLine:
    """A single transcript line, optionally with a timestamp."""

    timestamp: str | None
    text: str


@dataclass(frozen=True)
class EvidenceLine:
    """A transcript line selected as useful evidence for summarization."""

    timestamp: str | None
    text: str


@dataclass(frozen=True)
class SummaryPoint:
    """One important point from the lecture, with supporting evidence."""

    point: str
    evidence: list[EvidenceLine]


@dataclass(frozen=True)
class SummaryResult:
    """Structured lecture summary output."""

    source_path: str
    line_count: int
    evidence_count: int
    title: str
    one_sentence_summary: str
    key_points: list[SummaryPoint]
    fallback_note: str | None = None


def load_transcript(path: str) -> list[TranscriptLine]:
    """Load transcript lines from a transcript file or live-monitor log."""
    transcript_lines: list[TranscriptLine] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = _TIMESTAMPED_LINE_RE.match(line)
            if match:
                timestamp, text = match.groups()
                text = text.strip()
                if text:
                    transcript_lines.append(TranscriptLine(timestamp=timestamp, text=text))
                continue

            transcript_lines.append(TranscriptLine(timestamp=None, text=line))

    return transcript_lines


def resolve_log_path(course_id: str, log_dir: str = "logs", date_str: str | None = None) -> str:
    """Resolve the default live-monitor log path for a course and date."""
    if date_str is None:
        from datetime import date

        date_str = date.today().isoformat()
    return os.path.join(log_dir, f"{course_id}_{date_str}.txt")


def _normalize_term(term: str) -> str:
    return term.strip().lower()


def _extract_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text):
        normalized = _normalize_term(token)
        if not normalized:
            continue

        if normalized.isascii():
            if len(normalized) < 2 or normalized in _ASCII_STOPWORDS:
                continue
            terms.append(normalized)
            continue

        if normalized in _STOPWORDS:
            continue
        if len(normalized) <= 2:
            terms.append(normalized)
            continue

        terms.append(normalized)
        for i in range(len(normalized) - 1):
            bigram = normalized[i : i + 2]
            if bigram not in _STOPWORDS:
                terms.append(bigram)

    return terms


def _score_line(text: str, document_frequency: Counter[str], total_docs: int) -> float:
    terms = _extract_terms(text)
    if not terms:
        return 0.0

    unique_terms = set(terms)
    score = 0.0
    for term in unique_terms:
        df = document_frequency.get(term, 0)
        if df <= 0:
            continue
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        score += idf * (1.0 + min(len(term), 8) / 8.0)

    score += min(len(text) / 80.0, 1.5)
    return score


def _select_candidate_lines(
    transcript_lines: list[TranscriptLine],
    max_candidates: int = 18,
) -> list[EvidenceLine]:
    filtered_lines = [line for line in transcript_lines if line.text.strip()]
    if not filtered_lines:
        return []

    document_frequency: Counter[str] = Counter()
    for line in filtered_lines:
        document_frequency.update(set(_extract_terms(line.text)))

    scored_lines: list[tuple[float, int, TranscriptLine]] = []
    for index, line in enumerate(filtered_lines):
        score = _score_line(line.text, document_frequency, len(filtered_lines))
        scored_lines.append((score, index, line))

    selected: list[EvidenceLine] = []
    seen_texts: set[str] = set()
    for score, _, line in sorted(scored_lines, key=lambda item: (-item[0], item[1])):
        if score <= 0:
            continue
        normalized_text = line.text.strip()
        if normalized_text in seen_texts:
            continue
        selected.append(EvidenceLine(timestamp=line.timestamp, text=normalized_text))
        seen_texts.add(normalized_text)
        if len(selected) >= max_candidates:
            break

    return selected


def _format_evidence_lines(lines: list[EvidenceLine]) -> str:
    formatted: list[str] = []
    for line in lines:
        if line.timestamp:
            formatted.append(f"[{line.timestamp}] {line.text}")
        else:
            formatted.append(line.text)
    return "\n".join(formatted)


def _clean_llm_text(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _parse_plaintext_summary(payload: str, fallback_candidates: list[EvidenceLine]) -> SummaryResult:
    text = _clean_llm_text(payload)
    if not text:
        raise ValueError("LLM returned empty content")

    lines = [line.rstrip() for line in text.splitlines()]

    title = "课程总结"
    one_sentence_summary = ""
    key_points: list[SummaryPoint] = []
    current_point: SummaryPoint | None = None
    in_key_points = False
    in_evidence = False

    def flush_current_point() -> None:
        nonlocal current_point, key_points
        if current_point and current_point.point.strip():
            key_points.append(current_point)
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
            in_evidence = False
            continue
        if line.startswith("原文证据"):
            flush_current_point()
            in_key_points = False
            in_evidence = True
            continue

        if in_key_points:
            point_match = re.match(r"^(?:\d+[\.、]|[-•])\s*(.+)$", line)
            if point_match:
                flush_current_point()
                current_point = SummaryPoint(point=point_match.group(1).strip(), evidence=[])
            elif current_point and line.startswith("- "):
                # Ignore stray bullets inside the section.
                continue
            continue

        if in_evidence:
            if line.startswith("【") and line.endswith("】"):
                flush_current_point()
                current_point = SummaryPoint(point=line.strip("【】"), evidence=[])
                continue

            evidence_match = re.match(r"^[-*]\s*(?:\[(\d{2}:\d{2}:\d{2})\]\s*)?(.+)$", line)
            if evidence_match and current_point:
                timestamp = evidence_match.group(1)
                evidence_text = evidence_match.group(2).strip()
                if evidence_text:
                    current_point = SummaryPoint(
                        point=current_point.point,
                        evidence=current_point.evidence
                        + [EvidenceLine(timestamp=timestamp, text=evidence_text)],
                    )

    flush_current_point()

    if not key_points and fallback_candidates:
        key_points = [
            SummaryPoint(
                point="未能从模型输出中提取稳定的要点",
                evidence=[],
            )
        ]

    if not one_sentence_summary:
        one_sentence_summary = "未能生成一句话总结。"

    return SummaryResult(
        source_path="",
        line_count=len(fallback_candidates),
        evidence_count=0,
        title=title,
        one_sentence_summary=one_sentence_summary,
        key_points=key_points,
        fallback_note=None,
    )


def summarize_with_llm(
    transcript_lines: list[TranscriptLine],
    api_base: str,
    api_key: str,
    model: str,
    max_candidates: int = 18,
    max_points: int = 5,
) -> SummaryResult:
    """Summarize transcript lines by selecting evidence and asking an LLM."""
    from openai import OpenAI

    filtered_lines = [line for line in transcript_lines if line.text.strip()]
    candidates = [EvidenceLine(timestamp=line.timestamp, text=line.text.strip()) for line in filtered_lines]

    if not candidates:
        return SummaryResult(
            source_path="",
            line_count=0,
            evidence_count=0,
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
        f"{_format_evidence_lines(candidates)}"
    )

    client = OpenAI(api_key=api_key, base_url=api_base)
    request_kwargs = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8192,
        "timeout": 60,
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

    summary = _parse_plaintext_summary(raw, fallback_candidates=candidates)

    key_points = summary.key_points[:max_points]

    return SummaryResult(
        source_path="",
        line_count=len(filtered_lines),
        evidence_count=0,
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
    max_candidates: int = 18,
    max_points: int = 5,
) -> tuple[SummaryResult, str]:
    """Load a transcript file and return both structured and rendered summaries."""
    transcript_lines = load_transcript(path)
    summary = summarize_with_llm(
        transcript_lines,
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_candidates=max_candidates,
        max_points=max_points,
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
        