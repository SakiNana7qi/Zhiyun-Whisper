"""
Output formatting for transcription results.

Generates:
- .txt  — plain text, one paragraph per segment
- .srt  — SubRip subtitle format with timestamps
"""

from __future__ import annotations

import os
import re

from src.transcriber import Segment


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip()


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_txt(
    segments: list[Segment],
    title: str,
    output_dir: str = "output",
) -> str:
    """
    Save transcription as plain text.

    Returns the output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = _sanitize_filename(title)
    path = os.path.join(output_dir, f"{safe_name}.txt")

    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            if seg.text:
                f.write(seg.text + "\n")

    return path


def save_srt(
    segments: list[Segment],
    title: str,
    output_dir: str = "output",
) -> str:
    """
    Save transcription as SRT subtitle file.

    Format:
        1
        00:00:01,500 --> 00:00:04,200
        这是第一段文字

    Returns the output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = _sanitize_filename(title)
    path = os.path.join(output_dir, f"{safe_name}.srt")

    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            if not seg.text:
                continue
            start_ts = _format_srt_time(seg.start)
            end_ts = _format_srt_time(seg.end)
            f.write(f"{i}\n")
            f.write(f"{start_ts} --> {end_ts}\n")
            f.write(f"{seg.text}\n\n")

    return path
