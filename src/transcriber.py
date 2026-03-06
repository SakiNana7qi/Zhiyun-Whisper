"""
Speech-to-text transcription module.

Supports two backends:
- local: faster-whisper (CTranslate2, GPU/CPU)
- api:   OpenAI Whisper API
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass


@dataclass
class Segment:
    """A transcribed text segment with timestamps."""
    start: float  # seconds
    end: float    # seconds
    text: str


def transcribe_local(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = "zh",
) -> list[Segment]:
    """
    Transcribe audio using faster-whisper (local model).

    Args:
        audio_path: Path to audio file (WAV recommended)
        model_size: Whisper model size (tiny/base/small/medium/large-v3)
        device: "auto", "cuda", or "cpu"
        language: Language code for transcription

    Returns:
        List of Segment objects with timestamps and text
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    print(f"  Loading model: {model_size} (device={device}, compute={compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"  Transcribing: {audio_path}")
    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    from tqdm import tqdm

    duration = info.duration
    print(f"  Detected language: {info.language} (prob={info.language_probability:.2f})")
    print(f"  Audio duration: {duration / 60:.1f} min")

    total_sec = int(duration)
    pbar = tqdm(
        total=total_sec,
        unit="s",
        desc="  Transcribing",
        bar_format="  {desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    )

    segments = []
    last_pos = 0
    for seg in segments_gen:
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
        advance = min(seg.end, duration) - last_pos
        if advance > 0:
            pbar.update(advance)
            last_pos = min(seg.end, duration)

    pbar.update(total_sec - last_pos)
    pbar.close()

    print(f"  Transcription complete: {len(segments)} segments")
    return segments


# OpenAI Whisper API has a 25 MB file size limit
_API_MAX_FILE_SIZE = 25 * 1024 * 1024


def _split_audio_for_api(audio_path: str, max_size: int = _API_MAX_FILE_SIZE) -> list[str]:
    """
    Split a large audio file into chunks under max_size bytes.
    Returns list of chunk file paths.
    """
    import subprocess
    import shutil

    file_size = os.path.getsize(audio_path)
    if file_size <= max_size:
        return [audio_path]

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for splitting large audio files")

    # 16kHz mono 16-bit WAV = 32000 bytes/sec
    bytes_per_sec = 32000
    chunk_duration = max(60, int(max_size / bytes_per_sec))
    total_duration_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    total_duration = float(total_duration_result.stdout.strip())
    num_chunks = math.ceil(total_duration / chunk_duration)

    chunks = []
    base, ext = os.path.splitext(audio_path)
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = f"{base}_chunk{i:03d}{ext}"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(start_time), "-t", str(chunk_duration),
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             chunk_path],
            capture_output=True, text=True,
        )
        chunks.append(chunk_path)

    return chunks


def transcribe_api(
    audio_path: str,
    api_key: str,
    language: str = "zh",
) -> list[Segment]:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to audio file
        api_key: OpenAI API key
        language: Language code

    Returns:
        List of Segment objects with timestamps and text
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    chunks = _split_audio_for_api(audio_path)

    all_segments = []
    time_offset = 0.0

    for i, chunk_path in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Transcribing chunk {i + 1}/{len(chunks)}: {chunk_path}")
        else:
            print(f"  Transcribing via API: {audio_path}")

        with open(chunk_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                all_segments.append(Segment(
                    start=seg["start"] + time_offset,
                    end=seg["end"] + time_offset,
                    text=seg["text"].strip(),
                ))
        elif hasattr(response, "text") and response.text:
            all_segments.append(Segment(
                start=time_offset,
                end=time_offset + 30.0,
                text=response.text.strip(),
            ))

        if len(chunks) > 1 and chunk_path != audio_path:
            # Calculate offset for next chunk from file duration
            import subprocess
            dur_result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", chunk_path],
                capture_output=True, text=True,
            )
            time_offset += float(dur_result.stdout.strip())

    # Clean up chunk files
    for chunk_path in chunks:
        if chunk_path != audio_path and os.path.exists(chunk_path):
            os.remove(chunk_path)

    print(f"  Transcription complete: {len(all_segments)} segments")
    return all_segments
