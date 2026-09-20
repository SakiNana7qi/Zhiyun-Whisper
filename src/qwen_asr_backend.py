"""Qwen3-ASR adapter using the official qwen-asr Transformers backend."""

from __future__ import annotations

import wave

from src.transcriber import Segment


_LANGUAGES = dict(zip(
    "zh en yue ar de fr es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs fil fa el hu mk ro".split(),
    ("Chinese English Cantonese Arabic German French Spanish Portuguese Indonesian Italian "
     "Korean Russian Thai Vietnamese Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish "
     "Polish Czech Filipino Persian Greek Hungarian Macedonian Romanian").split(),
))
_LANGUAGES.update({"zh-cn": "Chinese", "zh-tw": "Chinese"})


def _language_name(language: str | None) -> str | None:
    if not language or language.lower() == "auto":
        return None
    return _LANGUAGES.get(language.lower(), language.capitalize())


def _aligned_segments(text: str, time_stamps) -> list[Segment]:
    """Group aligned words into captions while preserving original punctuation.

    The aligner strips punctuation from its tokens. Map those tokens back to
    the transcript rather than joining them with spaces (which breaks Chinese).
    Times already include the SDK's offsets for long recordings.
    """
    items = list(time_stamps) if time_stamps is not None else []
    if not items:
        raise RuntimeError("Qwen3-ASR returned text without alignment timestamps")

    positions = [i for i, char in enumerate(text) if char.isalnum() or char == "'"]
    normalized = "".join(text[i] for i in positions)
    token_cursor = text_cursor = 0
    segments = []
    parts = []
    start = 0.0
    for index, item in enumerate(items):
        word = "".join(char for char in item.text if char.isalnum() or char == "'")
        if not word or not normalized.startswith(word, token_cursor):
            raise RuntimeError("Qwen3-ASR alignment does not match the transcription")
        token_cursor += len(word)
        boundary = positions[token_cursor] if token_cursor < len(positions) else len(text)
        part = text[text_cursor:boundary]
        text_cursor = boundary
        if not parts:
            start = float(item.start_time)
        parts.append(part)
        end = float(item.end_time)
        gap = index + 1 < len(items) and items[index + 1].start_time - end >= 1.0
        if (any(char in part for char in "。！？!?；;\n.") or gap
                or end - start >= 8.0 or sum(map(len, parts)) >= 42
                or index == len(items) - 1):
            segments.append(Segment(start, end, "".join(parts).strip()))
            parts = []
    if token_cursor != len(normalized):
        raise RuntimeError("Qwen3-ASR alignment is missing part of the transcription")
    return segments


class QwenTranscriber:
    """Load once and reuse for recordings or successive live WAV chunks."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        batch_size: int = 1,
        return_timestamps: bool = True,
    ):
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        kwargs = dict(
            dtype=dtype, device_map=device,
            max_inference_batch_size=batch_size,
            # The SDK splits long recordings; allow enough tokens per audio chunk.
            max_new_tokens=4096,
        )
        if return_timestamps:
            kwargs.update(
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs={"dtype": dtype, "device_map": device},
            )
        print(f"  Loading Qwen3-ASR: {model_id} (device={device}, batch_size={batch_size})")
        if return_timestamps:
            print("  Loading Qwen3-ForcedAligner-0.6B for subtitle timestamps")
        self.model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
        self.return_timestamps = return_timestamps

    def transcribe(self, audio_path: str, language: str = "zh") -> list[Segment]:
        print(f"  Transcribing with Qwen3-ASR: {audio_path}")
        results = self.model.transcribe(
            audio=audio_path,
            language=_language_name(language),
            return_time_stamps=self.return_timestamps,
        )
        if len(results) != 1:
            raise RuntimeError("Qwen3-ASR must return one result for one audio file")
        result = results[0]
        text = result.text.strip()
        if not text:
            return []
        if self.return_timestamps:
            segments = _aligned_segments(text, result.time_stamps)
        else:
            # Live chunks are PCM WAVs produced by ffmpeg; no aligner is needed.
            with wave.open(audio_path, "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            segments = [Segment(0.0, duration, text)]
        print(f"  Transcription complete: {len(segments)} segments")
        return segments
