"""ASR routing, Qwen SDK contracts, subtitles, and monitor reuse without weights."""

import io
import tempfile
import unittest
import wave
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner

from main import cli
from src.crawler import Lesson
from src.live_monitor import monitor_loop
from src.qwen_asr_backend import QwenTranscriber, _aligned_segments, _language_name
from src.transcriber import DEFAULT_MODEL, Segment, load_local_model, transcribe_local


def stamp(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


@contextmanager
def fake_qwen(cuda=False, bf16=False):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda, is_bf16_supported=lambda: bf16,
    )
    torch.float32, torch.float16, torch.bfloat16 = "float32", "float16", "bfloat16"
    sdk = ModuleType("qwen_asr")
    sdk.Qwen3ASRModel = Mock()
    with patch.dict("sys.modules", {"torch": torch, "qwen_asr": sdk}):
        yield sdk.Qwen3ASRModel.from_pretrained


class LocalModelTests(unittest.TestCase):
    def test_qwen_default_and_size_aliases(self):
        for alias, size in (
            (DEFAULT_MODEL, "1.7B"), ("1.7b", "1.7B"), ("0.6b", "0.6B"),
            ("qwen3-asr-0.6b", "0.6B"), ("Qwen/Qwen3-ASR-1.7B", "1.7B"),
            ("Qwen/Qwen3-ASR-0.6B", "0.6B"),
        ):
            with self.subTest(alias=alias), patch("src.qwen_asr_backend.QwenTranscriber") as qwen:
                model = load_local_model(alias)
                self.assertIs(model, qwen.return_value)
                qwen.assert_called_once_with(
                    f"Qwen/Qwen3-ASR-{size}", device="auto", batch_size=1, return_timestamps=True,
                )

    def test_whisper_remains_available_without_qwen(self):
        sdk = ModuleType("faster_whisper")
        sdk.WhisperModel = Mock()
        with patch.dict("sys.modules", {"faster_whisper": sdk, "qwen_asr": None}), redirect_stdout(io.StringIO()):
            model = load_local_model("small", device="cpu")
            sdk.WhisperModel.assert_called_once_with("small", device="cpu", compute_type="int8")
            with patch("src.transcriber.transcribe_with_model", return_value=[]) as transcribe:
                self.assertEqual(model.transcribe("chunk.wav"), [])
                transcribe.assert_called_once_with(sdk.WhisperModel.return_value, "chunk.wav", "zh", 16)

    def test_recordings_request_timestamps_and_pass_options(self):
        with patch("src.transcriber.load_local_model") as load:
            transcribe_local("recording.wav", model_size="0.6b", batch_size=2, language="en")
            load.assert_called_once_with("0.6b", "auto", 2, return_timestamps=True)
            load.return_value.transcribe.assert_called_once_with("recording.wav", "en")

    def test_invalid_batch_and_qwen_size_are_rejected(self):
        for options in ({"batch_size": 0}, {"batch_size": -1}, {"model_size": "qwen3-asr-7b"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_local_model(**options)


class QwenAdapterTests(unittest.TestCase):
    def test_devices_dtypes_and_recording_aligner(self):
        for cuda, bf16, device, dtype in (
            (False, False, "cpu", "float32"),
            (True, False, "cuda:0", "float16"),
            (True, True, "cuda:0", "bfloat16"),
        ):
            with self.subTest(device=device, dtype=dtype), fake_qwen(cuda, bf16) as load, redirect_stdout(io.StringIO()):
                QwenTranscriber("Qwen/Qwen3-ASR-1.7B", batch_size=2)
                load.assert_called_once_with(
                    "Qwen/Qwen3-ASR-1.7B", device_map=device, dtype=dtype,
                    max_inference_batch_size=2, max_new_tokens=4096,
                    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                    forced_aligner_kwargs={"device_map": device, "dtype": dtype},
                )

    def test_recording_converts_sdk_timestamps(self):
        with fake_qwen() as load, redirect_stdout(io.StringIO()):
            load.return_value.transcribe.return_value = [SimpleNamespace(
                text="你好。", time_stamps=[stamp("你", 2, 2.2), stamp("好", 2.2, 2.5)],
            )]
            model = QwenTranscriber("Qwen/Qwen3-ASR-0.6B")
            self.assertEqual(model.transcribe("recording.wav"), [Segment(2, 2.5, "你好。")])
            load.return_value.transcribe.assert_called_once_with(
                audio="recording.wav", language="Chinese", return_time_stamps=True,
            )

    def test_live_chunks_reuse_model_without_aligner(self):
        with tempfile.TemporaryDirectory() as tmp, fake_qwen() as load, redirect_stdout(io.StringIO()):
            audio_path = str(Path(tmp) / "chunk.wav")
            with wave.open(audio_path, "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 8000)
            load.return_value.transcribe.return_value = [SimpleNamespace(text="测试。", time_stamps=None)]
            model = QwenTranscriber("Qwen/Qwen3-ASR-1.7B", return_timestamps=False)
            for _ in range(2):
                self.assertEqual(model.transcribe(audio_path), [Segment(0, 0.5, "测试。")])
            load.assert_called_once()
            self.assertNotIn("forced_aligner", load.call_args.kwargs)
            self.assertEqual(load.return_value.transcribe.call_count, 2)
            self.assertFalse(load.return_value.transcribe.call_args.kwargs["return_time_stamps"])

    def test_silence_does_not_require_alignment(self):
        with fake_qwen() as load, redirect_stdout(io.StringIO()):
            load.return_value.transcribe.return_value = [SimpleNamespace(text=" ", time_stamps=None)]
            self.assertEqual(QwenTranscriber("model").transcribe("silence.wav"), [])

    def test_language_codes_and_auto_detection(self):
        for value, expected in (("zh", "Chinese"), ("en", "English"), ("yue", "Cantonese"),
                                ("zh-TW", "Chinese"), ("Japanese", "Japanese"), ("auto", None)):
            with self.subTest(value=value):
                self.assertEqual(_language_name(value), expected)

    def test_missing_sdk_has_install_instructions(self):
        with patch.dict("sys.modules", {"qwen_asr": None}), self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
            QwenTranscriber("model")


class QwenSubtitlesTests(unittest.TestCase):
    def test_mixed_text_preserves_punctuation_and_long_audio_offsets(self):
        segments = _aligned_segments("你好，World！再见。", [
            stamp("你", 180.1, 180.3), stamp("好", 180.3, 180.6),
            stamp("World", 180.6, 181), stamp("再", 190.1, 190.3), stamp("见", 190.3, 190.5),
        ])
        self.assertEqual(segments, [Segment(180.1, 181, "你好，World！"), Segment(190.1, 190.5, "再见。")])

    def test_aligner_punctuation_removal_does_not_damage_transcript(self):
        text = "It's 3.14, isn't it?"
        segments = _aligned_segments(text, [
            stamp("It's", 0, 0.2), stamp("314", 0.2, 0.4),
            stamp("isn't", 0.4, 0.6), stamp("it", 0.6, 0.8),
        ])
        self.assertEqual(" ".join(s.text for s in segments), text)

    def test_long_unpunctuated_chinese_is_grouped_into_readable_captions(self):
        text = "课" * 100
        segments = _aligned_segments(text, [stamp("课", i * .1, (i + 1) * .1) for i in range(100)])
        self.assertEqual("".join(s.text for s in segments), text)
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(s.text) <= 42 for s in segments))

    def test_missing_or_mismatched_alignment_fails_instead_of_losing_text(self):
        for items in (None, [], [stamp("错", 0, 1)], [stamp("你", 0, 1)]):
            with self.subTest(items=items), self.assertRaises(RuntimeError):
                _aligned_segments("你好", items)


class AsrCommandTests(unittest.TestCase):
    def test_recording_default_and_api_routing(self):
        for mode in ("local", "api"):
            with (
                self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp,
                patch("main._get_session"),
                patch("src.crawler.fetch_lessons", return_value=[Lesson("1", "Lecture", "https://example.invalid/video.mp4")]),
                patch("src.crawler.download_audio", return_value="audio.wav"),
                patch("src.transcriber.transcribe_local", return_value=[]) as local,
                patch("src.transcriber.transcribe_api", return_value=[]) as api,
                patch.dict("os.environ", {"OPENAI_API_KEY": "test"}),
            ):
                result = CliRunner().invoke(cli, ["transcribe", "https://example.invalid/?course_id=1", "--mode", mode, "-o", tmp])
                self.assertEqual(result.exit_code, 0, result.output)
                if mode == "local":
                    self.assertEqual(local.call_args.kwargs["model_size"], "qwen3-asr-1.7b")
                    self.assertIsNone(local.call_args.kwargs["batch_size"])
                    api.assert_not_called()
                else:
                    api.assert_called_once_with(audio_path="audio.wav", api_key="test", language="zh")
                    local.assert_not_called()

    def test_monitor_defaults_and_model_override(self):
        auth = ModuleType("src.auth")
        auth.refresh_token = Mock()
        config = {"ZJU_TOKEN": "test", "DINGTALK_WEBHOOK": "test", "DINGTALK_SECRET": "test",
                  "LLM_API_BASE": "test", "LLM_API_KEY": "test"}
        for args, model, batch in (([], "qwen3-asr-1.7b", None), (["--model", "0.6b", "--batch-size", "2"], "0.6b", 2),
                                   (["--model", "small"], "small", None)):
            with (
                self.subTest(args=args), patch.dict("sys.modules", {"src.auth": auth}), patch.dict("os.environ", config),
                patch("main._get_session"), patch("src.live_monitor.monitor_loop") as monitor,
            ):
                result = CliRunner().invoke(cli, ["monitor", "--course-id", "1", *args])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(monitor.call_args.kwargs["model_size"], model)
                self.assertEqual(monitor.call_args.kwargs["batch_size"], batch)

    def test_invalid_batch_is_rejected_before_startup(self):
        for command in (["monitor"], ["transcribe", "https://example.invalid"]):
            result = CliRunner().invoke(cli, [*command, "--batch-size", "0"])
            self.assertEqual(result.exit_code, 2)


class MonitorModelReuseTests(unittest.TestCase):
    def test_monitor_loads_once_and_transcribes_multiple_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"chunk_{i}.wav" for i in range(2)]
            for path in paths:
                path.touch()
            live = ("https://example.invalid/live.m3u8", "sub1")
            with (
                patch("src.live_monitor.fetch_live_url", side_effect=[live, live, None]),
                patch("src.live_monitor.stream_audio_chunks", return_value=iter(map(str, paths))),
                patch("src.transcriber.load_local_model") as load,
                patch("src.live_monitor.is_stream_ended", return_value=False),
                patch("src.live_monitor.check_keywords_pinyin", return_value=None),
                patch("time.sleep"), redirect_stdout(io.StringIO()),
            ):
                load.return_value.transcribe.return_value = [Segment(0, 1, "课堂内容")]
                monitor_loop(Mock(), "course1", [], 30, DEFAULT_MODEL, {}, {}, log_dir=tmp)
            load.assert_called_once_with(model_size=DEFAULT_MODEL, batch_size=None, return_timestamps=False)
            self.assertEqual(load.return_value.transcribe.call_count, 2)
            self.assertTrue(all(not path.exists() for path in paths))
            logs = list(Path(tmp).glob("course1_*.txt"))
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].read_text().count("课堂内容"), 2)


if __name__ == "__main__":
    unittest.main()
