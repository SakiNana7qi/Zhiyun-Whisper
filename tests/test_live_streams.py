"""Live URL discovery for both generations of Zhiyun Classroom, without I/O."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import requests

from src.crawler import CATALOGUE_API
from src.live_monitor import (
    GET_SUB_INFO_API,
    INTERACTIVE_STREAMS_API,
    TokenExpiredError,
    fetch_live_url,
    monitor_loop,
)


COURSE_ID = "86830"
SUB_ID = "1970365"
TEACHER_URL = "https://media.example.invalid/teacher.m3u8?auth_key=test-signature"
PPT_URL = "https://media.example.invalid/ppt.m3u8?auth_key=test-ppt"
ILIVE_INFO = {
    "code": 0,
    "data": {
        "sub_type": "ilive", "sub_data_type": "ilive", "sub_status": "1",
        "is_m3u8": "no", "play_msg": "获取直播信息失败", "playurl": {"0": ""},
        "address_list": {"0": {"stream_address": "rtmp://10.0.0.1/live/internal"}},
    },
}


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


def catalogue(live=True, kind="ilive"):
    return {"success": True, "result": {"data": [{
        "sub_id": SUB_ID, "status": "1" if live else "0", "type": kind,
        "title": "Test lecture",
    }]}}


def stream_payload(streams=None):
    if streams is None:
        streams = [
            {"type": 2, "stream_m3u8": PPT_URL},
            {"type": 3, "stream_m3u8": TEACHER_URL, "voice_track": "0", "video_track": "0"},
        ]
    return {"success": True, "result": {"err": 0, "data": streams}}


def make_session(info=ILIVE_INFO, streams=None, kind="ilive"):
    session = Mock()
    session.get.side_effect = [
        response(catalogue(kind=kind)), response(info),
        response(stream_payload() if streams is None else streams),
    ]
    return session


class LiveDiscoveryTests(unittest.TestCase):
    def setUp(self):
        capture = redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def test_legacy_hls_is_returned_without_requesting_interactive_api(self):
        info = {"code": 0, "data": {"live_url": {"output": {"m3u8": TEACHER_URL}}}}
        for kind in ("live", "ilive"):
            with self.subTest(kind=kind):
                session = make_session(info=info, kind=kind)
                self.assertEqual(fetch_live_url(session, COURSE_ID), (TEACHER_URL, SUB_ID))
                self.assertEqual(session.get.call_count, 2)

    def test_ilive_uses_signed_teacher_hls_even_when_flags_say_no_audio_or_hls(self):
        session = make_session()
        self.assertEqual(fetch_live_url(session, COURSE_ID), (TEACHER_URL, SUB_ID))
        self.assertEqual(session.get.call_args, call(
            INTERACTIVE_STREAMS_API, params={"sub_id": SUB_ID, "clear_cache": 1}, timeout=10,
        ))

    def test_ilive_can_be_identified_from_catalogue_or_sub_info(self):
        for info, kind in (({"code": 0, "data": {}}, "ilive"), (ILIVE_INFO, "")):
            with self.subTest(info=info, kind=kind):
                self.assertEqual(fetch_live_url(make_session(info=info, kind=kind), COURSE_ID), (TEACHER_URL, SUB_ID))

    def test_string_stream_type_and_indexed_object_are_supported(self):
        streams = stream_payload({"0": {"type": "2", "stream_m3u8": PPT_URL},
                                  "1": {"type": "3", "stream_m3u8": TEACHER_URL}})
        self.assertEqual(fetch_live_url(make_session(streams=streams), COURSE_ID), (TEACHER_URL, SUB_ID))

    def test_absent_teacher_hls_does_not_select_ppt_rtmp_or_webrtc(self):
        for url in (None, "", " ", {}, "webrtc://media.example.invalid/live/id", "rtmp://10.0.0.1/live/id", "file:///tmp/audio.wav", "https:///missing-host"):
            streams = stream_payload([
                {"type": 2, "stream_m3u8": PPT_URL},
                {"type": 3, "stream_m3u8": url, "stream_play": "webrtc://media.example.invalid/live/id"},
            ])
            with self.subTest(url=url), self.assertLogs("src.live_monitor", level="WARNING"):
                self.assertIsNone(fetch_live_url(make_session(streams=streams), COURSE_ID))

    def test_missing_or_malformed_streams_are_retryable(self):
        for streams in ([], {"success": False, "result": {}}, {"success": True, "result": None},
                        stream_payload([]), stream_payload([None, {}]), stream_payload("unavailable")):
            with self.subTest(streams=streams), self.assertLogs("src.live_monitor", level="WARNING"):
                self.assertIsNone(fetch_live_url(make_session(streams=streams), COURSE_ID))

    def test_interactive_api_failure_is_retryable(self):
        session = make_session()
        session.get.side_effect = [response(catalogue()), response(ILIVE_INFO), requests.Timeout("test timeout")]
        with self.assertLogs("src.live_monitor", level="ERROR"):
            self.assertIsNone(fetch_live_url(session, COURSE_ID))

    def test_expired_token_is_reported_to_existing_refresh_flow(self):
        unauthorized = {"success": False, "result": {"name": "Unauthorized", "status": 401}}
        for last_response in (response(unauthorized), response({}, status=401)):
            session = make_session()
            session.get.side_effect = [response(catalogue()), response(ILIVE_INFO), last_response]
            with self.assertRaises(TokenExpiredError):
                fetch_live_url(session, COURSE_ID)

    def test_old_api_auth_failure_is_preserved(self):
        session = make_session(info={"code": 500, "msg": "用户认证失败"})
        with self.assertRaises(TokenExpiredError):
            fetch_live_url(session, COURSE_ID)
        self.assertEqual(session.get.call_count, 2)

    def test_no_live_lesson_does_not_request_streams(self):
        session = Mock()
        session.get.return_value = response(catalogue(live=False))
        self.assertIsNone(fetch_live_url(session, COURSE_ID))
        session.get.assert_called_once_with(CATALOGUE_API, params={"course_id": COURSE_ID})

    def test_missing_legacy_url_does_not_probe_interactive_streams(self):
        session = make_session(info={"code": 0, "data": {}}, kind="live")
        with self.assertLogs("src.live_monitor", level="WARNING"):
            self.assertIsNone(fetch_live_url(session, COURSE_ID))
        self.assertEqual(session.get.call_count, 2)


class InteractiveMonitorTests(unittest.TestCase):
    def test_reconnect_refetches_signed_hls_and_reuses_existing_chunk_transcription(self):
        session = Mock()
        catalogue_count = stream_count = 0

        def get(url, **kwargs):
            nonlocal catalogue_count, stream_count
            if url == CATALOGUE_API:
                catalogue_count += 1
                return response(catalogue(live=catalogue_count <= 3))
            if url == GET_SUB_INFO_API:
                return response(ILIVE_INFO)
            if url == INTERACTIVE_STREAMS_API:
                self.assertEqual(kwargs["params"], {"sub_id": SUB_ID, "clear_cache": 1})
                stream_count += 1
                return response(stream_payload([{
                    "type": 3, "stream_m3u8": f"{TEACHER_URL}-{stream_count}",
                }]))
            raise AssertionError(f"Unexpected API: {url}")

        session.get.side_effect = get
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"chunk_{n}.wav" for n in range(2)]
            for path in paths:
                path.touch()
            with (
                patch("src.live_monitor.stream_audio_chunks", side_effect=[iter([str(p)]) for p in paths]) as chunks,
                patch("src.transcriber.load_local_model") as load,
                patch("src.live_monitor.is_stream_ended", return_value=False),
                patch("src.live_monitor.check_keywords_pinyin", return_value=None),
                patch("src.live_monitor.time.sleep"), redirect_stdout(io.StringIO()),
            ):
                load.return_value.transcribe.return_value = [SimpleNamespace(text="新版课堂音频")]
                monitor_loop(session, COURSE_ID, [], 30, "qwen3-asr-1.7b", {}, {}, log_dir=tmp)
            self.assertEqual(stream_count, 3)
            self.assertEqual([args.args[0] for args in chunks.call_args_list], [f"{TEACHER_URL}-2", f"{TEACHER_URL}-3"])
            load.assert_called_once()
            self.assertEqual(load.return_value.transcribe.call_count, 2)
            self.assertTrue(all(not path.exists() for path in paths))
            log = next(Path(tmp).glob(f"{COURSE_ID}_*.txt")).read_text(encoding="utf-8")
            self.assertEqual(log.count("新版课堂音频"), 2)


if __name__ == "__main__":
    unittest.main()
