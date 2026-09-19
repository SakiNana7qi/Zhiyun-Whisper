"""Offline regressions for legacy and interactivemeta recording support."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from main import cli
from src.crawler import fetch_lessons, parse_url
from src.transcriber import Segment


LEGACY_URL = (
    "https://classroom.zju.edu.cn/livingroom"
    "?course_id=86161&sub_id=1966460&tenant_code=112"
)
REPLAY_URL = (
    "https://interactivemeta.cmc.zju.edu.cn/"
    "#/replay?course_id=86830&sub_id=1967175&tenant_code=112"
)
VIDEO_URL = "https://media.example/lecture.mp4"
OTHER_VIDEO_URL = "https://media.example/other.mp4"


def catalogue_session(items):
    session = Mock()
    session.get.return_value.json.return_value = {
        "success": True,
        "result": {"data": items},
    }
    return session


def catalogue_item(sub_id, content, title="Lecture", lesson_type="ilive"):
    return {
        "sub_id": sub_id,
        "title": title,
        "type": lesson_type,
        "content": json.dumps(content),
    }


class ReplayUrlTests(unittest.TestCase):
    def test_legacy_and_hash_route_links(self):
        for url, course_id, sub_id in (
            (LEGACY_URL, "86161", "1966460"),
            (LEGACY_URL + "#player", "86161", "1966460"),
            (REPLAY_URL, "86830", "1967175"),
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    parse_url(url),
                    {"course_id": course_id, "sub_id": sub_id, "tenant_code": "112"},
                )

    def test_course_only_links(self):
        for url in (
            "https://classroom.zju.edu.cn/coursedetail?course_id=86830",
            "https://interactivemeta.cmc.zju.edu.cn/#/replay?course_id=86830",
        ):
            with self.subTest(url=url):
                self.assertEqual(parse_url(url), {"course_id": "86830"})

    def test_route_parameters_override_outer_query(self):
        url = (
            "https://interactivemeta.cmc.zju.edu.cn/?course_id=old&tenant_code=112"
            "#/replay?course_id=86830&sub_id=1967175"
        )
        self.assertEqual(
            parse_url(url),
            {"course_id": "86830", "sub_id": "1967175", "tenant_code": "112"},
        )

    def test_missing_course_id_is_rejected(self):
        for url in (
            "https://classroom.zju.edu.cn/livingroom?sub_id=1966460",
            "https://interactivemeta.cmc.zju.edu.cn/#/replay?sub_id=1967175",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "course_id"):
                parse_url(url)


class CatalogueTests(unittest.TestCase):
    def test_old_and_new_playback_shapes(self):
        cases = [
            ({"playback": {"url": VIDEO_URL}}, VIDEO_URL),
            ({"playback": {"url": [VIDEO_URL], "selected": 0}}, VIDEO_URL),
            ({"playback": {"url": [VIDEO_URL, OTHER_VIDEO_URL], "selected": 1}}, OTHER_VIDEO_URL),
            ({"playback": {"url": [VIDEO_URL, OTHER_VIDEO_URL], "selected": "1"}}, OTHER_VIDEO_URL),
            ({"playback": {"url": [VIDEO_URL, OTHER_VIDEO_URL]}}, VIDEO_URL),
            ({"url": VIDEO_URL}, VIDEO_URL),
            ({"playback": {"url": VIDEO_URL}, "url": OTHER_VIDEO_URL}, VIDEO_URL),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                session = catalogue_session([catalogue_item("1967175", content)])
                lessons = fetch_lessons(session, "86830")
                self.assertEqual(lessons[0].video_url, expected)

    def test_invalid_selection_uses_first_usable_url(self):
        for selected in (None, "invalid", 99, -1, 0):
            with self.subTest(selected=selected):
                content = {
                    "playback": {
                        "url": [None, "", " ", {}, VIDEO_URL, OTHER_VIDEO_URL],
                        "selected": selected,
                    }
                }
                session = catalogue_session([catalogue_item("1967175", content)])
                self.assertEqual(fetch_lessons(session, "86830")[0].video_url, VIDEO_URL)

    def test_unavailable_playback_preserves_other_lessons(self):
        unavailable = [
            {}, None, [],
            {"playback": None},
            {"playback": {"url": []}},
            {"playback": {"url": [None, "", {}]}},
            {"playback": {"url": {"unexpected": VIDEO_URL}}},
        ]
        items = [catalogue_item(str(i), content) for i, content in enumerate(unavailable)]
        items.append({"sub_id": "broken", "title": "Broken", "content": "{invalid"})
        items.append(catalogue_item("ready", {"playback": {"url": [VIDEO_URL]}}))
        lessons = fetch_lessons(catalogue_session(items), "86830")
        self.assertEqual(len(lessons), len(items))
        self.assertTrue(all(lesson.video_url is None for lesson in lessons[:-1]))
        self.assertEqual(lessons[-1].video_url, VIDEO_URL)

    def test_unavailable_playback_can_use_legacy_fallback(self):
        content = {"playback": {"url": []}, "url": VIDEO_URL}
        session = catalogue_session([catalogue_item("1967175", content)])
        self.assertEqual(fetch_lessons(session, "86830")[0].video_url, VIDEO_URL)


class TranscribeCommandTests(unittest.TestCase):
    def test_old_and_new_links_produce_transcripts(self):
        for url, sub_id, playback in (
            (LEGACY_URL, "1966460", {"url": VIDEO_URL}),
            (REPLAY_URL, "1967175", {"url": [VIDEO_URL], "selected": 0}),
        ):
            with self.subTest(url=url), tempfile.TemporaryDirectory() as output_dir:
                session = catalogue_session([
                    catalogue_item("unavailable", {}),
                    catalogue_item(sub_id, {"playback": playback}),
                ])
                audio_path = str(Path(output_dir) / "Lecture.wav")
                with (
                    patch("main._get_session", return_value=session),
                    patch("src.crawler.download_audio", return_value=audio_path) as download,
                    patch("src.transcriber.transcribe_local", return_value=[
                        Segment(0.0, 1.5, "课程转录测试"),
                    ]) as transcribe,
                ):
                    result = CliRunner().invoke(cli, [
                        "transcribe", url, "--output-dir", output_dir,
                        "--model", "tiny", "--batch-size", "1",
                    ])
                self.assertEqual(result.exit_code, 0, result.output)
                download.assert_called_once_with(
                    video_url=VIDEO_URL, title="Lecture", output_dir=output_dir,
                )
                transcribe.assert_called_once_with(
                    audio_path=audio_path, model_size="tiny", language="zh", batch_size=1,
                )
                self.assertEqual(
                    (Path(output_dir) / "Lecture.txt").read_text(encoding="utf-8"),
                    "课程转录测试\n",
                )
                self.assertIn(
                    "00:00:00,000 --> 00:00:01,500\n课程转录测试",
                    (Path(output_dir) / "Lecture.srt").read_text(encoding="utf-8"),
                )

    def test_missing_sub_id_selects_latest_available_new_replay(self):
        session = catalogue_session([
            catalogue_item("older", {"playback": {"url": VIDEO_URL}}, "Older"),
            catalogue_item("latest", {"playback": {"url": [OTHER_VIDEO_URL]}}, "Latest"),
            catalogue_item("future", {}, "Future"),
        ])
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch("main._get_session", return_value=session),
            patch("src.crawler.download_audio", return_value="audio.wav") as download,
            patch("src.transcriber.transcribe_local", return_value=[]),
        ):
            result = CliRunner().invoke(cli, [
                "transcribe", "https://interactivemeta.cmc.zju.edu.cn/#/replay?course_id=86830",
                "--output-dir", output_dir,
            ])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(download.call_args.kwargs["video_url"], OTHER_VIDEO_URL)
        self.assertEqual(download.call_args.kwargs["title"], "Latest")


if __name__ == "__main__":
    unittest.main()
