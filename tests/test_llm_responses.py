"""Alert decisions and delivery without live APIs, a GPU or an installed SDK."""

import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.live_monitor import (
    _extract_llm_answer,
    evaluate_alert_with_llm,
    monitor_loop,
)


CONFIG = {"api_base": "https://example.invalid/v1", "api_key": "test", "model": "test"}
KEYWORDS = ["点到", "小测"]
QUIZ_TEXT = "提醒大家，下周有小测。"
PDP_TEXT = (
    "然后把它留给我们好了，是吧？啊，那我们继续往下走啊，我们来到我们的一个主第二个主题的活动啊。"
    "这个昨天我们做一点自我探索啊，刚才不是你们大家都选了自己的这个这个彩卡了，"
    "然后这个其实是一个很简单的一种测试，叫PDP啊PDP的啊PDP的性格测试啊，"
    "嗯，它就是啊有一个那个性格的测试的一个，哎，那个佳蕊你可以分享到我们今天。"
)


def decision_json(should_alert=True, **overrides):
    data = {
        "should_alert": should_alert,
        "matched_keywords": ["小测"] if should_alert else [],
        "evidence": "下周有小测" if should_alert else "",
        "analysis": "老师预告下周有小测。" if should_alert else "未检测到相关内容。",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def completion(content, reasoning_content="独立思考内容", finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning_content),
        finish_reason=finish_reason,
    )])


@contextmanager
def mock_completion(content, reasoning_content="独立思考内容", finish_reason="stop"):
    sdk = ModuleType("openai")
    sdk.OpenAI = Mock()
    create = sdk.OpenAI.return_value.chat.completions.create
    create.return_value = completion(content, reasoning_content, finish_reason)
    with patch.dict("sys.modules", {"openai": sdk}):
        yield create


class ExtractAnswerTests(unittest.TestCase):
    def test_plain_and_thinking_wrapped_answers(self):
        cases = [
            (" 否\n", "否"),
            ("是", "是"),
            ("否</think>否", "否"),
            ("否</think>是", "是"),
            ("是</think>否", "否"),
            ("</think> 是", "是"),
            ("<think>思考\n还有一行</think>\n是", "是"),
            ("<think>第一次</think><think>第二次</think>否", "否"),
            ("<think>内部推理</think>老师宣布小测。\n请准备答题。", "老师宣布小测。\n请准备答题。"),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                self.assertEqual(_extract_llm_answer(content), expected)

    def test_missing_or_unfinished_answer_is_rejected(self):
        for content in (None, "", " \n", "<think>是", "<think>是</think>", "否</think> ",
                        "<think>第一段</think><think>尚未完成"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                _extract_llm_answer(content)


class LlmResponseTests(unittest.TestCase):
    def evaluate(self, text=QUIZ_TEXT, recent_entries=None, **kwargs):
        return evaluate_alert_with_llm(
            text, recent_entries or [f"[15:31:09] {text}"], KEYWORDS, **CONFIG, **kwargs,
        )

    def test_decision_uses_only_final_json_and_leaves_thinking_defaults_unchanged(self):
        for prefix in ("", "内部推理</think>", "<think>内部推理</think>"):
            output = io.StringIO()
            with self.subTest(prefix=prefix), mock_completion(prefix + decision_json()) as create, redirect_stdout(output):
                result = self.evaluate(debug=True)
                self.assertTrue(result.should_alert)
                self.assertEqual(result.matched_keywords, ("小测",))
                self.assertEqual(result.evidence, "下周有小测")
                self.assertEqual(result.analysis, "老师预告下周有小测。")
                self.assertEqual(output.getvalue(), f"[debug] LLM decision: {decision_json()}\n")
                create.assert_called_once()
                self.assertEqual(set(create.call_args.kwargs), {
                    "model", "messages", "max_tokens", "temperature", "timeout",
                })

    def test_negative_verdict_is_distinct_from_failure_and_ignores_reasoning(self):
        with mock_completion(decision_json(False), reasoning_content=decision_json()):
            result = self.evaluate()
        self.assertIsNotNone(result)
        self.assertFalse(result.should_alert)
        self.assertEqual(result.matched_keywords, ())

    def test_only_two_previous_chunks_and_latest_text_are_sent_as_data(self):
        entries = ["太早的片段1", "太早的片段2", "[15:29:58] 分享", "[15:30:31] 拍照", "[15:31:09] " + PDP_TEXT]
        with mock_completion(decision_json(False)) as create:
            self.evaluate(text=PDP_TEXT, recent_entries=entries)
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(json.loads(messages[1]["content"]), {
            "keywords": KEYWORDS,
            "previous_transcripts": entries[-3:-1],
            "latest_transcript": PDP_TEXT,
        })

    def test_fenced_json_and_whitespace_differences_in_evidence_are_accepted(self):
        answer = "```json\n" + decision_json(matched_keywords=["小测", "小测"]) + "\n```"
        with mock_completion(answer):
            result = self.evaluate(text="提醒大家，下周有 小测。")
        self.assertTrue(result.should_alert)
        self.assertEqual(result.matched_keywords, ("小测",))

    def test_invalid_decisions_are_confirmation_failures(self):
        cases = [
            None, "", "<think>是", "<think>是</think>", "是", "否", "[]", "null", "{}",
            decision_json()[:-1],
            decision_json(should_alert="false"), decision_json(should_alert=1),
            decision_json(matched_keywords="小测"), decision_json(matched_keywords=[123]),
            decision_json(matched_keywords=["编造的关键词"]), decision_json(matched_keywords=[]),
            decision_json(evidence=""), decision_json(evidence=None),
            decision_json(evidence="现在点名"),
            decision_json(analysis=""), decision_json(analysis=[]),
            decision_json(False, matched_keywords=["小测"]),
            decision_json(False, evidence="下周有小测"),
        ]
        for content in cases:
            with self.subTest(content=content), mock_completion(content, reasoning_content=decision_json()):
                with self.assertLogs("src.live_monitor", level="ERROR"):
                    self.assertIsNone(self.evaluate())

    def test_evidence_from_only_an_old_chunk_is_rejected(self):
        with mock_completion(decision_json()), self.assertLogs("src.live_monitor", level="ERROR"):
            self.assertIsNone(self.evaluate(text=PDP_TEXT, recent_entries=[QUIZ_TEXT, PDP_TEXT]))

    def test_truncated_response_is_failure_even_if_json_is_complete(self):
        with mock_completion(decision_json(), finish_reason="length"), self.assertLogs("src.live_monitor", level="ERROR"):
            self.assertIsNone(self.evaluate())

    def test_api_error_is_a_confirmation_failure(self):
        with mock_completion(None) as create, self.assertLogs("src.live_monitor", level="ERROR"):
            create.side_effect = TimeoutError("test timeout")
            self.assertIsNone(self.evaluate())


class MonitorAlertTests(unittest.TestCase):
    def run_monitor(self, texts, candidates=None, send_result=True):
        """Run real decision parsing and delivery routing; stub only I/O/ASR."""
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"chunk_{i}.wav" for i in range(len(texts))]
            for path in paths:
                path.touch()
            live = ("https://example.invalid/live.m3u8", "sub1")
            with (
                patch("src.live_monitor.fetch_live_url", side_effect=[live, live, None]),
                patch("src.live_monitor.stream_audio_chunks", return_value=iter(map(str, paths))),
                patch("src.transcriber.load_local_model") as load,
                patch("src.live_monitor.is_stream_ended", return_value=False),
                patch("src.live_monitor.check_keywords_pinyin", side_effect=candidates or [("点到", 88)] * len(texts)),
                patch("src.notifier.send_dingtalk", return_value=send_result) as send,
                patch("src.live_monitor.time.sleep"), patch("src.live_monitor.time.time", return_value=1000),
                redirect_stdout(output),
            ):
                load.return_value.transcribe.side_effect = [[SimpleNamespace(text=text)] for text in texts]
                monitor_loop(
                    Mock(), "87063", KEYWORDS, 30, "qwen3-asr-1.7b",
                    {"webhook": "test", "secret": "test", "at_mobiles": ["test-mobile"]},
                    CONFIG, log_dir=tmp, course_title="测试课程", debug=True,
                )
            self.assertTrue(all(not path.exists() for path in paths))
            log_text = next(Path(tmp).glob("87063_*.txt")).read_text(encoding="utf-8")
            for text in texts:
                self.assertIn(text, log_text)
        return send, output.getvalue()

    def test_pdp_negative_decision_never_notifies(self):
        reason = "正在介绍 PDP 性格测试，未提及点名、考勤或课堂小测。"
        with mock_completion(decision_json(False, analysis=reason)) as create:
            send, output = self.run_monitor([PDP_TEXT])
        create.assert_called_once()
        send.assert_not_called()
        self.assertIn(reason, output)

    def test_literal_match_also_requires_semantic_confirmation(self):
        with mock_completion(decision_json(False, analysis="正在讲解点到原点的距离，与考勤无关。")) as create:
            send, _ = self.run_monitor(["计算这个点到原点的距离。"], candidates=[("点到", 100)])
        create.assert_called_once()
        send.assert_not_called()

    def test_notification_uses_confirmed_keywords_evidence_and_analysis_from_one_call(self):
        with mock_completion("内部思考</think>" + decision_json()) as create:
            send, output = self.run_monitor([QUIZ_TEXT])
        create.assert_called_once()
        send.assert_called_once()
        message = send.call_args.kwargs["message"]
        self.assertIn("[智云直播监控] 触发关键词：小测", message)
        self.assertNotIn("触发关键词：点到", message)
        self.assertIn("证据：下周有小测", message)
        self.assertIn("分析：老师预告下周有小测。", message)
        self.assertIn("测试课程（87063）", message)
        self.assertIn(QUIZ_TEXT, message)
        self.assertNotIn("内部思考", message + output)
        self.assertEqual(send.call_args.kwargs["at_mobiles"], ["test-mobile"])

    def test_roll_call_confirmation_sends_once(self):
        text = "现在开始点到，听到名字请回答。"
        with mock_completion(decision_json(
            matched_keywords=["点到"], evidence="现在开始点到", analysis="老师开始点到，要求听到名字后回答。",
        )) as create:
            send, _ = self.run_monitor([text], candidates=[("点到", 100)])
        create.assert_called_once()
        send.assert_called_once()
        self.assertIn("触发关键词：点到", send.call_args.kwargs["message"])

    def test_failed_confirmation_sends_only_an_explicitly_unconfirmed_alert(self):
        for response in ("是", decision_json(evidence="杜撰的原文"), None):
            with self.subTest(response=response), mock_completion(response) as create:
                if response is None:
                    create.side_effect = TimeoutError("test timeout")
                with self.assertLogs("src.live_monitor", level="ERROR"):
                    send, _ = self.run_monitor([PDP_TEXT])
            create.assert_called_once()
            send.assert_called_once()
            message = send.call_args.kwargs["message"]
            self.assertIn("疑似命中（语义确认失败）", message)
            self.assertIn("候选关键词：点到", message)
            self.assertNotIn("触发关键词", message)
            self.assertNotIn("杜撰的原文", message)
            self.assertIn(PDP_TEXT, message)

    def test_negative_decision_does_not_consume_cooldown(self):
        with mock_completion(None) as create:
            create.side_effect = [completion(decision_json(False)), completion(decision_json())]
            send, _ = self.run_monitor([PDP_TEXT, QUIZ_TEXT])
        self.assertEqual(create.call_count, 2)
        send.assert_called_once()

    def test_successful_delivery_applies_cooldown_to_confirmed_and_unconfirmed_alerts(self):
        for response in (decision_json(), "invalid"):
            with self.subTest(response=response), mock_completion(response) as create:
                if response == "invalid":
                    with self.assertLogs("src.live_monitor", level="ERROR"):
                        send, output = self.run_monitor([QUIZ_TEXT, QUIZ_TEXT])
                else:
                    send, output = self.run_monitor([QUIZ_TEXT, QUIZ_TEXT])
            create.assert_called_once()
            send.assert_called_once()
            self.assertIn("in cooldown, skipping", output)

    def test_failed_delivery_does_not_consume_cooldown(self):
        with mock_completion(decision_json()) as create:
            send, output = self.run_monitor([QUIZ_TEXT, QUIZ_TEXT], send_result=False)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(send.call_count, 2)
        self.assertIn("Alert delivery failed", output)

    def test_no_fuzzy_candidate_makes_no_llm_call_or_notification(self):
        with mock_completion(decision_json()) as create:
            send, _ = self.run_monitor([QUIZ_TEXT], candidates=[None])
        create.assert_not_called()
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
