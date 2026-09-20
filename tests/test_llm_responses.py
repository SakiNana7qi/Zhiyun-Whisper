"""LLM final-answer handling without live API calls or an installed SDK."""

import io
import unittest
from contextlib import contextmanager, redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from src.live_monitor import (
    _extract_llm_answer,
    analyze_context_with_llm,
    confirm_with_llm,
)


CONFIG = {"api_base": "https://example.invalid/v1", "api_key": "test", "model": "test"}


@contextmanager
def mock_completion(content, reasoning_content="独立思考内容"):
    sdk = ModuleType("openai")
    sdk.OpenAI = Mock()
    create = sdk.OpenAI.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning_content),
    )])
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
    def test_confirmation_uses_final_verdict_and_prints_only_final_text(self):
        for content, expected in (
            ("否</think>否", False),
            ("否</think>是", True),
            ("是</think>否", False),
            ("<think>内部思考</think>是", True),
        ):
            output = io.StringIO()
            with self.subTest(content=content), mock_completion(content) as create, redirect_stdout(output):
                result = confirm_with_llm("测试转录", **CONFIG, keywords=["小测"], debug=True)
                self.assertEqual(result, expected)
                self.assertEqual(output.getvalue(), f"[debug] LLM response: {'是' if expected else '否'}\n")
                # Response cleanup must leave the provider's thinking defaults alone.
                self.assertEqual(set(create.call_args.kwargs), {
                    "model", "messages", "max_tokens", "temperature", "timeout",
                })

    def test_separate_reasoning_is_not_used_as_the_verdict(self):
        with mock_completion("否", reasoning_content="是"):
            self.assertFalse(confirm_with_llm("测试转录", **CONFIG))

    def test_missing_final_answer_uses_existing_failure_policy(self):
        for content in (None, "", "<think>是", "<think>否</think>"):
            for fail_open in (True, False):
                with self.subTest(content=content, fail_open=fail_open), mock_completion(content):
                    with self.assertLogs("src.live_monitor", level="ERROR"):
                        result = confirm_with_llm("测试转录", **CONFIG, fail_open=fail_open)
                    self.assertEqual(result, fail_open)

    def test_context_analysis_returns_only_final_text(self):
        answer = "老师宣布进行小测。"
        output = io.StringIO()
        with mock_completion(f"内部推理</think>{answer}") as create, redirect_stdout(output):
            result = analyze_context_with_llm(["测试转录"], ["小测"], **CONFIG, debug=True)
        self.assertEqual(result, answer)
        self.assertEqual(output.getvalue(), f"[debug] LLM analysis: {answer}\n")
        self.assertEqual(set(create.call_args.kwargs), {
            "model", "messages", "max_tokens", "temperature", "timeout",
        })

    def test_analysis_without_final_text_uses_existing_failure_message(self):
        with mock_completion("<think>内部思考</think>"), self.assertLogs("src.live_monitor", level="ERROR"):
            answer = analyze_context_with_llm(["测试转录"], ["小测"], **CONFIG)
        self.assertEqual(answer, "（LLM分析失败）")


if __name__ == "__main__":
    unittest.main()
