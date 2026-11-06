"""Schedule authentication recovery and credential-safe errors, without network I/O."""

import base64
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import requests
from click.testing import CliRunner

from main import cli
from src.live_monitor import SCHEDULE_API, TokenExpiredError, fetch_live_courses


TOKEN = "header." + base64.urlsafe_b64encode(
    json.dumps({"sub": 123, "tenant_id": 112}).encode()
).decode().rstrip("=") + ".private-signature"
LIVE = {"success": True, "result": {"list": [{"course": [
    {"course_id": "86161", "course_title": "Test lecture", "status": "1"},
    {"course_id": "87381", "status": "0"},
]}]}}


def response(status=200, body=LIVE):
    result = requests.Response()
    result.status_code = status
    result.url = SCHEDULE_API + "?token=" + TOKEN
    result._content = json.dumps(body).encode()
    return result


class ScheduleTests(unittest.TestCase):
    def fetch(self, result=None, error=None):
        with patch("src.live_monitor._make_schedule_session") as session:
            if error:
                session.return_value.get.side_effect = error
            else:
                session.return_value.get.return_value = result
            return fetch_live_courses(TOKEN)

    def test_http_auth_failures_trigger_refresh_without_exposing_token(self):
        for status in (401, 403):
            with self.subTest(status=status), self.assertRaises(TokenExpiredError) as caught:
                self.fetch(response(status))
            self.assertNotIn(TOKEN, str(caught.exception))
            self.assertIn(str(status), str(caught.exception))

    def test_json_auth_failures_trigger_refresh(self):
        for result in (
            {"status": 401}, {"status": "403"}, {"name": "Unauthorized"},
            {"name": "Forbidden"}, {"data": "认证失败"},
        ):
            with self.subTest(result=result), self.assertRaises(TokenExpiredError):
                self.fetch(response(body={"success": False, "result": result}))

    def test_http_server_error_is_not_auth_failure_and_hides_token(self):
        with self.assertRaises(RuntimeError) as caught:
            self.fetch(response(503))
        self.assertIn("503", str(caught.exception))
        self.assertNotIn(TOKEN, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_network_errors_hide_request_url(self):
        for error in (requests.Timeout, requests.ConnectionError):
            with self.subTest(error=error), self.assertRaises(RuntimeError) as caught:
                self.fetch(error=error(SCHEDULE_API + "?token=" + TOKEN))
            self.assertNotIn(TOKEN, str(caught.exception))
            self.assertIn(error.__name__, str(caught.exception))

    def test_successful_schedule_filters_non_live_items(self):
        self.assertEqual(self.fetch(response()), [{"course_id": "86161", "title": "Test lecture"}])

    def test_failed_response_is_not_mistaken_for_an_empty_schedule(self):
        with self.assertRaisesRegex(RuntimeError, "unsuccessful"):
            self.fetch(response(body={"success": False, "result": {"message": "service error"}}))

    def test_empty_successful_schedule_remains_valid(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.fetch(response(body={"success": True, "result": {"list": []}})), [])


class ScheduleRefreshTests(unittest.TestCase):
    def invoke(self, failures, credentials=True):
        env = {
            "ZJU_TOKEN": TOKEN, "DINGTALK_WEBHOOK": "test", "DINGTALK_SECRET": "test",
            "LLM_API_BASE": "test", "LLM_API_KEY": "test",
        }
        if credentials:
            env.update(ZJU_USERNAME="student", ZJU_PASSWORD="password")
        session = Mock()
        results = [response(403)] * failures + [response()]
        with (
            patch.dict("os.environ", env, clear=True),
            patch("src.live_monitor.check_llm_apis"),
            patch("main._get_session"),
            patch("src.live_monitor._make_schedule_session") as schedule,
            patch("src.auth.refresh_token", return_value=(session, TOKEN + "-fresh")) as refresh,
            patch("src.live_monitor.monitor_loop") as monitor,
        ):
            schedule.return_value.get.side_effect = results
            result = CliRunner().invoke(cli, ["monitor", "--llm-all-chunks"])
        self.assertNotIn(TOKEN, result.output)
        return result, refresh, monitor, schedule, session

    def test_http_403_refreshes_and_resumes_monitor_with_new_session_and_token(self):
        result, refresh, monitor, schedule, session = self.invoke(1)
        self.assertEqual(result.exit_code, 0, result.output)
        refresh.assert_called_once_with("student", "password")
        self.assertEqual(schedule.return_value.get.call_args.kwargs["params"]["token"], TOKEN + "-fresh")
        monitor.assert_called_once()
        self.assertIs(monitor.call_args.kwargs["session"], session)
        self.assertEqual(monitor.call_args.kwargs["course_id"], "86161")
        self.assertTrue(monitor.call_args.kwargs["llm_all_chunks"])

    def test_persistent_403_stops_after_three_refreshes(self):
        result, refresh, monitor, _, _ = self.invoke(4)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(refresh.call_count, 3)
        self.assertIn("after 3 refresh attempts", result.output)
        monitor.assert_not_called()

    def test_no_credentials_explains_how_to_enable_refresh(self):
        result, refresh, monitor, _, _ = self.invoke(1, credentials=False)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Set ZJU_USERNAME and ZJU_PASSWORD", result.output)
        refresh.assert_not_called()
        monitor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
