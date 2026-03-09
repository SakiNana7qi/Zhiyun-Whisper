"""
DingTalk robot notification with HMAC-SHA256 signing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
import urllib.parse

import requests

logger = logging.getLogger(__name__)


def send_dingtalk(
    webhook: str,
    secret: str,
    message: str,
    at_mobiles: list[str] | None = None,
) -> bool:
    """
    Send a text message via DingTalk robot with signed security.

    Args:
        webhook: DingTalk robot webhook URL (contains access_token)
        secret:  DingTalk robot signing secret (starts with "SEC")
        message: Message content
        at_mobiles: List of mobile numbers to @mention; None means no @

    Returns:
        True on success, False on any error (does not raise).
    """
    try:
        timestamp = str(round(time.time() * 1000))
        sign_str = f"{timestamp}\n{secret}"
        hmac_code = hmac.new(
            secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

        url = f"{webhook}&timestamp={timestamp}&sign={sign}"

        at_mobiles = at_mobiles or []
        payload = {
            "msgtype": "text",
            "text": {"content": message},
            "at": {
                "atMobiles": at_mobiles,
                "isAtAll": False,
            },
        }

        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()

        data = resp.json()
        if data.get("errcode", 0) != 0:
            logger.error("DingTalk API error: %s", data)
            return False

        return True

    except Exception as exc:
        logger.error("Failed to send DingTalk notification: %s", exc)
        return False
