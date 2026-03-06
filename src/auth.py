"""
ZJU Unified Authentication (CAS) login module.

Flow:
1. GET login page -> extract `execution` token
2. GET /cas/v2/getPubKey -> RSA public key (exponent, modulus) + _pv0 cookie
3. RSA-encrypt the password
4. POST login form with encrypted password
5. Follow redirects to obtain classroom.zju.edu.cn session cookies
"""

import re
import requests
from bs4 import BeautifulSoup
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import base64

CAS_LOGIN_URL = "https://zjuam.zju.edu.cn/cas/login"
CAS_PUBKEY_URL = "https://zjuam.zju.edu.cn/cas/v2/getPubKey"
CLASSROOM_SERVICE = "https://classroom.zju.edu.cn"


def _get_execution_token(session: requests.Session) -> str:
    """Fetch the CAS login page and extract the hidden `execution` field."""
    resp = session.get(CAS_LOGIN_URL, params={"service": CLASSROOM_SERVICE})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    execution_input = soup.find("input", {"name": "execution"})
    if not execution_input:
        raise RuntimeError("Cannot find 'execution' token on the CAS login page")
    return execution_input["value"]


def _get_rsa_pubkey(session: requests.Session) -> tuple[int, int]:
    """Fetch RSA public key from CAS server. Returns (exponent, modulus)."""
    resp = session.get(CAS_PUBKEY_URL)
    resp.raise_for_status()
    data = resp.json()
    modulus = int(data["modulus"], 16)
    exponent = int(data["exponent"], 16)
    return exponent, modulus


def _rsa_encrypt(message: str, exponent: int, modulus: int) -> str:
    """Encrypt a message with RSA public key, return base64-encoded ciphertext."""
    rsa_key = RSA.construct((modulus, exponent))
    cipher = PKCS1_v1_5.new(rsa_key)
    encrypted = cipher.encrypt(message.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def login(username: str, password: str) -> requests.Session:
    """
    Perform ZJU CAS login and return an authenticated requests.Session
    with cookies valid for classroom.zju.edu.cn.

    Args:
        username: ZJU student ID
        password: ZJU password

    Returns:
        An authenticated requests.Session

    Raises:
        RuntimeError: If login fails
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    execution = _get_execution_token(session)
    exponent, modulus = _get_rsa_pubkey(session)
    encrypted_password = _rsa_encrypt(password, exponent, modulus)

    login_data = {
        "username": username,
        "password": encrypted_password,
        "authcode": "",
        "execution": execution,
        "_eventId": "submit",
    }

    resp = session.post(
        CAS_LOGIN_URL,
        params={"service": CLASSROOM_SERVICE},
        data=login_data,
        allow_redirects=True,
    )
    resp.raise_for_status()

    if "统一身份认证" in resp.text and "credentials" in resp.text.lower():
        raise RuntimeError(
            "Login failed — check your username and password in .env"
        )

    return session
