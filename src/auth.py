"""
ZJU Unified Authentication (CAS) login module.

Flow:
1. Follow CLASSROOM_AUTH_INIT through redirects to CAS login page
2. Extract `execution` token from the login form
3. GET /cas/v2/getPubKey -> RSA public key (exponent, modulus)
4. RSA-encrypt the password (ZJU custom algorithm: reverse + raw RSA, hex output)
5. POST login form, follow redirects through OAuth callback back to classroom
"""

import json
import requests
from bs4 import BeautifulSoup

CAS_LOGIN_URL = "https://zjuam.zju.edu.cn/cas/login"
CAS_PUBKEY_URL = "https://zjuam.zju.edu.cn/cas/v2/getPubKey"
CLASSROOM_SERVICE = "https://classroom.zju.edu.cn"
# Real auth entry point: tgmedia → yjapi/casapi → CAS OAuth → CAS login page
CLASSROOM_AUTH_INIT = (
    "https://yjapi.cmc.zju.edu.cn/casapi/index.php"
    "?r=auth/login&forward=https://classroom.zju.edu.cn/&tenant_code=112"
)
GET_SUB_INFO_API = (
    "https://classroom.zju.edu.cn/courseapi/v3/portal-home-setting/get-sub-info"
)


def _get_execution_token(session: requests.Session) -> tuple[str, str]:
    """
    Follow the real classroom auth redirect chain to the CAS login page
    and extract the hidden `execution` field.

    Flow: CLASSROOM_AUTH_INIT → yjapi/casapi → CAS OAuth authorize → CAS login page

    Returns (execution_token, cas_login_url) where cas_login_url is the
    actual URL to POST the login form to (includes the service param).
    """
    resp = session.get(CLASSROOM_AUTH_INIT, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    execution_input = soup.find("input", {"name": "execution"})
    if not execution_input:
        raise RuntimeError(
            f"Cannot find 'execution' token on the CAS login page. "
            f"Final URL: {resp.url}\n"
            f"Page snippet: {resp.text[:500]}"
        )
    return execution_input["value"], resp.url


def _get_rsa_pubkey(session: requests.Session) -> tuple[str, str]:
    """Fetch RSA public key from CAS server. Returns (exponent_hex, modulus_hex)."""
    resp = session.get(CAS_PUBKEY_URL)
    resp.raise_for_status()
    data = resp.json()
    return data["exponent"], data["modulus"]


def _rsa_encrypt(password: str, exponent_hex: str, modulus_hex: str) -> str:
    """
    Encrypt password using ZJU CAS custom RSA algorithm (matches security.js + login.js).

    Steps (matching the JavaScript exactly):
    1. Reverse the password string (login.js line: reversedPwd = password.split("").reverse().join(""))
    2. Convert to char codes
    3. Pad with zero bytes to multiple of chunkSize
       where chunkSize = 2 * biHighIndex(modulus) = 2 * (num_16bit_words - 1)
    4. For each chunk: interpret bytes as little-endian integer, compute block^e mod n (raw RSA)
    5. Convert result to hex (padded to multiple of 4 hex chars), join chunks with spaces
    """
    n = int(modulus_hex, 16)
    e = int(exponent_hex, 16)

    # chunkSize = 2 * biHighIndex(n) where biHighIndex = (number of 16-bit words) - 1
    n_words = (n.bit_length() + 15) // 16
    chunk_size = 2 * (n_words - 1)

    # Reverse password and convert to byte values
    a = [ord(c) for c in password[::-1]]

    # Pad to multiple of chunk_size
    while len(a) % chunk_size != 0:
        a.append(0)

    result_parts = []
    for i in range(0, len(a), chunk_size):
        chunk = bytes(a[i : i + chunk_size])
        block = int.from_bytes(chunk, "little")
        encrypted = pow(block, e, n)

        # biToHex: output hex padded to 4-char (16-bit) groups, no leading zeros
        hex_str = format(encrypted, "x")
        if len(hex_str) % 4:
            hex_str = "0" * (4 - len(hex_str) % 4) + hex_str
        result_parts.append(hex_str)

    return " ".join(result_parts)


def login(username: str, password: str) -> requests.Session:
    """
    Perform ZJU CAS login via the classroom OAuth flow and return an
    authenticated requests.Session with cookies valid for classroom.zju.edu.cn.

    Flow:
    1. Follow CLASSROOM_AUTH_INIT through redirects to CAS login page
    2. Extract `execution` token from the login form
    3. GET /cas/v2/getPubKey -> RSA public key
    4. POST login form with RSA-encrypted password
    5. Follow redirects: CAS → OAuth callback → tgmedia → classroom.zju.edu.cn

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

    execution, cas_login_url = _get_execution_token(session)
    exponent_hex, modulus_hex = _get_rsa_pubkey(session)
    encrypted_password = _rsa_encrypt(password, exponent_hex, modulus_hex)

    login_data = {
        "username": username,
        "password": encrypted_password,
        "authcode": "",
        "execution": execution,
        "_eventId": "submit",
    }

    # POST to the actual CAS login URL (includes correct service param)
    resp = session.post(
        cas_login_url,
        data=login_data,
        allow_redirects=True,
    )
    resp.raise_for_status()

    if "统一身份认证" in resp.text and resp.url.startswith("https://zjuam.zju.edu.cn"):
        raise RuntimeError(
            "Login failed — check your username and password in .env"
        )

    return session


def _find_jwt_in_obj(obj) -> str | None:
    """Recursively search a JSON object for the first JWT string (eyJ...eyJ...xxx)."""
    if isinstance(obj, str):
        if obj.startswith("eyJ") and obj.count(".") >= 2:
            return obj
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            result = _find_jwt_in_obj(v)
            if result:
                return result
    if isinstance(obj, list):
        for item in obj:
            result = _find_jwt_in_obj(item)
            if result:
                return result
    return None


def fetch_token_from_session(session: requests.Session) -> str:
    """从 CAS 认证后的 session 中获取 JWT token。"""
    resp = session.get(GET_SUB_INFO_API, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # JWT is in the top-level "token" field when code=0
    token = data.get("token", "")
    if token and token.startswith("eyJ"):
        return token
    # Fallback: recursively search the response in case the API structure changes
    token = _find_jwt_in_obj(data)
    if not token:
        raise RuntimeError(
            "Cannot find JWT token in GET_SUB_INFO_API response. "
            f"Raw: {json.dumps(data, ensure_ascii=False)[:500]}"
        )
    return token


def refresh_token(username: str, password: str) -> tuple[requests.Session, str]:
    """CAS 重新登录并获取新 JWT token。"""
    session = login(username, password)
    token = fetch_token_from_session(session)
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session, token
