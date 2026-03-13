# ZJU 智云课堂认证流程逆向记录

本文记录如何通过分析前端 JS 代码，还原智云课堂的 CAS 登录流程与 JWT 获取方式。对应实现在 `src/auth.py`。

## 背景

`ZJU_TOKEN` 是一个 JWT，约 24 小时过期。过期后调用课表 API（`SCHEDULE_API`）会返回：

```json
{"success": true, "result": {"err": 0, "errMsg": "", "data": "请求用户接口错误,用户认证失败！"}}
```

为了让监控进程自动刷新 Token，需要从代码层面实现完整的 ZJU 统一身份认证（CAS）登录流程。

---

## 第一步：找到真正的登录入口

直接访问 `https://classroom.zju.edu.cn` 不会触发服务器端跳转（它是 SPA），无法通过跟踪 `Location` 头获取 CAS service URL。

通过请求 `https://classroom.zju.edu.cn/static/config.js`，找到关键字段：

```javascript
window.CONFIG = {
  LOGIN_URL: "https://tgmedia.cmc.zju.edu.cn",   // 登录地址
  CASAPI: "https://tgmedia.cmc.zju.edu.cn",       // CAS API 地址
  YJAPI: "https://yjapi.cmc.zju.edu.cn",
  TENANT_ID: 112,
  ...
}
```

### 完整 Redirect 链

访问 `https://tgmedia.cmc.zju.edu.cn`，跟随所有 302 跳转：

```
GET https://tgmedia.cmc.zju.edu.cn/
  → 302 → https://yjapi.cmc.zju.edu.cn/casapi/index.php
              ?r=auth/login
              &forward=https://classroom.zju.edu.cn/
              &tenant_code=112

  → 302 → https://zjuam.zju.edu.cn/cas/oauth2.0/authorize
              ?response_type=code
              &client_id=ObXIv4FvjcC1e9hVcS
              &redirect_uri=https://tgmedia.cmc.zju.edu.cn/index.php?r=auth/get-info&url=https://classroom.zju.edu.cn/

  → 302 → https://zjuam.zju.edu.cn/cas/login
              ?service=http://zjuam.zju.edu.cn/cas/oauth2.0/callbackAuthorize
```

最终落到标准的 ZJU CAS 登录页，其中 `service` 参数是 OAuth 2.0 回调地址，**不是** `https://classroom.zju.edu.cn`。

因此，代码中的 auth 入口定义为：

```python
CLASSROOM_AUTH_INIT = (
    "https://yjapi.cmc.zju.edu.cn/casapi/index.php"
    "?r=auth/login&forward=https://classroom.zju.edu.cn/&tenant_code=112"
)
```

登录前先 `GET CLASSROOM_AUTH_INIT`（`allow_redirects=True`），让 session 跟完整个跳转链，再从最终落点的页面提取 `execution` token，并把该 URL 作为 POST 目标。这样 session 中已存好各个中间节点的 cookie，后续的 OAuth 回调才能正常处理。

---

## 第二步：搞清楚密码加密算法

CAS 登录页通过 RSA 对密码加密后再提交，避免明文传输。加载的 JS 文件：

```
js/login/security.js   — RSA 实现（自定义 BigInt + 无填充 raw RSA）
js/login/login.js      — 登录逻辑
```

### login.js 中的关键代码

```javascript
jQuery.getJSON("v2/getPubKey", function(data) {
    Modulus = data["modulus"];
    public_exponent = data["exponent"];
});

// 提交时：
var password = $("#password").val();
var key = new RSAUtils.getKeyPair(public_exponent, "", Modulus);
var reversedPwd = password.split("").reverse().join("");   // ← 密码先反转！
var encrypedPwd = RSAUtils.encryptedString(key, reversedPwd);
$("#password").val(encrypedPwd);
```

**第一个坑：密码在加密前需要反转字符串**（`"abc"` → `"cba"`）。

### security.js 中的加密逻辑

`encryptedString(key, s)` 的工作过程：

```
1. 将字符串转成 char code 数组
2. 用零字节补齐到 chunkSize 的整数倍
   chunkSize = 2 * biHighIndex(modulus)
             = 2 * (模数的 16-bit 字数 - 1)
             例：512-bit 密钥 → chunkSize = 62
3. 对每个 chunk（chunkSize 字节）：
   - 按小端序（little-endian）解释为一个大整数
   - 计算 block^e mod n（raw RSA，无填充）
   - 结果转为十六进制，按 4 字符（16-bit）对齐
4. 所有 chunk 的十六进制串用空格拼接，作为最终密码值提交
```

**第二个坑：这不是标准的 PKCS#1 v1.5 加密。** 用 `pycryptodome` 的 `PKCS1_v1_5` 会产生完全不同的结果，导致服务器报「用户名或密码错误」。

Python 实现只需用内置的 `pow(block, e, n)` 即可：

```python
n = int(modulus_hex, 16)
e = int(exponent_hex, 16)
n_words = (n.bit_length() + 15) // 16
chunk_size = 2 * (n_words - 1)

reversed_pwd = password[::-1]
a = [ord(c) for c in reversed_pwd]
while len(a) % chunk_size != 0:
    a.append(0)

parts = []
for i in range(0, len(a), chunk_size):
    block = int.from_bytes(bytes(a[i:i+chunk_size]), "little")
    enc = pow(block, e, n)
    h = format(enc, "x")
    if len(h) % 4:
        h = "0" * (4 - len(h) % 4) + h
    parts.append(h)

encrypted_password = " ".join(parts)
```

---

## 第三步：登录后拿到 JWT

POST 登录成功后，session 跟随 redirect 链完成 OAuth 回调，最终落在 `classroom.zju.edu.cn`。此时 session 中存有 `yjapi.cmc.zju.edu.cn` 等域的认证 cookie。

直接请求：

```
GET https://classroom.zju.edu.cn/courseapi/v3/portal-home-setting/get-sub-info
```

返回：

```json
{
  "code": 0,
  "msg": "token合法",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2NvdW50Ij...",
  "userinfo": [...]
}
```

顶层 `token` 字段即为所需的 JWT，可直接用于 `Authorization: Bearer <token>` 请求头。

---

## 完整流程总结

```
CLASSROOM_AUTH_INIT (GET, allow_redirects=True)
  → session 获得所有中间 cookie
  → 落在 CAS 登录页，提取 execution token 和 POST URL

getPubKey (GET)
  → 获得 modulus、exponent（hex 字符串）

RSA 加密：
  reversed = password[::-1]
  encrypted = custom_raw_rsa(reversed, exponent, modulus)   # 非 PKCS1

POST /cas/login?service=http://zjuam.zju.edu.cn/cas/oauth2.0/callbackAuthorize
  → 302 → OAuth callback → tgmedia/get-info → classroom.zju.edu.cn

GET /courseapi/v3/portal-home-setting/get-sub-info
  → response["token"] = JWT
```

---

## 参考

- CAS 登录页脚本：`https://zjuam.zju.edu.cn/cas/js/login/login.js`
- RSA 实现：`https://zjuam.zju.edu.cn/cas/js/login/security.js`
- 前端配置：`https://classroom.zju.edu.cn/static/config.js`
- OAuth client_id（截至 2026-03）：`ObXIv4FvjcC1e9hVcS`
