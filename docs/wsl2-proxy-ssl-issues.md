# WSL2 + 代理环境下的 SSL 问题排查记录

## 背景

在 Windows 上运行正常的 `monitor` 命令，迁移到 WSL2 后出现一系列 SSL 连接失败。
本文记录排查过程和最终解决方案。

---

## 问题一：`DH_KEY_TOO_SMALL` — Schedule API 请求失败

### 现象

```
Error fetching schedule: Schedule API request failed:
HTTPSConnectionPool(host='yjapi.cmc.zju.edu.cn', port=443): Max retries exceeded
(Caused by SSLError(SSLError(1, '[SSL: DH_KEY_TOO_SMALL] dh key too small (_ssl.c:1016)')))
```

### 分析

1. **直连测试**：`openssl s_client -connect yjapi.cmc.zju.edu.cn:443` 成功，服务器使用 TLS 1.3 + X25519，无 DHE。
2. **代理检测**：环境变量中存在 `https_proxy=http://172.27.112.1:7890`（Windows 宿主机的 Clash 代理）。
3. **根因**：Python 的 `requests` 走代理时，代理（Clash）与服务器协商出了 TLS 1.2 + DHE 密码套件，DHE 密钥过短（< 2048 bit），被 OpenSSL 3.x 拒绝。`openssl s_client` 不读系统代理，直连时服务器选了 TLS 1.3/ECDH，绕开了这个问题。

### 尝试过但无效的方案

| 方案 | 结果 | 原因 |
|------|------|------|
| `ctx.set_ciphers("DEFAULT:@SECLEVEL=1")` | 失败 | SECLEVEL=1 最低 DHE 仍为 1024 bit，不够 |
| `ctx.set_ciphers("DEFAULT:@SECLEVEL=0")` 仅挂载 `init_poolmanager` | 失败 | 代理连接走 `proxy_manager_for`，该方法未被覆盖 |
| `ctx.minimum_version = ssl.TLSVersion.TLSv1_3` | 失败 | 代理侧强制降级到 TLS 1.2 |
| `ssl.OP_LEGACY_SERVER_CONNECT` | 失败 | 仅针对 renegotiation，不影响 DHE key size 检查 |
| `s.proxies = {"no_proxy": "yjapi.cmc.zju.edu.cn"}` | 失败 | requests 的 `no_proxy` 只读环境变量，不认 session.proxies 中的该 key |
| `s.trust_env = False`（绕过代理直连） | 连通但触发 WAF | 服务器部署了阿里云 WAF，直连被 JS challenge 拦截，返回 HTML 而非 JSON |

### 最终解决方案

自定义 `HTTPAdapter`，同时覆盖 `init_poolmanager`（直连路径）和 `proxy_manager_for`（代理路径），对两条路径都注入 `SECLEVEL=0` 的 SSL context：

```python
class _DHFixAdapter(HTTPAdapter):
    def _ctx(self):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
        return ctx

    def init_poolmanager(self, *args, **kwargs):
        kwargs.setdefault("ssl_context", self._ctx())
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs.setdefault("ssl_context", self._ctx())
        return super().proxy_manager_for(proxy, **proxy_kwargs)
```

`SECLEVEL=0` 只放宽密钥强度限制，不禁用证书验证，对安全性影响极小。

---

## 问题二：auth 流程中多个子域名均受影响

### 现象

修复 Schedule API 后，`refresh_token` 触发登录，报新错：

```
SSLError: HTTPSConnectionPool(host='tgmedia.cmc.zju.edu.cn', port=443): ...
(Caused by SSLError(SSLError(1, '[SSL: DH_KEY_TOO_SMALL] dh key too small')))
```

### 分析

ZJU CAS 认证的重定向链经过多个 `*.cmc.zju.edu.cn` 子域：

```
yjapi.cmc.zju.edu.cn → zjuam.zju.edu.cn (CAS) → tgmedia.cmc.zju.edu.cn → classroom.zju.edu.cn
```

所有 `*.cmc.zju.edu.cn` 均部署在同一套服务器上，都有相同的弱 DHE 问题。

### 解决方案

将 `_DHFixAdapter` 挂载到 `https://` 前缀（匹配所有 HTTPS 请求），而非逐一列出受影响的子域。auth session 和 schedule session 都统一调用 `mount_legacy_ssl(session)`：

```python
def mount_legacy_ssl(session: requests.Session) -> requests.Session:
    session.mount("https://", _DHFixAdapter())
    return session
```

封装在 `src/session_utils.py` 中，供 `auth.py` 和 `live_monitor.py` 共用。

---

## 总结

| 组件 | 问题 | 修复位置 |
|------|------|----------|
| `fetch_live_courses` | Schedule API 走代理报 SSL 错 | `live_monitor._make_schedule_session` |
| `login` / `refresh_token` | CAS 认证重定向链报 SSL 错 | `auth.login` |
| 共用 | DHFix adapter 逻辑重复 | 提取为 `src/session_utils.mount_legacy_ssl` |

## 注意事项

- 此问题**仅在通过代理访问时出现**，Windows 原生环境直连无此问题。
- 若未来代理更换或代理配置更新（升级 TLS 版本），可移除 `mount_legacy_ssl` 调用。
- `SECLEVEL=0` 不影响证书链验证（`verify=True` 仍然生效）。
