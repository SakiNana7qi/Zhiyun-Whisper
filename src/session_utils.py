"""
Shared session utilities.

Provides a requests HTTPAdapter that works around the DH_KEY_TOO_SMALL SSL
error on ZJU servers (*.cmc.zju.edu.cn) when accessed through a proxy.
The proxy causes TLS 1.2 DHE negotiation with weak parameters;
SECLEVEL=0 allows Python to accept them.
"""

import ssl

import requests
from requests.adapters import HTTPAdapter


class _DHFixAdapter(HTTPAdapter):
    """Accept weak DH keys for both direct and proxied connections."""

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


def mount_legacy_ssl(session: requests.Session) -> requests.Session:
    """Mount the DHFix adapter for all HTTPS on this session. Returns the session."""
    session.mount("https://", _DHFixAdapter())
    return session
