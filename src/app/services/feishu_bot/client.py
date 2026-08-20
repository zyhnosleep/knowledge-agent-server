"""
client.py —— 飞书开放平台 API 客户端（机器人用）
==================================================

职责：
- tenant_access_token 获取与缓存（过期前 60 秒提前刷新）。
- 向会话发文本消息（``im.message:send_as_bot`` 权限）。
- 下载消息资源文件（``im:resource`` 权限，入库用）。

风格：与项目其余服务一致（ai.py / auth.py），同步 ``httpx.Client``；
异步 ws 事件循环侧经 ``asyncio.to_thread`` 调用。
"""

from __future__ import annotations

import json
import time

import httpx


class FeishuApiError(Exception):
    """飞书开放平台业务错误：HTTP 200 但 ``code != 0``。"""


class FeishuClient:
    """飞书机器人 API 客户端。

    参数：
    - ``app_id`` / ``app_secret``：开放平台应用凭证（与 OAuth 登录同一应用）。
    - ``base_url``：开放平台根地址（国内 open.feishu.cn，海外 open.larksuite.com）。
    - ``timeout``：请求超时秒数。
    - ``transport``：测试注入用（httpx.MockTransport）。
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        base_url: str = "https://open.feishu.cn",
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._app_id = app_id
        self._app_secret = app_secret
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _request_tenant_access_token(self) -> str:
        """POST /open-apis/auth/v3/tenant_access_token/internal 换 token。"""
        response = self._client.post(
            "/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise FeishuApiError(f"tenant_access_token failed: {payload.get('msg')}")
        data = payload.get("data") or {}
        token = data.get("tenant_access_token")
        if not token:
            raise FeishuApiError("tenant_access_token missing from response")
        self._token = token
        # expire 秒有效；留 60 秒余量提前刷新，避免临界期 401
        self._token_expires_at = time.monotonic() + float(data.get("expire", 7200)) - 60
        return token

    def get_tenant_access_token(self) -> str:
        """返回缓存未过期的 token，否则重新获取。"""
        if self._token is not None and time.monotonic() < self._token_expires_at:
            return self._token
        return self._request_tenant_access_token()

    def send_text(self, chat_id: str, text: str) -> None:
        """向 ``chat_id`` 会话发送纯文本消息。"""
        response = self._client.post(
            "/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            headers={"Authorization": f"Bearer {self.get_tenant_access_token()}"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise FeishuApiError(f"send message failed: {payload.get('msg')}")

    def download_file(self, message_id: str, file_key: str) -> bytes:
        """下载消息内文件资源（``?type=file``），返回原始字节。

        失败时飞书返回 JSON ``{code, msg}``；成功是二进制内容。
        """
        response = self._client.get(
            f"/open-apis/im/v1/messages/{message_id}/resources/{file_key}",
            params={"type": "file"},
            headers={"Authorization": f"Bearer {self.get_tenant_access_token()}"},
        )
        response.raise_for_status()
        if "json" in response.headers.get("content-type", ""):
            payload = response.json()
            if payload.get("code") != 0:
                raise FeishuApiError(f"download file failed: {payload.get('msg')}")
        return response.content

    def fetch_ws_url(self, region: str = "cn") -> str:
        """获取长连接 WebSocket 地址（POST /open-apis/ws/v1/endpoint）。

        返回 ``data.ws_url``（形如 ``wss://...?service_id=...&device_id=...``）。
        """
        response = self._client.post(
            "/open-apis/ws/v1/endpoint",
            params={"region": region},
            headers={"Authorization": f"Bearer {self.get_tenant_access_token()}"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise FeishuApiError(f"fetch ws endpoint failed: {payload.get('msg')}")
        data = payload.get("data") or {}
        ws_url = data.get("ws_url")
        if not ws_url:
            raise FeishuApiError("ws_url missing from endpoint response")
        return ws_url

    def get_bot_open_id(self) -> str:
        """查询机器人自身 open_id（GET /open-apis/bot/v3/info）。

        用于 @ 提及判断：事件 mentions 里含该 open_id 才算 @ 了机器人。
        """
        response = self._client.get(
            "/open-apis/bot/v3/info",
            headers={"Authorization": f"Bearer {self.get_tenant_access_token()}"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise FeishuApiError(f"get bot info failed: {payload.get('msg')}")
        bot = (payload.get("data") or {}).get("bot") or {}
        open_id = bot.get("open_id")
        if not open_id:
            raise FeishuApiError("bot open_id missing from response")
        return open_id
