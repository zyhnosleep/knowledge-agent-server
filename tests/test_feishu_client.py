"""FeishuClient 单测：tenant_access_token 缓存 / 发文本消息 / 下载文件资源。

所有 HTTP 调用通过 httpx.MockTransport 注入，不触网。
"""

import json

import httpx
import pytest

from app.services.feishu_bot.client import FeishuApiError, FeishuClient


def _make_client(handler) -> FeishuClient:
    transport = httpx.MockTransport(handler)
    return FeishuClient(
        app_id="cli_app",
        app_secret="secret",
        base_url="https://open.feishu.cn",
        transport=transport,
    )


def _token_payload(expire: int = 7200) -> dict:
    # 飞书真实响应为平铺结构：tenant_access_token 在顶层，不在 data 里。
    # （2026-08-24 现场修复：原 mock 用嵌套 data 结构，测试全绿但线上必挂）
    return {
        "code": 0,
        "msg": "ok",
        "tenant_access_token": "t-123",
        "expire": expire,
    }


def test_get_tenant_access_token_parses_data():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_token_payload())

    client = _make_client(handler)
    assert client.get_tenant_access_token() == "t-123"
    assert len(requests) == 1
    assert requests[0].url.path == "/open-apis/auth/v3/tenant_access_token/internal"
    assert json.loads(requests[0].content) == {"app_id": "cli_app", "app_secret": "secret"}


def test_token_is_cached_within_expiry():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_token_payload(expire=7200))

    client = _make_client(handler)
    assert client.get_tenant_access_token() == "t-123"
    assert client.get_tenant_access_token() == "t-123"
    assert len(requests) == 1


def test_token_refetched_after_expiry(monkeypatch):
    import time

    requests = []
    real_monotonic = time.monotonic
    current = [real_monotonic()]

    def fake_monotonic() -> float:
        return current[0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_token_payload(expire=2))

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    client = _make_client(handler)
    assert client.get_tenant_access_token() == "t-123"
    # 推 3 秒，越过 2 秒有效期（实现留 60 秒余量，此刻必然过期）→ 应重新请求
    current[0] += 3
    assert client.get_tenant_access_token() == "t-123"
    assert len(requests) == 2


def test_token_business_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 10003, "msg": "invalid secret"})

    client = _make_client(handler)
    with pytest.raises(FeishuApiError, match="invalid secret"):
        client.get_tenant_access_token()


def test_send_text_posts_to_chat():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json=_token_payload())
        assert request.method == "POST"
        assert request.url.path == "/open-apis/im/v1/messages"
        assert request.url.params["receive_id_type"] == "chat_id"
        assert request.headers["Authorization"] == "Bearer t-123"
        assert json.loads(request.content) == {
            "receive_id": "oc_chat_1",
            "msg_type": "text",
            "content": json.dumps({"text": "hello"}, ensure_ascii=False),
        }
        return httpx.Response(200, json={"code": 0, "msg": "ok"})

    client = _make_client(handler)
    client.send_text(chat_id="oc_chat_1", text="hello")
    assert len(requests) == 2  # 先取 token，再发消息


def test_send_text_business_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_token_payload()) if request.url.path.endswith(
            "tenant_access_token/internal"
        ) else httpx.Response(200, json={"code": 230002, "msg": "bot not in chat"})

    client = _make_client(handler)
    with pytest.raises(FeishuApiError, match="bot not in chat"):
        client.send_text(chat_id="oc_chat_1", text="hi")


def test_download_file_returns_bytes():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json=_token_payload())
        assert request.method == "GET"
        assert request.url.path == "/open-apis/im/v1/messages/om_1/resources/file_abc"
        assert request.url.params["type"] == "file"
        assert request.headers["Authorization"] == "Bearer t-123"
        return httpx.Response(200, content=b"%PDF-1.4 fake")

    client = _make_client(handler)
    assert client.download_file(message_id="om_1", file_key="file_abc") == b"%PDF-1.4 fake"
    assert len(requests) == 2


def test_download_file_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json=_token_payload())
        return httpx.Response(200, json={"code": 1061002, "msg": "file expired"})

    client = _make_client(handler)
    with pytest.raises(FeishuApiError, match="file expired"):
        client.download_file(message_id="om_1", file_key="file_abc")


def test_fetch_ws_url_requests_endpoint():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        # endpoint 接口用 AppID/AppSecret 直接鉴权，不请求 token
        assert request.method == "POST"
        assert request.url.path == "/callback/ws/endpoint"
        assert request.headers["locale"] == "zh"
        assert json.loads(request.content) == {
            "AppID": "cli_app",
            "AppSecret": "secret",
            "ClientAssertion": "",
        }
        return httpx.Response(
            200,
            json={"code": 0, "msg": "ok", "data": {"URL": "wss://ws.feishu.cn/x?service_id=s"}},
        )

    client = _make_client(handler)
    assert client.fetch_ws_url() == "wss://ws.feishu.cn/x?service_id=s"
    assert len(requests) == 1


def test_get_bot_open_id():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json=_token_payload())
        assert request.method == "GET"
        assert request.url.path == "/open-apis/bot/v3/info"
        assert request.headers["Authorization"] == "Bearer t-123"
        return httpx.Response(
            200, json={"code": 0, "msg": "ok", "bot": {"open_id": "ou_bot_1"}}
        )

    client = _make_client(handler)
    assert client.get_bot_open_id() == "ou_bot_1"
    assert len(requests) == 2


def test_get_bot_open_id_business_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json=_token_payload())
        return httpx.Response(200, json={"code": 10003, "msg": "bot disabled"})

    client = _make_client(handler)
    with pytest.raises(FeishuApiError, match="bot disabled"):
        client.get_bot_open_id()


def test_fetch_ws_url_business_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json=_token_payload())
        return httpx.Response(200, json={"code": 99991663, "msg": "long connection closed"})

    client = _make_client(handler)
    with pytest.raises(FeishuApiError, match="long connection closed"):
        client.fetch_ws_url()
