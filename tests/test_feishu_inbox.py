"""T2 文件入库链路单测：ServerApi（本地 API 客户端）+ InboxIngestor（下载→上传→轮询→回执）。

HTTP 调用经 httpx.MockTransport 注入；下载经 FeishuClient 替身。
"""

import httpx
import pytest

from app.services.feishu_bot.client import FeishuApiError
from app.services.feishu_bot.events import parse_message_event
from app.services.feishu_bot.inbox import InboxIngestor, IngestResult, ServerApi


def _file_event(**overrides) -> dict:
    base = {
        "schema": "2.0",
        "header": {
            "event_id": "evt_1",
            "event_type": "im.message.receive_v1",
            "create_time": "1700000000000",
            "token": "tok",
            "app_id": "cli_app",
            "tenant_key": "tenant_a",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": "ou_user", "union_id": "u", "user_id": "uid"},
                "sender_type": "user",
                "tenant_key": "tenant_a",
            },
            "message": {
                "message_id": "om_1",
                "root_id": "",
                "parent_id": "",
                "create_time": "1700000000000",
                "chat_id": "oc_group",
                "chat_type": "group",
                "message_type": "file",
                "content": '{"file_key":"file_abc","file_name":"报告.pdf","file_size":12345}',
                "mentions": [],
            },
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


class _FakeFeishu:
    """FeishuClient 替身：download_file 可控。"""

    def __init__(self, content: bytes | None = None, error: str | None = None) -> None:
        self.content = content
        self.error = error

    def download_file(self, message_id: str, file_key: str) -> bytes:
        if self.error:
            raise FeishuApiError(self.error)
        return self.content or b""


def _make_server_api(handler) -> ServerApi:
    return ServerApi(
        base_url="http://127.0.0.1:8002",
        transport=httpx.MockTransport(handler),
    )


# ---- ServerApi ----

def test_upload_posts_multipart_and_returns_document_id():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/ingest/upload"
        assert request.url.params["project_slug"] == "feishu-inbox"
        assert request.url.params["project_name"] == "飞书收件箱"
        assert "报告.pdf" in request.content.decode("utf-8", errors="replace")
        assert b"%PDF-1.4" in request.content
        return httpx.Response(
            200,
            json={"document_id": "doc_1", "status": "pending", "project_slug": "feishu-inbox"},
        )

    api = _make_server_api(handler)
    document_id = api.upload(
        project_slug="feishu-inbox",
        project_name="飞书收件箱",
        filename="报告.pdf",
        content=b"%PDF-1.4 fake",
    )
    assert document_id == "doc_1"
    assert len(requests) == 1


def test_upload_business_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "File name is required."})

    api = _make_server_api(handler)
    with pytest.raises(RuntimeError, match="File name is required"):
        api.upload("feishu-inbox", "飞书收件箱", "报告.pdf", b"x")


def test_document_status_returns_status():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/documents/doc_1"
        return httpx.Response(200, json={"id": "doc_1", "status": "ready"})

    api = _make_server_api(handler)
    assert api.document_status("doc_1") == "ready"


# ---- InboxIngestor ----

def test_ingest_success_downloads_uploads_polls():
    """全链路：下载 → 上传 → 轮询到 ready → ok 结果。"""
    states = iter(["pending", "processing", "ready"])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ingest/upload":
            return httpx.Response(
                200, json={"document_id": "doc_1", "status": "pending"}
            )
        assert request.url.path == "/api/documents/doc_1"
        return httpx.Response(200, json={"id": "doc_1", "status": next(states)})

    ingestor = InboxIngestor(
        feishu=_FakeFeishu(content=b"%PDF-1.4"),
        api=_make_server_api(handler),
        inbox_project="feishu-inbox",
        inbox_project_name="飞书收件箱",
        poll_interval=0.01,
        poll_timeout=5,
    )
    result = ingestor.ingest_file(parse_message_event(_file_event()))
    assert result.ok is True
    assert result.document_id == "doc_1"
    assert result.file_name == "报告.pdf"
    assert result.error is None


def test_ingest_download_failure_reports_reason():
    ingestor = InboxIngestor(
        feishu=_FakeFeishu(error="file expired"),
        api=_make_server_api(lambda req: httpx.Response(500)),
        inbox_project="feishu-inbox",
        inbox_project_name="飞书收件箱",
        poll_interval=0.01,
        poll_timeout=5,
    )
    result = ingestor.ingest_file(parse_message_event(_file_event()))
    assert result.ok is False
    assert "file expired" in result.error


def test_ingest_upload_failure_reports_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "Unsupported file type"})

    ingestor = InboxIngestor(
        feishu=_FakeFeishu(content=b"x"),
        api=_make_server_api(handler),
        inbox_project="feishu-inbox",
        inbox_project_name="飞书收件箱",
        poll_interval=0.01,
        poll_timeout=5,
    )
    result = ingestor.ingest_file(parse_message_event(_file_event()))
    assert result.ok is False
    assert "Unsupported file type" in result.error


def test_ingest_terminal_failure_status_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ingest/upload":
            return httpx.Response(200, json={"document_id": "doc_1", "status": "pending"})
        return httpx.Response(200, json={"id": "doc_1", "status": "parse_failed"})

    ingestor = InboxIngestor(
        feishu=_FakeFeishu(content=b"x"),
        api=_make_server_api(handler),
        inbox_project="feishu-inbox",
        inbox_project_name="飞书收件箱",
        poll_interval=0.01,
        poll_timeout=5,
    )
    result = ingestor.ingest_file(parse_message_event(_file_event()))
    assert result.ok is False
    assert "parse_failed" in result.error


def test_ingest_poll_timeout_reports_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ingest/upload":
            return httpx.Response(200, json={"document_id": "doc_1", "status": "pending"})
        return httpx.Response(200, json={"id": "doc_1", "status": "processing"})

    ingestor = InboxIngestor(
        feishu=_FakeFeishu(content=b"x"),
        api=_make_server_api(handler),
        inbox_project="feishu-inbox",
        inbox_project_name="飞书收件箱",
        poll_interval=0.01,
        poll_timeout=0.05,
    )
    result = ingestor.ingest_file(parse_message_event(_file_event()))
    assert result.ok is False
    assert "超时" in result.error
