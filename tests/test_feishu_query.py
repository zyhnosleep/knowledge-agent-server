"""T3 文本问答链路单测：ServerApi.query + Answerer（@ 提问 → 答案推送）。"""

import httpx
import pytest

from app.services.feishu_bot.inbox import ServerApi
from app.services.feishu_bot.query import Answerer, QueryResult


def _make_server_api(handler) -> ServerApi:
    return ServerApi(
        base_url="http://127.0.0.1:8002",
        transport=httpx.MockTransport(handler),
    )


def _completed_payload() -> dict:
    # citations 是 Citation 形状（schemas/common.py）：页面标题在 page_title
    return {
        "request_id": "req_1",
        "session_id": "sess_1",
        "status": "completed",
        "final_answer": "结论是……",
        "citations": [
            {"document_id": "doc_1", "page_title": "报告.pdf", "score": 0.9, "excerpt": "x"},
            {"document_id": "doc_1", "page_title": "报告.pdf", "score": 0.8, "excerpt": "y"},
            {"document_id": "doc_2", "page_title": "附录.docx", "score": 0.7, "excerpt": "z"},
        ],
        "steps": [],
        "usage": {},
    }


# ---- ServerApi.query ----

def test_query_posts_to_agent_endpoint():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/agent/query"
        body = request.read().decode("utf-8")
        assert '"project_slug":"feishu-inbox"' in body
        assert '"query":"xxx"' in body
        return httpx.Response(200, json=_completed_payload())

    api = _make_server_api(handler)
    payload = api.query("feishu-inbox", "xxx")
    assert payload["status"] == "completed"
    assert payload["final_answer"] == "结论是……"
    assert len(requests) == 1


def test_query_business_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "Agent service is not enabled"})

    api = _make_server_api(handler)
    with pytest.raises(RuntimeError, match="Agent service is not enabled"):
        api.query("feishu-inbox", "xxx")


# ---- Answerer ----

def test_answerer_completed_returns_answer_and_unique_citations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completed_payload())

    answerer = Answerer(_make_server_api(handler), project_slug="feishu-inbox")
    result = answerer.answer("xxx")
    assert result.ok is True
    assert result.answer == "结论是……"
    # 引用去重且保序
    assert result.citations == ["报告.pdf", "附录.docx"]


@pytest.mark.parametrize(
    "status,expected",
    [
        ("timeout", "超时"),
        ("error", "问答失败"),
        ("max_steps", "问答失败"),
    ],
)
def test_answerer_non_completed_status_fails_with_reason(status, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _completed_payload()
        payload["status"] = status
        return httpx.Response(200, json=payload)

    answerer = Answerer(_make_server_api(handler), project_slug="feishu-inbox")
    result = answerer.answer("xxx")
    assert result.ok is False
    assert expected in result.error


def test_answerer_api_exception_fails_with_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "boom"})

    answerer = Answerer(_make_server_api(handler), project_slug="feishu-inbox")
    result = answerer.answer("xxx")
    assert result.ok is False
    assert "boom" in result.error


def test_answerer_empty_question_reports_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty question should not reach API")

    answerer = Answerer(_make_server_api(handler), project_slug="feishu-inbox")
    result = answerer.answer("   ")
    assert result.ok is False
    assert "问题" in result.error


def test_answerer_truncates_very_long_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _completed_payload()
        payload["final_answer"] = "长" * 5000
        return httpx.Response(200, json=payload)

    answerer = Answerer(_make_server_api(handler), project_slug="feishu-inbox")
    result = answerer.answer("xxx")
    assert result.ok is True
    assert len(result.answer) <= 4000
    assert "截断" in result.answer
