import pytest

from scripts.verify_standalone import check_agent_result, parse_sse_final, main


def result(backend):
    return {"status": "completed", "final_answer": "a", "answer_model": "qwen3-vl:4b",
        "metadata": {"execution_mode": "static", "retrieval_backend": backend}, "steps": [
            {"step_type": "retrieve", "metadata": {"retrieval_backend": backend}}]}


def test_json_fallback_cannot_pass_pgvector_smoke():
    with pytest.raises(ValueError, match="pgvector_not_executed"):
        check_agent_result(result("json_cosine"), require_pgvector=True)
    assert check_agent_result(result("pgvector"), require_pgvector=True)["backend"] == "pgvector"


def test_pixel_claim_requires_actual_image_evidence():
    with pytest.raises(ValueError, match="pixels_not_executed"):
        check_agent_result(result("pgvector"), require_pixels=True)


def test_sse_final_is_required_and_not_just_http_success():
    with pytest.raises(ValueError, match="sse_final_missing"):
        parse_sse_final(["event: step", 'data: {"status":"completed"}', ""])
    with pytest.raises(ValueError, match="sse_multiple_finals"):
        parse_sse_final(["event: final", 'data: {}', "", "event: final", 'data: {}', ""])
    assert parse_sse_final(["event: final", 'data: {"status":"completed"}', ""]) == {"status": "completed"}


def test_failed_smoke_exit_code_and_report(tmp_path, monkeypatch):
    import scripts.verify_standalone as module
    monkeypatch.setattr(module, "run_smoke", lambda **kwargs: {"passed": False, "checks": [{"error": "pgvector_not_executed"}]})
    code = main(["--base-url", "http://127.0.0.1:18002", "--project", "p", "--output", str(tmp_path / "result")])
    assert code == 1
    assert (tmp_path / "result/report.json").is_file()


def test_smoke_consumes_real_plain_response_contract(monkeypatch):
    import json
    import httpx
    import scripts.verify_standalone as module
    from scripts.evaluate_adaptive_agent import Case
    agent = result("pgvector")
    def handler(request):
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "models": {"vector_store": {"status": "ready"}}})
        if request.url.path == "/api/query":
            return httpx.Response(200, json={"answer_markdown": "A [1]", "citations": [], "verification_status": "local-only"})
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, text="event: final\ndata: " + json.dumps(agent) + "\n\n")
        return httpx.Response(200, json=agent)
    real_client = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(module, "capture_context", lambda project: {"identity": "one"})
    case = Case(id="text", question="q", document_scope=None, reference_facts=["A"], required_evidence=[], unanswerable=False)
    assert module.run_smoke(base_url="http://127.0.0.1:18002", project="p", cases=[case])["passed"] is True


def test_smoke_cross_turn_uses_separate_json_and_sse_sessions(monkeypatch):
    import json
    import httpx
    import scripts.verify_standalone as module
    from scripts.evaluate_adaptive_agent import Case
    seen = []

    def handler(request):
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "models": {"vector_store": {"status": "ready"}}})
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        is_sse = request.url.path.endswith("/stream")
        if payload["query"] == "首轮":
            assert "session_id" not in payload
        else:
            assert payload["session_id"] == ("sse-session" if is_sse else "json-session")
        agent = {**result("pgvector"), "session_id": "sse-session" if is_sse else "json-session"}
        if is_sse:
            return httpx.Response(200, text="event: final\ndata: " + json.dumps(agent) + "\n\n")
        return httpx.Response(200, json=agent)

    real_client = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(module, "capture_context", lambda project: {"identity": "one"})
    case = Case(id="cross", question="那它呢", setup_questions=["首轮"], group="cross-turn",
        document_scope=None, reference_facts=["A"], required_evidence=[], unanswerable=False)
    report = module.run_smoke(base_url="http://127.0.0.1:18002", project="p", cases=[case])
    assert report["passed"] is True
    assert [payload["query"] for _, payload in seen] == ["首轮", "那它呢", "首轮", "那它呢"]
    assert report["checks"][0]["plain_skipped"] == "stateless_endpoint"


def test_smoke_keeps_failed_pixel_response_with_safe_reason(monkeypatch):
    import httpx
    import scripts.verify_standalone as module
    from scripts.evaluate_adaptive_agent import Case

    def handler(request):
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "models": {"vector_store": {"status": "ready"}}})
        if request.url.path == "/api/query":
            return httpx.Response(200, json={"answer_markdown": "A", "citations": [], "verification_status": "local-only"})
        return httpx.Response(200, json=result("pgvector"))

    real_client = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(module, "capture_context", lambda project: {})
    case = Case(id="pixels", question="图像?", group="pixels", document_scope=None,
        reference_facts=[], required_evidence=[], unanswerable=False)
    report = module.run_smoke(base_url="http://127.0.0.1:18002", project="p", cases=[case])
    row = report["checks"][0]
    assert report["passed"] is False
    assert row["error"] == "pixels_not_executed"
    assert row["json_response"]["final_answer"] == "a"
    assert row["fact_correct"] is None
