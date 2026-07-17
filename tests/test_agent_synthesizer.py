from __future__ import annotations

import pytest

from app.services.agent_synthesizer import AgentSynthesizer


class FakeOllamaClient:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def generate_structured(self, schema, **kwargs):
        self.calls.append({"schema": schema, **kwargs})
        if self.error is not None:
            raise self.error
        return schema.model_validate(self.result)


def _local_result(**overrides):
    result = {
        "answer_markdown": "Entropy measures the number of accessible states [0].",
        "cited_indexes": [0],
        "warnings": [],
        "confidence": 0.9,
    }
    result.update(overrides)
    return result


def test_synthesize_auto_without_external_api_uses_local_ollama(monkeypatch) -> None:
    """Auto mode uses Ollama when the external API is disabled."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_auto_disabled)
    ollama = FakeOllamaClient(result=_local_result())
    syn = AgentSynthesizer(ollama_client=ollama)
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    assert result["provider"] == "local"
    assert result["model"] == "qwen3:14b"
    assert result["answer_markdown"] == "Entropy measures the number of accessible states [0]."
    assert len(result["warnings"]) == 0
    assert ollama.calls[0]["model"] == "qwen3:14b"
    assert "entropy defined" in ollama.calls[0]["user_prompt"]


def test_synthesize_local_returns_structured_ollama_answer(monkeypatch) -> None:
    """The local provider synthesizes evidence through Ollama."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient(result=_local_result()))
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    assert result["provider"] == "local"
    assert result["model"] == "qwen3:14b"
    assert result["answer_markdown"] == "Entropy measures the number of accessible states [0]."


def test_synthesize_local_failure_returns_evidence_fallback_with_warning(monkeypatch) -> None:
    """Ollama failures preserve the evidence answer and report degraded mode."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(
        ollama_client=FakeOllamaClient(error=RuntimeError("Ollama unavailable"))
    )

    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert result["provider"] == "local"
    assert result["model"] == "local-fallback"
    assert result["answer_markdown"] == "Entropy is a measure of disorder."
    assert any("ollama" in warning.lower() for warning in result["warnings"])


def test_synthesize_local_sanitizes_cited_indexes(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(
        ollama_client=FakeOllamaClient(
            result=_local_result(cited_indexes=[0, 2, -1, "0"])
        )
    )

    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert result["cited_indexes"] == [0]


def test_synthesize_local_without_evidence_skips_ollama(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(result=_local_result())
    syn = AgentSynthesizer(ollama_client=ollama)

    result = syn.synthesize(
        query="Say hello",
        route="simple_rag",
        rag_answer="No matching evidence was found.",
        citations=[],
        evidence_pack={"status": "empty", "items": []},
    )

    assert result["answer_markdown"] == "No matching evidence was found."
    assert result["model"] == "local-fallback"
    assert ollama.calls == []


def test_synthesize_external_api_when_enabled_returns_structured_result(monkeypatch) -> None:
    """When provider is 'external_api' and API is enabled, call external API and return structured result."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_external)
    # Stub the httpx call to return a structured synthesis
    import httpx

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"answer_markdown": "Synthesized answer.", "cited_indexes": [0], "warnings": [], "confidence": 0.9}'
                        }
                    }
                ]
            }

    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: FakeResponse())
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    assert result["provider"] == "external_api"
    assert result["model"] == "gpt-4o-mini"
    assert "answer_markdown" in result
    assert "cited_indexes" in result
    assert "confidence" in result


def test_synthesize_external_api_failure_returns_fallback_with_warning(monkeypatch) -> None:
    """When external API call fails, return fallback with a warning instead of crashing."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_external)
    import httpx

    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: (_ for _ in ()).throw(ValueError("API error")))
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    assert result["provider"] == "external_api"
    assert result["answer_markdown"] == "Entropy is a measure of disorder."
    assert any("fallback" in w.lower() or "failure" in w.lower() or "error" in w.lower() for w in result["warnings"])


def test_synthesize_external_api_empty_answer_returns_fallback(monkeypatch) -> None:
    """An external empty answer never leaves the Agent UI blank."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_external)
    import httpx

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"answer_markdown": "", "cited_indexes": [0], "warnings": ["weak evidence"], "confidence": 0.2}'
                        }
                    }
                ]
            }

    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: FakeResponse())
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    assert result["provider"] == "external_api"
    assert result["answer_markdown"] == "Entropy is a measure of disorder."
    assert any("empty answer" in w.lower() for w in result["warnings"])


def test_synthesize_result_has_required_fields(monkeypatch) -> None:
    """Result dict contains all required keys: answer_markdown, cited_indexes, warnings, confidence, provider, model."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient(result=_local_result()))
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    for field in ("answer_markdown", "cited_indexes", "warnings", "confidence", "provider", "model"):
        assert field in result, f"Missing required field: {field}"


def test_synthesize_sanitizes_cited_indexes(monkeypatch) -> None:
    """Nonexistent citation indexes are removed from cited_indexes."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient(result=_local_result()))
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    # With the local fallback, only valid indexes within citations range should remain
    if result.get("cited_indexes"):
        for idx in result["cited_indexes"]:
            assert 0 <= idx < 1, f"Cited index {idx} out of range for 1 citation"


def _fake_settings_auto_disabled():
    from app.core.config import Settings

    s = Settings()
    s.agent_synthesis_provider = "auto"
    s.external_api_enabled = False
    s.external_api_key = None
    s.external_api_model = "gpt-4o-mini"
    s.external_api_base_url = "https://api.openai.com/v1"
    s.external_api_timeout = 90
    return s


def _fake_settings_local():
    from app.core.config import Settings

    s = Settings()
    s.agent_synthesis_provider = "local"
    s.external_api_enabled = False
    s.external_api_key = None
    s.external_api_model = "gpt-4o-mini"
    s.external_api_base_url = "https://api.openai.com/v1"
    s.external_api_timeout = 90
    return s


# ------------------------------------------------------------------
# evidence pack synthesis tests
# ------------------------------------------------------------------


def test_synthesize_accepts_optional_evidence_pack(monkeypatch) -> None:
    """synthesize() accepts optional evidence_pack parameter without error."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_local
    )
    syn = AgentSynthesizer()
    evidence_pack = {
        "status": "ok",
        "items": [
            {
                "index": 0,
                "document_id": "d1",
                "score": 0.95,
                "excerpt": "entropy defined",
                "evidence_kind": "source_chunk",
                "source_stage": "source_chunk",
                "support_hint": "direct",
            }
        ],
    }
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
        evidence_pack=evidence_pack,
    )
    assert result["answer_markdown"] == "Entropy is a measure of disorder."
    assert result["provider"] == "local"


def test_synthesize_local_fallback_identical_with_evidence_pack(monkeypatch) -> None:
    """Local fallback returns identical result whether evidence_pack is provided or not."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_local
    )
    syn = AgentSynthesizer()

    kwargs = {
        "query": "What is entropy?",
        "route": "simple_rag",
        "conversation_summary": "",
        "rag_answer": "Entropy is a measure of disorder.",
        "citations": [{"document_id": "d1", "excerpt": "entropy defined"}],
    }

    result_without = syn.synthesize(**kwargs)
    result_with = syn.synthesize(
        **kwargs,
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "entropy defined",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    assert result_with["answer_markdown"] == result_without["answer_markdown"]
    assert result_with["provider"] == result_without["provider"]


def test_synthesize_external_with_evidence_pack_includes_items(monkeypatch) -> None:
    """External synthesis includes evidence pack item fields in prompt:
    support_hint, source_stage, evidence_kind, and excerpt text."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    captured_prompt = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"answer_markdown": "Synthesized.", "cited_indexes": [0], "warnings": [], "confidence": 0.9}'
                        }
                    }
                ]
            }

    def capture_post(self, url, json, headers, **kw):
        captured_prompt["user"] = json["messages"][1]["content"]
        return FakeResponse()

    monkeypatch.setattr(httpx.Client, "post", capture_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "evidence excerpt text",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    assert result["provider"] == "external_api"
    prompt_text = captured_prompt.get("user", "")
    # Evidence pack excerpt should appear in the prompt
    assert "evidence excerpt text" in prompt_text
    # support_hint must be present in the evidence-pack section
    assert "hint=direct" in prompt_text, (
        f"support_hint not found in prompt: {prompt_text[:500]}"
    )
    # source_stage must be present
    assert "stage=source_chunk" in prompt_text, (
        f"source_stage not found in prompt: {prompt_text[:500]}"
    )
    # evidence_kind must be present
    assert "kind=source_chunk" in prompt_text, (
        f"evidence_kind not found in prompt: {prompt_text[:500]}"
    )
    # doc identity must be present
    assert "doc=d1" in prompt_text, (
        f"document_id not found in prompt: {prompt_text[:500]}"
    )


def test_synthesize_without_evidence_pack_still_works(monkeypatch) -> None:
    """synthesize() without evidence_pack is backward compatible."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_local
    )
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        conversation_summary="",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )
    assert result["provider"] == "local"
    assert "answer_markdown" in result
    for field in ("answer_markdown", "cited_indexes", "warnings", "confidence", "provider", "model"):
        assert field in result, f"Missing required field: {field}"


def _fake_settings_external():
    from app.core.config import Settings

    s = Settings()
    s.agent_synthesis_provider = "external_api"
    s.external_api_enabled = True
    s.external_api_key = "test-key"
    s.external_api_model = "gpt-4o-mini"
    s.external_api_base_url = "https://api.openai.com/v1"
    s.external_api_timeout = 90
    return s


# ------------------------------------------------------------------
# coverage retry tests (Phase 3)
# ------------------------------------------------------------------


def test_coverage_retry_happens_when_anchors_missing(monkeypatch) -> None:
    """On evidence_required route, when first answer omits evidence anchors,
    a second API call (retry) is made."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    call_count = [0]
    second_called = [False]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            # First answer: omits the anchor "NMR"
            return _fake_response(
                '{"answer_markdown": "The structure was analyzed.", '
                '"cited_indexes": [0], "warnings": [], "confidence": 0.7}'
            )
        else:
            # Second answer: includes the anchor
            second_called[0] = True
            return _fake_response(
                '{"answer_markdown": "The structure was analyzed using NMR spectroscopy.", '
                '"cited_indexes": [0], "warnings": [], "confidence": 0.85}'
            )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique analyzed the structure?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="The structure was analyzed.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy was used to analyze the molecular structure."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy was used to analyze the molecular structure.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    assert second_called[0], "Expected a second API call for coverage retry"
    assert call_count[0] == 2, f"Expected 2 API calls, got {call_count[0]}"
    assert "NMR" in result["answer_markdown"]
    assert result["provider"] == "external_api"


def test_coverage_retry_skipped_when_anchors_covered(monkeypatch) -> None:
    """When the first answer already covers evidence anchors, no retry is made."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        return _fake_response(
            '{"answer_markdown": "NMR spectroscopy at 7.5 kcal/mol was used.", '
            '"cited_indexes": [0, 1], "warnings": [], "confidence": 0.9}'
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique and energy?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="NMR was used.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy was used."},
            {"document_id": "d2", "excerpt": "Binding energy was 7.5 kcal/mol."},
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                },
                {
                    "index": 1,
                    "document_id": "d2",
                    "score": 0.90,
                    "excerpt": "Binding energy was 7.5 kcal/mol.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                },
            ],
        },
    )
    # Only 1 call — anchors were covered
    assert call_count[0] == 1, f"Expected 1 API call (no retry), got {call_count[0]}"
    assert result["provider"] == "external_api"


def test_coverage_retry_skipped_on_simple_rag_route(monkeypatch) -> None:
    """Coverage retry is not triggered for simple_rag route even with missing anchors."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        return _fake_response(
            '{"answer_markdown": "General answer without specifics.", '
            '"cited_indexes": [0], "warnings": [], "confidence": 0.7}'
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique?",
        route="simple_rag",  # NOT evidence-heavy
        conversation_summary="",
        rag_answer="Some analysis was done.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy was used."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy was used.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    assert call_count[0] == 1, "simple_rag should not trigger coverage retry"
    assert result["provider"] == "external_api"


def test_coverage_retry_skipped_without_evidence_pack(monkeypatch) -> None:
    """When evidence_pack is None, coverage retry is skipped."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        return _fake_response(
            '{"answer_markdown": "General answer.", '
            '"cited_indexes": [0], "warnings": [], "confidence": 0.7}'
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR was used."}
        ],
        # No evidence_pack passed
    )
    assert call_count[0] == 1, "No retry when evidence_pack is missing"
    assert result["provider"] == "external_api"


def test_coverage_retry_skipped_on_local_fallback(monkeypatch) -> None:
    """Local provider never triggers coverage retry."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_local
    )
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR was used."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy was used.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    # Local fallback — no API call, no retry
    assert result["provider"] == "local"
    assert result["answer_markdown"] == "Some analysis."


def test_coverage_retry_failure_falls_back_gracefully(monkeypatch) -> None:
    """When the retry API call itself fails, fall back to first answer."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return _fake_response(
                '{"answer_markdown": "The structure was analyzed using standard methods.", '
                '"cited_indexes": [0], "warnings": [], "confidence": 0.6}'
            )
        else:
            raise ValueError("Retry API failure")

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy at 7.5 kcal/mol was used."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy at 7.5 kcal/mol was used.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    # Falls back to first answer
    assert "standard methods" in result["answer_markdown"]
    # Should have a warning about retry not producing usable result
    assert any("retry" in w.lower() or "coverage" in w.lower() for w in result["warnings"]), (
        f"Expected retry/coverage warning, got warnings={result['warnings']}"
    )


def test_coverage_retry_bounded_to_one_retry(monkeypatch) -> None:
    """Even if retry answer still misses anchors, only one retry is performed."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        # Both answers omit the anchor
        return _fake_response(
            '{"answer_markdown": "General analysis was performed.", '
            '"cited_indexes": [0], "warnings": [], "confidence": 0.6}'
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="Analysis was done.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy at 7.5 kcal/mol."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy at 7.5 kcal/mol.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    # Exactly 2 calls: first + one retry (bounded)
    assert call_count[0] == 2, f"Expected exactly 2 calls, got {call_count[0]}"


def test_extract_evidence_anchors_numeric_and_abbrev() -> None:
    """_extract_evidence_anchors finds numeric patterns and abbreviations."""
    evidence_pack = {
        "status": "ok",
        "items": [
            {
                "index": 0,
                "document_id": "d1",
                "score": 0.95,
                "excerpt": "Binding affinity was -2.4 kcal/mol as measured by NMR.",
                "evidence_kind": "table",
                "source_stage": "document_table",
                "support_hint": "direct",
            }
        ],
    }
    citations = [
        {"document_id": "d2", "excerpt": "The GLH mutation showed FXA inhibition at 0.5 nM."}
    ]
    anchors = AgentSynthesizer._extract_evidence_anchors(evidence_pack, citations)
    # Should capture numeric patterns and abbreviations
    assert len(anchors) >= 2, f"Expected at least 2 anchors, got {anchors}"
    anchor_lower = [a.lower() for a in anchors]
    # Key numeric value
    assert any("2.4" in a for a in anchor_lower), f"Numeric anchor missing from {anchors}"
    # Key abbreviation
    assert any("nmr" in a for a in anchor_lower) or any("glh" in a for a in anchor_lower) or any("fxa" in a for a in anchor_lower), f"No abbreviation in {anchors}"


def test_extract_evidence_anchors_empty_input() -> None:
    """_extract_evidence_anchors returns empty list for missing/empty inputs."""
    assert AgentSynthesizer._extract_evidence_anchors(None, []) == []
    assert AgentSynthesizer._extract_evidence_anchors({}, []) == []
    assert AgentSynthesizer._extract_evidence_anchors({"status": "ok", "items": []}, []) == []
    assert AgentSynthesizer._extract_evidence_anchors(
        {"status": "ok", "items": [{"excerpt": "", "evidence_kind": "source_chunk"}]},
        [],
    ) == []


def test_should_retry_for_coverage_simple_rag_skips() -> None:
    """_should_retry_for_coverage returns False for simple_rag route."""
    assert AgentSynthesizer._should_retry_for_coverage(
        route="simple_rag",
        evidence_pack={"status": "ok", "items": [{"excerpt": "NMR", "evidence_kind": "source_chunk"}]},
        citations=[{"excerpt": "NMR"}],
        answer_text="No NMR here.",
    ) is False


def test_should_retry_for_coverage_empty_pack_skips() -> None:
    """_should_retry_for_coverage returns False when evidence pack is empty."""
    assert AgentSynthesizer._should_retry_for_coverage(
        route="evidence_required",
        evidence_pack={"status": "ok", "items": []},
        citations=[],
        answer_text="any answer",
    ) is False


def test_should_retry_for_coverage_all_covered_skips() -> None:
    """_should_retry_for_coverage returns False when all anchors are covered."""
    assert AgentSynthesizer._should_retry_for_coverage(
        route="evidence_required",
        evidence_pack={
            "status": "ok",
            "items": [{"excerpt": "NMR spectroscopy at 7.5 kcal/mol.", "evidence_kind": "source_chunk"}],
        },
        citations=[{"excerpt": "NMR spectroscopy at 7.5 kcal/mol."}],
        answer_text="NMR spectroscopy at 7.5 kcal/mol was used.",
    ) is False


def test_should_retry_for_coverage_missing_anchors_triggers() -> None:
    """_should_retry_for_coverage returns True when anchors are missing on evidence-heavy route."""
    assert AgentSynthesizer._should_retry_for_coverage(
        route="evidence_required",
        evidence_pack={
            "status": "ok",
            "items": [{"excerpt": "NMR spectroscopy at 7.5 kcal/mol.", "evidence_kind": "table"}],
        },
        citations=[{"excerpt": "NMR spectroscopy at 7.5 kcal/mol."}],
        answer_text="The structure was analyzed.",  # Neither NMR nor 7.5 kcal/mol present
    ) is True


# ------------------------------------------------------------------
# evidence pack prompt formatting tests (Phase 3 rework)
# ------------------------------------------------------------------


def test_format_evidence_pack_section_includes_all_fields() -> None:
    """_format_evidence_pack_section emits support_hint, source_stage,
    evidence_kind, document_id, index, and excerpt for each item."""
    evidence_pack = {
        "status": "ok",
        "items": [
            {
                "index": 0,
                "document_id": "d1",
                "score": 0.95,
                "excerpt": "Binding energy was 7.5 kcal/mol.",
                "evidence_kind": "table",
                "source_stage": "document_table",
                "support_hint": "direct",
            }
        ],
    }
    result = AgentSynthesizer._format_evidence_pack_section(evidence_pack)
    assert "hint=direct" in result, f"Missing support_hint in: {result}"
    assert "stage=document_table" in result, f"Missing source_stage in: {result}"
    assert "kind=table" in result, f"Missing evidence_kind in: {result}"
    assert "doc=d1" in result, f"Missing document_id in: {result}"
    assert "[0]" in result, f"Missing index in: {result}"
    assert "7.5 kcal/mol" in result, f"Missing excerpt in: {result}"


def test_format_evidence_pack_section_caps_items() -> None:
    """When evidence pack has more than MAX_EVIDENCE_PACK_ITEMS,
    only the first MAX_EVIDENCE_PACK_ITEMS are emitted and a
    truncation note is included."""
    max_n = AgentSynthesizer.MAX_EVIDENCE_PACK_ITEMS
    items = []
    for i in range(max_n + 5):
        items.append({
            "index": i,
            "document_id": f"d{i}",
            "score": 0.9,
            "excerpt": f"Excerpt text for item {i}.",
            "evidence_kind": "source_chunk",
            "source_stage": "source_chunk",
            "support_hint": "direct",
        })
    evidence_pack = {"status": "ok", "items": items}
    result = AgentSynthesizer._format_evidence_pack_section(evidence_pack)
    # Only max_n items should have their excerpts in the output
    for i in range(max_n):
        assert f"d{i}" in result, f"Expected item d{i} to be present"
    # Items beyond the cap should not appear
    assert f"d{max_n}" not in result, (
        f"Item d{max_n} should be omitted, got: {result}"
    )
    # Truncation note should be present
    assert "omitted" in result.lower(), (
        f"Expected truncation note, got: {result}"
    )
    # Header should mention the count
    assert f"({max_n} of {max_n + 5})" in result, (
        f"Expected count header, got: {result}"
    )


def test_format_evidence_pack_section_truncates_long_excerpts() -> None:
    """Excerpts longer than MAX_EXCERPT_CHARS are truncated with ellipsis."""
    max_chars = AgentSynthesizer.MAX_EXCERPT_CHARS
    long_excerpt = "x" * (max_chars + 100)
    evidence_pack = {
        "status": "ok",
        "items": [
            {
                "index": 0,
                "document_id": "d1",
                "score": 0.95,
                "excerpt": long_excerpt,
                "evidence_kind": "source_chunk",
                "source_stage": "source_chunk",
                "support_hint": "direct",
            }
        ],
    }
    result = AgentSynthesizer._format_evidence_pack_section(evidence_pack)
    # The truncated version should end with ellipsis
    assert result.rstrip().endswith("…"), (
        f"Expected truncated excerpt ending with …, got: ...{result[-50:]}"
    )
    # The full long excerpt should NOT appear
    assert long_excerpt not in result
    # The truncated prefix should be present
    assert "x" * (max_chars - 5) in result, (
        f"Expected truncated prefix, got: ...{result[-100:]}"
    )


def test_format_evidence_pack_section_empty_inputs() -> None:
    """_format_evidence_pack_section returns empty string for missing/empty inputs."""
    assert AgentSynthesizer._format_evidence_pack_section(None) == ""
    assert AgentSynthesizer._format_evidence_pack_section({}) == ""
    assert AgentSynthesizer._format_evidence_pack_section(
        {"status": "ok", "items": []}
    ) == ""
    assert AgentSynthesizer._format_evidence_pack_section(
        {"status": "ok", "items": [{"excerpt": "", "evidence_kind": "source_chunk"}]}
    ) == ""


def test_retry_uses_same_evidence_pack_format(monkeypatch) -> None:
    """The coverage retry prompt includes evidence-pack section with
    all required fields (support_hint, stage, kind, doc, excerpt)."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    captured_retry_prompt = {}
    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            # First answer omits NMR anchor
            return _fake_response(
                '{"answer_markdown": "Standard analysis was performed.", '
                '"cited_indexes": [0], "warnings": [], "confidence": 0.5}'
            )
        else:
            captured_retry_prompt["user"] = json["messages"][1]["content"]
            captured_retry_prompt["system"] = json["messages"][0]["content"]
            return _fake_response(
                '{"answer_markdown": "NMR spectroscopy was used.", '
                '"cited_indexes": [0], "warnings": [], "confidence": 0.8}'
            )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What technique?",
        route="evidence_required",
        conversation_summary="",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy was used."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.95,
                    "excerpt": "NMR spectroscopy at 7.5 kcal/mol.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
    )
    assert call_count[0] == 2, f"Expected retry, got {call_count[0]} calls"
    assert result["provider"] == "external_api"
    retry_text = captured_retry_prompt.get("user", "")
    # Retry prompt must include all evidence-pack fields
    assert "hint=direct" in retry_text, (
        f"Retry prompt missing support_hint: {retry_text[:500]}"
    )
    assert "stage=source_chunk" in retry_text, (
        f"Retry prompt missing source_stage: {retry_text[:500]}"
    )
    assert "kind=source_chunk" in retry_text, (
        f"Retry prompt missing evidence_kind: {retry_text[:500]}"
    )
    assert "doc=d1" in retry_text, (
        f"Retry prompt missing doc id: {retry_text[:500]}"
    )
    assert "7.5 kcal/mol" in retry_text, (
        f"Retry prompt missing excerpt: {retry_text[:500]}"
    )


def _fake_response(content: str):
    """Create a fake httpx response with the given JSON content string."""
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": content
                        }
                    }
                ]
            }

    return FakeResponse()
