from __future__ import annotations

from app.core.config import Settings
from app.services.agent_model_router import AgentModelRouter


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        OLLAMA_FAST_BASE_URL="http://fast:11435",
        OLLAMA_DEEP_BASE_URL="http://deep:11436",
        OLLAMA_FAST_MODEL="fast-model",
        OLLAMA_DEEP_MODEL="deep-model",
        OLLAMA_FAST_CONTEXT_LENGTH=16384,
        OLLAMA_DEEP_CONTEXT_LENGTH=32768,
    )


def test_auto_routes_simple_and_evidence_queries_to_fast() -> None:
    router = AgentModelRouter(_settings())

    for route in ("simple_rag", "evidence_required"):
        target = router.select("auto", route)
        assert target.profile == "fast"
        assert target.base_url == "http://fast:11435"
        assert target.model == "fast-model"
        assert target.context_length == 16384


def test_auto_routes_complex_queries_to_deep() -> None:
    router = AgentModelRouter(_settings())

    for route in ("table_or_metric", "multi_source_compare", "complex_multi_hop"):
        target = router.select("auto", route)
        assert target.profile == "deep"
        assert target.base_url == "http://deep:11436"
        assert target.model == "deep-model"
        assert target.context_length == 32768


def test_manual_modes_override_policy_route() -> None:
    router = AgentModelRouter(_settings())

    assert router.select("fast", "complex_multi_hop").profile == "fast"
    assert router.select("deep", "simple_rag").profile == "deep"


def test_needs_clarification_does_not_select_a_model() -> None:
    target = AgentModelRouter(_settings()).select("auto", "needs_clarification")

    assert target.profile == "none"
    assert target.model == ""
    assert target.context_length == 0


def test_router_records_why_target_was_selected() -> None:
    router = AgentModelRouter(_settings())

    automatic = router.select("auto", "multi_source_compare")
    manual = router.select("fast", "multi_source_compare")

    assert "multi_source_compare" in automatic.reason
    assert "override" in manual.reason.lower()
