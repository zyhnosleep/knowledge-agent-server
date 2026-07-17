from __future__ import annotations

from app.core.config import Settings
from app.services.agent_model_router import AgentModelRouter


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        OLLAMA_GENERATION_BASE_URL="http://generation:11435",
        OLLAMA_GENERATION_MODEL="generation-model",
        OLLAMA_GENERATION_CONTEXT_LENGTH=32768,
    )


def test_all_answer_routes_use_one_generation_target() -> None:
    router = AgentModelRouter(_settings())

    for route in (
        "simple_rag",
        "evidence_required",
        "table_or_metric",
        "multi_source_compare",
        "complex_multi_hop",
    ):
        target = router.select(route)
        assert target.profile == "generation"
        assert target.base_url == "http://generation:11435"
        assert target.model == "generation-model"
        assert target.context_length == 32768


def test_needs_clarification_does_not_select_a_model() -> None:
    target = AgentModelRouter(_settings()).select("needs_clarification")

    assert target.profile == "none"
    assert target.model == ""
    assert target.context_length == 0


def test_router_records_why_target_was_selected() -> None:
    router = AgentModelRouter(_settings())

    target = router.select("multi_source_compare")

    assert "multi_source_compare" in target.reason
    assert "generation" in target.reason.lower()
