from __future__ import annotations

import pytest

from app.services.agent_model_router import InferenceTarget
from app.services.agent_synthesizer import AgentSynthesizer, SynthesisPayload


class FakeOllamaClient:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def generate_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "content": self.result["answer_markdown"],
            "model": kwargs["model"],
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

    def generate_structured(self, *args, **kwargs):
        raise AssertionError("Local synthesis must not request structured JSON")

    def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        text = self.result["answer_markdown"]
        midpoint = max(1, len(text) // 2)
        yield {"content": text[:midpoint], "model": kwargs["model"], "done": False}
        yield {"content": text[midpoint:], "model": kwargs["model"], "done": True}


def _generation_target() -> InferenceTarget:
    return InferenceTarget(
        profile="generation",
        base_url="http://generation:11434",
        model="qwen3.5:9b",
        context_length=32768,
        reason="test generation",
    )


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
        target=_generation_target(),
    )
    assert result["provider"] == "local"
    assert result["model"] == "qwen3.5:9b"
    assert result["answer_markdown"] == "Entropy measures the number of accessible states [0]."
    assert len(result["warnings"]) == 0
    assert ollama.calls[0]["model"] == "qwen3.5:9b"
    assert ollama.calls[0]["context_length"] == 32768
    assert ollama.calls[0]["max_output_tokens"] == 768
    assert "entropy defined" in ollama.calls[0]["messages"][1]["content"]
    assert "MUST use the same language" in ollama.calls[0]["messages"][0]["content"]


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
        target=_generation_target(),
    )
    assert result["provider"] == "local"
    assert result["model"] == "qwen3.5:9b"
    assert result["answer_markdown"] == "Entropy measures the number of accessible states [0]."


def test_synthesize_ollama_provider_dispatches_to_ollama_synthesis(monkeypatch) -> None:
    """An explicit ``ollama`` provider must not fall through to external/fallback."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )
    calls: list[dict] = []

    def fake_ollama_synthesize(self, **kwargs):
        calls.append(kwargs)
        return {
            "answer_markdown": "Synthesized by Ollama [0].",
            "cited_indexes": [0],
            "warnings": [],
            "confidence": 0.9,
            "provider": "ollama",
            "model": "qwen3.5:9b-synthesis",
        }

    monkeypatch.setattr(AgentSynthesizer, "_ollama_synthesize", fake_ollama_synthesize)
    syn = AgentSynthesizer()
    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert calls and calls[0]["query"] == "What is entropy?"
    assert result["provider"] == "ollama"
    assert result["model"] == "qwen3.5:9b-synthesis"
    assert result["model"] != "local-fallback"


def test_synthesize_ollama_provider_uses_injected_client_and_structured_model(monkeypatch) -> None:
    """The explicit provider uses the configured Ollama synthesis client."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )

    class FakeStructuredOllama:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_structured(self, schema, **kwargs):
            self.calls.append({"schema": schema, **kwargs})
            return SynthesisPayload(
                answer_markdown="Synthesized by Ollama [0].",
                cited_indexes=[0],
                warnings=[],
                confidence=0.9,
            )

    ollama = FakeStructuredOllama()
    result = AgentSynthesizer(ollama_client=ollama).synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert result["provider"] == "ollama"
    assert result["answer_markdown"] == "Synthesized by Ollama [0]."
    assert ollama.calls and ollama.calls[0]["model"] == "qwen3.5:9b-synthesis"


def test_ollama_evidence_route_retries_when_exact_anchor_is_omitted(monkeypatch) -> None:
    """Ollama synthesis must retry once when evidence-heavy output drops an anchor."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )

    class SequencedStructuredOllama:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.responses = [
                SynthesisPayload(
                    answer_markdown="The structure was analyzed.",
                    cited_indexes=[0],
                    warnings=[],
                    confidence=0.7,
                ),
                SynthesisPayload(
                    answer_markdown="The structure was analyzed using NMR spectroscopy.",
                    cited_indexes=[0],
                    warnings=[],
                    confidence=0.85,
                ),
            ]

        def generate_structured(self, schema, **kwargs):
            self.calls.append({"schema": schema, **kwargs})
            return self.responses.pop(0)

    ollama = SequencedStructuredOllama()
    result = AgentSynthesizer(ollama_client=ollama).synthesize(
        query="What technique analyzed the structure?",
        route="evidence_required",
        rag_answer="The structure was analyzed.",
        citations=[
            {
                "document_id": "d1",
                "excerpt": "NMR spectroscopy was used to analyze the molecular structure.",
            }
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

    assert len(ollama.calls) == 2
    assert "NMR" in result["answer_markdown"]
    assert result["provider"] == "ollama"


def test_ollama_first_prompt_requires_exact_evidence_for_table_routes(monkeypatch) -> None:
    """The first Ollama pass must preserve table labels, terms, and values."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )

    class CapturingStructuredOllama:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_structured(self, schema, **kwargs):
            self.calls.append({"schema": schema, **kwargs})
            return SynthesisPayload(
                answer_markdown="Table 5 reports 21.0 and 95.3.",
                cited_indexes=[0],
                warnings=[],
                confidence=0.9,
            )

    ollama = CapturingStructuredOllama()
    AgentSynthesizer(ollama_client=ollama).synthesize(
        query="Compare Table 5 metrics and explain the QM/NMR evidence.",
        route="table_or_metric",
        rag_answer="The table reports the exact values.",
        citations=[
            {
                "document_id": "d1",
                "page_title": "Paper",
                "excerpt": "Table 5: QM and NMR validation values are 21.0 and 95.3.",
            }
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: QM and NMR validation values are 21.0 and 95.3.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
            "table_facts": [
                {"table_id": "t5", "table_label": "Table 5", "row_label": "QM", "column": "NMR", "value": "21.0"},
            ],
        },
    )

    prompt = ollama.calls[0]["system_prompt"] + "\n" + ollama.calls[0]["user_prompt"]
    assert "preserve" in prompt.lower()
    assert "exact numeric" in prompt.lower()
    assert "table labels" in prompt.lower()
    assert "canonical table facts" in prompt.lower()
    assert "do not summarize" in prompt.lower()


def test_answer_rules_follow_query_language_and_ban_metadata(monkeypatch) -> None:
    """T3：synthesize 约束语言跟随 / 元数据禁止 / 假设标记（R29/R36/R26）。"""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )

    chinese = AgentSynthesizer._answer_rules("为什么作者选择了 C36 而不是 C22/CMAP？")
    assert "Answer in Chinese" in chinese
    assert "简体中文" in chinese
    assert "Never output internal document metadata" in chinese
    assert "submission IDs" in chinese
    assert "mark it explicitly as inference" in chinese
    assert "readable Unicode text" in chinese
    assert "no backslash escapes of any kind" in chinese
    assert "## 结论" in chinese
    assert "## 证据" in chinese
    assert "## 不确定性" in chinese

    english = AgentSynthesizer._answer_rules("Why did the authors choose C36?")
    assert "Answer in English" in english
    assert "Answer in Chinese" not in english
    assert "Never output internal document metadata" in english
    assert "## Conclusion" in english
    assert "## Evidence" in english
    assert "## Uncertainty" in english
    assert "## 结论" not in english


def test_ollama_synthesis_prompt_includes_t3_answer_rules(monkeypatch) -> None:
    """Ollama 综合 system prompt 必须携带 T3 约束块（语言/元数据/假设）。"""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )

    class CapturingStructuredOllama:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_structured(self, schema, **kwargs):
            self.calls.append({"schema": schema, **kwargs})
            return SynthesisPayload(
                answer_markdown="作者采用了 C36，因为其拟合质量更好。",
                cited_indexes=[0],
                warnings=[],
                confidence=0.9,
            )

    ollama = CapturingStructuredOllama()
    AgentSynthesizer(ollama_client=ollama).synthesize(
        query="为什么作者选择了 C36 而不是 C22/CMAP？",
        route="evidence_required",
        rag_answer="C36 拟合质量更好。",
        citations=[{"document_id": "d1", "excerpt": "C36 provides better fitting."}],
    )

    system_prompt = ollama.calls[0]["system_prompt"]
    assert "Answer in Chinese" in system_prompt
    assert "Never output internal document metadata" in system_prompt
    assert "mark it explicitly as inference" in system_prompt


def test_synthesize_generation_target_calls_9b_once_as_plain_markdown(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(result=_local_result())
    syn = AgentSynthesizer(ollama_client=ollama)
    target = InferenceTarget(
        profile="generation",
        base_url="http://generation:11434",
        model="qwen3.5:9b",
        context_length=32768,
        reason="test generation",
    )

    result = syn.synthesize(
        query="Compare the documents",
        route="multi_source_compare",
        rag_answer="Draft",
        citations=[{"document_id": "d1", "excerpt": "evidence"}],
        target=target,
    )

    assert len(ollama.calls) == 1
    assert ollama.calls[0]["model"] == "qwen3.5:9b"
    assert ollama.calls[0]["context_length"] == 32768
    assert result["model"] == "qwen3.5:9b"


def test_synthesize_stream_forwards_tokens_then_citation(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient(result=_local_result()))
    events: list[tuple[str, dict]] = []

    result = syn.synthesize_stream(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Draft",
        citations=[{"document_id": "d1", "excerpt": "entropy evidence"}],
        target=_generation_target(),
        event_sink=lambda name, data: events.append((name, data)),
    )

    assert [name for name, _ in events] == ["token", "token", "citation"]
    assert "".join(data["delta"] for name, data in events if name == "token") == result[
        "answer_markdown"
    ]
    assert events[-1][1]["index"] == 0


# ------------------------------------------------------------------
# local synthesis fidelity tests (Task 16 agent synthesis fidelity fix)
# ------------------------------------------------------------------


class _SequencedLocalOllama:
    """Non-streaming local client with a scripted sequence of answers."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate_chat(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.responses.pop(0)["answer_markdown"]
        return {
            "content": answer,
            "model": kwargs["model"],
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

    def generate_structured(self, *args, **kwargs):
        raise AssertionError("Local synthesis must not request structured JSON")

    def stream_chat(self, **kwargs):
        raise AssertionError("Non-stream local synthesis must not stream")


class _SequencedStreamingOllama:
    """Streaming client whose first stream misses anchors and retry fixes it."""

    def __init__(self, *, stream_text: str, retry_text: str) -> None:
        self.stream_text = stream_text
        self.retry_text = retry_text
        self.stream_calls: list[dict] = []
        self.retry_calls: list[dict] = []

    def stream_chat(self, **kwargs):
        self.stream_calls.append(kwargs)
        midpoint = max(1, len(self.stream_text) // 2)
        yield {"content": self.stream_text[:midpoint], "model": kwargs["model"], "done": False}
        yield {"content": self.stream_text[midpoint:], "model": kwargs["model"], "done": True}

    def generate_chat(self, **kwargs):
        self.retry_calls.append(kwargs)
        return {
            "content": self.retry_text,
            "model": kwargs["model"],
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }

    def generate_structured(self, *args, **kwargs):
        raise AssertionError("Local synthesis must not request structured JSON")


def test_local_prompt_includes_precision_rules_and_table_facts(monkeypatch) -> None:
    """The local prompt must carry the same fidelity controls as ollama/external."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(
            answer_markdown="Table 5: NMR spectroscopy at 7.5 kcal/mol was used [0]."
        )
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    syn.synthesize(
        query="Compare Table 5 metrics and explain the QM/NMR evidence.",
        route="table_or_metric",
        rag_answer="The table reports the exact values.",
        citations=[
            {
                "document_id": "d1",
                "page_title": "Paper",
                "excerpt": "Table 5: QM and NMR validation values are 21.0 and 95.3.",
            }
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: QM and NMR validation values are 21.0 and 95.3.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
            "table_facts": [
                {"table_id": "t5", "row_label": "QM", "column": "NMR", "value": "21.0"},
            ],
        },
        target=_generation_target(),
    )
    prompt = ollama.calls[0]["messages"][0]["content"] + "\n" + ollama.calls[0]["messages"][1]["content"]
    assert "exact numeric" in prompt.lower()
    assert "table labels" in prompt.lower()
    assert "canonical table facts" in prompt.lower()
    assert "do not summarize" in prompt.lower()
    assert "value=21.0" in prompt


def test_streaming_local_prompt_includes_precision_rules_and_table_facts(monkeypatch) -> None:
    """The streaming local prompt must carry fidelity controls too."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(
            answer_markdown="Table 5: NMR spectroscopy at 7.5 kcal/mol was used [0]."
        )
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    events: list[tuple[str, dict]] = []
    syn.synthesize_stream(
        query="Compare Table 5 metrics and explain the QM/NMR evidence.",
        route="table_or_metric",
        rag_answer="NMR spectroscopy at 7.5 kcal/mol was used.",
        citations=[
            {
                "document_id": "d1",
                "page_title": "Paper",
                "excerpt": "Table 5: QM and NMR validation values are 21.0 and 95.3.",
            }
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: QM and NMR validation values are 21.0 and 95.3.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
            "table_facts": [
                {"table_id": "t5", "row_label": "QM", "column": "NMR", "value": "21.0"},
            ],
        },
        target=_generation_target(),
        event_sink=lambda name, data: events.append((name, data)),
    )
    prompt = ollama.calls[0]["messages"][0]["content"] + "\n" + ollama.calls[0]["messages"][1]["content"]
    assert "exact numeric" in prompt.lower()
    assert "canonical table facts" in prompt.lower()
    assert "value=21.0" in prompt


def test_local_evidence_route_retries_when_exact_anchor_is_omitted(monkeypatch) -> None:
    """The local path must retry once when an evidence-heavy answer drops an anchor."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = _SequencedLocalOllama(
        responses=[
            {"answer_markdown": "The structure was analyzed."},
            {"answer_markdown": "The structure was analyzed using NMR spectroscopy."},
        ]
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    result = syn.synthesize(
        query="What technique analyzed the structure?",
        route="evidence_required",
        rag_answer="The structure was analyzed.",
        citations=[
            {
                "document_id": "d1",
                "excerpt": "NMR spectroscopy was used to analyze the molecular structure.",
            }
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "NMR spectroscopy was used to analyze the molecular structure.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
    )
    assert len(ollama.calls) == 2
    assert "NMR" in result["answer_markdown"]
    assert result["provider"] == "local"
    assert result["model"] == "qwen3.5:9b"
    assert any("retry" in w.lower() for w in result["warnings"])


def test_local_coverage_retry_bounded_to_one_retry(monkeypatch) -> None:
    """Even a retry that still misses anchors never triggers a second retry."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)

    class AlwaysMissingOllama:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_chat(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "content": "General analysis was performed.",
                "model": kwargs["model"],
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

        def generate_structured(self, *args, **kwargs):
            raise AssertionError("Local synthesis must not request structured JSON")

        def stream_chat(self, **kwargs):
            raise AssertionError("Non-stream local synthesis must not stream")

    ollama = AlwaysMissingOllama()
    syn = AgentSynthesizer(ollama_client=ollama)
    result = syn.synthesize(
        query="What technique?",
        route="evidence_required",
        rag_answer="Analysis was done.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy at 7.5 kcal/mol was used."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "NMR spectroscopy at 7.5 kcal/mol was used.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
    )
    assert len(ollama.calls) == 2, f"Expected exactly 2 calls, got {len(ollama.calls)}"
    assert any("retry" in w.lower() for w in result["warnings"])


def test_local_retry_prompt_uses_evidence_anchors_not_benchmark_terms(monkeypatch) -> None:
    """The retry prompt lists generic evidence anchors, never benchmark ids/terms."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = _SequencedLocalOllama(
        responses=[
            {"answer_markdown": "The structure was analyzed."},
            {"answer_markdown": "The structure was analyzed using NMR spectroscopy."},
        ]
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    syn.synthesize(
        query="What technique analyzed the structure?",
        route="evidence_required",
        rag_answer="The structure was analyzed.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy was used to analyze the structure."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "NMR spectroscopy was used to analyze the structure.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
    )
    assert len(ollama.calls) == 2
    retry_prompt = ollama.calls[1]["messages"][0]["content"] + "\n" + ollama.calls[1]["messages"][1]["content"]
    # The only anchor available in the evidence is NMR — it must drive the retry.
    assert "NMR" in retry_prompt
    # No benchmark case ids / domain anchor lists may appear.
    assert "opls5_table_metrics" not in retry_prompt
    assert "charmm36" not in retry_prompt


def test_local_synthesis_introducing_unsupported_number_falls_back(monkeypatch) -> None:
    """9.3.3：合成答案引入证据不支持的数值（999.0）必须回退 RAG 草稿。

    draft_fidelity 守卫只检查 anchor 覆盖下降；本场景合成保留 21.0（覆盖不降）
    但新增 999.0——需要独立的新数字打回校验。
    """
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(
            answer_markdown="The accuracy is 21.0% and improved to 999.0% [0]."
        )
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The accuracy is 21.0%."
    result = syn.synthesize(
        query="What is the accuracy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 7: accuracy 21.0%."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "accuracy",
                    "value": "21.0",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
    )
    assert "999.0" not in result["answer_markdown"]
    assert result["answer_markdown"] == draft
    assert result["provider"] == "local"
    assert result["model"] == "local-fallback"


def test_local_synthesis_keeps_number_format_variants(monkeypatch) -> None:
    """9.3.3 数值规范化：'21' 与 '21.0' 数值等价，不应触发假阳性回退。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: accuracy 21.0% [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The accuracy is 21.0%."
    result = syn.synthesize(
        query="What is the accuracy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 7: accuracy 21."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "accuracy",
                    "value": "21",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
    )
    # 合成答案保留 21.0（证据是 21，数值等价）→ 不被视为编造
    assert "21.0" in result["answer_markdown"]
    assert result["answer_markdown"] != draft


def test_local_synthesis_rounded_number_falls_back(monkeypatch) -> None:
    """9.3.3 数值规范化：80 与 80.5 数值不等，取整编造必须回退。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="The value is 80 [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The value is 80.5."
    result = syn.synthesize(
        query="What is the value?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 1: value 80.5."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "table_facts": [
                {
                    "table_id": "table-1",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "80.5",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
    )
    # 取整编造被回退：答案必须回退到含 80.5 的草稿
    assert result["answer_markdown"] == draft
    assert result["provider"] == "local"
    assert result["model"] == "local-fallback"


def test_local_synthesis_dropping_table_label_falls_back(monkeypatch) -> None:
    """9.3.3：合成答案删除证据中的表号（Table 7）必须回退。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="The accuracy is 21.0% [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "Table 7: accuracy 21.0%."
    result = syn.synthesize(
        query="What is the accuracy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 7: accuracy 21.0%."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "accuracy",
                    "value": "21.0",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
    )
    # 答案删除了 Table 7 表号 → 回退 RAG 草稿（保留表号）
    assert "Table 7" in result["answer_markdown"]
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"


def test_local_synthesis_missing_fact_value_falls_back(monkeypatch) -> None:
    """9.3.2b：期望 facts（table_facts values）中 draft 覆盖但合成答案遗漏
    （1.18）时回退。citations 不含 1.18，隔离 draft_fidelity 的 anchor 覆盖。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    # 合成答案保留表号（避免表号守卫拦截），但遗漏 draft 覆盖的 1.18
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: value 21.0 [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The value is 21.0 and 1.18 kcal/mol."
    result = syn.synthesize(
        query="What are the values?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 7: value 21.0."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "inventory": [_inventory_entry()],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "21.0",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                },
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "other",
                    "column": "value",
                    "value": "1.18",
                    "row_index": 1,
                    "source_chunk_ids": ["c1"],
                },
            ],
        },
        target=_generation_target(),
    )
    # 合成答案删掉了 draft 覆盖的 1.18 → 回退 RAG 草稿
    assert "1.18" in result["answer_markdown"]
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"


def test_local_narrow_context_excludes_evidence_pack_items(monkeypatch) -> None:
    """9.3.1：narrow_context=True 时 prompt 不含 evidence_pack items 摘录。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(result=_local_result())
    syn = AgentSynthesizer(ollama_client=ollama)
    result = syn.synthesize(
        query="那个的值是多少？",
        route="table_or_metric",
        conversation_summary="User previously asked about charmm36m.",
        rag_answer="The value is 21.0.",
        # 摘录不带句号：_answer_numbers 不提取"21.0." 这类句尾数字
        citations=[{"document_id": "d1", "excerpt": "Table 7: value 21.0"}],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "score": 0.9,
                    "excerpt": "full narrative evidence paragraph that should be excluded",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
            "inventory": [_inventory_entry()],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "21.0",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
        narrow_context=True,
    )
    assert result["provider"] == "local"
    prompt_text = ollama.calls[0]["messages"][1]["content"]
    # evidence-item 摘录与完整 inventory 被排除，citations excerpt 与
    # conversation_summary 保留
    assert "evidence-item-0" not in prompt_text
    assert "excluded" not in prompt_text
    assert "inv-child-1" not in prompt_text
    assert "Table 7: value 21.0" in prompt_text
    assert "User previously asked about charmm36m." in prompt_text


def test_local_synthesis_dropping_fact_unit_falls_back(monkeypatch) -> None:
    """9.3.2：合成答案删除期望 facts 的单位（%），draft 覆盖但合成遗漏 → 回退。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    # 合成保留数值 21.0 与表号，但删除单位 %（隔离 value/表号守卫）
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: value 21.0 [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "Table 7: value 21.0%."
    result = syn.synthesize(
        query="What is the value?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 7: value 21.0."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "inventory": [_inventory_entry()],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "21.0",
                    "unit": "%",
                    "term": "value",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
    )
    # 合成答案删除 % → 回退 RAG 草稿（保留单位）
    assert "%" in result["answer_markdown"]
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"


def test_local_draft_fidelity_guard_returns_draft_when_synthesis_drops_anchors(monkeypatch) -> None:
    """A synthesis that loses draft-covered anchors must fall back to the draft."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(
            answer_markdown="Table 5: NMR spectroscopy at 7.5 kcal/mol was used [0]."
        )
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used."
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
    )
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"
    assert result["provider"] == "local"
    assert any("coverage" in w.lower() for w in result["warnings"])


def test_local_draft_fidelity_keeps_synthesis_when_coverage_preserved(monkeypatch) -> None:
    """A synthesis that preserves draft anchor coverage is kept."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    synthesized = "Table 5: NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used [0]."
    ollama = FakeOllamaClient(result=_local_result(answer_markdown=synthesized))
    syn = AgentSynthesizer(ollama_client=ollama)
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer="NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used.",
        citations=[
            {"document_id": "d1", "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
    )
    assert result["answer_markdown"] == synthesized
    assert result["model"] == "qwen3.5:9b"
    assert not any("coverage" in w.lower() for w in result["warnings"])


def test_streaming_local_coverage_retry_recovers_missing_anchor(monkeypatch) -> None:
    """The streaming path retries once (bounded) when the stream drops an anchor."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = _SequencedStreamingOllama(
        stream_text="The structure was analyzed using standard methods.",
        retry_text="The structure was analyzed using NMR spectroscopy.",
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    events: list[tuple[str, dict]] = []
    result = syn.synthesize_stream(
        query="What technique?",
        route="evidence_required",
        rag_answer="The structure was analyzed.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy was used."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "NMR spectroscopy was used.",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
        event_sink=lambda name, data: events.append((name, data)),
    )
    assert len(ollama.stream_calls) == 1
    assert len(ollama.retry_calls) == 1
    assert "NMR" in result["answer_markdown"]
    assert result["provider"] == "local"
    assert result["model"] == "qwen3.5:9b"
    assert any("retry" in w.lower() for w in result["warnings"])


def test_streaming_draft_fidelity_guard_returns_draft(monkeypatch) -> None:
    """The streaming selection path falls back to the draft on coverage loss."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(
            answer_markdown="Table 5: NMR spectroscopy at 7.5 kcal/mol was used [0]."
        )
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used."
    events: list[tuple[str, dict]] = []
    result = syn.synthesize_stream(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
        },
        target=_generation_target(),
        event_sink=lambda name, data: events.append((name, data)),
    )
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"
    assert any("coverage" in w.lower() for w in result["warnings"])


def test_streaming_synthesis_introducing_unsupported_number_falls_back(monkeypatch) -> None:
    """9.3.3：流式路径同样拦截证据外数值（999.0）并回退 RAG 草稿。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(
            answer_markdown="The accuracy is 21.0% and improved to 999.0% [0]."
        )
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The accuracy is 21.0%."
    events: list[tuple[str, dict]] = []
    result = syn.synthesize_stream(
        query="What is the accuracy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {"document_id": "d1", "excerpt": "Table 7: accuracy 21.0%."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "accuracy",
                    "value": "21.0",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
        event_sink=lambda name, data: events.append((name, data)),
    )
    assert "999.0" not in result["answer_markdown"]
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"
    assert any("not supported" in w.lower() for w in result["warnings"])


# ------------------------------------------------------------------
# provider-consistency draft-fidelity tests (Task 16 provider fidelity rework)
# ------------------------------------------------------------------


class _StructuredOllama:
    """Structured-output client with a scripted SynthesisPayload sequence."""

    def __init__(self, payloads: list[SynthesisPayload]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def generate_structured(self, schema, **kwargs):
        self.calls.append({"schema": schema, **kwargs})
        return self.payloads.pop(0)


def _provider_consistency_case() -> dict:
    """Shared evidence inputs for the provider-consistency guard tests."""
    return {
        "citations": [
            {
                "document_id": "d1",
                "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
            }
        ],
        "evidence_pack": {
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
                    "evidence_kind": "table",
                    "source_stage": "document_table",
                    "support_hint": "direct",
                }
            ],
        },
    }


def test_ollama_draft_fidelity_guard_returns_draft_when_synthesis_drops_anchors(monkeypatch) -> None:
    """An Ollama synthesis that loses draft-covered anchors must fall back to the draft."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )
    ollama = _StructuredOllama(
        [
            SynthesisPayload(
                answer_markdown="Table 5: NMR spectroscopy at 7.5 kcal/mol was used [0].",
                cited_indexes=[0],
                warnings=[],
                confidence=0.85,
            )
        ]
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    case = _provider_consistency_case()
    draft = "NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used."
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=case["citations"],
        evidence_pack=case["evidence_pack"],
    )
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"
    assert result["provider"] == "local"
    assert any("coverage" in w.lower() for w in result["warnings"])


def test_ollama_draft_fidelity_keeps_synthesis_when_coverage_preserved(monkeypatch) -> None:
    """An Ollama synthesis that preserves draft anchor coverage is kept."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )
    synthesized = "Table 5: NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used [0]."
    ollama = _StructuredOllama(
        [
            SynthesisPayload(
                answer_markdown=synthesized,
                cited_indexes=[0],
                warnings=[],
                confidence=0.9,
            )
        ]
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    case = _provider_consistency_case()
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer="NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used.",
        citations=case["citations"],
        evidence_pack=case["evidence_pack"],
    )
    assert result["answer_markdown"] == synthesized
    assert result["provider"] == "ollama"
    assert result["model"] == "qwen3.5:9b-synthesis"
    assert not any("coverage" in w.lower() for w in result["warnings"])


def test_external_draft_fidelity_guard_returns_draft_when_synthesis_drops_anchors(monkeypatch) -> None:
    """An external synthesis that loses draft-covered anchors must fall back to the draft."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    def fake_post(self, url, json, headers, **kw):
        return _fake_response(
            '{"answer_markdown": "NMR spectroscopy at 7.5 kcal/mol was used.", '
            '"cited_indexes": [0], "warnings": [], "confidence": 0.85}'
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    case = _provider_consistency_case()
    draft = "NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used."
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=case["citations"],
        evidence_pack=case["evidence_pack"],
    )
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"
    assert result["provider"] == "local"
    assert any("coverage" in w.lower() for w in result["warnings"])


def test_external_draft_fidelity_keeps_synthesis_when_coverage_preserved(monkeypatch) -> None:
    """An external synthesis that preserves draft anchor coverage is kept."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    synthesized = "Table 5: NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used."
    fake_response_content = (
        '{"answer_markdown": "Table 5: NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used.", '
        '"cited_indexes": [0], "warnings": [], "confidence": 0.9}'
    )

    def fake_post(self, url, json, headers, **kw):
        return _fake_response(fake_response_content)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    case = _provider_consistency_case()
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer="NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used.",
        citations=case["citations"],
        evidence_pack=case["evidence_pack"],
    )
    assert result["answer_markdown"] == synthesized
    assert result["provider"] == "external_api"
    assert result["model"] == "gpt-4o-mini"
    assert not any("coverage" in w.lower() for w in result["warnings"])


def test_draft_fidelity_fallback_generic_coverage_math() -> None:
    """_draft_fidelity_fallback keeps synthesis when coverage is not lower."""
    syn = AgentSynthesizer()
    # Synthesis preserves every draft anchor -> kept (None).
    assert syn._draft_fidelity_fallback(
        rag_answer="NMR at 7.5 kcal/mol for GLH mutant.",
        citations=[{"excerpt": "NMR at 7.5 kcal/mol for GLH mutant."}],
        evidence_pack={
            "status": "ok",
            "items": [{"excerpt": "NMR at 7.5 kcal/mol for GLH mutant."}],
        },
        synthesized_answer="NMR at 7.5 kcal/mol for GLH mutant was used.",
        warnings=[],
    ) is None
    # Synthesis drops GLH (draft-covered) -> draft fallback.
    fallback = syn._draft_fidelity_fallback(
        rag_answer="NMR at 7.5 kcal/mol for GLH mutant.",
        citations=[{"excerpt": "NMR at 7.5 kcal/mol for GLH mutant."}],
        evidence_pack={
            "status": "ok",
            "items": [{"excerpt": "NMR at 7.5 kcal/mol for GLH mutant."}],
        },
        synthesized_answer="NMR at 7.5 kcal/mol was used.",
        warnings=["prior warning"],
    )
    assert fallback is not None
    assert fallback["answer_markdown"] == "NMR at 7.5 kcal/mol for GLH mutant."
    assert fallback["model"] == "local-fallback"
    assert "prior warning" in fallback["warnings"]
    assert any("coverage" in w.lower() for w in fallback["warnings"])


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
        target=_generation_target(),
    )

    assert result["provider"] == "local"
    assert result["model"] == "local-fallback"
    assert result["answer_markdown"] == "Entropy is a measure of disorder."
    assert any("ollama" in warning.lower() for warning in result["warnings"])


def test_synthesize_local_sanitizes_cited_indexes(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    syn = AgentSynthesizer(
        ollama_client=FakeOllamaClient(
            result=_local_result(
                answer_markdown="Supported statement [0][2][-1][0]."
            )
        )
    )

    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
        target=_generation_target(),
    )

    assert result["cited_indexes"] == [0]
    assert "[2]" not in result["answer_markdown"]


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
        target=_generation_target(),
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


def _fake_settings_deepseek():
    from app.core.config import Settings

    s = Settings(_env_file=None)
    s.agent_synthesis_provider = "deepseek"
    s.generation_provider = "deepseek"
    s.deepseek_api_key = "test-key"
    s.deepseek_base_url = "https://api.deepseek.com/v1"
    s.deepseek_model = "deepseek-chat"
    s.generation_max_retries = 0
    s.generation_retry_backoff_seconds = 0
    return s


def test_synthesize_deepseek_returns_usage_and_structured_result(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_deepseek)

    def fake_generate(self, **kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        return {
            "content": '{"answer_markdown":"DeepSeek answer [0]", "cited_indexes":[0], "warnings":[], "confidence":0.8}',
            "model": "deepseek-chat",
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            "usage_source": "provider",
        }

    monkeypatch.setattr("app.services.agent_synthesizer.DeepSeekClient.generate_chat", fake_generate)
    syn = AgentSynthesizer()

    result = syn.synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-chat"
    assert result["usage"]["total_tokens"] == 28
    assert result["usage_source"] == "provider"


def test_comparison_synthesis_allows_grounded_paraphrase_without_matrix_anchor_rollback(
    monkeypatch,
) -> None:
    """Comparison prose need not repeat every raw excerpt verbatim.

    The side-aware matrix guard still requires both paper citations, while the
    generic draft-anchor guard is intentionally skipped for matrix drafts.
    """
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_deepseek)
    monkeypatch.setattr(
        "app.services.agent_synthesizer.DeepSeekClient.generate_chat",
        lambda self, **kwargs: {
            "content": '{"answer_markdown":"Paper A uses self-reflection [0], while Paper B uses an external evaluator and corrective actions [1].","cited_indexes":[0,1],"warnings":[],"confidence":0.9}',
            "model": "deepseek-chat",
            "usage": {"total_tokens": 12},
            "usage_source": "provider",
        },
    )
    evidence_pack = {
        "status": "ok",
        "items": [
            {"index": 0, "document_id": "d1", "excerpt": "raw anchor A that should be paraphrased"},
            {"index": 1, "document_id": "d2", "excerpt": "raw anchor B that should be paraphrased"},
        ],
        "comparison": {
            "papers": [{"document_id": "d1"}, {"document_id": "d2"}],
            "cells": [
                {"paper_id": "d1", "dimension": "method", "status": "supported", "evidence_indexes": [0], "citation_indexes": [0]},
                {"paper_id": "d2", "dimension": "method", "status": "supported", "evidence_indexes": [1], "citation_indexes": [1]},
            ],
        },
    }
    result = AgentSynthesizer().synthesize(
        query="compare the methods",
        route="multi_source_compare",
        rag_answer="## Comparison evidence matrix\n- d1: raw anchor A that should be paraphrased [0]\n- d2: raw anchor B that should be paraphrased [1]",
        citations=[
            {"document_id": "d1", "chunk_id": "c1", "excerpt": "raw anchor A that should be paraphrased"},
            {"document_id": "d2", "chunk_id": "c2", "excerpt": "raw anchor B that should be paraphrased"},
        ],
        evidence_pack=evidence_pack,
    )

    assert result["answer_markdown"].startswith("Paper A uses self-reflection")
    assert result["provider"] == "deepseek"
    assert not any("lost evidence-anchor" in warning for warning in result["warnings"])


def test_answer_numbers_ignores_arxiv_identifiers() -> None:
    assert AgentSynthesizer._answer_numbers("arXiv:2401.15884 reports 84.3") == {"84.3"}


def test_deepseek_fidelity_fallback_keeps_remote_provider_identity(monkeypatch) -> None:
    """A rejected DeepSeek answer must not be mislabeled as local Ollama."""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_deepseek)

    monkeypatch.setattr(
        "app.services.agent_synthesizer.DeepSeekClient.generate_chat",
        lambda self, **kwargs: {
            "content": '{"answer_markdown":"Only the method is reported.","cited_indexes":[0],"warnings":[],"confidence":0.6}',
            "model": "deepseek-chat",
            "usage": {"total_tokens": 10},
            "usage_source": "provider",
        },
    )
    syn = AgentSynthesizer()
    draft = "NMR spectroscopy at 7.5 kcal/mol for the GLH mutant was used."
    result = syn.synthesize(
        query="What technique and energy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[
            {
                "document_id": "d1",
                "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
            }
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "Table 5: NMR at 7.5 kcal/mol for GLH mutant.",
                    "evidence_kind": "table",
                }
            ],
        },
    )

    assert result["answer_markdown"] == draft
    assert result["provider"] == "deepseek"
    assert result["model"] == "deepseek-chat"
    assert any("fidelity guard" in warning.lower() for warning in result["warnings"])


def test_parse_deepseek_json_accepts_fence_and_reasoning_wrapper() -> None:
    payload = AgentSynthesizer._parse_deepseek_json(
        "<think>brief reasoning</think>\n```json\n"
        '{"answer_markdown":"Grounded answer","cited_indexes":[0]}\n```'
    )
    assert payload["answer_markdown"] == "Grounded answer"
    assert payload["cited_indexes"] == [0]


def test_deepseek_singleton_object_array_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_deepseek)
    monkeypatch.setattr(
        "app.services.agent_synthesizer.DeepSeekClient.generate_chat",
        lambda self, **kwargs: {
            "content": '[{"answer_markdown":"Wrapped answer [0]","cited_indexes":[0],"warnings":[],"confidence":0.8}]',
            "model": "deepseek-chat",
            "usage": {"total_tokens": 4},
            "usage_source": "provider",
        },
    )

    result = AgentSynthesizer().synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert result["answer_markdown"] == "Wrapped answer [0]"
    assert result["provider"] == "deepseek"


def test_deepseek_retries_once_after_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_deepseek)
    calls = {"count": 0}

    def fake_generate(self, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"content": "[invalid]", "model": "deepseek-chat"}
        return {
            "content": '{"answer_markdown":"Recovered answer [0]","cited_indexes":[0],"warnings":[],"confidence":0.8}',
            "model": "deepseek-chat",
            "usage": {"total_tokens": 4},
            "usage_source": "provider",
        }

    monkeypatch.setattr("app.services.agent_synthesizer.DeepSeekClient.generate_chat", fake_generate)
    result = AgentSynthesizer().synthesize(
        query="What is entropy?",
        route="simple_rag",
        rag_answer="Entropy is a measure of disorder.",
        citations=[{"document_id": "d1", "excerpt": "entropy defined"}],
    )

    assert calls["count"] == 2
    assert result["answer_markdown"] == "Recovered answer [0]"
    assert any("parse retry" in warning for warning in result["warnings"])


def test_synthesize_auto_selects_deepseek_when_generation_provider_is_configured(
    monkeypatch,
) -> None:
    settings = _fake_settings_deepseek()
    settings.agent_synthesis_provider = "auto"
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.services.agent_synthesizer.DeepSeekClient.generate_chat",
        lambda self, **kwargs: {
            "content": '{"answer_markdown":"DeepSeek answer", "cited_indexes":[], "warnings":[], "confidence":0.7}',
            "model": "deepseek-chat",
            "usage": None,
            "usage_source": "unknown",
        },
    )

    result = AgentSynthesizer().synthesize(
        query="Say hello",
        route="simple_rag",
        rag_answer="Hello",
        citations=[],
    )

    assert result["provider"] == "deepseek"


def test_synthesize_deepseek_without_key_uses_grounded_fallback(monkeypatch) -> None:
    settings = _fake_settings_deepseek()
    settings.deepseek_api_key = None
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", lambda: settings)

    result = AgentSynthesizer().synthesize(
        query="Say hello",
        route="simple_rag",
        rag_answer="Hello",
        citations=[],
    )

    assert result["provider"] == "deepseek"
    assert result["answer_markdown"] == "Hello"
    assert any("not configured" in warning for warning in result["warnings"])


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


def _fake_settings_ollama():
    s = _fake_settings_local()
    s.agent_synthesis_provider = "ollama"
    s.ollama_synthesis_model = "qwen3.5:9b-synthesis"
    return s


# ------------------------------------------------------------------
# evidence pack synthesis tests
# ------------------------------------------------------------------


def test_synthesize_accepts_optional_evidence_pack(monkeypatch) -> None:
    """synthesize() accepts optional evidence_pack parameter without error."""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_local
    )
    syn = AgentSynthesizer(
        ollama_client=FakeOllamaClient(error=RuntimeError("Ollama unavailable"))
    )
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
    syn = AgentSynthesizer(
        ollama_client=FakeOllamaClient(error=RuntimeError("Ollama unavailable"))
    )

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


def test_table_facts_prompt_keeps_human_readable_table_labels() -> None:
    """Exact facts must remain associated with their source table label."""
    evidence_pack = {
        "status": "ok",
        "items": [
            {
                "table_id": "table-2",
                "excerpt": "Table 2. Hydration free energies for small aromatic molecules.",
            }
        ],
        "table_facts": [
            {
                "table_id": "table-2",
                "row_label": "RMS error",
                "column": "OPLS4",
                "value": "0.76",
            },
            {
                "table_id": "table-2",
                "row_label": "RMS error",
                "column": "OPLS5",
                "value": "0.46",
            },
        ],
    }

    rendered = AgentSynthesizer._format_table_facts_section(evidence_pack)

    assert "label=Table 2" in rendered
    assert "caption=Hydration free energies for small aromatic molecules" in rendered
    assert "table=table-2" in rendered
    assert "row=RMS error column=OPLS4 value=0.76" in rendered
    assert "row=RMS error column=OPLS5 value=0.46" in rendered


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
    syn = AgentSynthesizer(
        ollama_client=FakeOllamaClient(error=RuntimeError("Ollama unavailable"))
    )
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


def test_table_facts_bypass_generic_excerpt_cap() -> None:
    evidence_pack = {
        "status": "ok",
        "items": [{"index": 0, "evidence_kind": "table", "excerpt": "short"}],
        "table_facts": [
            {
                "table_id": "table-7",
                "row_label": "OPLS5",
                "column": "C6",
                "value": "8.95",
            }
        ],
    }

    prompt = AgentSynthesizer._format_table_facts_section(evidence_pack)

    assert "table-7" in prompt
    assert "OPLS5" in prompt
    assert "C6" in prompt
    assert "8.95" in prompt


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


def test_local_synthesis_unknown_expected_facts_skips_missing_field_check(
    monkeypatch,
) -> None:
    """9.8：问题无可解析需求目标（指代追问）→ expected_facts_status=unknown，
    不触发 facts 遗漏回退（draft-fidelity 锚点守卫仍生效）。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    # 合成答案保留 draft 全部锚点（21.0/1.18/kcal/mol），仅隔离 facts 校验
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: the value is 21.0 and 1.18 kcal/mol [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The value is 21.0 and 1.18 kcal/mol."
    result = syn.synthesize(
        query="那 C36 呢？",
        route="table_or_metric",
        rag_answer=draft,
        # 摘录不带句号：_answer_numbers 不提取"21.0." 这类句尾数字
        citations=[{"document_id": "d1", "excerpt": "Table 7: value 21.0"}],
        evidence_pack={
            "status": "ok",
            "items": [],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "1.18",
                    "row_index": 1,
                }
            ],
        },
        target=_generation_target(),
    )
    # 期望集合不可判定 → facts 遗漏校验短路；合成结果保留、状态可审计
    assert result["answer_markdown"] == "Table 7: the value is 21.0 and 1.18 kcal/mol [0]."
    assert result["expected_facts_status"] == "unknown"


def test_local_synthesis_missing_expected_facts_does_not_rollback(
    monkeypatch,
) -> None:
    """9.8：需求目标解析但 facts 零命中 → expected_facts_status=missing，
    不触发遗漏回退（需求存在但供给未返回，不是 synthesis 的错）。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    # 合成答案保留 draft 全部锚点与 Table 7 标签，仅隔离 facts 校验；
    # 摘录不带句号（_answer_numbers 不提取"21.0." 这类句尾数字）
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: the value is 21.0 and 1.18 kcal/mol [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "The value is 21.0 and 1.18 kcal/mol."
    result = syn.synthesize(
        query="What about the accuracy?",
        route="table_or_metric",
        rag_answer=draft,
        citations=[{"document_id": "d1", "excerpt": "Table 7: value 21.0"}],
        evidence_pack={
            "status": "ok",
            "items": [],
            "inventory": [_inventory_entry()],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "value",
                    "value": "1.18",
                    "row_index": 1,
                }
            ],
        },
        target=_generation_target(),
    )
    # 无事实可对照 → 保留合成结果、状态可审计（需求存在但供给未返回）
    assert result["answer_markdown"] == "Table 7: the value is 21.0 and 1.18 kcal/mol [0]."
    assert result["expected_facts_status"] == "missing"


def test_expected_facts_matches_explicit_table_reference() -> None:
    """9.8/任务9：显式 Table N + 列需求经 typed inventory 解析 →
    与已返回 facts 全部精确命中 → complete。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    evidence_pack = {
        "inventory": [_inventory_entry()],
        "table_facts": [
            _table_fact(
                row_label="OPLS5",
                column="binding RMSE",
                value="1.18",
                row_index=2,
            )
        ],
    }
    status, targets = syn._expected_facts_status(
        "Table 7 的 binding RMSE 是多少？", evidence_pack
    )
    assert status == "complete"
    assert targets and targets[0]["column"] == "binding rmse"
    assert targets[0]["table_id"] == "table-7"
    assert targets[0]["document_id"] == "d1"
    assert targets[0]["parse_version"] == "canonical-v4"
    assert targets[0]["row_index"] is None
    assert targets[0]["identity"] == "d1|canonical-v4|table-7|any|binding rmse"


def test_expected_facts_unknown_without_parseable_target() -> None:
    """9.8/任务9：无术语且无表引用的指代追问 → unknown，不移除任何 facts 校验。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    status, targets = syn._expected_facts_from_question(
        "那 C36 呢？",
        [_inventory_entry()],
    )
    assert status == "unknown"
    assert targets == []


# ------------------------------------------------------------------
# 9.8/任务9：需求侧 expected facts（typed inventory + exact join）
# ------------------------------------------------------------------


def _inventory_entry(
    table_id: str = "table-7",
    document_id: str = "d1",
    parse_version: str = "canonical-v4",
    row_indices: list[int] | None = None,
) -> dict:
    """构造 typed inventory（TableCoverage dict 形态）测试数据。"""
    row_indices = row_indices or []
    return {
        "document_id": document_id,
        "parse_version": parse_version,
        "table_id": table_id,
        "row_count": len(row_indices),
        "source_block_ids": ["sb-1"],
        "child_ids": ["inv-child-1"],
        "child_count": 1,
        "parent_ids": ["p-1"],
        "row_indices": row_indices,
    }


def _table_fact(
    table_id: str = "table-7",
    document_id: str = "d1",
    parse_version: str = "canonical-v4",
    row_index: int = 0,
    row_label: str = "",
    column: str = "value",
    value: str = "21.0",
    unit: str = "",
    term: str | None = None,
) -> dict:
    """构造已返回 table_facts（TableFactEvidence dict 形态）测试数据。"""
    return {
        "table_id": table_id,
        "document_id": document_id,
        "parse_version": parse_version,
        "row_label": row_label,
        "column": column,
        "value": value,
        "row_index": row_index,
        "source_chunk_ids": ["c1"],
        "fact_id": f"{table_id}|{row_index}|{column}|{value}",
        "unit": unit,
        "term": term or column,
    }


def test_expected_facts_from_question_requires_unique_inventory() -> None:
    """9.8/任务9：inventory 缺失或无法唯一解析（多表/多版本）→ unknown。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    assert syn._expected_facts_from_question("What is X?", None)[0] == "unknown"
    assert syn._expected_facts_from_question("What is X?", [])[0] == "unknown"
    # 无表引用 + 多表 inventory → 无法唯一解析目标表 → unknown
    multi = [_inventory_entry(), _inventory_entry(table_id="table-2")]
    assert (
        syn._expected_facts_from_question("What about the accuracy?", multi)[0]
        == "unknown"
    )
    # 同一表号出现在两个解析版本 → 无法唯一解析 → unknown
    two_versions = [_inventory_entry(), _inventory_entry(parse_version="v3")]
    assert (
        syn._expected_facts_from_question("Table 7 的 accuracy 是多少？", two_versions)[0]
        == "unknown"
    )
    # 请求的表不在 inventory 中 → 无法唯一解析 → unknown
    only_other = [_inventory_entry(table_id="table-2")]
    assert (
        syn._expected_facts_from_question("Table 7 的 accuracy 是多少？", only_other)[0]
        == "unknown"
    )


def test_expected_facts_from_question_returns_demand_targets() -> None:
    """9.8/任务9：返回需求目标（含 join 身份），不是已召回 facts。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    status, targets = syn._expected_facts_from_question(
        "Table 7 的 binding RMSE 是多少？",
        [_inventory_entry(row_indices=[2])],
    )
    assert status == "resolved"
    assert len(targets) == 1
    target = targets[0]
    assert target["table_ref"] == "Table 7"
    assert target["table_id"] == "table-7"
    assert target["document_id"] == "d1"
    assert target["parse_version"] == "canonical-v4"
    assert target["column"] == "binding rmse"
    assert target["row_index"] is None
    assert target["identity"] == "d1|canonical-v4|table-7|any|binding rmse"


def test_expected_facts_multicolumn_demand_partial_supply_is_missing() -> None:
    """9.8/任务9：问题要求 Asp、C6、exptl，仅返回 Asp=21.0 →
    missing，单个 fact 命中不能把多列需求判为 complete。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    evidence_pack = {
        "inventory": [_inventory_entry()],
        "table_facts": [_table_fact(column="Asp", value="21.0")],
    }
    status, targets = syn._expected_facts_status(
        "Table 7 的 Asp、C6、exptl 值是多少？", evidence_pack
    )
    assert status == "missing"
    assert [t["column"] for t in targets] == ["asp", "c6", "exptl"]


def test_expected_facts_all_demanded_columns_returned_is_complete() -> None:
    """9.8/任务9：多列需求全部命中 → complete（join 的正向对照）。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    evidence_pack = {
        "inventory": [_inventory_entry()],
        "table_facts": [
            _table_fact(column="Asp", value="21.0"),
            _table_fact(column="C6", value="1.18"),
            _table_fact(column="exptl", value="2.5"),
        ],
    }
    status, _ = syn._expected_facts_status(
        "Table 7 的 Asp、C6、exptl 值是多少？", evidence_pack
    )
    assert status == "complete"


def test_expected_facts_wrong_table_facts_do_not_satisfy() -> None:
    """9.8/任务9：明确请求 Table 7，只返回 Table 2 facts → missing，
    错表 fact 不能当作 Table 7 需求的满足项。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    evidence_pack = {
        "inventory": [_inventory_entry(), _inventory_entry(table_id="table-2")],
        "table_facts": [
            _table_fact(table_id="table-2", column="binding RMSE", value="1.18")
        ],
    }
    status, targets = syn._expected_facts_status(
        "Table 7 的 binding RMSE 是多少？", evidence_pack
    )
    assert status == "missing"
    assert targets[0]["table_id"] == "table-7"


def test_expected_facts_exact_join_on_document_version_table_row() -> None:
    """9.8/任务9：exact join 按 document_id + parse_version + table_id +
    row_index/row_label + column/term；任一维度不匹配即 missing。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    inventory = [_inventory_entry(row_indices=[2, 3])]
    question = "Table 7 的 row 3 binding RMSE 是多少？"
    # 行号 + 列全部匹配 → complete
    ok = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(row_index=3, column="binding RMSE", value="1.18")
        ],
    }
    assert syn._expected_facts_status(question, ok)[0] == "complete"
    # 行不匹配 → missing
    wrong_row = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(row_index=2, column="binding RMSE", value="1.12")
        ],
    }
    assert syn._expected_facts_status(question, wrong_row)[0] == "missing"
    # 版本不匹配 → missing
    wrong_version = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(
                row_index=3, parse_version="v3", column="binding RMSE", value="1.18"
            )
        ],
    }
    assert syn._expected_facts_status(question, wrong_version)[0] == "missing"
    # 文档不匹配 → missing
    wrong_doc = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(
                row_index=3, document_id="d2", column="binding RMSE", value="1.18"
            )
        ],
    }
    assert syn._expected_facts_status(question, wrong_doc)[0] == "missing"


def test_expected_facts_empty_facts_parseable_inventory_is_missing() -> None:
    """9.8/任务9：table_facts=[] 但 inventory 可解析 → demand targets 仍
    生成且状态为 missing，不能静默变成 unknown 或 complete。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    evidence_pack = {
        "inventory": [_inventory_entry()],
        "table_facts": [],
    }
    status, targets = syn._expected_facts_status(
        "Table 7 的 Asp、C6、exptl 值是多少？", evidence_pack
    )
    assert status == "missing"
    assert [t["column"] for t in targets] == ["asp", "c6", "exptl"]
    assert all(t["table_id"] == "table-7" for t in targets)


def test_expected_facts_row_label_demand_complete_and_missing() -> None:
    """9.8/任务9：问题明确提到行标签（"the total row"）→ 需求 target
    携带 row_label 并参与 exact identity join：同标签 fact 满足 →
    complete，其他标签（mean）不能满足 → missing，无行标签 fact
    同样不满足（行维度不放松为 any）。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    inventory = [_inventory_entry()]
    # 自然语言 "the total row" 形式（英文）
    question = "What is the binding free energy in the total row of Table 7?"
    same_label = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(row_label="total", column="binding free energy", value="1.18")
        ],
    }
    status, targets = syn._expected_facts_status(question, same_label)
    assert status == "complete"
    assert targets[0]["row_label"] == "total"
    assert targets[0]["column"] == "binding free energy"
    assert (
        targets[0]["identity"]
        == "d1|canonical-v4|table-7|label:total|binding free energy"
    )
    # 其他行标签 fact 不能满足同一需求
    other_label = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(row_label="mean", column="binding free energy", value="1.18")
        ],
    }
    assert syn._expected_facts_status(question, other_label)[0] == "missing"
    # 无行标签 fact（行维度为 any）同样不满足行标签需求
    any_row = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(row_label="", column="binding free energy", value="1.18")
        ],
    }
    assert syn._expected_facts_status(question, any_row)[0] == "missing"


@pytest.mark.parametrize("alias", ["total", "average", "mean", "overall"])
def test_expected_facts_row_label_aliases_join_exactly(alias: str) -> None:
    """9.8/任务9：total/average/mean/overall 行别名（"X row" 形式）均映射
    到 canonical row label 并参与 exact join；同标签返回 complete。"""
    syn = AgentSynthesizer(ollama_client=FakeOllamaClient())
    inventory = [_inventory_entry()]
    question = f"Table 7 的 {alias} row 的 binding free energy 是多少？"
    matched = {
        "inventory": inventory,
        "table_facts": [
            _table_fact(row_label=alias, column="binding free energy", value="1.18")
        ],
    }
    status, targets = syn._expected_facts_status(question, matched)
    assert status == "complete"
    assert targets[0]["row_label"] == alias
    assert targets[0]["identity"].endswith(f"label:{alias}|binding free energy")


def test_local_synthesis_multicolumn_demand_partial_facts_status_missing(
    monkeypatch,
) -> None:
    """9.8/任务9：端到端——问题要求 Asp、C6、exptl，只返回 Asp=21.0 →
    expected_facts_status=missing，不因单个 fact 命中而 complete，也不
    触发遗漏回退（供给缺失不是 synthesis 的错）。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: Asp=21.0 [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "Table 7: Asp=21.0."
    result = syn.synthesize(
        query="Table 7 的 Asp、C6、exptl 值是多少？",
        route="table_or_metric",
        rag_answer=draft,
        citations=[{"document_id": "d1", "excerpt": "Table 7: Asp=21.0"}],
        evidence_pack={
            "status": "ok",
            "items": [],
            "inventory": [_inventory_entry()],
            "table_facts": [_table_fact(column="Asp", value="21.0")],
        },
        target=_generation_target(),
    )
    # 其余守卫通过（锚点/数值/表号均保留），仅需求侧状态为 missing
    assert result["answer_markdown"] == "Table 7: Asp=21.0 [0]."
    assert result["expected_facts_status"] == "missing"


def test_local_synthesis_dropping_fact_term_falls_back(monkeypatch) -> None:
    """9.3.2/任务9：合成答案删除期望 facts 的术语（RMS error），
    draft 覆盖但合成遗漏 → 回退。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    # 合成保留数值 21.0 与表号，但删除术语 RMS error（隔离 value/表号守卫）
    ollama = FakeOllamaClient(
        result=_local_result(answer_markdown="Table 7: RMS 21.0 [0].")
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    draft = "Table 7: RMS error is 21.0%."
    result = syn.synthesize(
        query="Table 7 的 RMS error 值是多少？",
        route="table_or_metric",
        rag_answer=draft,
        citations=[{"document_id": "d1", "excerpt": "Table 7: RMS error is 21.0%"}],
        evidence_pack={
            "status": "ok",
            "items": [],
            "inventory": [_inventory_entry()],
            "table_facts": [
                {
                    "table_id": "table-7",
                    "document_id": "d1",
                    "parse_version": "canonical-v4",
                    "row_label": "total",
                    "column": "RMS error",
                    "value": "21.0",
                    "unit": "%",
                    "term": "RMS error",
                    "row_index": 0,
                    "source_chunk_ids": ["c1"],
                }
            ],
        },
        target=_generation_target(),
    )
    # 合成答案删除术语 → 回退 RAG 草稿（保留术语）
    assert result["answer_markdown"] == draft
    assert result["model"] == "local-fallback"


# ------------------------------------------------------------------
# 9.3.1/任务9：四路径 narrow_context 收窄矩阵（local/ollama/external）
# ------------------------------------------------------------------


def test_local_narrow_context_retry_second_call_also_narrowed(monkeypatch) -> None:
    """9.3.1/任务9：local coverage retry 的第二次调用同样收窄（不含
    evidence items 摘录与完整 inventory）。"""
    monkeypatch.setattr("app.services.agent_synthesizer.get_settings", _fake_settings_local)
    ollama = _SequencedLocalOllama(
        responses=[
            {"answer_markdown": "Standard analysis was performed."},
            {"answer_markdown": "NMR spectroscopy at 7.5 kcal/mol was used."},
        ]
    )
    syn = AgentSynthesizer(ollama_client=ollama)
    syn.synthesize(
        query="What technique?",
        route="evidence_required",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy at 7.5 kcal/mol."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "full narrative evidence paragraph excluded",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
            "inventory": [_inventory_entry()],
        },
        target=_generation_target(),
        narrow_context=True,
    )
    assert len(ollama.calls) == 2, f"Expected first + retry, got {len(ollama.calls)}"
    for call in ollama.calls:
        prompt = (
            call["messages"][0]["content"] + "\n" + call["messages"][1]["content"]
        )
        assert "evidence-item-0" not in prompt
        assert "full narrative evidence paragraph excluded" not in prompt
        assert "inv-child-1" not in prompt


def test_ollama_narrow_context_excludes_items_and_inventory_retry(monkeypatch) -> None:
    """9.3.1/任务9：ollama 路径 narrow_context=True 时首轮与 coverage
    retry 第二次调用均不含 evidence items 摘录与完整 inventory，
    结构化 table_facts 保留。"""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_ollama
    )

    class SequencedStructuredOllama:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.responses = [
                SynthesisPayload(
                    answer_markdown="The structure was analyzed.",
                    cited_indexes=[0],
                    warnings=[],
                    confidence=0.7,
                ),
                SynthesisPayload(
                    answer_markdown="The structure was analyzed using NMR spectroscopy.",
                    cited_indexes=[0],
                    warnings=[],
                    confidence=0.85,
                ),
            ]

        def generate_structured(self, schema, **kwargs):
            self.calls.append({"schema": schema, **kwargs})
            return self.responses.pop(0)

    ollama = SequencedStructuredOllama()
    AgentSynthesizer(ollama_client=ollama).synthesize(
        query="What technique analyzed the structure?",
        route="evidence_required",
        rag_answer="The structure was analyzed.",
        citations=[
            {
                "document_id": "d1",
                "excerpt": "NMR spectroscopy was used to analyze the molecular structure.",
            }
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "full narrative evidence paragraph excluded",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
            "inventory": [_inventory_entry()],
            "table_facts": [_table_fact()],
        },
        narrow_context=True,
    )
    assert len(ollama.calls) == 2, f"Expected first + retry, got {len(ollama.calls)}"
    for call in ollama.calls:
        prompt = (
            str(call.get("system_prompt") or "") + "\n" + str(call.get("user_prompt") or "")
        )
        assert "full narrative evidence paragraph excluded" not in prompt
        assert "inv-child-1" not in prompt
        # 结构化 table_facts 在收窄时仍保留（只去 items/inventory）
        assert "21.0" in prompt


def test_external_narrow_context_excludes_items_and_inventory_retry(monkeypatch) -> None:
    """9.3.1/任务9：external 路径 narrow_context=True 时首轮与 coverage
    retry 第二次调用均不含 evidence items 摘录与完整 inventory，
    结构化 table_facts 保留。"""
    monkeypatch.setattr(
        "app.services.agent_synthesizer.get_settings", _fake_settings_external
    )
    import httpx

    captured: list[str] = []
    call_count = [0]

    def fake_post(self, url, json, headers, **kw):
        call_count[0] += 1
        captured.append(json["messages"][1]["content"])
        if call_count[0] == 1:
            return _fake_response(
                '{"answer_markdown": "Standard analysis was performed.", '
                '"cited_indexes": [0], "warnings": [], "confidence": 0.5}'
            )
        return _fake_response(
            '{"answer_markdown": "NMR spectroscopy at 7.5 kcal/mol was used.", '
            '"cited_indexes": [0], "warnings": [], "confidence": 0.8}'
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    syn = AgentSynthesizer()
    syn.synthesize(
        query="What technique?",
        route="evidence_required",
        rag_answer="Some analysis.",
        citations=[
            {"document_id": "d1", "excerpt": "NMR spectroscopy at 7.5 kcal/mol."}
        ],
        evidence_pack={
            "status": "ok",
            "items": [
                {
                    "index": 0,
                    "document_id": "d1",
                    "excerpt": "full narrative evidence paragraph excluded",
                    "evidence_kind": "source_chunk",
                    "source_stage": "source_chunk",
                    "support_hint": "direct",
                }
            ],
            "inventory": [_inventory_entry()],
            "table_facts": [_table_fact()],
        },
        narrow_context=True,
    )
    assert call_count[0] == 2, f"Expected first + retry, got {call_count[0]} calls"
    for prompt in captured:
        assert "full narrative evidence paragraph excluded" not in prompt
        assert "inv-child-1" not in prompt
        # 结构化 table_facts 在收窄时仍保留（只去 items/inventory）
        assert "21.0" in prompt
