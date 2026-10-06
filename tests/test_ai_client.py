import pytest
import httpx

from app.services import ai
from app.services.ai import (
    ContextualizationOllamaClient,
    DocumentPagePayload,
    HeadAnalysisPayload,
    OllamaClient,
)


def test_keep_alive_numeric_strings_are_normalized_for_ollama_compatibility(monkeypatch) -> None:
    monkeypatch.setattr(ai.settings, "ollama_keep_alive", "-1", raising=False)

    payload = OllamaClient._with_keep_alive({"model": "test"})

    assert payload["keep_alive"] == -1


def test_parse_structured_content_extracts_json_from_text() -> None:
    content = """
    Here is the structured result:

    ```json
    {"head_entity":"Hypertension","summary":"Follow-up needed","triples":[]}
    ```
    """

    parsed = OllamaClient._parse_structured_content(HeadAnalysisPayload, content)

    assert parsed.head_entity == "Hypertension"
    assert parsed.summary == "Follow-up needed"


def test_client_uses_injected_generation_and_embedding_urls() -> None:
    client = OllamaClient(
        base_url="http://deep:11436/",
        embedding_base_url="http://embed:11435/",
    )

    assert client.base_url == "http://deep:11436"
    assert client.embedding_base_url == "http://embed:11435"


def test_generate_chat_sends_context_keep_alive_and_disables_thinking(monkeypatch) -> None:
    client = OllamaClient(base_url="http://deep:11436")
    captured: list[dict] = []

    def fake_post_chat(payload):
        captured.append(payload)
        return {
            "model": "deep-model",
            "message": {"content": "answer [0]"},
            "prompt_eval_count": 11,
            "eval_count": 5,
            "prompt_eval_duration": 100,
            "eval_duration": 200,
        }

    monkeypatch.setattr(client, "_post_chat", fake_post_chat)
    result = client.generate_chat(
        messages=[{"role": "user", "content": "answer"}],
        model="deep-model",
        context_length=32768,
        max_output_tokens=512,
    )

    assert result["content"] == "answer [0]"
    assert result["model"] == "deep-model"
    assert result["prompt_eval_count"] == 11
    assert captured == [
        {
            "model": "deep-model",
            "messages": [{"role": "user", "content": "answer"}],
            "stream": False,
            "think": False,
            "options": {"num_ctx": 32768, "num_predict": 512},
        }
    ]


def test_stream_chat_parses_ndjson_and_stops_when_cancelled(monkeypatch) -> None:
    import threading

    from app.services import ai

    captured: dict = {}
    cancel = threading.Event()

    class FakeStreamResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield '{"message":{"content":"first "},"done":false}'
            cancel.set()
            yield '{"message":{"content":"second"},"done":false}'

    class FakeStreamContext:
        def __enter__(self):
            return FakeStreamResponse()

        def __exit__(self, exc_type, exc, traceback):
            return None

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def stream(self, method, url, json):
            captured.update({"method": method, "url": url, "json": json})
            return FakeStreamContext()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    client = OllamaClient(base_url="http://deep:11436")

    chunks = list(
        client.stream_chat(
            messages=[{"role": "user", "content": "answer"}],
            model="deep-model",
            context_length=32768,
            cancel_event=cancel,
        )
    )

    assert [chunk["content"] for chunk in chunks] == ["first "]
    assert captured["url"] == "http://deep:11436/api/chat"
    assert captured["json"]["stream"] is True
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_ctx"] == 32768


def test_embed_uses_dedicated_embedding_url(monkeypatch) -> None:
    from app.services import ai

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2]]}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def post(self, url, json):
            captured.update({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(ai.settings, "ollama_embedding_model", "embed-model")
    client = OllamaClient(
        base_url="http://deep:11436",
        embedding_base_url="http://embed:11435",
    )

    assert client.embed(["alpha"]) == [[0.1, 0.2]]
    assert captured["url"] == "http://embed:11435/api/embed"
    assert captured["json"]["model"] == "embed-model"


def test_embed_uses_openai_compatible_qwen_api_and_restores_input_order(monkeypatch) -> None:
    from app.services import ai

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def post(self, url, *, headers, json):
            captured.update({"url": url, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(ai.settings, "embedding_provider", "openai-compatible")
    monkeypatch.setattr(ai.settings, "embedding_api_base_url", "https://embed.example/v1/")
    monkeypatch.setattr(ai.settings, "embedding_api_key", "test-secret")
    monkeypatch.setattr(ai.settings, "embedding_api_model", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(ai.settings, "embedding_api_timeout", 12)
    monkeypatch.setattr(ai.settings, "ollama_embedding_dimensions", 2)

    result = OllamaClient().embed(["alpha", "beta"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["url"] == "https://embed.example/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["json"] == {
        "model": "Qwen/Qwen3-Embedding-4B",
        "input": ["alpha", "beta"],
    }
    assert captured["timeout"] == 12


def test_embed_batches_openai_compatible_requests_at_configured_limit(monkeypatch) -> None:
    from app.services import ai

    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, payload: dict):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def post(self, url, *, headers, json):
            calls.append({"url": url, "headers": headers, "json": json})
            # Return each batch in reverse order to verify batch-local index
            # sorting while the caller concatenates batches in input order.
            offset = (len(calls) - 1) * 100
            rows = [
                {"index": index, "embedding": [float(offset + index), 1.0]}
                for index in range(len(json["input"]))
            ]
            return FakeResponse({"data": list(reversed(rows))})

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(ai.settings, "embedding_provider", "openai-compatible")
    monkeypatch.setattr(ai.settings, "embedding_api_base_url", "https://embed.example/v1")
    monkeypatch.setattr(ai.settings, "embedding_api_key", "test-secret")
    monkeypatch.setattr(ai.settings, "embedding_api_model", "qwen3.7-text-embedding")
    monkeypatch.setattr(ai.settings, "embedding_api_batch_size", 20)
    monkeypatch.setattr(ai.settings, "embedding_dimensions", 2)

    texts = [f"text-{index}" for index in range(21)]
    result = OllamaClient().embed(texts)

    assert len(calls) == 2
    assert [len(call["json"]["input"]) for call in calls] == [20, 1]
    assert result == [
        [float(index), 1.0] for index in range(20)
    ] + [[100.0, 1.0]]


def test_embed_openai_compatible_requires_server_credentials(monkeypatch) -> None:
    from app.services import ai

    monkeypatch.setattr(ai.settings, "embedding_provider", "openai-compatible")
    monkeypatch.setattr(ai.settings, "embedding_api_base_url", None)
    monkeypatch.setattr(ai.settings, "embedding_api_key", None)

    with pytest.raises(ValueError, match="EMBEDDING_API_BASE_URL"):
        OllamaClient().embed(["alpha"])


def test_embed_openai_compatible_rejects_wrong_dimensions(monkeypatch) -> None:
    from app.services import ai

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1]}]}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def post(self, url, *, headers, json):
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(ai.settings, "embedding_provider", "openai-compatible")
    monkeypatch.setattr(ai.settings, "embedding_api_base_url", "https://embed.example/v1")
    monkeypatch.setattr(ai.settings, "embedding_api_key", "test-secret")
    monkeypatch.setattr(ai.settings, "ollama_embedding_dimensions", 2)

    with pytest.raises(ValueError, match="dimensions"):
        OllamaClient().embed(["alpha"])


def test_deepseek_client_posts_openai_compatible_payload_and_preserves_usage(monkeypatch) -> None:
    from app.services.ai import DeepSeekClient

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "deepseek-chat",
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            }

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, *, json, headers):
            captured.update({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    client = DeepSeekClient(
        base_url="https://api.deepseek.com/v1/",
        api_key="test-key",
        model="deepseek-chat",
        timeout=7,
        max_retries=0,
    )

    result = client.generate_chat(
        messages=[{"role": "user", "content": "hello"}],
        max_output_tokens=128,
        response_format={"type": "json_object"},
    )

    assert result["content"] == '{"answer":"ok"}'
    assert result["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    assert result["usage_source"] == "provider"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["max_tokens"] == 128
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 7


def test_deepseek_client_does_not_retry_authentication_errors(monkeypatch) -> None:
    from app.services.ai import DeepSeekAPIError, DeepSeekClient

    calls = 0

    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise RuntimeError("unauthorized")

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)

    with pytest.raises(DeepSeekAPIError) as exc_info:
        DeepSeekClient(api_key="test-key", max_retries=3, retry_backoff_seconds=0).generate_chat(
            messages=[]
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False
    assert calls == 1


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_deepseek_client_retries_transient_http_errors(monkeypatch, status_code: int) -> None:
    from app.services.ai import DeepSeekAPIError, DeepSeekClient

    calls = 0

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return FakeResponse(status_code) if calls == 1 else FakeResponse(200)

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    result = DeepSeekClient(api_key="test-key", max_retries=1, retry_backoff_seconds=0).generate_chat(
        messages=[]
    )

    assert result["content"] == "ok"
    assert calls == 2


def test_deepseek_client_exhausts_transient_errors_and_retries_timeout(monkeypatch) -> None:
    from app.services.ai import DeepSeekAPIError, DeepSeekClient

    calls = 0

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)

    with pytest.raises(DeepSeekAPIError) as exc_info:
        DeepSeekClient(api_key="test-key", max_retries=1, retry_backoff_seconds=0).generate_chat(
            messages=[]
        )

    assert exc_info.value.error_code == "transport_error"
    assert exc_info.value.retryable is True
    assert calls == 2


def test_deepseek_client_marks_missing_usage_unknown(monkeypatch) -> None:
    from app.services.ai import DeepSeekClient

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    result = DeepSeekClient(api_key="test-key", max_retries=0).generate_chat(messages=[])

    assert result["usage"] is None
    assert result["usage_source"] == "unknown"
    assert result["prompt_tokens"] is None


def test_generate_structured_retries_empty_schema_response(monkeypatch) -> None:
    client = OllamaClient()
    calls: list[dict] = []

    def fake_post_chat(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"content": ""}}
        return {"message": {"content": '{"head_entity":"Hypertension","summary":"Recovered","triples":[]}'}}

    monkeypatch.setattr(client, "_post_chat", fake_post_chat)

    parsed = client.generate_structured(
        HeadAnalysisPayload,
        system_prompt="Return JSON.",
        user_prompt="Analyze this head.",
        model="fake-model",
    )

    assert parsed.head_entity == "Hypertension"
    assert parsed.summary == "Recovered"
    assert calls[0]["format"] != "json"
    assert calls[1]["format"] == "json"


def test_generate_structured_disables_thinking_by_default(monkeypatch) -> None:
    """结构化生成默认关闭推理链：与 generate_chat 对齐，避免 qwen3.5 思考链拖慢生成。"""
    client = OllamaClient()
    calls: list[dict] = []

    def fake_post_chat(payload):
        calls.append(payload)
        return {"message": {"content": '{"head_entity":"Hypertension","summary":"Recovered","triples":[]}'}}

    monkeypatch.setattr(client, "_post_chat", fake_post_chat)

    client.generate_structured(
        HeadAnalysisPayload,
        system_prompt="Return JSON.",
        user_prompt="Analyze this head.",
        model="fake-model",
    )

    assert calls[0]["think"] is False


def test_generate_structured_retry_json_mode_also_disables_thinking(monkeypatch) -> None:
    """Schema 约束失败后的 JSON 模式重试同样默认关闭推理链。"""
    client = OllamaClient()
    calls: list[dict] = []

    def fake_post_chat(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"content": ""}}
        return {"message": {"content": '{"head_entity":"Hypertension","summary":"Recovered","triples":[]}'}}

    monkeypatch.setattr(client, "_post_chat", fake_post_chat)

    client.generate_structured(
        HeadAnalysisPayload,
        system_prompt="Return JSON.",
        user_prompt="Analyze this head.",
        model="fake-model",
    )

    assert len(calls) == 2
    assert calls[0]["think"] is False
    assert calls[1]["think"] is False


def test_generate_structured_honors_explicit_think_override(monkeypatch) -> None:
    """显式传 think=True 时仍允许覆盖默认值（保留推理链能力）。"""
    client = OllamaClient()
    calls: list[dict] = []

    def fake_post_chat(payload):
        calls.append(payload)
        return {"message": {"content": '{"head_entity":"Hypertension","summary":"Recovered","triples":[]}'}}

    monkeypatch.setattr(client, "_post_chat", fake_post_chat)

    client.generate_structured(
        HeadAnalysisPayload,
        system_prompt="Return JSON.",
        user_prompt="Analyze this head.",
        model="fake-model",
        think=True,
    )

    assert calls[0]["think"] is True


def test_generate_structured_sends_keep_alive_to_all_chat_requests(monkeypatch) -> None:
    monkeypatch.setattr(ai.settings, "ollama_keep_alive", "0", raising=False)
    client = OllamaClient()
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, data: dict) -> None:
            self.data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.data

    class FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url: str, json: dict) -> FakeResponse:
            calls.append(json)
            if len(calls) == 1:
                return FakeResponse({"message": {"content": ""}})
            return FakeResponse(
                {"message": {"content": '{"head_entity":"Hypertension","summary":"Recovered","triples":[]}'}}
            )

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)

    client.generate_structured(
        HeadAnalysisPayload,
        system_prompt="Return JSON.",
        user_prompt="Analyze this head.",
        model="fake-model",
    )

    assert calls[0]["keep_alive"] == 0
    assert calls[1]["keep_alive"] == 0


def test_embed_sends_keep_alive_to_single_batch_embedding_request(monkeypatch) -> None:
    monkeypatch.setattr(ai.settings, "ollama_keep_alive", "0", raising=False)
    monkeypatch.setattr(ai.settings, "ollama_embedding_model", "fake-embedding", raising=False)
    client = OllamaClient()
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url: str, json: dict) -> FakeResponse:
            calls.append(json)
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)

    assert client.embed(["alpha", "beta"]) == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert calls == [{
        "model": "fake-embedding",
        "input": ["alpha", "beta"],
        "keep_alive": "0",
        "options": {"num_ctx": 16384},
    }]


def test_unload_loaded_models_releases_each_resident_model_once(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None]] = []

    class FakeResponse:
        def __init__(self, data: dict) -> None:
            self.data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.data

    class FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def get(self, url: str) -> FakeResponse:
            calls.append(("GET", url, None))
            if url.startswith("http://shared"):
                return FakeResponse(
                    {"models": [{"name": "qwen3.5:9b"}, {"model": "qwen3-embedding:4b"}]}
                )
            return FakeResponse({"models": []})

        def post(self, url: str, json: dict) -> FakeResponse:
            calls.append(("POST", url, json))
            return FakeResponse({})

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    client = OllamaClient(
        base_url="http://shared:11435",
        embedding_base_url="http://shared:11435",
    )

    assert client.unload_loaded_models() == ["qwen3.5:9b", "qwen3-embedding:4b"]
    assert calls == [
        ("GET", "http://shared:11435/api/ps", None),
        (
            "POST",
            "http://shared:11435/api/generate",
            {"model": "qwen3.5:9b", "keep_alive": 0},
        ),
        (
            "POST",
            "http://shared:11435/api/generate",
            {"model": "qwen3-embedding:4b", "keep_alive": 0},
        ),
    ]


def test_json_mode_payload_uses_compact_schema_shape() -> None:
    payload = OllamaClient._json_mode_payload(
        schema=HeadAnalysisPayload,
        model="fake-model",
        system_prompt="Return JSON.",
        user_prompt="Analyze this head.",
    )

    content = payload["messages"][1]["content"]

    assert "head_entity" in content
    assert "triples" in content
    assert "$defs" not in content
    assert "model_json_schema" not in content


def test_document_page_provenance_is_absent_from_model_schema_and_retry_shape() -> None:
    schema = DocumentPagePayload.model_json_schema()
    compact = OllamaClient._compact_schema_shape(DocumentPagePayload)

    assert "analysis_source" not in schema["properties"]
    assert "analysis_source" not in compact


def test_vision_json_retry_accepts_sections_without_provenance(monkeypatch) -> None:
    client = OllamaClient()
    calls: list[dict] = []

    def fake_post_chat(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"message": {"content": ""}}
        return {
            "message": {
                "content": '{"page_label":"1","sections":["Recovered section"]}'
            }
        }

    monkeypatch.setattr(client, "_post_chat", fake_post_chat)

    parsed = client.generate_structured_with_images(
        DocumentPagePayload,
        system_prompt="Return page JSON.",
        user_prompt="Analyze page.",
        images=[b"page"],
        model="fake-model",
    )

    assert parsed.sections == ["Recovered section"]
    assert "analysis_source" not in calls[1]["messages"][1]["content"]


def test_contextualization_client_uses_dedicated_transport_and_strict_schema(monkeypatch) -> None:
    from app.services import ai
    from app.services.contextualization import ContextualPrefixBatch

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {
                    "content": '{"items":[{"child_id":"c1","prefix":"该部分说明方法关系。"}]}'
                }
            }

    class FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            captured["timeout"] = kwargs["timeout"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url: str, json: dict) -> FakeResponse:
            captured.update({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    client = ContextualizationOllamaClient(
        base_url="http://context-only:11500/",
        model="context-model",
        timeout=37,
        prompt_version="context-v9",
        keep_alive="0",
    )

    result = client.generate_contextualization(
        ContextualPrefixBatch,
        system_prompt="contextualize",
        user_prompt="batch",
    )

    assert result.items[0].child_id == "c1"
    assert client.prompt_version == "context-v9"
    assert captured["timeout"] == 37
    assert captured["url"] == "http://context-only:11500/api/chat"
    assert captured["json"]["model"] == "context-model"
    assert captured["json"]["stream"] is False
    assert captured["json"]["think"] is False
    assert captured["json"]["keep_alive"] == 0
    assert captured["json"]["format"] == ContextualPrefixBatch.model_json_schema()


def test_contextualization_client_can_explicitly_disable_global_keep_alive(monkeypatch) -> None:
    from app.services import ai

    monkeypatch.setattr(ai.settings, "ollama_keep_alive", "5m", raising=False)
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": '{"items":[]}'}}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url: str, json: dict) -> FakeResponse:
            captured.update(json)
            return FakeResponse()

    monkeypatch.setattr(ai.httpx, "Client", FakeHttpClient)
    client = ContextualizationOllamaClient(keep_alive=None)
    from app.services.contextualization import ContextualPrefixBatch

    client.generate_contextualization(
        ContextualPrefixBatch,
        system_prompt="contextualize",
        user_prompt="batch",
    )

    assert "keep_alive" not in captured
