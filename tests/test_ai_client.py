from app.services import ai
from app.services.ai import HeadAnalysisPayload, OllamaClient


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
    assert calls == [{"model": "fake-embedding", "input": ["alpha", "beta"], "keep_alive": 0}]


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
