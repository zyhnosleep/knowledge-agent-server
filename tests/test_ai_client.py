from app.services import ai
from app.services.ai import HeadAnalysisPayload, OllamaClient


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

    assert calls[0]["keep_alive"] == "0"
    assert calls[1]["keep_alive"] == "0"


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
    assert calls == [{"model": "fake-embedding", "input": ["alpha", "beta"], "keep_alive": "0"}]


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
