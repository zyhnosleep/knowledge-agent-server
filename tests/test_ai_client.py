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
