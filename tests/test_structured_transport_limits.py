import json

import httpx
import pytest

from app.services import ai
from app.services.ai import ContextualizationOllamaClient, DocumentPagePayload, OllamaClient


@pytest.mark.parametrize("mode", ["text", "image", "contextualization", "retry"])
def test_nonrequest_structured_calls_send_configured_output_limit(monkeypatch, mode):
    # The model service otherwise defaults to 512 tokens, truncating complete
    # table JSON during ingestion (which has no request ExecutionBudget).
    monkeypatch.setattr(ai.settings, "generation_max_output_tokens", 2048)
    monkeypatch.setattr(ai.settings, "ollama_generation_context_length", 32768)
    calls = []
    real_client = httpx.Client

    def serve(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if mode == "retry" and len(calls) == 1:
            content = "not JSON"
        elif payload.get("options", {}).get("num_predict", 512) < 2048:
            content = '{"page_label":"5","tables":['
        else:
            content = json.dumps({"page_label": "5", "tables": ["| Method | EM |\n| --- | --- |\n| ReAct | 27.4 |"]})
        return httpx.Response(200, json={"message": {"content": content}, "done": True})

    monkeypatch.setattr(ai.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs))
    client = ContextualizationOllamaClient() if mode == "contextualization" else OllamaClient()
    kwargs = {"system_prompt": "extract", "user_prompt": "page"}
    if mode in {"image", "retry"}:
        result = client.generate_structured_with_images(DocumentPagePayload, images=[b"image"], **kwargs)
    elif mode == "contextualization":
        result = client.generate_contextualization(DocumentPagePayload, **kwargs)
    else:
        result = client.generate_structured(DocumentPagePayload, **kwargs)
    assert result.tables == ["| Method | EM |\n| --- | --- |\n| ReAct | 27.4 |"]
    assert len(calls) == (2 if mode == "retry" else 1)
    assert all(p["options"]["num_predict"] == 2048 and p["options"]["num_ctx"] == 32768 for p in calls)


def test_explicit_smaller_limits_are_not_expanded_without_request_budget(monkeypatch):
    monkeypatch.setattr(ai.settings, "generation_max_output_tokens", 2048)
    with ai._model_request({"options": {"num_predict": 64, "num_ctx": 1024, "temperature": 0}}, 90, ollama=True) as (payload, _, reservation):
        assert payload["options"] == {"num_predict": 64, "num_ctx": 1024, "temperature": 0}
        assert reservation is None
