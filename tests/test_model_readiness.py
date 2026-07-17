from __future__ import annotations

from app.core.config import Settings
from app.services.model_readiness import ModelReadiness


class FakeResponse:
    def __init__(self, models):
        self._models = models

    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"model": model} for model in self._models]}


def test_readiness_requires_available_and_prewarmed_models(monkeypatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            calls.append(url)
            if "11435" in url and url.endswith("/api/tags"):
                return FakeResponse(["qwen3:14b", "qwen3-embedding:8b"])
            if "11435" in url:
                return FakeResponse(["qwen3:14b"])
            return FakeResponse(["qwen3.6:27b"])

    monkeypatch.setattr("app.services.model_readiness.httpx.Client", FakeClient)
    settings = Settings(
        OLLAMA_FAST_BASE_URL="http://127.0.0.1:11435",
        OLLAMA_DEEP_BASE_URL="http://127.0.0.1:11436",
    )
    readiness = ModelReadiness(settings, cache_seconds=30)

    first = readiness.check()
    second = readiness.check()

    assert first["status"] == "degraded"
    assert first["models"]["fast"]["status"] == "ready"
    assert first["models"]["deep"]["status"] == "ready"
    assert first["models"]["embedding"]["status"] == "not_loaded"
    assert second == first
    assert len(calls) == 4
