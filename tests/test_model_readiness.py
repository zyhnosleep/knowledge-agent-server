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


def test_readiness_treats_available_unloaded_embedding_as_healthy_idle(monkeypatch) -> None:
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
            if url.endswith("/api/tags"):
                return FakeResponse(["qwen3.5:9b", "qwen3-embedding:4b"])
            return FakeResponse(["qwen3.5:9b"])

    monkeypatch.setattr("app.services.model_readiness.httpx.Client", FakeClient)
    settings = Settings(
        OLLAMA_GENERATION_BASE_URL="http://127.0.0.1:11435",
        OLLAMA_EMBEDDING_BASE_URL="http://127.0.0.1:11435",
    )
    readiness = ModelReadiness(settings, cache_seconds=30)

    first = readiness.check()
    second = readiness.check()

    assert first["status"] == "ok"
    assert set(first["models"]) == {"generation", "embedding"}
    assert first["models"]["generation"]["status"] == "ready"
    assert first["models"]["embedding"]["status"] == "idle"
    assert second == first
    assert len(calls) == 2


def test_readiness_still_degrades_when_a_configured_model_is_missing(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            if url.endswith("/api/tags"):
                return FakeResponse(["qwen3.5:9b"])
            return FakeResponse([])

    monkeypatch.setattr("app.services.model_readiness.httpx.Client", FakeClient)
    readiness = ModelReadiness(Settings(), cache_seconds=0)

    result = readiness.check()

    assert result["status"] == "degraded"
    assert result["models"]["embedding"]["status"] == "missing"


def test_readiness_does_not_call_remote_embedding_api(monkeypatch) -> None:
    """Remote Qwen embedding readiness is configuration-only (no quota use)."""
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
            if url.endswith("/api/tags"):
                return FakeResponse(["qwen3.5:9b"])
            return FakeResponse(["qwen3.5:9b"])

    monkeypatch.setattr("app.services.model_readiness.httpx.Client", FakeClient)
    settings = Settings(
        EMBEDDING_PROVIDER="openai-compatible",
        EMBEDDING_API_BASE_URL="https://embedding.example/v1",
        EMBEDDING_API_KEY="test-secret",
    )

    result = ModelReadiness(settings, cache_seconds=0).check()

    assert result["models"]["embedding"]["status"] == "configured"
    assert result["models"]["embedding"]["provider"] == "openai-compatible"
    assert result["models"]["embedding"]["model"] == "Qwen/Qwen3-Embedding-4B"
    assert result["models"]["embedding"]["dimensions"] == 2560
    assert result["models"]["embedding"]["probe"] == "configuration-only"
    assert all("embeddings" not in url for url in calls)


def test_readiness_marks_remote_embedding_missing_without_credentials(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            if url.endswith("/api/tags"):
                return FakeResponse(["qwen3.5:9b"])
            return FakeResponse(["qwen3.5:9b"])

    monkeypatch.setattr("app.services.model_readiness.httpx.Client", FakeClient)
    settings = Settings(
        EMBEDDING_PROVIDER="openai-compatible",
        EMBEDDING_API_BASE_URL="https://embedding.example/v1",
        EMBEDDING_API_KEY=None,
    )

    result = ModelReadiness(settings, cache_seconds=0).check()

    assert result["status"] == "degraded"
    assert result["models"]["embedding"]["status"] == "missing"


def test_readiness_skips_ollama_when_generation_and_embedding_are_remote(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs):
            raise AssertionError("remote-only readiness must not call Ollama")

    monkeypatch.setattr("app.services.model_readiness.httpx.Client", FailingClient)
    settings = Settings(
        GENERATION_PROVIDER="deepseek",
        DEEPSEEK_BASE_URL="https://api.deepseek.com/v1",
        DEEPSEEK_API_KEY="test-deepseek-key",
        EMBEDDING_PROVIDER="openai-compatible",
        EMBEDDING_API_BASE_URL="https://embedding.example/v1",
        EMBEDDING_API_KEY="test-embedding-key",
    )

    result = ModelReadiness(settings, cache_seconds=0).check()

    assert result["status"] == "ok"
    assert result["models"]["generation"]["status"] == "configured"
    assert result["models"]["generation"]["provider"] == "deepseek"
    assert result["models"]["embedding"]["status"] == "configured"
