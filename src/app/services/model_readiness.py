from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import Settings, get_settings


class ModelReadiness:
    """Short-lived, non-loading readiness checks for configured Ollama models."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache_seconds: float = 5.0,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache_seconds = cache_seconds
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None

    def check(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._cached is not None and now - self._cached_at < self._cache_seconds:
                return self._cached
            result = self._probe()
            self._cached = result
            self._cached_at = now
            return result

    def _probe(self) -> dict[str, Any]:
        runtime = self._probe_endpoint(self._settings.ollama_generation_base_url)
        profiles = {
            "generation": self._model_status(
                runtime,
                self._settings.ollama_generation_model,
                context_length=self._settings.ollama_generation_context_length,
            ),
            "embedding": self._model_status(
                runtime,
                self._settings.ollama_embedding_model,
                dimensions=self._settings.ollama_embedding_dimensions,
            ),
        }
        healthy_statuses = {"ready", "idle"}
        overall = (
            "ok"
            if all(item["status"] in healthy_statuses for item in profiles.values())
            else "degraded"
        )
        return {"status": overall, "models": profiles}

    def _probe_endpoint(self, base_url: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                tags_response = client.get(f"{base_url.rstrip('/')}/api/tags")
                tags_response.raise_for_status()
                ps_response = client.get(f"{base_url.rstrip('/')}/api/ps")
                ps_response.raise_for_status()
            return {
                "available": self._model_names(tags_response.json()),
                "loaded": self._model_names(ps_response.json()),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"available": set(), "loaded": set(), "error": str(exc)}

    @staticmethod
    def _model_names(payload: dict[str, Any]) -> set[str]:
        return {
            str(item.get("model") or item.get("name"))
            for item in payload.get("models", [])
            if item.get("model") or item.get("name")
        }

    @staticmethod
    def _model_status(
        probe: dict[str, Any], model: str, **metadata: int
    ) -> dict[str, Any]:
        if probe["error"]:
            status = "unreachable"
        elif model not in probe["available"]:
            status = "missing"
        elif model not in probe["loaded"]:
            status = "idle"
        else:
            status = "ready"
        result: dict[str, Any] = {"status": status, "model": model, **metadata}
        if probe["error"]:
            result["error"] = probe["error"]
        return result


@lru_cache(maxsize=1)
def get_model_readiness() -> ModelReadiness:
    return ModelReadiness()
