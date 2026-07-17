from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings, get_settings

AnswerMode = Literal["auto", "fast", "deep"]


@dataclass(frozen=True)
class InferenceTarget:
    profile: Literal["none", "fast", "deep"]
    base_url: str
    model: str
    context_length: int
    reason: str


class AgentModelRouter:
    """Select one configured generation profile without another model call."""

    DEEP_ROUTES = frozenset(
        {"table_or_metric", "multi_source_compare", "complex_multi_hop"}
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def select(self, answer_mode: AnswerMode, route: str) -> InferenceTarget:
        if route == "needs_clarification":
            return InferenceTarget(
                profile="none",
                base_url="",
                model="",
                context_length=0,
                reason="needs_clarification does not require generation",
            )

        if answer_mode == "fast":
            return self._fast("Manual fast override")
        if answer_mode == "deep":
            return self._deep("Manual deep override")
        if answer_mode != "auto":
            raise ValueError(f"Unsupported answer mode: {answer_mode}")

        if route in self.DEEP_ROUTES:
            return self._deep(f"Automatic deep route: {route}")
        return self._fast(f"Automatic fast route: {route}")

    def _fast(self, reason: str) -> InferenceTarget:
        return InferenceTarget(
            profile="fast",
            base_url=self._settings.ollama_fast_base_url.rstrip("/"),
            model=self._settings.ollama_fast_model,
            context_length=self._settings.ollama_fast_context_length,
            reason=reason,
        )

    def _deep(self, reason: str) -> InferenceTarget:
        return InferenceTarget(
            profile="deep",
            base_url=self._settings.ollama_deep_base_url.rstrip("/"),
            model=self._settings.ollama_deep_model,
            context_length=self._settings.ollama_deep_context_length,
            reason=reason,
        )
