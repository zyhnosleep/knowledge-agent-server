from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings, get_settings

@dataclass(frozen=True)
class InferenceTarget:
    profile: Literal["none", "generation"]
    base_url: str
    model: str
    context_length: int
    reason: str


class AgentModelRouter:
    """Select the configured generation model without another model call."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def select(self, route: str) -> InferenceTarget:
        if route == "needs_clarification":
            return InferenceTarget(
                profile="none",
                base_url="",
                model="",
                context_length=0,
                reason="needs_clarification does not require generation",
            )

        return InferenceTarget(
            profile="generation",
            base_url=self._settings.ollama_generation_base_url.rstrip("/"),
            model=self._settings.ollama_generation_model,
            context_length=self._settings.ollama_generation_context_length,
            reason=f"Single generation target for route: {route}",
        )
