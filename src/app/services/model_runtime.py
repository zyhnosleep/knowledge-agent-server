from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from itertools import count

from app.core.config import get_settings


class ModelRequestCancelled(RuntimeError):
    """Raised when a queued generation request is cancelled before acquisition."""


class ModelRuntime:
    """Process-wide, per-profile FIFO leases for bounded model generation."""

    def __init__(self, capacities: Mapping[str, int]) -> None:
        if not capacities or any(capacity <= 0 for capacity in capacities.values()):
            raise ValueError("Model capacities must be positive")
        self._capacities = dict(capacities)
        self._active = {profile: 0 for profile in capacities}
        self._queues = {profile: deque() for profile in capacities}
        self._condition = threading.Condition()
        self._tickets = count()

    @contextmanager
    def acquire(
        self,
        profile: str,
        *,
        on_queue: Callable[[int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[None]:
        """Acquire one profile slot in FIFO order and always release it."""
        if profile not in self._capacities:
            raise ValueError(f"Unknown model profile: {profile}")
        if cancel_event is not None and cancel_event.is_set():
            raise ModelRequestCancelled(f"{profile} model request was cancelled")

        ticket = next(self._tickets)
        acquired = False
        last_position: int | None = None
        with self._condition:
            queue = self._queues[profile]
            queue.append(ticket)
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise ModelRequestCancelled(
                            f"{profile} model request was cancelled"
                        )
                    is_first = bool(queue) and queue[0] == ticket
                    has_capacity = (
                        self._active[profile] < self._capacities[profile]
                    )
                    if is_first and has_capacity:
                        queue.popleft()
                        self._active[profile] += 1
                        acquired = True
                        break

                    position = list(queue).index(ticket) + 1
                    if on_queue is not None and position != last_position:
                        on_queue(position)
                        last_position = position
                    self._condition.wait(timeout=0.1)
            except BaseException:
                if not acquired and ticket in queue:
                    queue.remove(ticket)
                    self._condition.notify_all()
                raise

        try:
            yield
        finally:
            if acquired:
                with self._condition:
                    self._active[profile] -= 1
                    self._condition.notify_all()

    def wake_waiters(self) -> None:
        """Wake queued requests so they can observe cancellation promptly."""
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return an immutable-by-copy view for health and diagnostics."""
        with self._condition:
            return {
                profile: {
                    "capacity": self._capacities[profile],
                    "active": self._active[profile],
                    "queued": len(self._queues[profile]),
                }
                for profile in self._capacities
            }


@lru_cache(maxsize=1)
def get_model_runtime() -> ModelRuntime:
    settings = get_settings()
    return ModelRuntime(
        {
            "fast": settings.ollama_fast_parallelism,
            "deep": settings.ollama_deep_parallelism,
        }
    )
