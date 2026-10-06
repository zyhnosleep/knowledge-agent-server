"""Request-local admission/accounting, not a general task scheduler.

Text uses a conservative UTF-8 byte ceiling (no tokenizer dependency).
Unknown image processors reserve the bounded input context; only a caller
with verified processor limits may provide a smaller image token ceiling.
Unreported usage retains the reservation: it is not exact billing.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from app.schemas.agent import AgentConstraints


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f'Execution budget exceeded: {reason}')


@dataclass
class ModelReservation:
    request_id: int
    input_upper_bound: int
    max_output_tokens: int
    timeout_seconds: float
    settled: bool = False


class ExecutionBudget:
    def __init__(self, constraints: AgentConstraints, *, max_answer_retries: int = 0,
                 clock: Callable[[], float] = time.monotonic,
                 cancel_event: threading.Event | None = None):
        self.constraints = constraints
        self.clock = clock
        self.cancel_event = cancel_event
        self.deadline = clock() + constraints.timeout_seconds
        self.max_answer_retries = max(0, max_answer_retries)
        self.answer_retries = 0
        self.format_retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_requests = 0
        self._reservations: dict[int, ModelReservation] = {}
        self._sources: dict[int, str] = {}

    def set_answer_retry_limit(self, max_retries: int) -> None:
        self.max_answer_retries = max(0, max_retries)

    def check_deadline(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise BudgetExceeded('cancelled')
        if self.clock() >= self.deadline:
            raise BudgetExceeded('deadline')

    def http_timeout(self, configured_timeout: float) -> float:
        self.check_deadline()
        return min(float(configured_timeout), self.deadline - self.clock())

    def tool_block_reason(self, *, step_count: int, tool_calls: int) -> str | None:
        self.check_deadline()
        if step_count >= self.constraints.max_steps:
            return 'step_limit'
        if tool_calls >= self.constraints.max_tool_calls:
            return 'tool_limit'
        return None

    def consume_answer_retry(self, *, empty_answer: bool = False) -> bool:
        self.check_deadline()
        limit = max(self.max_answer_retries, 1 if empty_answer else 0)
        if self.answer_retries >= limit:
            return False
        self.answer_retries += 1
        return True

    def consume_format_retry(self) -> bool:
        self.check_deadline()
        if self.format_retries:
            return False
        self.format_retries += 1
        return True

    def reserve_model_call(self, *, messages: list[dict[str, Any]], model: str,
                           max_output_tokens: int, context_length: int,
                           image_token_upper_bound: int | None = None) -> ModelReservation:
        self.check_deadline()
        text_bytes = 0
        image_count = 0
        for message in messages:
            image_count += len(message.get('images') or [])
            content = message.get('content', '')
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') in ('image_url', 'input_image'):
                        image_count += 1
                    else:
                        text_parts.append(part)
                content = json.dumps(text_parts, ensure_ascii=False)
            text_bytes += len(str(content).encode('utf-8')) + len(str(message.get('role', '')).encode()) + 16
        input_bound = text_bytes
        if image_count:
            input_bound = max(text_bytes, context_length) if image_token_upper_bound is None else text_bytes + image_token_upper_bound
        elif input_bound >= context_length:
            raise BudgetExceeded('context_tokens')
        remaining = self.constraints.budget_tokens - self.prompt_tokens - self.completion_tokens - input_bound
        output_limit = min(max_output_tokens, remaining)
        if not image_count or image_token_upper_bound is not None:
            output_limit = min(output_limit, context_length - input_bound)
        if output_limit <= 0:
            raise BudgetExceeded('token_limit')
        self.model_requests += 1
        reservation = ModelReservation(self.model_requests, input_bound, output_limit,
                                       self.deadline - self.clock())
        self._reservations[reservation.request_id] = reservation
        self.prompt_tokens += input_bound
        self.completion_tokens += output_limit
        self._sources[reservation.request_id] = 'estimated'
        return reservation

    def finish_model_call(self, reservation: ModelReservation, *, prompt_tokens: int | None,
                          completion_tokens: int | None) -> None:
        if self._reservations.get(reservation.request_id) is not reservation:
            raise ValueError('Reservation does not belong to this budget')
        if reservation.settled:
            raise ValueError('Reservation already settled')
        reservation.settled = True
        if all(type(value) is int and value >= 0 for value in (prompt_tokens, completion_tokens)):
            self.prompt_tokens += prompt_tokens - reservation.input_upper_bound
            self.completion_tokens += completion_tokens - reservation.max_output_tokens
            self._sources[reservation.request_id] = 'provider'

    def usage_metadata(self) -> dict[str, Any]:
        sources = set(self._sources.values())
        source = next(iter(sources)) if len(sources) == 1 else 'mixed' if sources else 'none'
        return {
            'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens,
            'model_requests': self.model_requests, 'answer_retries': self.answer_retries,
            'format_retries': self.format_retries, 'usage_source': source,
            'budget_tokens': self.constraints.budget_tokens,
            'remaining_tokens': max(0, self.constraints.budget_tokens - self.prompt_tokens - self.completion_tokens),
            'remaining_seconds': max(0.0, self.deadline - self.clock()),
            'unsettled_requests': sum(not r.settled for r in self._reservations.values()),
        }


_current: ContextVar[ExecutionBudget | None] = ContextVar('execution_budget', default=None)


def current_execution_budget() -> ExecutionBudget | None:
    return _current.get()


@contextmanager
def execution_budget_scope(budget: ExecutionBudget) -> Iterator[None]:
    token = _current.set(budget)
    try:
        yield
    finally:
        _current.reset(token)
