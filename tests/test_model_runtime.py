from __future__ import annotations

import threading
import time

import pytest

from app.services.model_runtime import (
    ModelRequestCancelled,
    ModelRuntime,
)


def test_fast_and_deep_capacities_are_independent() -> None:
    runtime = ModelRuntime({"fast": 1, "deep": 1})

    with runtime.acquire("fast"):
        with runtime.acquire("deep"):
            snapshot = runtime.snapshot()
            assert snapshot["fast"]["active"] == 1
            assert snapshot["deep"]["active"] == 1


def test_waiters_acquire_in_fifo_order_and_report_positions() -> None:
    runtime = ModelRuntime({"deep": 1})
    order: list[str] = []
    queued: dict[str, list[int]] = {"second": [], "third": []}
    started = threading.Event()

    def worker(name: str) -> None:
        started.set()
        with runtime.acquire(
            "deep", on_queue=lambda position: queued[name].append(position)
        ):
            order.append(name)
            time.sleep(0.02)

    with runtime.acquire("deep"):
        second = threading.Thread(target=worker, args=("second",))
        third = threading.Thread(target=worker, args=("third",))
        second.start()
        started.wait(timeout=1)
        time.sleep(0.02)
        third.start()
        time.sleep(0.05)

    second.join(timeout=2)
    third.join(timeout=2)
    assert order == ["second", "third"]
    assert queued["second"][0] == 1
    assert queued["third"][0] == 2


def test_cancelled_waiter_never_acquires() -> None:
    runtime = ModelRuntime({"deep": 1})
    cancel = threading.Event()
    acquired = threading.Event()
    cancelled: list[bool] = []

    def waiter() -> None:
        try:
            with runtime.acquire("deep", cancel_event=cancel):
                acquired.set()
        except ModelRequestCancelled:
            cancelled.append(True)

    with runtime.acquire("deep"):
        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.03)
        cancel.set()
        runtime.wake_waiters()

    thread.join(timeout=2)
    assert cancelled == [True]
    assert not acquired.is_set()


def test_lease_releases_after_body_exception() -> None:
    runtime = ModelRuntime({"fast": 1})

    with pytest.raises(RuntimeError):
        with runtime.acquire("fast"):
            raise RuntimeError("generation failed")

    with runtime.acquire("fast"):
        assert runtime.snapshot()["fast"]["active"] == 1


def test_unknown_profile_is_rejected() -> None:
    runtime = ModelRuntime({"fast": 1})

    with pytest.raises(ValueError, match="Unknown model profile"):
        with runtime.acquire("deep"):
            pass
