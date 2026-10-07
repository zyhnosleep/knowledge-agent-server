"""模型运行时：按配置档位管理模型生成的并发与排队。

本模块提供进程级、按 profile（配置档位）隔离的并发控制——:class:`ModelRuntime`。
其目标是限制"生成模型"同时运行的请求数量不超过配置的并行度上限
（capacity），超出上限的请求进入 FIFO 队列等待，先到先得。

核心机制：
- **容量（capacity）**：每个 profile 允许同时运行的最大请求数。
- **活跃计数（active）**：当前正在执行的请求数。
- **FIFO 队列（queue）**：每个 profile 一个 deque，按到达顺序排票。
- **条件变量（Condition）**：保护上述状态，并用于在容量释放/取消发生时
  唤醒等待线程，避免忙等待。
- **票据（ticket）**：来自全局递增计数器的唯一序号，用于判定队首与队列位置。

与线程模型的关系：请求通常由并发 worker 线程发起；``acquire`` 是上下文管理器，
保证无论成功与否都能安全释放"占用"的槽位（lease）。通过 ``cancel_event``
支持协作式取消：排队中被取消的请求会立即从队列移除并抛出
:class:`ModelRequestCancelled`。

应用全局实例通过 :func:`get_model_runtime` 以单例形式提供，容量取自配置的
``ollama_generation_parallelism``。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from itertools import count

from app.core.config import get_settings
from app.services.execution_budget import BudgetExceeded, current_execution_budget


class ModelRequestCancelled(RuntimeError):
    """排队中的生成请求在获得槽位之前被取消时抛出。

    Raised when a queued generation request is cancelled before acquisition.
    """


_request_runtime: ContextVar[ModelRuntime | None] = ContextVar('request_model_runtime', default=None)


@contextmanager
def model_runtime_scope(runtime):
    """Trusted executor injection; no client-selected capacity or runtime."""
    token = _request_runtime.set(runtime)
    try:
        yield
    finally:
        _request_runtime.reset(token)


@contextmanager
def answer_generation_lease():
    runtime = _request_runtime.get() or get_model_runtime()
    budget = current_execution_budget()
    try:
        with runtime.acquire('generation', cancel_event=budget.cancel_event if budget else None,
                             deadline=budget.deadline if budget else None):
            yield
    except ModelRequestCancelled:
        if budget:
            raise BudgetExceeded('cancelled') from None
        raise


class ModelRuntime:
    """进程级、按 profile 的 FIFO 租约，用于有界（限流）的模型生成。

    Process-wide, per-profile FIFO leases for bounded model generation.

    每个 profile 拥有独立的容量与队列；一个请求要获得执行权必须同时满足
    "排到队首" 与 "有空闲容量" 两个条件。
    """

    def __init__(self, capacities: Mapping[str, int]) -> None:
        """初始化运行时。

        :param capacities: 每个 profile 的最大并发容量，如
            ``{"generation": 2}``。每个值必须为正整数。
        :raises ValueError: 容量为空或存在非正容量时。
        """
        if not capacities or any(capacity <= 0 for capacity in capacities.values()):
            raise ValueError("Model capacities must be positive")
        self._capacities = dict(capacities)
        # 各 profile 当前活跃（正在执行）的请求计数，初始为 0。
        self._active = {profile: 0 for profile in capacities}
        # 各 profile 的等待队列（FIFO），初始为空 deque。
        self._queues = {profile: deque() for profile in capacities}
        # 共享条件变量：保护 _active / _queues 的并发访问，
        # 并让等待线程在容量释放或被唤醒时立即重新检查条件。
        self._condition = threading.Condition()
        # 全局递增的票据生成器：为每个到达的请求分配唯一序号，保证 FIFO 顺序。
        self._tickets = count()

    @contextmanager
    def acquire(
        self,
        profile: str,
        *,
        on_queue: Callable[[int], None] | None = None,
        cancel_event: threading.Event | None = None,
        deadline: float | None = None,
    ) -> Iterator[None]:
        """按 FIFO 顺序获取一个 profile 槽位，并在退出时始终释放。

        Acquire one profile slot in FIFO order and always release it.

        作为上下文管理器使用：``with runtime.acquire("generation"): ...``。
        - 有可用容量且排到队首时立即获得槽位；
        - 否则阻塞等待，等待期间周期性检查取消事件；
        - 退出 with 块时（无论正常/异常）都会释放槽位；
        - 排队期间被取消（cancel_event 置位）会抛出
          :class:`ModelRequestCancelled`，并从队列移除自身。

        :param profile: 要获取的配置档位名（必须存在于 capacities）。
        :param on_queue: 可选的排队进度回调，参数为当前队列位置（从 1 开始）；
            仅在位置发生变化时调用，用于向调用方上报等待进度。
        :param cancel_event: 可选的取消事件；置位后排队中的请求立即放弃等待。
        :raises ValueError: profile 未知时。
        :raises ModelRequestCancelled: 获取前请求已被取消时。
        """
        if profile not in self._capacities:
            raise ValueError(f"Unknown model profile: {profile}")
        budget = current_execution_budget()
        if budget:
            budget.check_deadline()
            deadline = budget.deadline if deadline is None else min(deadline, budget.deadline)
        # 进入前先做一次取消检查，避免已经取消的请求还去排队。
        if cancel_event is not None and cancel_event.is_set():
            raise ModelRequestCancelled(f"{profile} model request was cancelled")

        # 领取唯一票据：按此序号判断队列顺序（先进先出）。
        ticket = next(self._tickets)
        acquired = False
        last_position: int | None = None
        # 以下操作全部在条件变量保护下进行，保证与其它线程互斥。
        with self._condition:
            queue = self._queues[profile]
            queue.append(ticket)
            try:
                while True:
                    if budget:
                        budget.check_deadline()
                    clock = budget.clock if budget else time.monotonic
                    if deadline is not None and clock() >= deadline:
                        raise BudgetExceeded('deadline')
                    # 每次被唤醒都重新检查取消事件（支持协作式取消）。
                    if cancel_event is not None and cancel_event.is_set():
                        raise ModelRequestCancelled(
                            f"{profile} model request was cancelled"
                        )
                    # 获得槽位的条件：① 自己是队首；② 还有空闲容量。
                    is_first = bool(queue) and queue[0] == ticket
                    has_capacity = (
                        self._active[profile] < self._capacities[profile]
                    )
                    if is_first and has_capacity:
                        # 出队并占用一个活跃槽位。
                        queue.popleft()
                        self._active[profile] += 1
                        acquired = True
                        break

                    # 尚未获得：计算自己的排队位置（列表下标 + 1）。
                    position = list(queue).index(ticket) + 1
                    # 位置变化时才回调 on_queue，避免高频重复通知。
                    if on_queue is not None and position != last_position:
                        on_queue(position)
                        last_position = position
                    # 条件等待：带 0.1s 超时，以便周期醒来检查取消事件。
                    self._condition.wait(timeout=min(0.1, max(0.0, deadline - clock())) if deadline is not None else 0.1)
            except BaseException:
                # 任一异常（如取消）路径：若尚未获得槽位且自己仍在队列中，
                # 把自己从队列移除并广播唤醒，避免残留请求堵住队列。
                if not acquired and ticket in queue:
                    queue.remove(ticket)
                    self._condition.notify_all()
                raise

        # 已获得槽位：进入用户的 with 代码块。
        try:
            yield
        finally:
            # 无论正常退出还是抛异常，都释放槽位并唤醒等待者，
            # 让下一位排队者有机会获得执行权。
            if acquired:
                with self._condition:
                    self._active[profile] -= 1
                    self._condition.notify_all()

    def wake_waiters(self) -> None:
        """唤醒所有排队请求，使其能立即观察到取消状态。

        Wake queued requests so they can observe cancellation promptly.

        用于外部主动取消大批请求时，减少最长 0.1s 的取消感知延迟。
        """
        with self._condition:
            self._condition.notify_all()

    def snapshot(self) -> dict[str, dict[str, int]]:
        """返回按值拷贝的运行时快照，供健康检查与诊断使用。

        Return an immutable-by-copy view for health and diagnostics.

        :return: ``{profile: {"capacity": int, "active": int, "queued": int}}``。
        """
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
    """返回进程级单例 :class:`ModelRuntime`（lru_cache 缓存，仅一份）。

    容量配置来自应用配置的 ``ollama_generation_parallelism``，即生成模型
    允许的最大并发请求数。
    """
    settings = get_settings()
    return ModelRuntime(
        {"generation": settings.ollama_generation_parallelism}
    )
