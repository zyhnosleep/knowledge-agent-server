"""
queue.py —— 任务队列（RQ / Redis）模块
=====================================

职责：
- 基于 Redis + RQ（Redis Queue）封装任务队列：创建连接、分阶段队列、
  任务入队、任务去重与 worker 创建。
- 是 ingestion 管线异步化的基础设施：管线调用方通过
  ``JobDispatcher.enqueue_stage`` 把某个 ingestion 阶段作为独立 RQ 任务
  提交，由 worker 进程异步执行，从而支持阶段级重试与水平扩展。

设计说明：
- 队列拓扑：
  - 默认队列 ``"ingest"``：保留的旧版/通用 ingest 队列。
  - 阶段队列 ``STAGE_QUEUES``：每个 ingestion 阶段一个专用队列
    （如 ``ingestion_stages`` 中定义的 stage -> queue 映射）。
- 当 Redis 未配置（``redis_url`` 为空）时，整个模块退化为"同步直调"
  模式：任务不经过队列，直接在调用进程内执行。这便于本地开发与测试。
- ``enqueue_stage`` 通过确定性 job_id（``ingestion-{doc}-{ver}-{stage}``）
  实现阶段任务去重；同一 (文档, 版本, 阶段) 组合不会产生重复任务。
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from redis import Redis
from rq import Queue, Worker
from rq.exceptions import DuplicateJobError

from app.core.config import get_settings
from app.services.ingestion_stages import INGESTION_STAGES, STAGE_QUEUES

# 模块级加载全局配置单例
settings = get_settings()


def redis_connection() -> Redis | None:
    """创建 Redis 连接；未配置时返回 None。

    说明：
    - 读取 ``settings.redis_url``；为空表示队列功能未启用，
      调用方应退化为同步执行模式。
    - 使用 ``Redis.from_url`` 解析连接串（含认证、端口等信息）。
    """
    if not settings.redis_url:
        return None
    return Redis.from_url(settings.redis_url)


class JobDispatcher:
    """任务分发器：负责把可调用对象或 ingestion 阶段入队到 RQ。

    封装了默认队列与各阶段队列的初始化，以及两种入队方式：
    - ``enqueue_or_run``：通用入队（未启用 Redis 时直接同步执行）。
    - ``enqueue_stage``：阶段专用入队，带幂等去重与失败恢复逻辑。
    """

    def __init__(self) -> None:
        """初始化队列连接与队列对象。

        1. 建立 Redis 连接（可能为 None）。
        2. 有连接时创建默认 ``"ingest"`` 队列，并为 ``STAGE_QUEUES``
           中的每个阶段创建同名专属队列（映射到 ``stage_queues``）。
        3. 无连接时 ``queue`` 为 None、``stage_queues`` 为空字典，
           后续入队操作会退化为同步直调。
        """
        self.connection = redis_connection()
        self.queue = Queue("ingest", connection=self.connection) if self.connection is not None else None
        # 为每个阶段创建专属队列
        self.stage_queues = (
            {
                stage: Queue(queue_name, connection=self.connection)
                for stage, queue_name in STAGE_QUEUES.items()
            }
            if self.connection is not None
            else {}
        )

    def enqueue_or_run(self, func: Callable[..., object] | str, *args, **kwargs):
        """入队并返回任务；Redis 未启用时直接同步调用。

        参数：
        - ``func``：可调用对象，或 ``"module.path.func"`` 形式的点路径
          字符串（此时会按需导入后再执行/入队）。
        - ``*args, **kwargs``：传给任务函数的参数。

        逻辑：
        - 若队列不可用（Redis 未配置）：解析字符串形式的函数并同步调用，
          返回其结果，保证调用方无需感知部署模式。
        - 若队列可用：``queue.enqueue`` 提交异步任务，并设置超时
          ``settings.queue_job_timeout``。
        """
        if self.queue is None:
            # 退化模式：解析点路径并同步执行
            if isinstance(func, str):
                module_name, attr_name = func.rsplit(".", 1)
                imported = getattr(importlib.import_module(module_name), attr_name)
                return imported(*args, **kwargs)
            return func(*args, **kwargs)
        # 正常模式：入队异步执行
        return self.queue.enqueue(func, *args, job_timeout=settings.queue_job_timeout, **kwargs)

    def enqueue_stage(self, document_id: str, version_key: str, stage: str):
        """把 ingestion 阶段任务入队（带幂等去重与失败恢复）。

        参数：
        - ``document_id``：文档 ID。
        - ``version_key``：文档版本键。
        - ``stage``：阶段名，必须是 ``STAGE_QUEUES`` 中已注册的阶段。

        实现要点：
        1. 未知阶段直接抛 ``ValueError``。
        2. 任务函数固定为 ``app.workers.jobs.run_ingestion_stage``；
           使用确定性 job_id 防止同一任务重复入队（``unique=True``）。
        3. 未启用 Redis 时同步调用该 worker 函数。
        4. 入队遇到 ``DuplicateJobError``（job_id 冲突）时：
           - 查询现有任务：
             - 正在排队/运行中（queued/started/deferred/scheduled）
               → 直接复用，返回已有任务。
             - 已失败（failed）→ ``requeue()`` 重新入队。
             - 已结束（finished/stopped/canceled）→ 删除旧任务后重试
               重新入队。
           - 最多尝试 3 次，仍无法解决则抛 ``RuntimeError``。
        """
        if stage not in STAGE_QUEUES:
            raise ValueError(f"Unknown ingestion stage {stage!r}.")
        # 阶段任务统一由 worker 层的 run_ingestion_stage 执行
        function = "app.workers.jobs.run_ingestion_stage"
        job_id = ingestion_stage_job_id(document_id, version_key, stage)
        queue = self.stage_queues.get(stage)
        if queue is None:
            # 退化模式：同步直调 worker 函数
            module_name, attr_name = function.rsplit(".", 1)
            imported = getattr(importlib.import_module(module_name), attr_name)
            return imported(document_id, version_key, stage)
        # 入队参数：unique=True 与确定性 job_id 共同保证幂等
        enqueue_kwargs = {
            "func": function,
            "args": (document_id, version_key, stage),
            "timeout": settings.queue_job_timeout,
            "job_id": job_id,
            "unique": True,
        }
        # 处理重复 job_id 冲突，最多尝试 3 次
        for attempt in range(3):
            try:
                return queue.enqueue_call(**enqueue_kwargs)
            except DuplicateJobError as exc:
                existing = queue.fetch_job(job_id)
                if existing is None:
                    # 冲突但查不到任务，可能是竞态；重试，最后仍失败则报错
                    if attempt < 2:
                        continue
                    raise RuntimeError(
                        f"Duplicate RQ job {job_id!r} could not be resolved."
                    ) from exc
                # 读取任务状态（兼容状态枚举与字符串两种形态）
                status = existing.get_status()
                status = getattr(status, "value", status)
                if status in {"queued", "started", "deferred", "scheduled"}:
                    # 任务已在队列中或正在执行：直接复用
                    return existing
                if status == "failed":
                    # 上次失败：重新入队
                    existing.requeue()
                    return existing
                if status in {"finished", "stopped", "canceled"}:
                    # 已结束/被取消：清除旧任务后重试入队
                    existing.delete(remove_from_queue=True)
                    if attempt < 2:
                        continue
                if attempt == 2:
                    raise RuntimeError(
                        f"Duplicate RQ job {job_id!r} has unresolved status {status!r}."
                    ) from exc
        raise RuntimeError(f"Duplicate RQ job {job_id!r} could not be resolved.")


def ingestion_stage_job_id(document_id: str, version_key: str, stage: str) -> str:
    """生成 ingestion 阶段任务的确定性 job_id。

    规则：``ingestion-{document_id}-{version_key}-{stage}``。

    确定性意味着：同一 (文档, 版本, 阶段) 组合在任何进程/时刻生成
    的 job_id 都一致，从而配合 RQ 的 ``unique`` 参数实现全局去重。

    参数：
    - ``document_id``：文档 ID。
    - ``version_key``：文档版本键。
    - ``stage``：阶段名（须已注册，否则抛 ``ValueError``）。
    """
    if stage not in STAGE_QUEUES:
        raise ValueError(f"Unknown ingestion stage {stage!r}.")
    return f"ingestion-{document_id}-{version_key}-{stage}"


def create_worker() -> Worker:
    """创建并配置 ingestion 专用的 RQ Worker。

    流程与约束：
    1. 队列列表：按 ``INGESTION_STAGES`` 顺序生成各阶段队列名，
       末尾追加旧版 ``"ingest"`` 队列。
    2. 若设置了 ``INGESTION_WORKER_QUEUES`` 环境变量，则要求其精确等于
       上述队列列表（顺序一致），否则报错——用于防止 worker 配置漂移。
    3. ``INGESTION_WORKER_CONCURRENCY`` 必须为 ``"1"``：ingestion 阶段
       有共享资源/顺序要求，禁止多并发 worker。
    4. 必须配置 ``REDIS_URL``，否则无法创建 worker。

    返回：配置完成的 RQ ``Worker`` 实例。
    """
    # 队列顺序：所有阶段队列 + 旧版 ingest 队列
    queues = [*(STAGE_QUEUES[stage] for stage in INGESTION_STAGES), "ingest"]
    # 可选的环境变量强约束：校验队列列表完全一致
    configured_queues = os.getenv("INGESTION_WORKER_QUEUES")
    if configured_queues is not None:
        parsed_queues = [
            value.strip() for value in configured_queues.split(",") if value.strip()
        ]
        if parsed_queues != queues:
            raise RuntimeError(
                "INGESTION_WORKER_QUEUES must list all stage queues in order "
                "followed by the legacy ingest queue."
            )
    # 强制单并发：ingestion 阶段共享状态，不允许并发执行
    concurrency = os.getenv("INGESTION_WORKER_CONCURRENCY", "1")
    if concurrency != "1":
        raise RuntimeError("INGESTION_WORKER_CONCURRENCY must be 1.")
    connection = redis_connection()
    if connection is None:
        raise RuntimeError("REDIS_URL is not configured.")
    return Worker(queues, connection=connection)
