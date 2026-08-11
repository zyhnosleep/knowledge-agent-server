"""后台 worker 任务定义层（RQ worker 实际执行的入口函数）。

本模块是「异步队列任务」与「业务服务」之间的桥梁：RQ（Redis Queue）队列里
保存的是函数引用字符串（如 ``"app.workers.jobs.run_ingestion_stage"``），
worker 进程取出任务后通过 importlib 动态导入并调用本模块中的函数真正执行。

本模块暴露两个任务入口：

- ``run_ingestion_stage``：在 worker 中执行单个、按文档解析版本（version）
  隔离的 ingestion 阶段（parse / repair / canonicalize / semantic_split /
  contextualize / embed / index / activate），基于 checkpoint 支持断点续跑，
  是分阶段流水线任务的最小执行单元。
- ``run_document_ingestion``：在未配置 Redis（直连模式）下，对整篇文档同步
  执行完整的 ingestion 流水线（``IngestionPipeline.process_document``），
  返回一条 ``PipelineRun`` 记录。

设计约定：
- 两个函数都自行创建并负责关闭数据库 session（而非依赖上层传入），因此可以
  被 RQ worker 在独立进程中以纯函数的方式安全调用。
- 任务失败时异常向上抛出，由 RQ 负责把任务标记为 failed 并在重试机制下重跑，
  本模块不吞掉任何异常。
"""

from __future__ import annotations

from app.db.session import SessionLocal
from app.services.ingestion_stages import IngestionStageRunner
from app.services.pipeline import IngestionPipeline
from app.services.queue import JobDispatcher


def run_ingestion_stage(document_id: str, version_key: str, stage: str) -> str:
    """执行单个 ingestion 阶段（worker 内的最小任务单元）。

    在一个独立的数据库 session 中，把「分阶段 ingestion 的执行器」装配起来，
    运行指定的一个阶段，并返回该阶段执行后解析版本的 id。

    Args:
        document_id: 目标文档的主键 id。
        version_key: 文档的解析版本标识（同一文档可有多个版本，版本间相互隔离）。
        stage: 要执行的 ingestion 阶段名（须在 ``INGESTION_STAGES`` 白名单内）。

    Returns:
        执行完该阶段后的 ``DocumentParseVersion.id``（字符串）。

    Raises:
        ValueError: 文档或解析版本不存在，或阶段不在白名单内。
        StageAlreadyClaimed: 该阶段已被其他 worker 认领且 lease 尚未过期。
        StageHandlerUnavailable: 当前环境未配置该阶段的生产 handler。

    说明：
    - 用 try/finally 保证 session 一定被关闭，避免数据库连接泄漏。
    - 成功时 ``run_stage`` 内部已完成 commit；失败时异常沿调用栈抛出，
      由 RQ 记录失败状态，后续按重试策略重跑本任务。
    - 阶段处理器（handlers）取自 ``IngestionPipeline.ingestion_stage_handlers()``，
      即生产环境基于规范（canonical）产物的实现。
    """
    # 每个任务独立开启一个数据库 session，任务生命周期内独占使用
    db = SessionLocal()
    try:
        # 装配流水线对象：持有文档/版本操作所需的服务与存储
        pipeline = IngestionPipeline(db)
        # IngestionStageRunner 负责阶段执行的完整生命周期：
        # 加行锁加载文档/版本、前置校验、写 running checkpoint、调用 handler、
        # 写 completed/failed checkpoint，并在完成后投递下一阶段任务
        runner = IngestionStageRunner(
            db,
            # 各阶段的生产 handler（parse/repair/canonicalize/.../activate）
            handlers=pipeline.ingestion_stage_handlers(),
            # 阶段间衔接：当前阶段完成后，向下一阶段对应的队列投递任务
            dispatcher=JobDispatcher(),
            # 阶段前置校验：校验该解析版本的 manifest 与当前 ingestion 配置一致，
            # 防止配置漂移后仍在旧版本上继续处理
            pre_stage_validator=pipeline.validate_ingestion_identity,
        )
        # 执行指定阶段；内部若发现已完成会直接返回（幂等，可安全重试）
        version = runner.run_stage(document_id, version_key, stage)
        return version.id
    finally:
        # 无论成功失败，任务结束时都关闭本次 session
        db.close()


def run_document_ingestion(document_id: str) -> str:
    """对一篇文档执行完整的 ingestion 流水线（同步/直连路径）。

    当系统未配置 Redis（``settings.redis_url`` 为空）时，文档 ingestion 不走
    RQ 队列，而是由调用方（例如 API 请求线程）直接同步执行完整流水线；
    该场景下队列模块中的 ``enqueue_or_run`` / ``enqueue_stage`` 也会回退到
    直接调用本函数。

    Args:
        document_id: 目标文档的主键 id。

    Returns:
        本次处理产生的 ``PipelineRun.id``（字符串）。

    Raises:
        ValueError: 文档不存在时抛出。

    说明：
    - ``IngestionPipeline.process_document`` 会创建/复用一条 ``PipelineRun``
      记录，并将其推进到第一个可执行阶段（无 Redis 时为 legacy 同步路径）。
    - session 的生命周期同样由本函数管理（try/finally 关闭）。
    """
    # 每个任务独立开启一个数据库 session，任务生命周期内独占使用
    db = SessionLocal()
    try:
        # 装配流水线对象，处理完整文档 ingestion
        pipeline = IngestionPipeline(db)
        # 同步执行整条 ingestion 流水线，返回本次运行的记录
        run = pipeline.process_document(document_id)
        return run.id
    finally:
        # 无论成功失败，任务结束时都关闭本次 session
        db.close()
