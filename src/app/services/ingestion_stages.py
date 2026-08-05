"""
ingestion_stages.py —— Ingestion 阶段编排器（阶段机 + 持久化检查点）模块
======================================================================

职责：
- 定义 ingestion 管线的 8 个阶段：parse -> repair -> canonicalize ->
  semantic_split -> contextualize -> embed -> index -> activate，
  以及每个阶段对应的 RQ 队列名（``STAGE_QUEUES``）。
- 提供 ``IngestionStageRunner``：以"版本作用域 + 可持久化、可恢复的
  检查点（checkpoint）"方式串行执行各阶段，并把阶段状态推进/失败回退
  映射到解析版本状态机（见 ``parse_versions``）。
- 通过 RQ 任务队列异步串接阶段（每个阶段独立入队，worker 单并发执行），
  支持阶段级重试、故障恢复与并发认领（claim）保护。

核心机制：
- **检查点（checkpoint）**：每个阶段的运行现场（input / output / 状态 /
  认领人 / 租约过期时间 / 尝试次数）以字典形式存于
  ``version.stage_state[stage]``，序列化前经 ``_json_safe`` 规范化，
  并受 ``max_checkpoint_bytes`` 上限约束。
- **认领（claim）**：某 worker 开始执行阶段时会写入
  ``claim_owner`` + ``lease_expires_at``；其他 worker 看到"running 且租约
  未过期"即认为已被认领（抛 ``StageAlreadyClaimed``），租约过期才允许
  接管——以此实现"同一阶段只有一个 worker 在执行"的保证。
- **顺序校验**：执行阶段前必须保证前面所有阶段都已 completed
  （``_validate_order``）；版本状态由 ``_prepare_status_for_attempt``
  推进到对应 running 状态。
- **失败恢复**：阶段失败后版本进入失败态，重试时经
  ``FAILED_STAGE_RETRIES`` 回退到重试阶段，再用 ``_status_path`` BFS
  沿合法转移边恢复到目标 running 状态。
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import (
    Document,
    DocumentParseVersion,
    PipelineRun,
    RunStatus,
)
from app.services.parse_versions import (
    ALLOWED_TRANSITIONS,
    FAILED_STAGE_RETRIES,
    ParseVersionService,
)


# 阶段执行顺序（activate 可选；run_until_blocked 默认排除它）
INGESTION_STAGES = (
    "parse",
    "repair",
    "canonicalize",
    "semantic_split",
    "contextualize",
    "embed",
    "index",
    "activate",
)

# 每个阶段对应的 RQ 队列名：ingest.<stage>
STAGE_QUEUES = {stage: f"ingest.{stage}" for stage in INGESTION_STAGES}

# 阶段错误信息入库的最大字节数（超出则截断并附哈希）
_MAX_STAGE_ERROR_BYTES = 4096

# 阶段完成时版本状态机到达的状态
_COMPLETED_STATUS = {
    "parse": "quality_checking",
    "repair": "canonicalizing",
    "canonicalize": "chunking",
    "semantic_split": "contextualizing",
    "contextualize": "embedding",
    "embed": "indexing",
    "index": "ready_to_activate",
    "activate": "active",
}
# 阶段开始执行（running）时版本状态机到达的状态
_RUNNING_STATUS = {
    "parse": "parsing",
    "repair": "repairing",
    "canonicalize": "canonicalizing",
    "semantic_split": "chunking",
    "contextualize": "contextualizing",
    "embed": "embedding",
    "index": "indexing",
    "activate": "ready_to_activate",
}
# 阶段失败时版本状态机到达的失败状态
_FAILED_STATUS = {
    "parse": "parse_failed",
    "repair": "table_repair_failed",
    "canonicalize": "parse_failed",
    "semantic_split": "parse_failed",
    "contextualize": "contextualization_failed",
    "embed": "embedding_failed",
    "index": "embedding_failed",
    "activate": "activation_failed",
}


class StageAlreadyClaimed(RuntimeError):
    """Raised when another worker has committed a claim for the same stage.

    当其他 worker 已提交了同一阶段的认领（claim）时抛出，
    防止重复执行同一阶段。
    """


class StageHandlerUnavailable(RuntimeError):
    """Raised instead of recording a stage as complete without real work.

    当某阶段没有配置实际执行器（handler）时抛出，而不是伪造完成。
    """


class CapabilityUnavailable(RuntimeError):
    """Raised when a stage depends on a capability that is not available yet.

    当阶段依赖的能力（如模型、服务）尚不可用时抛出。
    """


class StageCheckpointTooLarge(ValueError):
    """Raised before committing a checkpoint that exceeds the durable JSON cap.

    在提交超出"持久化 JSON 大小上限"的检查点之前抛出。
    """


@dataclass(frozen=True)
class IngestionStageContext:
    """阶段执行器收到的上下文（不可变数据类）。

    字段：
    - ``db``：数据库会话。
    - ``document``：当前文档。
    - ``version``：当前解析版本。
    - ``stage``：阶段名。
    - ``input``：该阶段的上游输入（含上一阶段输出）。
    - ``attempt``：当前是第几次尝试。
    """

    db: Session
    document: Document
    version: DocumentParseVersion
    stage: str
    input: dict[str, Any]
    attempt: int


# 阶段执行器类型：接收上下文并返回任意可 JSON 化结果
StageHandler = Callable[[IngestionStageContext], Any]
# 阶段前校验器类型：抛异常即阻止该阶段执行
PreStageValidator = Callable[[Document, DocumentParseVersion, str], None]


class IngestionStageRunner:
    """Run version-scoped ingestion stages with durable, resumable checkpoints.

    以"版本作用域 + 可持久化、可恢复检查点"方式运行 ingestion 阶段。
    负责：认领保护、顺序校验、状态机推进、检查点持久化、失败记录、
    下一阶段入队与批量激活。
    """

    def __init__(
        self,
        db: Session,
        *,
        handlers: Mapping[str, StageHandler] | None = None,
        pre_stage_validator: PreStageValidator | None = None,
        dispatcher: Any | None = None,
        clock: Callable[[], datetime] | None = None,
        claim_owner: str | None = None,
        claim_ttl_seconds: int | None = None,
        max_checkpoint_bytes: int = 256 * 1024,
    ) -> None:
        """初始化阶段运行器。

        参数：
        - ``db``：数据库会话。
        - ``handlers``：阶段名 -> 执行器 的映射；未知阶段名会报错。
        - ``pre_stage_validator``：可选的阶段前校验器（如检查能力是否
          就绪），抛异常则中止阶段。
        - ``dispatcher``：可选的 ``JobDispatcher``，用于把下一阶段入队。
        - ``clock``：可注入的时钟（便于测试）；默认 UTC 时间。
        - ``claim_owner``：本运行器的认领标识；默认随机 UUID。
        - ``claim_ttl_seconds``：认领租约 TTL；默认 = 队列任务超时 + 300 秒。
        - ``max_checkpoint_bytes``：检查点 JSON 大小上限，默认 256 KiB。
        """
        self.db = db
        self.handlers = dict(handlers or {})
        self.pre_stage_validator = pre_stage_validator
        # 校验：传入的 handler 名必须属于已知阶段
        unknown_handlers = set(self.handlers) - set(INGESTION_STAGES)
        if unknown_handlers:
            names = ", ".join(sorted(unknown_handlers))
            raise ValueError(f"Unknown ingestion stage handlers: {names}.")
        self.dispatcher = dispatcher
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.claim_owner = claim_owner or uuid4().hex
        if claim_ttl_seconds is None:
            # 默认租约 TTL 比单个任务超时长一些，允许任务超时后仍可接管
            claim_ttl_seconds = get_settings().queue_job_timeout + 300
        if claim_ttl_seconds <= 0:
            raise ValueError("claim_ttl_seconds must be positive")
        if max_checkpoint_bytes <= 0:
            raise ValueError("max_checkpoint_bytes must be positive")
        self.claim_ttl_seconds = claim_ttl_seconds
        self.max_checkpoint_bytes = max_checkpoint_bytes
        # 版本状态机服务
        self.versions = ParseVersionService(db)

    def run_until_blocked(
        self,
        document_id: str,
        version_key: str,
        *,
        include_activation: bool = False,
    ) -> DocumentParseVersion:
        """同步依次执行全部阶段（不入队后续阶段）。

        用于同步/测试路径：按 ``INGESTION_STAGES`` 顺序逐个调用
        ``run_stage`` 且 ``enqueue_next=False``；``include_activation``
        控制是否包含最后的 activate 阶段。

        返回：最终版本对象（最后一个被执行的阶段产物）。
        """
        stages = INGESTION_STAGES if include_activation else INGESTION_STAGES[:-1]
        version: DocumentParseVersion | None = None
        for stage in stages:
            version = self.run_stage(
                document_id,
                version_key,
                stage,
                enqueue_next=False,
            )
        if version is None:  # pragma: no cover - the stage contract is never empty
            raise RuntimeError("No ingestion stages are configured.")
        return version

    def activate_batch(
        self,
        version_by_document: dict[str, str],
    ) -> list[DocumentParseVersion]:
        """批量执行 activate 阶段并激活多个文档版本（单事务）。

        参数：``{document_id: version_key}`` 映射。

        流程：
        1. 清洗输入（非空校验）。
        2. 必须有 activate 执行器，否则抛 ``StageHandlerUnavailable``。
        3. 按文档 ID 升序逐个行锁加载 (document, version)。
        4. 逐个准备：前置校验器、认领检查（已有 running 且租约未过期则
           抛 ``StageAlreadyClaimed``）、顺序校验、状态推进、组装输入与
           尝试次数。
        5. 对每个准备好的文档执行 activate 执行器，收集输出。
        6. 统一调用 ``versions.batch_activate`` 原子激活；成功后把所有
           文档置 ``ready``、写入 completed 检查点。
        7. 任一步异常则整体 rollback 并重抛。
        """
        requested = {
            str(document_id): str(version_key)
            for document_id, version_key in version_by_document.items()
            if str(document_id) and str(version_key)
        }
        if not requested or len(requested) != len(version_by_document):
            raise ValueError(
                "Batch activation requires non-empty document/version keys."
            )
        handler = self.handlers.get("activate")
        if handler is None:
            raise StageHandlerUnavailable(
                "No production handler is configured for ingestion stage 'activate'."
            )

        # 按文档 ID 升序加锁，保持全局一致加锁顺序（防死锁）
        locked: list[tuple[Document, DocumentParseVersion]] = []
        try:
            for document_id in sorted(requested):
                document, version = self._load_locked(
                    document_id,
                    requested[document_id],
                )
                locked.append((document, version))

            # 准备阶段：校验 + 认领 + 状态推进
            prepared: list[
                tuple[Document, DocumentParseVersion, dict[str, Any], int]
            ] = []
            for document, version in locked:
                if self.pre_stage_validator is not None:
                    self.pre_stage_validator(document, version, "activate")
                checkpoint = dict((version.stage_state or {}).get("activate") or {})
                # 若已有人认领且租约未过期，说明本批次与他人冲突
                if checkpoint.get("status") == "running" and not self._claim_expired(
                    checkpoint
                ):
                    raise StageAlreadyClaimed(
                        "Ingestion stage 'activate' is already claimed for "
                        f"document {document.id!r}, version {version.version_key!r}."
                    )
                self._validate_order(version, "activate")
                self._prepare_status_for_attempt(version, "activate")
                prepared.append(
                    (
                        document,
                        version,
                        self._stage_input(version, "activate"),
                        int(checkpoint.get("attempts") or 0) + 1,  # 尝试次数 +1
                    )
                )

            # 执行阶段：逐个调用 handler
            outputs: dict[str, dict[str, Any]] = {}
            for document, version, input_checkpoint, attempt in prepared:
                output = handler(
                    IngestionStageContext(
                        db=self.db,
                        document=document,
                        version=version,
                        stage="activate",
                        input=input_checkpoint,
                        attempt=attempt,
                    )
                )
                outputs[document.id] = {
                    "input": input_checkpoint,
                    "output": _json_safe(output),
                    "attempt": attempt,
                }

            # 原子批量激活；成功后落 checkpoints
            activated = self.versions.batch_activate(requested)
            completed_at = self._timestamp()
            for document, version in locked:
                prepared_output = outputs[document.id]
                document.status = "ready"
                self._set_pipeline_progress(document.id, "activate", completed=True)
                self._set_checkpoint(
                    version,
                    "activate",
                    {
                        "status": "completed",
                        "attempts": prepared_output["attempt"],
                        "progress": 100,
                        "input": prepared_output["input"],
                        "output": prepared_output["output"],
                        "error": None,
                        "started_at": completed_at,
                        "completed_at": completed_at,
                        "failed_at": None,
                        "next_enqueued": False,
                        "claim_owner": self.claim_owner,
                        "lease_expires_at": None,
                    },
                )
            self.db.commit()
            return activated
        except Exception:
            self.db.rollback()
            raise

    def run_stage(
        self,
        document_id: str,
        version_key: str,
        stage: str,
        *,
        enqueue_next: bool = True,
    ) -> DocumentParseVersion:
        """执行单个 ingestion 阶段（含认领、执行、落检查点、失败记录）。

        参数：
        - ``document_id`` / ``version_key``：定位 (document, version)。
        - ``stage``：要执行的阶段名。
        - ``enqueue_next``：完成后是否入队下一阶段。

        执行流程：
        1. 校验阶段合法；行锁加载 (document, version)。
        2. 阶段前校验器（异常则回滚中止）。
        3. 读取既有检查点：
           - completed：直接提交并（按需）入队下一阶段返回。
           - running 且租约未过期：抛 ``StageAlreadyClaimed``。
           - running 且租约已过期 / 无检查点：视为可接管/新执行。
        4. 顺序校验 + 状态推进（``_prepare_status_for_attempt``）+
           更新文档状态 + 写 running 检查点 + 提交（对外可见认领）。
        5. 重新加锁加载，执行 handler：
           - 成功：按 ``_COMPLETED_STATUS`` 推进版本状态，更新文档状态，
             写 completed 检查点，提交。
           - 失败：回滚，走 ``_record_failure`` 记录失败状态与检查点，
             重抛异常。
        6. 按需入队下一阶段。

        返回：执行后的版本对象。
        """
        self._validate_stage(stage)
        document, version = self._load_locked(document_id, version_key)
        if self.pre_stage_validator is not None:
            try:
                self.pre_stage_validator(document, version, stage)
            except Exception:
                self.db.rollback()
                raise
        checkpoint = dict((version.stage_state or {}).get(stage) or {})

        # 已完成的阶段：直接提交并（可选）串接下一阶段
        if checkpoint.get("status") == "completed":
            self.db.commit()
            if enqueue_next and not checkpoint.get("next_enqueued", False):
                self._enqueue_next(version, stage)
            return version
        # running 且租约未过期：已被其他 worker 认领
        if checkpoint.get("status") == "running":
            if not self._claim_expired(checkpoint):
                self.db.rollback()
                raise StageAlreadyClaimed(
                    f"Ingestion stage {stage!r} is already claimed for "
                    f"document {document_id!r}, version {version_key!r}."
                )

        # 顺序校验 + 版本状态推进 + 文档状态推进
        self._validate_order(version, stage)
        self._prepare_status_for_attempt(version, stage)
        if self._can_update_document_status(document, version):
            document.status = self._document_running_status(stage)
        self._set_pipeline_progress(document.id, stage, completed=False)
        attempt = int(checkpoint.get("attempts") or 0) + 1
        input_checkpoint = self._stage_input(version, stage)
        # 写 running 检查点并提交：对外暴露"认领"
        running = {
            "status": "running",
            "attempts": attempt,
            "progress": 0,
            "input": input_checkpoint,
            "output": checkpoint.get("output"),
            "error": None,
            "started_at": self._timestamp(),
            "completed_at": None,
            "failed_at": None,
            "next_enqueued": False,
            "claim_owner": self.claim_owner,
            "lease_expires_at": self._timestamp(
                self._now() + timedelta(seconds=self.claim_ttl_seconds)
            ),
        }
        self._set_checkpoint(version, stage, running)
        self.db.commit()

        try:
            # 重新加锁加载（认领期间其他 worker 可能已修改）
            document, version = self._load_locked(document_id, version_key)
            handler = self.handlers.get(stage)
            if handler is None:
                raise StageHandlerUnavailable(
                    f"No production handler is configured for ingestion stage {stage!r}."
                )
            output = handler(
                IngestionStageContext(
                    db=self.db,
                    document=document,
                    version=version,
                    stage=stage,
                    input=input_checkpoint,
                    attempt=attempt,
                )
            )
            safe_output = _json_safe(output)
            if stage == "activate":
                # activate 阶段：激活版本并把文档置 ready
                self.versions.activate(document, version)
                document.status = "ready"
            else:
                # 其余阶段：把版本状态推进到完成态
                self.versions.transition(version, _COMPLETED_STATUS[stage])
                # 完成态不是 ready_to_activate 且允许更新文档状态时同步文档
                if (
                    _COMPLETED_STATUS[stage] != "ready_to_activate"
                    and self._can_update_document_status(document, version)
                ):
                    document.status = _COMPLETED_STATUS[stage]
            self._set_pipeline_progress(document.id, stage, completed=True)
            # 写 completed 检查点
            completed = {
                **running,
                "status": "completed",
                "progress": 100,
                "output": safe_output,
                "completed_at": self._timestamp(),
            }
            self._set_checkpoint(version, stage, completed)
            self.db.commit()
        except Exception as exc:
            # 失败：回滚后记录失败状态与检查点，再重抛
            self.db.rollback()
            self._record_failure(document_id, version_key, stage, running, exc)
            raise

        if enqueue_next:
            self._enqueue_next(version, stage)
        return version

    def _load_locked(
        self, document_id: str, version_key: str
    ) -> tuple[Document, DocumentParseVersion]:
        """行锁加载 (document, version)；任一不存在则抛 ValueError。"""
        document = self.db.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None:
            raise ValueError(f"Document {document_id!r} does not exist.")
        version = self.db.scalar(
            select(DocumentParseVersion)
            .where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == version_key,
            )
            .with_for_update()
        )
        if version is None:
            raise ValueError(
                f"Parse version {version_key!r} does not exist for document {document_id!r}."
            )
        return document, version

    @staticmethod
    def _validate_stage(stage: str) -> None:
        """校验阶段名在 ``INGESTION_STAGES`` 内。"""
        if stage not in INGESTION_STAGES:
            raise ValueError(f"Unknown ingestion stage {stage!r}.")

    @staticmethod
    def _validate_order(version: DocumentParseVersion, stage: str) -> None:
        """校验执行顺序：该阶段之前的所有阶段必须已完成。

        检查 ``version.stage_state`` 中前面各阶段的检查点状态是否为
        completed；有未完成的前置阶段则抛 ``ValueError``。
        """
        state = version.stage_state or {}
        stage_index = INGESTION_STAGES.index(stage)
        missing = [
            previous
            for previous in INGESTION_STAGES[:stage_index]
            if (state.get(previous) or {}).get("status") != "completed"
        ]
        if missing:
            raise ValueError(
                f"Ingestion stage {stage!r} is out of order; incomplete stages: "
                + ", ".join(missing)
                + "."
            )

    def _prepare_status_for_attempt(
        self, version: DocumentParseVersion, stage: str
    ) -> None:
        """把版本状态推进到该阶段对应的 running 状态（准备执行）。

        三种情况：
        1. 当前状态是可重试失败态（在 ``FAILED_STAGE_RETRIES`` 中）：
           先回退到重试阶段，再沿合法转移边（``_status_path`` BFS）
           推进到目标 running 状态。
        2. 当前已是目标状态：无需操作。
        3. 其他：直接按 ``ALLOWED_TRANSITIONS`` 推进到目标状态
           （不合法会抛 ValueError）。
        """
        expected = _RUNNING_STATUS[stage]
        if version.status in FAILED_STAGE_RETRIES:
            # 失败重试：回到可重试阶段后再恢复到目标 running 状态
            self.versions.retry_failed_stage(version)
            for target in self._status_path(version.status, expected):
                self.versions.transition(version, target)
            return
        if version.status == expected:
            return
        self.versions.transition(version, expected)

    @staticmethod
    def _status_path(source: str, target: str) -> list[str]:
        """在状态机中 BFS 搜索从 ``source`` 到 ``target`` 的合法中间状态路径。

        只在 ``ALLOWED_TRANSITIONS`` 的合法边上移动；跳过所有 ``*_failed``
        失败态与 ``active`` 终态（这些不能作为中间停靠点）。

        返回：中间状态列表（不含 source/target）；找不到路径抛 ValueError。
        """
        if source == target:
            return []
        pending: deque[tuple[str, list[str]]] = deque([(source, [])])
        visited = {source}
        while pending:
            current, path = pending.popleft()
            for candidate in ALLOWED_TRANSITIONS.get(current, set()):
                # 失败态与 active 不参与中间恢复路径
                if candidate.endswith("_failed") or candidate == "active":
                    continue
                candidate_path = [*path, candidate]
                if candidate == target:
                    return candidate_path
                if candidate not in visited:
                    visited.add(candidate)
                    pending.append((candidate, candidate_path))
        raise ValueError(
            f"Parse-version status {source!r} cannot resume at {target!r}."
        )

    @staticmethod
    def _stage_input(
        version: DocumentParseVersion, stage: str
    ) -> dict[str, Any]:
        """组装某阶段的输入检查点：文档/版本标识 + 上一阶段及其输出。

        首阶段（index 为 0）没有上一阶段，``previous_stage`` 为 None。
        """
        stage_index = INGESTION_STAGES.index(stage)
        previous_output = None
        previous_stage = None
        if stage_index:
            previous_stage = INGESTION_STAGES[stage_index - 1]
            previous_output = (version.stage_state or {}).get(previous_stage, {}).get(
                "output"
            )
        return _json_safe(
            {
                "document_id": version.document_id,
                "version_key": version.version_key,
                "stage": stage,
                "previous_stage": previous_stage,
                "previous_output": previous_output,
            }
        )

    def _record_failure(
        self,
        document_id: str,
        version_key: str,
        stage: str,
        running: dict[str, Any],
        exc: Exception,
    ) -> None:
        """阶段执行失败后的落库：推进失败状态、记录错误、写 failed 检查点。

        流程：
        1. 加锁重新加载 (document, version)。
        2. 把版本状态推进到该阶段对应的失败态（若尚未处于失败态）。
        3. 允许时同步文档状态为失败态。
        4. 生成有界错误信息（``_bounded_stage_error``）。
        5. 写 failed 检查点；若错误信息导致检查点超限，则退化为只保存
           截断摘要 + 错误哈希。
        6. 更新 pipeline 运行记录并提交。
        """
        document, version = self._load_locked(document_id, version_key)
        failed_status = _FAILED_STATUS[stage]
        if version.status != failed_status:
            self.versions.transition(version, failed_status)
        if self._can_update_document_status(document, version):
            document.status = failed_status
        # 有界错误：截断 + 哈希，避免大异常文本撑爆检查点
        error, error_sha256, error_truncated = _bounded_stage_error(exc)
        self._set_pipeline_failure(document.id, stage, error)
        failed = {
            **running,
            "status": "failed",
            "progress": 0,
            "error": error,
            "error_type": type(exc).__name__,
            "error_sha256": error_sha256,
            "error_truncated": error_truncated,
            "failed_at": self._timestamp(),
        }
        try:
            self._set_checkpoint(version, stage, failed)
        except StageCheckpointTooLarge:
            # 错误文本使检查点超限：退化为仅保存截断信息
            self._set_checkpoint(
                version,
                stage,
                {
                    "status": "failed",
                    "attempts": running["attempts"],
                    "progress": 0,
                    "error": (
                        f"{type(exc).__name__}: details omitted; "
                        f"sha256={error_sha256}"
                    ),
                    "error_type": type(exc).__name__,
                    "error_sha256": error_sha256,
                    "error_truncated": True,
                    "failed_at": self._timestamp(),
                },
            )
        self.db.commit()

    def _enqueue_next(self, version: DocumentParseVersion, stage: str) -> None:
        """把下一阶段入队（若存在），并标记本阶段已完成入队。

        - 已是最后一个阶段：仅标记 ``next_enqueued``，不再入队。
        - 无 dispatcher（未启用队列）：直接跳过（调用方应同步执行）。
        - 否则经 dispatcher 入队下一阶段，再标记。
        """
        stage_index = INGESTION_STAGES.index(stage)
        if stage_index == len(INGESTION_STAGES) - 1:
            self._mark_enqueued(version, stage)
            return
        if self.dispatcher is None:
            return
        next_stage = INGESTION_STAGES[stage_index + 1]
        self.dispatcher.enqueue_stage(
            version.document_id,
            version.version_key,
            next_stage,
        )
        self._mark_enqueued(version, stage)

    def _mark_enqueued(self, version: DocumentParseVersion, stage: str) -> None:
        """在检查点中标记 ``next_enqueued`` 并提交。

        重新加锁加载版本，更新该阶段检查点字段，防止与并发写冲突。
        """
        _document, locked_version = self._load_locked(
            version.document_id, version.version_key
        )
        checkpoint = dict((locked_version.stage_state or {}).get(stage) or {})
        checkpoint["next_enqueued"] = True
        checkpoint["next_enqueued_at"] = self._timestamp()
        self._set_checkpoint(locked_version, stage, checkpoint)
        self.db.commit()

    def _set_checkpoint(
        self,
        version: DocumentParseVersion, stage: str, checkpoint: dict[str, Any]
    ) -> None:
        """把某阶段的检查点写入 ``version.stage_state``（带大小上限校验）。

        检查点先经 ``_json_safe`` 规范化，再整体 JSON 序列化；若编码后的
        字节数超过 ``max_checkpoint_bytes`` 则抛 ``StageCheckpointTooLarge``，
        绝不把超限数据写入数据库。
        """
        state = dict(version.stage_state or {})
        state[stage] = _json_safe(checkpoint)
        # 紧凑、无 NaN、键排序无关的确定性编码（用于字节大小判定）
        encoded = json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > self.max_checkpoint_bytes:
            raise StageCheckpointTooLarge(
                f"Ingestion checkpoint is {len(encoded)} bytes; "
                f"maximum is {self.max_checkpoint_bytes} bytes."
            )
        version.stage_state = state

    def _now(self) -> datetime:
        """返回规范化到 UTC 的当前时间。"""
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _timestamp(self, value: datetime | None = None) -> str:
        """把 datetime 转为 ISO 8601 字符串（UTC，后缀 ``Z``）。"""
        value = value or self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _claim_expired(self, checkpoint: Mapping[str, Any]) -> bool:
        """判断认领租约是否已过期。

        - 无 ``lease_expires_at`` 字符串：视为未过期（False）。
        - 时间解析失败：视为未过期（保守）。
        - 过期时间（UTC）不晚于当前时间即认为已过期。
        """
        raw_expiry = checkpoint.get("lease_expires_at")
        if not isinstance(raw_expiry, str):
            return False
        try:
            expiry = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry.astimezone(timezone.utc) <= self._now()

    def _latest_pipeline_run(self, document_id: str) -> PipelineRun | None:
        """取文档最近一条 PipelineRun 记录（按创建时间倒序）。"""
        return self.db.scalar(
            select(PipelineRun)
            .where(PipelineRun.document_id == document_id)
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        )

    def _set_pipeline_progress(
        self, document_id: str, stage: str, *, completed: bool
    ) -> None:
        """更新 pipeline 运行记录的进度百分比与运行状态。

        进度计算：``100 * (阶段序号 + (completed ? 1 : 0)) / 总阶段数``，
        即开始某阶段时占满前一格、完成后占满当前格。

        - activate 完成时：run 状态置 ``completed``。
        - 其余情况：置 ``running``。
        """
        run = self._latest_pipeline_run(document_id)
        if run is None:
            return
        stage_index = INGESTION_STAGES.index(stage)
        percent = round(
            100 * (stage_index + (1 if completed else 0)) / len(INGESTION_STAGES)
        )
        report = dict(run.provider_report or {})
        report["progress"] = {
            "percent": percent,
            "stage": "completed" if stage == "activate" and completed else stage,
            "message": (
                f"Ingestion stage {stage} completed."
                if completed
                else f"Ingestion stage {stage} started."
            ),
        }
        run.provider_report = report
        run.status = (
            RunStatus.completed.value
            if stage == "activate" and completed
            else RunStatus.running.value
        )

    def _set_pipeline_failure(
        self, document_id: str, stage: str, error: str
    ) -> None:
        """把 pipeline 运行记录标记为失败，写入错误与进度。"""
        run = self._latest_pipeline_run(document_id)
        if run is None:
            return
        report = dict(run.provider_report or {})
        report["error"] = error
        report["progress"] = {
            "percent": round(
                100 * INGESTION_STAGES.index(stage) / len(INGESTION_STAGES)
            ),
            "stage": f"{stage}_failed",
            "message": error,
        }
        run.provider_report = report
        run.status = RunStatus.failed.value
        run.notes = error

    @staticmethod
    def _document_running_status(stage: str) -> str:
        """返回阶段 running 时文档应处的状态。

        activate 阶段比较特殊：它在运行期文档状态记为 ``indexing``
        （激活成功后才会置 ready）。
        """
        if stage == "activate":
            return "indexing"
        return _RUNNING_STATUS[stage]

    @staticmethod
    def _can_update_document_status(
        document: Document, version: DocumentParseVersion
    ) -> bool:
        """判断是否可以随阶段推进同步更新文档状态。

        仅当文档当前激活版本指针为空或正指向本版本时才允许——即只对
        "正在进行本版本管线"的文档更新状态，避免动到已激活的其他版本。
        """
        return document.active_parse_version in (None, version.version_key)


def _json_safe(value: Any) -> Any:
    """把任意值递归转换为 JSON 可序列化的结构（供检查点存储）。

    转换规则：
    - Pydantic ``BaseModel``：``model_dump(mode="json")``。
    - dataclass（非类本身）：``asdict``。
    - 标量（None/str/int/bool）：原样返回。
    - float：拒绝 NaN/±Inf（``allow_nan=False`` 的要求），其余原样。
    - datetime/date：ISO 字符串。
    - Path / Enum：转为字符串。
    - Mapping / 列表类：递归转换各元素，并做一次 ``allow_nan=False``
      的 ``json.dumps`` 试探以尽早发现不可序列化内容。
    - 其余类型抛 ``TypeError``。
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # 拒绝非有限数值（NaN / ±inf），保证可稳定 JSON 化
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Stage checkpoints cannot contain non-finite numbers.")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, Mapping):
        converted = {str(key): _json_safe(item) for key, item in value.items()}
        json.dumps(converted, allow_nan=False)
        return converted
    if isinstance(value, (list, tuple, set, frozenset)):
        converted = [_json_safe(item) for item in value]
        json.dumps(converted, allow_nan=False)
        return converted
    raise TypeError(f"Stage checkpoint value {type(value).__name__} is not JSON serializable.")


def _bounded_stage_error(exc: Exception) -> tuple[str, str, bool]:
    """把异常文本转换为"有界"错误信息：清洗、截断并计算哈希。

    步骤：
    1. 计算异常文本 UTF-8 字节的 SHA-256（无论截断与否，哈希都对原始
       文本计算，便于跨记录比对同一错误）。
    2. 清洗：把不可打印字符替换为 ``?``（保留换行/制表符）。
    3. 若清洗后文本在 ``_MAX_STAGE_ERROR_BYTES`` 内，直接返回
       (clean_text, digest, False)。
    4. 否则按预算截断前缀，追加 ``... [truncated; sha256=...]`` 后缀，
       返回 (prefix+suffix, digest, True)。

    返回：``(error_text, sha256_hex, truncated: bool)``。
    """
    raw = str(exc)
    raw_bytes = raw.encode("utf-8", errors="replace")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    # 清洗不可打印字符
    sanitized = "".join(
        character if character.isprintable() or character in "\n\t" else "?"
        for character in raw
    )
    encoded = sanitized.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_STAGE_ERROR_BYTES:
        return sanitized, digest, False
    # 截断：预留后缀空间，切出可用预算
    suffix = f"... [truncated; sha256={digest}]"
    budget = _MAX_STAGE_ERROR_BYTES - len(suffix.encode("ascii"))
    prefix = encoded[:budget].decode("utf-8", errors="ignore")
    return prefix + suffix, digest, True
