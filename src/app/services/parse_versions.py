"""
parse_versions.py —— 文档解析版本（ParseVersion）状态机与激活服务模块
===================================================================

职责：
- 管理 ``DocumentParseVersion``（文档解析版本）的生命周期状态机：
  版本创建、阶段推进、失败重试、激活/取代。
- 定义解析版本的状态转移表（``ALLOWED_TRANSITIONS``）与失败回退表
  （``FAILED_STAGE_RETRIES``）。
- 提供**并发安全**的版本激活（``activate``）与批量激活
  （``batch_activate``）：激活涉及"文档当前版本指针"与"新版本状态"两处
  数据的一致性更新，模块通过 ``SELECT ... FOR UPDATE`` 行锁、
  ``set_committed_value`` 状态快照与失败回滚来保证。

状态机（ALLOWED_TRANSITIONS，简化链）：
    queued -> parsing -> quality_checking -> repairing -> canonicalizing
    -> chunking -> contextualizing -> embedding -> indexing
    -> ready_to_activate -> active
    失败状态：parse_failed / table_repair_failed /
    contextualization_failed / embedding_failed / activation_failed，
    均由 FAILED_STAGE_RETRIES 回退到对应重试阶段。

并发安全设计（激活核心）：
- 整个"校验 + 变更"流程包在 ``db.no_autoflush`` 块内，避免自动 flush
  打乱事务。
- 对 document 与 version 行执行 ``with_for_update()`` 行锁；由于先锁定
  document 行再锁 version 行，可避免死锁。
- 对"已激活的其他版本"用 ``set_committed_value`` 保存数据库侧状态快照，
  提交失败时据此回滚（还原 active 指针、状态与 activated_at）。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import inspect as sa_inspect, or_, select
from sqlalchemy.orm import Session, object_session
from sqlalchemy.orm.attributes import set_committed_value

from app.models.records import Document, DocumentParseVersion

# 状态机转移表：{当前状态: 允许的下一状态集合}
# 正常推进路径：
#   queued -> parsing -> quality_checking -> repairing -> canonicalizing
#   -> chunking -> contextualizing -> embedding -> indexing
#   -> ready_to_activate -> active
# 各阶段允许直接跳转到对应失败态（如 parsing 可到 parse_failed）
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "queued": {"parsing", "parse_failed"},
    "parsing": {"quality_checking", "parse_failed"},
    "quality_checking": {"repairing", "canonicalizing", "parse_failed"},
    "repairing": {"canonicalizing", "table_repair_failed", "parse_failed"},
    "canonicalizing": {"chunking", "parse_failed"},
    "chunking": {"contextualizing", "parse_failed"},
    "contextualizing": {"embedding", "contextualization_failed"},
    "embedding": {"indexing", "embedding_failed"},
    "indexing": {"ready_to_activate", "embedding_failed"},
    "ready_to_activate": {"active", "activation_failed"},
}


class ActivationError(RuntimeError):
    """Raised when a staged parse version is not complete enough to publish.

    当某个分阶段（staged）的解析版本尚未完整到足以对外发布时抛出。
    """


# 失败状态回退表：{失败状态: 重试时回到的阶段}
# 例如 parse_failed 表示解析阶段失败，重试时回到 "parsing" 重新解析
FAILED_STAGE_RETRIES: dict[str, str] = {
    "parse_failed": "parsing",
    "table_repair_failed": "repairing",
    "contextualization_failed": "contextualizing",
    "embedding_failed": "embedding",
    "activation_failed": "ready_to_activate",
}


class ParseVersionService:
    """解析版本生命周期服务：创建、推进、重试、激活。

    所有方法依赖注入的 ``db``（可能为 None）。激活类方法在 ``db`` 为
    None 或传入对象不属于当前会话时抛 ``ValueError``/``RuntimeError``，
    强制调用方提供有效会话。
    """

    def __init__(self, db: Session | None) -> None:
        """保存数据库会话；可为 None（供只读/离线场景，但激活类操作会拒绝）。"""
        self.db = db

    def create(
        self,
        document_id: str,
        version_key: str,
        artifact_dir: str,
        parser_name: str | None = None,
        parser_version: str | None = None,
    ) -> DocumentParseVersion:
        """创建新的解析版本记录（初始状态 queued）。

        参数：
        - ``document_id``：所属文档 ID。
        - ``version_key``：版本键（见 ingestion_identity.build_parse_version_key）。
        - ``artifact_dir``：产物目录（解析结果落盘位置）。
        - ``parser_name`` / ``parser_version``：可选，记录解析器名称与版本。

        说明：只 ``add`` + ``flush``（不 commit），事务边界由调用方掌握；
        flush 使版本获得主键。
        """
        if self.db is None:
            raise RuntimeError("A database session is required to create a parse version.")
        version = DocumentParseVersion(
            document_id=document_id,
            version_key=version_key,
            artifact_dir=artifact_dir,
            parser_name=parser_name,
            parser_version=parser_version,
        )
        self.db.add(version)
        self.db.flush()
        return version

    def transition(
        self, version: DocumentParseVersion, target: str
    ) -> DocumentParseVersion:
        """按状态机把版本从当前状态推进到目标状态。

        若 ``target`` 不在 ``ALLOWED_TRANSITIONS[当前状态]`` 中则抛
        ``ValueError``。校验通过后直接修改 ``version.status`` 并返回
        同一实例（持久化由调用方/会话的 commit 完成）。
        """
        allowed = ALLOWED_TRANSITIONS.get(version.status, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid parse-version transition from {version.status!r} to {target!r}."
            )
        version.status = target
        return version

    def retry_failed_stage(
        self, version: DocumentParseVersion
    ) -> DocumentParseVersion:
        """把失败状态的版本回退到可重试阶段。

        依据 ``FAILED_STAGE_RETRIES`` 查找目标状态；非可重试状态（不在
        表中）抛 ``ValueError``。例如 ``parse_failed`` -> ``parsing``。
        """
        target = FAILED_STAGE_RETRIES.get(version.status)
        if target is None:
            raise ValueError(
                f"Parse version in status {version.status!r} is not retryable."
            )
        version.status = target
        return version

    def activate(
        self, document: Document, version: DocumentParseVersion
    ) -> DocumentParseVersion:
        """把 ``ready_to_activate`` 的解析版本激活为文档当前版本（并发安全）。

        原子语义：版本置为 ``active`` 且文档指针指向该版本；此前已激活的
        其他版本置为 ``superseded``（被取代）。任何失败回滚全部变更。

        关键步骤（均处于 ``no_autoflush`` 块内）：
        1. 前置检查：``document`` 与 ``version`` 必须持久化于当前会话
           （``_require_persistent_current_session``）。
        2. 行锁读取：``with_for_update()`` 锁定 document 行、version 行，
           并读文档当前 active 指针；先锁文档再锁版本，保持加锁顺序一致。
        3. 一致性校验：版本必须属于该文档；状态必须允许激活
           （``_validate_activation_status``）。
        4. 用 ``set_committed_value`` 把 version 的 document_id/version_key
           以及各旧 active 版本的状态快照"固化"为数据库当前值，避免后续
           flush 把这些字段当作脏数据误写。
        5. 提交变更：旧 active 版本 -> superseded；文档指针更新；
           新版本 -> active + activated_at。
        6. flush 失败则回滚所有变更（还原指针、状态、activated_at）。
        """
        if self.db is None:
            raise RuntimeError("A database session is required to activate a parse version.")
        self._require_persistent_current_session(document, "Document")
        self._require_persistent_current_session(version, "Parse version")

        with self.db.no_autoflush:
            # 行锁读取文档，防止并发激活时指针丢失更新
            locked_document = self.db.scalar(
                select(Document).where(Document.id == document.id).with_for_update()
            )
            if locked_document is None:
                raise ValueError(f"Document {document.id!r} does not exist.")
            # 行锁读取待激活版本
            locked_version = self.db.scalar(
                select(DocumentParseVersion)
                .where(DocumentParseVersion.id == version.id)
                .with_for_update()
            )
            if locked_version is None:
                raise ValueError(f"Parse version {version.id!r} does not exist.")
            # 读取文档当前激活指针（数据库侧真实值）
            database_pointer = self.db.execute(
                select(Document.active_parse_version)
                .where(Document.id == locked_document.id)
                .with_for_update()
            ).scalar_one()
            # 读取版本行的数据库侧字段（避免依赖会话中可能已过期的值）
            (
                database_document_id,
                database_version_key,
                database_status,
            ) = self.db.execute(
                select(
                    DocumentParseVersion.document_id,
                    DocumentParseVersion.version_key,
                    DocumentParseVersion.status,
                )
                .where(DocumentParseVersion.id == locked_version.id)
                .with_for_update()
            ).one()
            # 版本必须属于被激活的文档
            if database_document_id != locked_document.id:
                raise ValueError(
                    "Parse version and document must belong to the same document."
                )
            # 校验本地状态与数据库状态在激活路径上一致
            self._validate_activation_status(locked_version, database_status)
            # 固化只读字段：防止 flush 时把这些字段当作被修改项写回
            set_committed_value(
                locked_version, "document_id", database_document_id
            )
            set_committed_value(
                locked_version, "version_key", database_version_key
            )
            # 行锁读取该文档所有"已激活"的其他版本，用于将其取代
            previous_rows = self.db.execute(
                select(DocumentParseVersion.id, DocumentParseVersion.status)
                .where(
                    DocumentParseVersion.document_id == locked_document.id,
                    DocumentParseVersion.status == "active",
                    DocumentParseVersion.id != locked_version.id,
                )
                .with_for_update()
            ).all()
            previous_statuses: list[tuple[DocumentParseVersion, str]] = []
            for previous_id, previous_database_status in previous_rows:
                previous = self.db.get(DocumentParseVersion, previous_id)
                if previous is None:
                    raise ValueError(
                        f"Active parse version {previous_id!r} no longer exists."
                    )
                # 固化旧版本状态，记录回滚快照
                set_committed_value(
                    previous, "status", previous_database_status
                )
                previous_statuses.append((previous, previous_database_status))

        # 记录变更前的现场，供失败回滚
        activated_at = datetime.utcnow()
        version_status = locked_version.status
        version_activated_at = locked_version.activated_at
        # 执行变更：旧版本取代、文档指针更新、新版本激活
        for previous, _database_status in previous_statuses:
            previous.status = "superseded"
        locked_document.active_parse_version = database_version_key
        locked_version.status = "active"
        locked_version.activated_at = activated_at
        try:
            self.db.flush()
        except Exception:
            # 回滚：还原指针与所有被修改版本的状态/时间
            locked_document.active_parse_version = database_pointer
            for previous, status in previous_statuses:
                previous.status = status
            locked_version.status = version_status
            locked_version.activated_at = version_activated_at
            raise
        return locked_version

    def batch_activate(
        self,
        version_by_document: dict[str, str],
    ) -> list[DocumentParseVersion]:
        """批量激活多个文档的解析版本（单事务、按文档 ID 排序加锁防死锁）。

        参数：
        - ``version_by_document``：``{document_id: version_key}`` 映射。

        流程：
        1. 过滤空键并校验输入非空。
        2. 在 ``no_autoflush`` 内，按文档 ID 升序依次行锁读取文档，
           校验文档全部存在。
        3. 用 ``OR`` 组合条件行锁读取所有目标版本，校验存在且状态为
           ``ready_to_activate``。
        4. 行锁读取所有将被取代的旧 active 版本。
        5. 退出 ``no_autoflush`` 后：为新版本置 active、旧版本置
           superseded、文档指针更新；flush 失败则按快照回滚。
        6. 返回激活后的版本列表。

        说明：``zip(..., strict=True)`` 要求文档与版本一一对应，
        且都非 None（前面已保证），防止批量场景下标错位。
        """
        if self.db is None:
            raise RuntimeError("A database session is required to activate parse versions.")
        # 清洗输入：剔除空字符串键/值
        requested = {
            str(document_id): str(version_key)
            for document_id, version_key in version_by_document.items()
            if str(document_id) and str(version_key)
        }
        if not requested or len(requested) != len(version_by_document):
            raise ValueError("Batch activation requires non-empty document/version keys.")
        # 按文档 ID 排序，保证加锁顺序全局一致（防死锁）
        document_ids = sorted(requested)
        with self.db.no_autoflush:
            # 行锁读取全部目标文档（按 ID 升序）
            document_rows = self.db.execute(
                select(Document.id, Document.active_parse_version)
                .where(Document.id.in_(document_ids))
                .order_by(Document.id)
                .with_for_update()
            ).all()
            if [str(row.id) for row in document_rows] != document_ids:
                raise ValueError("Batch activation contains an unknown document.")
            # 行锁读取全部目标版本：以 (document_id, version_key) 组合匹配
            version_rows = self.db.execute(
                select(
                    DocumentParseVersion.id,
                    DocumentParseVersion.document_id,
                    DocumentParseVersion.version_key,
                    DocumentParseVersion.status,
                )
                .where(
                    or_(
                        *(
                            (
                                (DocumentParseVersion.document_id == document_id)
                                & (DocumentParseVersion.version_key == requested[document_id])
                            )
                            for document_id in document_ids
                        )
                    )
                )
                .order_by(DocumentParseVersion.document_id)
                .with_for_update()
            ).all()
            versions_by_document = {
                str(row.document_id): row for row in version_rows
            }
            # 逐个校验版本存在且处于可激活状态
            for document_id in document_ids:
                row = versions_by_document.get(document_id)
                version_key = requested[document_id]
                if row is None:
                    raise ValueError(
                        f"Parse version {version_key!r} for document {document_id!r} does not exist."
                    )
                if row.status != "ready_to_activate":
                    raise ValueError(
                        f"Parse version {version_key} must be in ready_to_activate status; "
                        f"database status is {row.status!r}."
                    )
            # 行锁读取所有将被取代的旧 active 版本
            previous_rows = self.db.scalars(
                select(DocumentParseVersion)
                .where(
                    DocumentParseVersion.document_id.in_(document_ids),
                    DocumentParseVersion.status == "active",
                )
                .order_by(DocumentParseVersion.document_id, DocumentParseVersion.id)
                .with_for_update()
            ).all()

        # 重新从会话取对象（已锁定的行，确保是最新且属于本会话）
        documents = [self.db.get(Document, document_id) for document_id in document_ids]
        staged = [
            self.db.get(DocumentParseVersion, versions_by_document[document_id].id)
            for document_id in document_ids
        ]
        if any(item is None for item in documents) or any(item is None for item in staged):
            raise ValueError("Batch activation rows changed while locked.")
        # 保存回滚快照
        document_snapshots = [
            (item, item.active_parse_version) for item in documents if item is not None
        ]
        version_snapshots = [
            (item, item.status, item.activated_at)
            for item in [*previous_rows, *staged]
            if item is not None
        ]
        activated_at = datetime.utcnow()
        # 排除本轮目标版本本身，其余旧 active 版本置 superseded
        target_ids = {item.id for item in staged if item is not None}
        for previous in previous_rows:
            if previous.id not in target_ids:
                previous.status = "superseded"
        # 逐个更新文档指针与新版本状态
        for document, version in zip(documents, staged, strict=True):
            assert document is not None and version is not None
            document.active_parse_version = version.version_key
            version.status = "active"
            version.activated_at = activated_at
        try:
            self.db.flush()
        except Exception:
            # 回滚所有变更
            for document, pointer in document_snapshots:
                document.active_parse_version = pointer
            for version, status, version_activated_at in version_snapshots:
                version.status = status
                version.activated_at = version_activated_at
            raise
        return [item for item in staged if item is not None]

    def _require_persistent_current_session(self, instance: object, label: str) -> None:
        """校验对象必须"持久化且属于当前会话"。

        检查：
        - ``object_session(instance) is self.db``：对象属于当前会话。
        - ``state.persistent``：对象处于持久化状态（已 flush 过）。
        - 未处于删除态（``state.deleted`` 或在本会话 ``deleted`` 集合中）。

        不满足时抛 ``ValueError``，防止对游离/新建/待删除对象做激活操作。
        """
        state = sa_inspect(instance)
        if (
            object_session(instance) is not self.db
            or not state.persistent
            or state.deleted
            or instance in self.db.deleted
        ):
            raise ValueError(
                f"{label} must be persistent in the current session and not pending deletion."
            )

    @staticmethod
    def _validate_activation_status(
        version: DocumentParseVersion, database_status: str
    ) -> None:
        """校验版本本地状态与数据库状态在"激活"路径上一致。

        目标状态必须是 ``ready_to_activate``，且根据本地状态的变更历史
        判断本次激活是否为合法的单步转移：

        1. 本地状态本身就是 ``ready_to_activate``：
           - 若本地无变更历史（``history.has_changes()`` 为 False），则
             要求数据库状态也是 ``ready_to_activate``。
           - 若有变更历史，则本地来源状态必须能通过合法边到达目标
             （``ALLOWED_TRANSITIONS`` 或 ``FAILED_STAGE_RETRIES``），
             且数据库状态必须等于本地来源状态（防止与其他会话的并发
             更新冲突）。
        2. 任一校验不通过即抛 ``ValueError``，中止激活。

        说明：这是"乐观并发"的校验——激活前确认会话中的本地状态变更
        与数据库现状没有冲突。
        """
        target = "ready_to_activate"
        if version.status != target:
            raise ValueError(
                "Parse version must be in ready_to_activate status before activation."
            )

        # 读取该字段的变更历史（仅检测到变化时才需校验转移边）
        history = sa_inspect(version).attrs.status.history
        if not history.has_changes():
            # 本地无变更：要求数据库本来就处于可激活状态
            if database_status != target:
                raise ValueError(
                    f"Parse version database status is {database_status!r}, not {target!r}; "
                    "activation aborted."
                )
            return

        # 有本地变更：解析来源状态，校验是否为合法单步转移
        sources = list(history.deleted)
        targets = list(history.added)
        source = sources[0] if len(sources) == 1 else None
        # 允许的边：目标在来源状态的正向转移表中，或是失败回退到目标
        is_allowed_edge = source is not None and (
            target in ALLOWED_TRANSITIONS.get(source, set())
            or FAILED_STAGE_RETRIES.get(source) == target
        )
        is_valid_local_transition = (
            len(sources) == 1
            and targets == [target]
            and database_status == source  # 数据库仍处于来源状态，无并发冲突
            and is_allowed_edge
        )
        if not is_valid_local_transition:
            raise ValueError(
                "Parse version local transition conflicts with database status "
                f"{database_status!r} (local source {source!r}); activation aborted."
            )
