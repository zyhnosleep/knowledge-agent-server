"""数据库 ORM 模型定义（SQLAlchemy 映射层）。

本模块集中定义系统所有持久化数据表的 SQLAlchemy ORM 模型，覆盖以下业务域：

- 项目与文档管理：``projects`` / ``documents``，以及文档的多个解析版本
  （``document_parse_versions``）与按层级/阅读顺序组织的内容分块
  （``document_chunks``）。
- 知识抽取与质量：实体（``entities``）、论断（``claims``）、人工复核项
  （``review_items``）。
- 处理流水线：流水线运行记录（``pipeline_runs``）、问答记录
  （``question_answers``）。
- 对话与会话附件：会话（``conversation_sessions``）、会话轮次
  （``conversation_turns``）、临时附件（``session_attachments``）及其分块。
- Agent 执行轨迹：Agent 运行（``agent_trace_runs``）与运行内步骤
  （``agent_trace_steps``）。
- 用户与认证：用户（``users``）与认证会话（``auth_sessions``）。

通用约定：

- 大部分表通过 ``TimestampMixin`` 提供 ``created_at`` / ``updated_at`` 两个时间戳
  列（``conversation_turns``、``conversation_sessions``、``agent_trace_runs``、
  ``agent_trace_steps``、``users``、``auth_sessions`` 自带等效字段）。
- 各表主键 ``id`` 均为 UUID 字符串，默认值由本模块的 ``_uuid`` 辅助函数生成。
- 使用 SQLAlchemy 2.0 风格的 ``Mapped[...]`` + ``mapped_column`` 声明式映射。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base


def _uuid() -> str:
    """生成一个新的 UUID 字符串，用作各表主键 ``id`` 列的默认值。"""
    return str(uuid.uuid4())


class TimestampMixin:
    """公共时间戳混入类。

    为继承它的 ORM 模型统一提供两个时间戳列：
    - ``created_at``：记录创建时间（插入时自动写入当前 UTC 时间）。
    - ``updated_at``：记录最近更新时间（插入时写入，更新时由 ``onupdate`` 自动刷新）。
    """

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentStatus(str, Enum):
    """文档处理流水线的全生命周期状态枚举。

    覆盖从上传到最终可检索的完整处理阶段：
    - 队列/处理中状态：``pending``、``processing``、``parsing``、
      ``quality_checking``、``repairing``、``canonicalizing``、``chunking``、
      ``contextualizing``、``embedding``、``indexing``。
    - 成功终态：``ready``（处理完成，可检索）。
    - 通用失败终态：``failed``。
    - 各阶段失败终态：``parse_failed``（解析失败）、``table_repair_failed``
      （表格修复失败）、``contextualization_failed``（上下文增强失败）、
      ``embedding_failed``（向量化失败）、``activation_failed``（版本激活失败）。
    """

    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"
    parsing = "parsing"
    quality_checking = "quality_checking"
    repairing = "repairing"
    canonicalizing = "canonicalizing"
    chunking = "chunking"
    contextualizing = "contextualizing"
    embedding = "embedding"
    indexing = "indexing"
    parse_failed = "parse_failed"
    table_repair_failed = "table_repair_failed"
    contextualization_failed = "contextualization_failed"
    embedding_failed = "embedding_failed"
    activation_failed = "activation_failed"


class ReviewSeverity(str, Enum):
    """复核项严重程度枚举：low（低）/ medium（中）/ high（高）。"""

    low = "low"
    medium = "medium"
    high = "high"


class ReviewStatus(str, Enum):
    """复核项处理状态枚举：pending（待处理）/ resolved（已解决）/ ignored（已忽略）。"""

    pending = "pending"
    resolved = "resolved"
    ignored = "ignored"


class RunType(str, Enum):
    """流水线运行类型枚举：ingest（摄取）/ query（查询）/ rebuild（重建）/ verify（校验）。"""

    ingest = "ingest"
    query = "query"
    rebuild = "rebuild"
    verify = "verify"


class RunStatus(str, Enum):
    """流水线运行状态枚举：queued（排队中）/ running（运行中）/ completed（完成）/ failed（失败）。"""

    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Project(Base, TimestampMixin):
    """项目表（``projects``）：系统内最高层级的业务隔离单元。

    一个项目（project）对应一个知识库工作区，通过唯一 ``slug`` 在 URL / API 中标识，
    并拥有自己的文档集合与实体集合。删除项目时，通过 ``cascade="all, delete-orphan"``
    级联删除其全部文档与实体。
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)  # 唯一项目标识（URL/API 中使用），如 "internal-pilot"
    name: Mapped[str] = mapped_column(String(255))  # 项目显示名称
    description: Mapped[str | None] = mapped_column(Text, nullable=True)  # 项目描述（可空）

    documents: Mapped[list["Document"]] = relationship(back_populates="project", cascade="all, delete-orphan")  # 项目下所有文档（删除项目时级联删除）
    entities: Mapped[list["Entity"]] = relationship(back_populates="project", cascade="all, delete-orphan")  # 项目下所有实体（删除项目时级联删除）


class Document(Base, TimestampMixin):
    """文档表（``documents``）：项目中的一个源文档。

    记录源文件的元信息、内容快照与处理状态。同一文档可被多次解析并产生多个解析版本
    （见 ``DocumentParseVersion``），当前生效版本由 ``active_parse_version`` 指向
    （对应某个 ``document_parse_versions.version_key``）；解析结果被切分为若干内容块
    （见 ``DocumentChunk``），并记录针对该文档的流水线运行（见 ``PipelineRun``）。
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    title: Mapped[str] = mapped_column(String(255))  # 文档标题
    file_name: Mapped[str] = mapped_column(String(255))  # 原始文件名
    sha256: Mapped[str] = mapped_column(String(64), index=True)  # 文件内容的 SHA-256 哈希（用于去重与完整性校验）
    source_type: Mapped[str] = mapped_column(String(40), default="file")  # 来源类型（如 file / url），默认 "file"
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)  # 来源 URI（如原始下载地址，可空）
    raw_path: Mapped[str] = mapped_column(Text)  # 原始文件在本地存储中的落盘路径
    object_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # 对象存储中的对象键（若使用对象存储，可空）
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # 提取出的原始文本快照（可空）
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)  # 任意附加元数据（JSON），如上传者、来源信息等
    status: Mapped[str] = mapped_column(String(40), default=DocumentStatus.pending.value)  # 文档处理状态（见 DocumentStatus），默认 pending
    active_parse_version: Mapped[str | None] = mapped_column(String(128), nullable=True)  # 当前生效的解析版本号（对应 document_parse_versions.version_key，可空）

    project: Mapped["Project"] = relationship(back_populates="documents")  # 反向关联所属项目
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")  # 文档的全部内容分块（删除文档时级联删除）
    parse_versions: Mapped[list["DocumentParseVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )  # 文档的全部解析版本（删除文档时级联删除；依赖数据库 ON DELETE 被动处理）
    runs: Mapped[list["PipelineRun"]] = relationship(back_populates="document", cascade="all, delete-orphan")  # 针对该文档的流水线运行记录（删除文档时级联删除）


class DocumentParseVersion(Base, TimestampMixin):
    """文档解析版本表（``document_parse_versions``）。

    同一文档可能以不同解析器 / 参数被多次解析，每次解析产生一个独立版本，用
    ``version_key`` 标识（同一文档内唯一，见联合唯一约束）。每个版本保存解析产物目录
    （``artifact_dir``）、解析器信息、产物清单（``manifest_json``）、质量评估
    （``quality_json``）与各处理阶段状态（``stage_state``）；被激活为当前版本的记录
    通过 ``activated_at`` 记录激活时间。当前生效版本由 ``documents.active_parse_version``
    指向。删除文档时该表记录通过外键 ``ondelete="CASCADE"`` 级联删除。
    """

    __tablename__ = "document_parse_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "version_key",
            name="uq_document_parse_versions_document_version",
        ),  # 同一文档（document_id）内版本号（version_key）唯一
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )  # 所属文档 ID（文档删除时该版本随之级联删除）
    version_key: Mapped[str] = mapped_column(String(128))  # 版本标识符（如 "v1"、时间戳或哈希），同一文档内唯一
    status: Mapped[str] = mapped_column(String(40), default="queued")  # 解析状态（如 queued / processing / ready / failed），默认 queued
    artifact_dir: Mapped[str] = mapped_column(Text)  # 该版本解析产物的存储目录
    parser_name: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 使用的解析器名称（可空）
    parser_version: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 使用的解析器版本（可空）
    manifest_json: Mapped[dict] = mapped_column(JSON, default=dict)  # 解析产物清单（JSON）：各产物文件的路径与类型等
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)  # 解析质量评估结果（JSON）：分数、接受/拒绝等
    stage_state: Mapped[dict] = mapped_column(JSON, default=dict)  # 流水线各处理阶段的运行状态快照（JSON）
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 该版本被激活为当前版本的时间（可空）

    document: Mapped["Document"] = relationship(back_populates="parse_versions")  # 反向关联所属文档


def _embedding_text_default(context) -> str:
    """``DocumentChunk.embedding_text`` 列的默认值生成函数。

    当插入分块时未显式提供 ``embedding_text``，则回退使用同一条记录的 ``text`` 字段
    内容（通过 ``context.get_current_parameters()`` 读取当前插入参数中的 ``text``），
    从而保证用于向量化的文本始终非空。
    """
    return str(context.get_current_parameters().get("text") or "")


class DocumentChunk(Base, TimestampMixin):
    """文档内容分块表（``document_chunks``）：检索与向量化的基本单元。

    存储某个解析版本（``parse_version``）下文档被切分出的内容块。分块通过两条结构线索
    组织：

    1. 父子层级（``parent_chunk_id`` / ``chunk_role``）：父块代表较粗粒度的区块
       （如章节、表格整体），子块代表其内部的细粒度片段（如段落、表格行）。
       ``chunk_role`` 取 ``"parent"`` / ``"child"`` 标识块在层级中的角色；
       ``parent`` / ``children`` 为同一张表上的自引用关系。
    2. 阅读顺序（``previous_chunk_id`` / ``next_chunk_id``）：用双向链表记录块在文档中
       的先后顺序，便于按序拼接上下文。

    每个块还携带定位信息（``heading``、``page_label``、``section_path``、
    ``source_block_ids``、``source_spans``）、用于检索的文本（``text``、
    ``embedding_text``、``embedding``）以及可选的上下文增强信息（``contextual_prefix``
    及 contextualization_* 系列字段）。
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index(
            "ix_document_chunks_document_parse_version_role",
            "document_id",
            "parse_version",
            "chunk_role",
        ),  # 联合索引：按“文档 + 解析版本 + 块角色”快速筛选
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)  # 所属文档 ID
    parse_version: Mapped[str] = mapped_column(String(128), default="legacy")  # 该块所属的解析版本号（缺省为旧版 "legacy"）
    parent_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=True
    )  # 父块 ID（自引用外键，NULL 表示顶层块；父块删除时子块级联删除）
    chunk_role: Mapped[str] = mapped_column(String(20), default="child")  # 块角色："parent"（父块/粗粒度区块）或 "child"（子块/细粒度片段）
    block_type: Mapped[str] = mapped_column(String(40), default="narrative")  # 内容块类型：如 "narrative"（正文）/ "table"（表格）/ "figure"（插图）
    ordinal: Mapped[int] = mapped_column(Integer)  # 在文档（或父块）内的顺序号（用于排序）
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 块所属的章节标题（可空）
    page_label: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 页码标签（如 "第 3 页" / "Page 3"，可空）
    section_path: Mapped[list[str]] = mapped_column(JSON, default=list)  # 章节路径（JSON 字符串数组）：从根到当前块的标题链
    source_block_ids: Mapped[list[str]] = mapped_column(JSON, default=list)  # 对应源解析产物中的块 ID 列表（JSON 字符串数组）
    source_spans: Mapped[list[dict]] = mapped_column(JSON, default=list)  # 源文本位置区间列表（JSON dict 数组）：每项含行/字符区间、页面等信息，用于精确溯源定位
    text: Mapped[str] = mapped_column(Text)  # 块的纯文本内容（用于展示与拼接上下文）
    contextual_prefix: Mapped[str | None] = mapped_column(Text, nullable=True)  # 上下文增强后的前缀文本（可空，用于弥补孤立块的语义缺失）
    embedding_text: Mapped[str] = mapped_column(Text, default=_embedding_text_default)  # 用于生成向量的文本（未显式提供时默认回退为 text 字段内容）
    contextualization_model: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 生成上下文所用的模型（可空）
    contextualization_version: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 生成上下文所用的模型版本（可空）
    contextualization_prompt_version: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 生成上下文所用的提示词（prompt）版本（可空）
    contextualized_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 完成上下文增强的时间（可空）
    parser_name: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 产生本块的解析器名称（可空）
    parser_version: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 产生本块的解析器版本（可空）
    splitter_name: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 切分器名称（可空）
    splitter_version: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 切分器版本（可空）
    splitting_model: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 用于语义切分的模型（可空）
    semantic_boundary_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 语义切分边界处的置信分数（可空）
    token_count: Mapped[int] = mapped_column(Integer, default=0)  # 实际 token 计数
    previous_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )  # 阅读顺序中前一块的 ID（自引用，其被删除时置 NULL）
    next_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )  # 阅读顺序中后一块的 ID（自引用，其被删除时置 NULL）
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)  # 估算的 token 数（用于预算控制）
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)  # 向量化结果（浮点数组，JSON 存储；可空表示尚未生成）

    document: Mapped["Document"] = relationship(back_populates="chunks")  # 反向关联所属文档
    parent: Mapped["DocumentChunk | None"] = relationship(
        remote_side=[id],
        foreign_keys=[parent_chunk_id],
        back_populates="children",
    )  # 自引用父块关系：remote_side 指向本表主键 id，通过 parent_chunk_id 关联
    children: Mapped[list["DocumentChunk"]] = relationship(
        foreign_keys=[parent_chunk_id],
        back_populates="parent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )  # 自引用子块集合：通过 parent_chunk_id 关联，父块删除时级联删除子块
    previous_chunk: Mapped["DocumentChunk | None"] = relationship(
        remote_side=[id], foreign_keys=[previous_chunk_id]
    )  # 阅读顺序中的前一块（自引用，remote_side 指向自身主键 id）
    next_chunk: Mapped["DocumentChunk | None"] = relationship(
        remote_side=[id], foreign_keys=[next_chunk_id]
    )  # 阅读顺序中的后一块（自引用，remote_side 指向自身主键 id）


class Entity(Base, TimestampMixin):
    """实体表（``entities``）：从文档中抽取 / 归纳的知识实体（概念、术语等）。

    实体归属于某个项目（``project_id``），且同一项目内实体名唯一（见联合唯一约束）。
    ``canonical_page_id`` 指向规范化百科页面（canonical wiki page），将实体与权威
    条目关联起来。
    """

    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_entities_project_name"),)  # 同一项目内实体名称（name）唯一

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    name: Mapped[str] = mapped_column(String(255), index=True)  # 实体名称（项目内唯一）
    entity_type: Mapped[str] = mapped_column(String(80), default="concept")  # 实体类型（如 concept / person / organization），默认 "concept"
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)  # 实体的别名列表（JSON 字符串数组）
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)  # 实体摘要（可空）
    canonical_page_id: Mapped[str | None] = mapped_column(String(36), nullable=True)  # 关联的规范化百科页面 ID（可空）

    project: Mapped["Project"] = relationship(back_populates="entities")  # 反向关联所属项目


class Claim(Base, TimestampMixin):
    """论断表（``claims``）：从文档中抽取的 (主语, 谓词, 宾语) 三元组式论断。

    每个论断关联来源文档与证据块（``evidence_chunk_id``），带抽取置信度
    （``confidence``）与验证状态（``verification_status``），供知识校验与问答使用。
    """

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)  # 来源文档 ID
    subject: Mapped[str] = mapped_column(String(255))  # 论断主语（subject）
    predicate: Mapped[str] = mapped_column(String(255))  # 论断谓词（predicate）
    object_text: Mapped[str] = mapped_column(Text)  # 论断宾语（object）文本
    evidence_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)  # 支撑该论断的证据块 ID（可空）
    confidence: Mapped[float] = mapped_column(Float, default=0.5)  # 抽取置信度（0~1），默认 0.5
    verification_status: Mapped[str] = mapped_column(String(40), default="unverified")  # 验证状态（如 unverified / verified / disputed），默认 unverified
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)  # 任意附加元数据（JSON）


class KnowledgeEdge(Base, TimestampMixin):
    """轻量知识关系边（``knowledge_edges``）。

    这是 SAC-KG 的可审计扩展，不是独立的图数据库：边两端可以是论文、实体、
    论断或证据块，且始终绑定项目、解析版本和可选的原文证据。比较工作台只在
    查询涉及的文档范围内按需写入/复用这些边；当 SAC-KG 关闭时，仅允许记录
    不含语义推断的论文配对 ``compared_with`` 元关系。
    """

    __tablename__ = "knowledge_edges"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source_type",
            "source_id",
            "relation_type",
            "target_type",
            "target_id",
            "parse_version",
            name="uq_knowledge_edges_identity",
        ),
        Index("ix_knowledge_edges_project_relation", "project_id", "relation_type"),
        Index("ix_knowledge_edges_document_version", "document_id", "parse_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(40))
    source_id: Mapped[str] = mapped_column(String(255))
    relation_type: Mapped[str] = mapped_column(String(60))
    target_type: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[str] = mapped_column(String(255))
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True
    )
    parse_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    evidence_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    extraction_version: Mapped[str] = mapped_column(String(80), default="comparison-v1")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ReviewItem(Base, TimestampMixin):
    """复核项表（``review_items``）：人工质量复核队列中的待处理项。

    记录处理过程中发现的可疑问题（如解析异常、低置信度论断、内容质量问题等），
    可关联文档（``document_id``）或论断（``claim_id``），并标注严重程度
    （``severity``，见 ReviewSeverity）与处理状态（``status``，见 ReviewStatus）。
    """

    __tablename__ = "review_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)  # 关联的文档 ID（可空）
    claim_id: Mapped[str | None] = mapped_column(ForeignKey("claims.id"), nullable=True)  # 关联的论断 ID（可空）
    title: Mapped[str] = mapped_column(String(255))  # 复核项标题（问题概要）
    detail: Mapped[str] = mapped_column(Text)  # 复核项详细说明
    severity: Mapped[str] = mapped_column(String(20), default=ReviewSeverity.medium.value)  # 严重程度（见 ReviewSeverity），默认 medium
    status: Mapped[str] = mapped_column(String(20), default=ReviewStatus.pending.value)  # 处理状态（见 ReviewStatus），默认 pending
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # 附加数据（JSON）：相关上下文、原始错误信息等


class PipelineRun(Base, TimestampMixin):
    """流水线运行表（``pipeline_runs``）：一次处理 / 查询任务的执行记录。

    ``run_type``（见 RunType）区分任务类型（摄取、查询、重建、校验），``status``
    （见 RunStatus）记录运行状态；``provider_report`` 保存服务商返回的报告 / 指标，
    ``notes`` 存放人类可读备注。
    """

    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)  # 关联的文档 ID（可空，如重建任务不针对单一文档）
    run_type: Mapped[str] = mapped_column(String(40), default=RunType.ingest.value)  # 运行类型（见 RunType），默认 ingest
    status: Mapped[str] = mapped_column(String(20), default=RunStatus.queued.value)  # 运行状态（见 RunStatus），默认 queued
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)  # 备注信息（可空）
    provider_report: Mapped[dict] = mapped_column(JSON, default=dict)  # 服务商运行报告 / 指标（JSON）

    document: Mapped["Document | None"] = relationship(back_populates="runs")  # 反向关联文档（可空）


class QuestionAnswer(Base, TimestampMixin):
    """问答记录表（``question_answers``）：已保存的用户提问与系统回答。

    保存问题、Markdown 格式回答、引用来源（``citations``）、风险等级
    （``risk_level``）与验证状态（``verification_status``），用于事后审计与回放。
    """

    __tablename__ = "question_answers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    question: Mapped[str] = mapped_column(Text)  # 用户提问内容
    answer_markdown: Mapped[str] = mapped_column(Text)  # 系统回答（Markdown 格式）
    citations: Mapped[list[dict]] = mapped_column(JSON, default=list)  # 引用来源列表（JSON dict 数组）
    risk_level: Mapped[str] = mapped_column(String(20), default="normal")  # 回答风险等级（如 normal / high），默认 normal
    verification_status: Mapped[str] = mapped_column(String(40), default="local-only")  # 验证状态（如 local-only / verified），默认 local-only


class ConversationTurn(Base):
    """会话轮次表（``conversation_turns``）：一条 Agent 对话会话中的单个轮次。

    会话由 ``session_id`` 标识，轮次通过 ``turn_index`` 排序。``role`` 标识角色
    （"user" / "agent" / "tool"）；若该轮为工具调用，则 ``tool_name`` / ``tool_args``
    / ``tool_result`` 记录调用详情，``step_type`` 记录 Agent 步骤类型。该表自带
    ``created_at``，因此未继承 TimestampMixin。
    """

    __tablename__ = "conversation_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # 所属会话 ID（索引）
    turn_index: Mapped[int] = mapped_column(Integer)  # 轮次序号（会话内递增）
    role: Mapped[str] = mapped_column(String(20))  # 角色："user" | "agent" | "tool"
    content: Mapped[str] = mapped_column(Text)  # 该轮内容（用户提问 / Agent 回答 / 工具输出等）
    tool_name: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 工具名称（仅工具调用轮次，可空）
    tool_args: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # 工具调用参数（JSON，可空）
    tool_result: Mapped[str | None] = mapped_column(Text, nullable=True)  # 工具调用返回结果（可空）
    step_type: Mapped[str | None] = mapped_column(String(40), nullable=True)  # Agent 步骤类型（可空）
    citations: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)  # 该轮引用的来源列表（JSON，可空）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 创建时间


class ConversationSession(Base):
    """对话会话表（``conversation_sessions``）：Agent 对话会话的元信息。

    会话归属用户（``owner_user_id``）并限定在某个项目（``project_slug``）及可选文档
    （``document_id``）范围内；``expires_at`` 决定会话过期时间。会话下可挂载临时附件
    （见 SessionAttachment）。
    """

    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 主键：会话 ID（由调用方生成，64 字符内）
    owner_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)  # 所属用户 ID（可空，索引）
    project_slug: Mapped[str] = mapped_column(String(120), index=True)  # 所属项目 slug（限定会话范围）
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)  # 限定到的文档 ID（可空）
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)  # 会话过期时间（索引）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 最近更新时间

    attachments: Mapped[list["SessionAttachment"]] = relationship(back_populates="session", cascade="all, delete-orphan")  # 会话下的全部临时附件（删除会话时级联删除）


class SessionAttachment(Base, TimestampMixin):
    """会话附件表（``session_attachments``）：挂载到对话会话上的临时文件。

    用于在会话中上传供 Agent 参考的临时材料（不进入正式文档管线）。记录文件元信息
    （文件名、SHA-256、字节数）与存储路径，状态由 ``status`` 管理；其内容被切分为
    若干块（见 SessionAttachmentChunk）。
    """

    __tablename__ = "session_attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    session_id: Mapped[str] = mapped_column(ForeignKey("conversation_sessions.id"), index=True)  # 所属会话 ID
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)  # 所属项目 ID
    file_name: Mapped[str] = mapped_column(String(255))  # 原始文件名
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 附件显示标题（可空）
    storage_path: Mapped[str] = mapped_column(Text)  # 附件在存储中的路径
    sha256: Mapped[str] = mapped_column(String(64), index=True)  # 文件内容的 SHA-256 哈希（索引，用于去重）
    byte_size: Mapped[int] = mapped_column(Integer, default=0)  # 文件字节数
    status: Mapped[str] = mapped_column(String(40), default="ready")  # 附件状态（如 ready / processing），默认 ready

    session: Mapped["ConversationSession"] = relationship(back_populates="attachments")  # 反向关联所属会话
    chunks: Mapped[list["SessionAttachmentChunk"]] = relationship(back_populates="attachment", cascade="all, delete-orphan")  # 附件的全部内容分块（删除附件时级联删除）


class SessionAttachmentChunk(Base, TimestampMixin):
    """会话附件分块表（``session_attachment_chunks``）：附件内容切分后的块。

    每个块保存文本、顺序号、章节标题与页码标签，并估算 token 数（``token_estimate``），
    供会话检索 / 上下文拼接使用。
    """

    __tablename__ = "session_attachment_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    attachment_id: Mapped[str] = mapped_column(ForeignKey("session_attachments.id"), index=True)  # 所属附件 ID
    ordinal: Mapped[int] = mapped_column(Integer)  # 在附件内的顺序号
    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 块所属标题（可空）
    page_label: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 页码标签（可空）
    text: Mapped[str] = mapped_column(Text)  # 块文本内容
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)  # 估算的 token 数

    attachment: Mapped["SessionAttachment"] = relationship(back_populates="chunks")  # 反向关联所属附件


class AgentTraceRun(Base):
    """Agent 执行轨迹表（``agent_trace_runs``）：一次 Agent 查询执行的完整轨迹。

    记录一次请求（``request_id``）在指定会话 / 项目内的执行概况：路由选择（``route``）、
    最终回答（``final_answer``）、引用、告警、性能指标（``latency_ms``、各 token 计数、
    工具调用数、步骤数）以及所用模型信息（``provider`` / ``model``）。具体步骤存放在
    子表 AgentTraceStep 中。该表自带 ``created_at``，未继承 TimestampMixin。
    """

    __tablename__ = "agent_trace_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    owner_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)  # 所属用户 ID（可空，索引）
    request_id: Mapped[str] = mapped_column(String(36), index=True)  # 本次执行的请求 ID（索引）
    session_id: Mapped[str] = mapped_column(String(64), index=True)  # 所属对话会话 ID（索引）
    project_slug: Mapped[str] = mapped_column(String(120))  # 所属项目 slug
    query: Mapped[str] = mapped_column(Text)  # 用户查询内容
    constraints: Mapped[dict] = mapped_column(JSON, default=dict)  # 本次执行的约束（JSON）：最大步数 / 工具数 / token 预算等
    route: Mapped[str | None] = mapped_column(String(40), nullable=True)  # 路由决策结果（可空）
    final_answer: Mapped[str] = mapped_column(Text)  # 最终回答内容
    citations: Mapped[list[dict]] = mapped_column(JSON, default=list)  # 回答引用的来源列表（JSON）
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)  # 执行过程中的告警信息列表（JSON 字符串数组）
    status: Mapped[str] = mapped_column(String(20))  # 执行状态（如 completed / error / timeout）
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)  # 总耗时（毫秒）
    provider: Mapped[str] = mapped_column(String(40), default="local")  # 使用的模型提供商（如 local / openai），默认 "local"
    model: Mapped[str] = mapped_column(String(120), default="local-fallback")  # 实际使用的模型名（默认 "local-fallback"）
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)  # 提示词（prompt）token 数
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)  # 生成（completion）token 数
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)  # 工具调用次数
    step_count: Mapped[int] = mapped_column(Integer, default=0)  # 步骤数
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 创建时间

    steps: Mapped[list["AgentTraceStep"]] = relationship(back_populates="run", cascade="all, delete-orphan",
                                                          order_by="AgentTraceStep.step_id")  # 该次执行的全部步骤（按 step_id 排序，删除轨迹时级联删除）


class AgentTraceStep(Base):
    """Agent 轨迹步骤表（``agent_trace_steps``）：一次 Agent 执行中的单个步骤。

    记录步骤类型（``step_type``，如 route / retrieve / tool_call / synthesize /
    finalize 等）、摘要、耗时、工具调用信息（``tool_name`` / ``tool_ok``）与任意元数据
    （``metadata_json``）。``step_id`` 为运行内递增序号。
    """

    __tablename__ = "agent_trace_steps"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_trace_runs.id"), index=True)  # 所属执行轨迹 ID
    step_id: Mapped[int] = mapped_column(Integer)  # 运行内递增的步骤序号
    step_type: Mapped[str] = mapped_column(String(40))  # 步骤类型（如 route / retrieve / tool_call / synthesize / finalize）
    summary: Mapped[str] = mapped_column(Text)  # 步骤摘要说明
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 该步骤耗时（毫秒，可空）
    tool_name: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 调用的工具名称（可空）
    tool_ok: Mapped[bool | None] = mapped_column(nullable=True)  # 工具调用是否成功（可空）
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)  # 步骤附加元数据（JSON）

    run: Mapped["AgentTraceRun"] = relationship(back_populates="steps")  # 反向关联所属执行轨迹


class User(Base):
    """用户表（``users``）：应用用户，基于飞书（Feishu/Lark）开放平台 SSO 认证。

    以飞书开放 ID（``feishu_open_id``，唯一）标识用户，并保存联合 ID
    （``feishu_union_id``）、租户键（``tenant_key``）与展示信息；``is_active`` 控制
    账号可用状态。该表自带 created_at / updated_at，未继承 TimestampMixin。
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    feishu_open_id: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)  # 飞书 open_id（唯一，非空，登录主标识）
    feishu_union_id: Mapped[str | None] = mapped_column(String(120), index=True, nullable=True)  # 飞书 union_id（跨应用统一 ID，可空）
    tenant_key: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 飞书租户键（tenant key，可空）
    display_name: Mapped[str] = mapped_column(String(255), default="")  # 显示名称（默认空字符串）
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # 头像 URL（可空）
    is_active: Mapped[bool] = mapped_column(default=True)  # 账号是否启用（默认 True）
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 最近登录时间（可空）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )  # 最近更新时间


class AuthSession(Base):
    """认证会话表（``auth_sessions``）：用户登录后创建的认证会话。

    保存会话令牌哈希（``token_hash``，唯一）与 CSRF 令牌哈希（``csrf_hash``），设置
    过期时间（``expires_at``），并通过 ``revoked_at`` 支持会话吊销。出于安全考虑，只
    存储哈希而非明文令牌。
    """

    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)  # 主键（UUID 字符串）
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)  # 所属用户 ID（非空，索引）
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)  # 会话令牌的哈希（唯一，非空，仅存哈希）
    csrf_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # CSRF 令牌的哈希（非空）
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)  # 会话过期时间（非空，索引）
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 最近活跃时间（可空）
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 吊销时间（非空表示该会话已吊销）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )  # 最近更新时间

    user: Mapped["User"] = relationship()  # 关联所属用户
