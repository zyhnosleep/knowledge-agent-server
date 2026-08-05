"""项目级公共 API 的 Pydantic Schema 定义。

本模块集中定义普通（非 Agent）REST API 使用到的公共数据模型，包括：

- 项目与文档管理：``ProjectCreate`` / ``ProjectRead`` / ``DocumentRead``、
  摄取响应 ``IngestResponse``。
- 解析进度与质量：``ParseProgress`` / ``ParseQualitySummary`` /
  ``CanonicalParseRead`` / ``CanonicalMarkdownRead``。
- 溯源定位：``PublicSourceMetadata`` / ``PublicSourceSpan`` /
  ``CitationLocationRead``。
- 查询问答：``QueryRequest`` / ``QueryResponse`` / ``Citation``。
- 复核项与健康检查：``ReviewItemRead`` / ``HealthResponse``。

其中 ``PublicSourceSpan`` 携带对 ``bbox`` / ``normalized_bbox`` / ``heading_path``
及行、字符区间的前置/交叉校验器，保证溯源定位数据始终合法、有界。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProjectCreate(BaseModel):
    """创建项目（``POST /api/projects``）的请求体。"""

    slug: str  # 唯一项目标识（必填）
    name: str  # 项目名称（必填）
    description: str | None = None  # 项目描述（可空）


class ProjectRead(BaseModel):
    """项目的只读视图（用于项目列表 / 详情响应）。"""

    id: str  # 项目 ID
    slug: str  # 项目 slug
    name: str  # 项目名称
    description: str | None = None  # 项目描述（可空）


class IngestResponse(BaseModel):
    """文档摄取请求（如 ``POST /api/ingest``）的响应体。"""

    document_id: str  # 生成的文档 ID
    project_id: str  # 所属项目 ID
    project_slug: str  # 所属项目 slug
    run_id: str  # 触发的流水线运行 ID
    status: str  # 初始处理状态（如 pending / processing）
    document_title: str  # 文档标题


class DocumentRead(BaseModel):
    """文档的只读视图（用于文档列表 / 详情响应）。"""

    id: str  # 文档 ID
    title: str  # 文档标题
    file_name: str  # 原始文件名
    status: str  # 文档处理状态（见 models.DocumentStatus）
    sha256: str  # 文件内容 SHA-256 哈希
    metadata_json: dict[str, Any] = Field(default_factory=dict)  # 文档元数据（JSON）


class ParseProgress(BaseModel):
    """解析进度信息（阶段名 + 完成百分比）。"""

    stage: str  # 当前阶段名称
    percent: int = Field(ge=0, le=100)  # 完成百分比（0~100）


class ParseQualitySummary(BaseModel):
    """解析质量评估摘要。"""

    status: str | None = None  # 质量评估状态（可空）
    accepted: bool | None = None  # 是否接受该解析版本（可空）
    score: float | None = None  # 质量分数（可空）


class CanonicalParseRead(BaseModel):
    """规范化解析版本的只读详情（用于解析状态 / 进度查询响应）。"""

    document_id: str  # 文档 ID
    version: str  # 解析版本号
    parser: str | None = None  # 解析器名称（可空）
    parser_version: str | None = None  # 解析器版本（可空）
    progress: ParseProgress  # 解析进度
    quality: ParseQualitySummary  # 解析质量摘要
    repair_pages: list[int] = Field(default_factory=list)  # 需要 / 已修复的页码列表
    warning_count: int = Field(ge=0)  # 告警数量（非负）
    download_available: bool  # 解析产物是否可下载


class CanonicalMarkdownRead(BaseModel):
    """规范化解析版本的 Markdown 内容（用于导出 / 下载响应）。"""

    document_id: str  # 文档 ID
    version: str  # 解析版本号
    markdown: str  # Markdown 文本内容


class PublicSourceMetadata(BaseModel):
    """源对象元数据（溯源用）。

    采用严格模式且忽略多余字段（``extra="ignore"``），仅接受有限的、有界长度字段，
    防止客户端构造超大或多余的元数据。
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    table_id: str | None = Field(default=None, max_length=512)  # 表格 ID（可空，最长 512）
    figure_id: str | None = Field(default=None, max_length=512)  # 插图 ID（可空，最长 512）
    formula_id: str | None = Field(default=None, max_length=512)  # 公式 ID（可空，最长 512）
    asset_id: str | None = Field(default=None, max_length=512)  # 资源 ID（可空，最长 512）
    source_role: str | None = Field(default=None, max_length=128)  # 源对象角色（可空，最长 128）
    structure_type: str | None = Field(default=None, max_length=128)  # 结构类型（可空，最长 128）


class PublicSourceSpan(BaseModel):
    """公开的源文本位置区间（span），用于精确溯源定位。

    描述证据 / 引用在原始文档中的物理位置，包括页码、边界框（bbox）、行 / 字符区间、
    关联的源块 / 表格 / 插图等。所有字段均有长度或取值约束；``bbox`` /
    ``normalized_bbox`` / ``heading_path`` 以及行 / 字符区间在构造时经校验器校验。
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    page_index: int | None = Field(default=None, ge=0)  # 页码索引（从 0 开始，可空）
    page_label: str | None = Field(default=None, max_length=128)  # 页码标签（可空，最长 128）
    bbox: tuple[float, float, float, float] | None = None  # 物理边界框 (x0, y0, x1, y1)（可空）
    normalized_bbox: tuple[float, float, float, float] | None = None  # 归一化边界框（坐标位于 0~1，可空）
    source_block_id: str | None = Field(default=None, max_length=512)  # 源块 ID（可空，最长 512）
    paragraph_id: str | None = Field(default=None, max_length=512)  # 段落 ID（可空，最长 512）
    table_id: str | None = Field(default=None, max_length=512)  # 表格 ID（可空，最长 512）
    row_index: int | None = Field(default=None, ge=0)  # 表格行索引（可空）
    column_index: int | None = Field(default=None, ge=0)  # 表格列索引（可空）
    image_relationship_id: str | None = Field(default=None, max_length=512)  # 图像关系 ID（可空，最长 512）
    xpath: str | None = Field(default=None, max_length=4096)  # XML XPath 定位（可空，最长 4096）
    css_selector: str | None = Field(default=None, max_length=4096)  # CSS 选择器定位（可空，最长 4096）
    element_id: str | None = Field(default=None, max_length=512)  # 元素 ID（可空，最长 512）
    heading_path: list[str] = Field(default_factory=list, max_length=64)  # 标题路径（元素个数不超过 64）
    line_start: int | None = Field(default=None, ge=0)  # 起始行号（可空）
    line_end: int | None = Field(default=None, ge=0)  # 结束行号（可空）
    char_start: int | None = Field(default=None, ge=0)  # 起始字符偏移（可空）
    char_end: int | None = Field(default=None, ge=0)  # 结束字符偏移（可空）
    metadata: PublicSourceMetadata = Field(default_factory=PublicSourceMetadata)  # 源对象元数据

    @field_validator("bbox", "normalized_bbox", mode="before")
    @classmethod
    def validate_bbox(cls, value):
        """前置校验 bbox / normalized_bbox：必须是 4 个有限数值且坐标顺序正确。"""
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain exactly four coordinates")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            for coordinate in value
        ):
            raise ValueError("bbox coordinates must be finite numbers")
        x0, y0, x1, y1 = (float(coordinate) for coordinate in value)
        if x1 < x0 or y1 < y0:
            raise ValueError("bbox coordinates are out of order")
        return (x0, y0, x1, y1)

    @field_validator("normalized_bbox")
    @classmethod
    def validate_normalized_bbox(cls, value):
        """校验归一化 bbox 坐标必须落在 0..1 区间。"""
        if value is not None and any(coordinate < 0 or coordinate > 1 for coordinate in value):
            raise ValueError("normalized bbox coordinates must be within 0..1")
        return value

    @field_validator("heading_path")
    @classmethod
    def validate_heading_path(cls, value: list[str]) -> list[str]:
        """校验标题路径中的每一项都是长度受限的字符串。"""
        if any(not isinstance(item, str) or len(item) > 512 for item in value):
            raise ValueError("heading path entries must be bounded strings")
        return value

    @model_validator(mode="after")
    def validate_ranges(self):
        """交叉校验：行 / 字符区间必须成对出现且保持顺序。"""
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("line range must contain both endpoints")
        if self.line_start is not None and self.line_end < self.line_start:
            raise ValueError("line range is out of order")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("character range must contain both endpoints")
        if self.char_start is not None and self.char_end < self.char_start:
            raise ValueError("character range is out of order")
        return self


class CitationLocationRead(BaseModel):
    """引用定位信息：一条引用的来源位置详情。"""

    document_id: str  # 文档 ID
    chunk_id: str  # 分块 ID
    parse_version: str  # 解析版本号
    source_type: str  # 来源类型
    source_url: str  # 来源 URL
    source_spans: list[PublicSourceSpan] = Field(default_factory=list)  # 源文本位置区间列表


class QueryRequest(BaseModel):
    """普通查询接口（``POST /api/query``）的请求体。"""

    project_slug: str  # 目标项目 slug（必填）
    question: str  # 用户问题（必填）
    save_answer: bool = True  # 是否保存回答记录（默认 True）
    document_id: str | None = None  # 限定文档 ID（可空）


class Citation(BaseModel):
    """一条引用来源（回答 / 证据中引用的文档位置）。"""

    document_id: str | None = None  # 文档 ID（可空）
    chunk_id: str | None = None  # 分块 ID（可空）
    attachment_id: str | None = None  # 会话附件 ID（可空）
    page_slug: str | None = None  # 规范化页面 slug（可空）
    page_title: str | None = None  # 页面标题（可空）
    page_kind: str | None = None  # 页面类型（可空）
    score: float  # 相关度分数
    page_label: str | None = None  # 页码标签（可空）
    excerpt: str  # 引用摘录文本
    parse_version: str | None = None  # 解析版本（可空）
    parent_chunk_id: str | None = None  # 父块 ID（可空）
    block_type: str | None = None  # 块类型（可空）
    source_spans: list[dict[str, Any]] = Field(default_factory=list)  # 源位置区间列表
    asset_id: str | None = None  # 资源 ID（可空）
    table_id: str | None = None  # 表格 ID（可空）
    figure_id: str | None = None  # 插图 ID（可空）
    formula_id: str | None = None  # 公式 ID（可空）


class QueryResponse(BaseModel):
    """普通查询接口的响应体。"""

    answer_markdown: str  # 回答（Markdown 格式）
    citations: list[Citation]  # 引用来源列表
    verification_status: str  # 回答验证状态


class ReviewItemRead(BaseModel):
    """复核项的只读视图（用于复核列表 / 详情响应）。"""

    id: str  # 复核项 ID
    title: str  # 标题
    detail: str  # 详细说明
    severity: str  # 严重程度（见 models.ReviewSeverity）
    status: str  # 处理状态（见 models.ReviewStatus）
    payload: dict[str, Any] = Field(default_factory=dict)  # 附加数据（JSON）


class HealthResponse(BaseModel):
    """健康检查接口（``GET /api/health``）的响应体。"""

    status: str  # 总体健康状态（如 ok / degraded）
    app_name: str  # 应用名称
    api_status: str = "ok"  # API 状态（默认 "ok"）
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)  # 各模型运行状态（按模型名）
    queues: dict[str, dict[str, int]] = Field(default_factory=dict)  # 各队列状态（按队列名，值为整数指标）
