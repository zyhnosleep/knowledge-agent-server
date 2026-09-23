"""
pipeline.py —— Ingestion 管线编排（canonical 阶段机 + 知识抽取）模块
===================================================================

职责：
- 编排文档 ingestion 全流程，支持两套路径：
  1. **canonical 阶段机路径**（主路径）：文档经 parse -> repair ->
     canonicalize -> semantic_split -> contextualize -> embed -> index ->
     activate 各阶段，每个阶段以持久化检查点（artifact）方式推进，
     阶段执行器经 ``ingestion_stage_handlers()`` 暴露给
     ``ingestion_stages.IngestionStageRunner``。
  2. **legacy 路径**：无 Redis 队列时的同步兜底（``_process_document_legacy``），
     解析 -> 分块 -> 嵌入 -> SAC-KG 知识抽取（实体/三元组/审阅项）。
- 提供文档注册（register_document）、解析版本管理、SAC-KG 抽取
  （以文档标题等候选 head 驱动模型生成三元组，经本地/外部验证过滤）、
  实体增长决策（grow/keep/prune）与审阅项生成。

核心概念：
- **版本化分块**：所有分块按 (document_id, parse_version) 版本隔离，
  向量索引同样版本化，激活（activate）时整体切换当前版本。
- **Stage artifact**：阶段产物以带 SHA-256 指纹的 JSON 落盘
  （``_write_stage_artifact`` / ``_load_previous_stage_artifact``），
  阶段间通过检查点传递并校验指纹，保证可重放与防篡改。
- **身份校验**：``validate_ingestion_identity`` 在语义切分等阶段校验
  当前配置与入队时的配置快照一致，防止新旧配置混用。

模块级常量说明：
- ``FACT_MARKERS`` / ``FACT_VALUE_PATTERN``：抽取关键事实的启发式。
- ``GROW_ENTITY_TYPES`` / ``PRUNE_ENTITY_TYPES``：实体增长决策类型。
- ``HEAD_*``：候选 head 生成相关的数量/预算常量。
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import (
    Claim,
    Document,
    DocumentChunk,
    DocumentParseVersion,
    DocumentStatus,
    Entity,
    PipelineRun,
    Project,
    ReviewItem,
    ReviewSeverity,
    ReviewStatus,
    RunStatus,
    RunType,
)
from app.services.ai import (
    DocumentAnalysisPayload,
    DocumentExtraction,
    DeepSeekClient,
    ExternalVerifier,
    ExtractedClaim,
    ExtractedEntity,
    GeneratedTriple,
    GrowthDecision,
    GrowthDecisionPayload,
    HeadAnalysisPayload,
    OllamaClient,
    cosine_similarity,
    safe_model_call,
)
from app.services.filesystem import compute_sha256, display_title_from_path, looks_like_internal_sample, readable_title_from_path, slugify, strip_upload_prefix
from app.services.contextualization_policy import (
    requires_contextualization,
    valid_contextualized_embedding,
    valid_plain_embedding,
)
from app.services.paper_profile import (
    ensure_paper_profile,
    ensure_source_identity,
    prepare_canonical_profile_source,
)
from app.services.ingestion_identity import (
    build_ingestion_config_snapshot,
    build_parse_version_key,
    canonical_ingestion_config_hash,
    require_matching_ingestion_config,
)
from app.services.parser import parse_document
from app.services.parse_versions import ActivationError, ParseVersionService
from app.services.repositories import get_or_create_project
from app.services.storage import ObjectStorage
from app.services.vector_store import ChunkVector, get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

FACT_MARKERS = (
    "建议",
    "复查",
    "随访",
    "诊断",
    "用药",
    "剂量",
    "治疗",
    "检查",
    "结果",
    "结论",
    "风险",
    "时间",
    "recommended",
    "recommendation",
    "follow-up",
    "follow up",
    "diagnosis",
    "medication",
    "dose",
    "treatment",
    "result",
    "conclusion",
    "risk",
)
FACT_VALUE_PATTERN = re.compile(
    r"(\d+\s*个?\s*(天|周|月|年|小时|分钟|mg|g|ml|%|次|days?|weeks?|months?|years?|hours?)|"
    r"[一二三四五六七八九十两]+个?(天|周|月|年|小时|分钟))",
    re.IGNORECASE,
)
GROW_ENTITY_TYPES = {
    "condition",
    "component",
    "concept",
    "disease",
    "drug",
    "entity",
    "institution",
    "method",
    "model",
    "module",
    "organization",
    "person",
    "procedure",
    "test",
    "therapy",
    "treatment",
}
PRUNE_ENTITY_TYPES = {"date", "time", "dose", "dosage", "measurement", "number", "value"}
GENERIC_TRIPLE_EXAMPLES = [
    {"subject": "Disease", "predicate": "requires_follow_up", "object_text": "scheduled monitoring"},
    {"subject": "Medication", "predicate": "has_dosage", "object_text": "specific dose guidance"},
    {"subject": "Treatment Plan", "predicate": "includes", "object_text": "follow-up examination"},
    {"subject": "Clinical Finding", "predicate": "supports", "object_text": "diagnosis"},
    {"subject": "Project", "predicate": "documents", "object_text": "key recommendation"},
]
HEAD_MAX_COUNT = 10
HEAD_REPROMPT_ERROR_THRESHOLD = 3
HEAD_SNIPPET_LIMIT = 6
HEAD_CONTEXT_CHAR_BUDGET = 3600


def _span_structured_id(span: Any, field: str) -> str | None:
    """从持久化的 source_span JSON 中读取结构 ID（table_id/figure_id/formula_id）。

    结构化分块会经 ``_annotate_structure_spans`` 把结构 ID 写入 span 的
    metadata（不覆盖 span 自身的顶层字段），因此同时检查 metadata 与顶层字段。
    """
    if not isinstance(span, dict):
        return None
    metadata = span.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(field)
        if isinstance(value, str) and value:
            return value
    value = span.get(field)
    if isinstance(value, str) and value:
        return value
    return None


def _is_footnote_only_chunk(metadata: Any) -> bool:
    """判断表格子块是否为"仅脚注"块（无数据行、仅引用上下文）。

    结构化证据构建器（``structured_evidence._missing_table_footnote_chunks``）
    为未被常规子块覆盖的脚注生成独立子块，其 metadata 携带脚注契约键
    ``footnote_index``（另含 ``footnote_part_index``/``footnote_part_count``）。
    这类子块为支撑引用上下文会携带完整 ``row_indices``，但本身不含任何
    数据行，因此不能参与行覆盖判定。
    """
    return isinstance(metadata, dict) and "footnote_index" in metadata


def canonical_tables_to_document_intelligence(tables: list[Any]) -> list[dict[str, str | None]]:
    """把 CanonicalTable 列表组装为 ``document_intelligence.tables`` 条目。

    检索表格证据（``QueryService._search_document_table_contexts``）只读取
    ``documents.metadata_json["document_intelligence"]["tables"]``；canonical-v4
    的表格数据以 table chunks 入库后从未写回该字段（2026-08-12 诊断：charmm36
    文档该字段为空，导致表格查询全部拿不到表格证据）。此处与旧 DI/mineru
    摄取路径（parser.py 的 tables 清单）保持一致：每项
    ``{"markdown": ..., "page_label": ...}``，markdown 优先取 source_markdown
    （caption + 规范化表体），退到 normalized_markdown。
    """
    entries: list[dict[str, str | None]] = []
    for table in tables:
        markdown = (table.source_markdown or table.normalized_markdown or "").strip()
        if not markdown:
            continue
        page_label = None
        for span in table.source_spans or []:
            if getattr(span, "page_label", None):
                page_label = str(span.page_label).strip()
                break
        entries.append({"markdown": markdown, "page_label": page_label})
    return entries


def _is_valid_row_index(value: Any) -> bool:
    """是否为合法的非负整数行下标。

    Python 中 ``bool`` 是 ``int`` 的子类，``isinstance(True, int)`` 成立；
    若不显式排除，``True/False`` 会被当作行 1/0 计入覆盖。
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def _normalize_row_indices(values: list[Any]) -> set[int]:
    """规范化行下标集合：仅保留合法的非负整数，忽略布尔值与其他噪音。"""
    return {int(value) for value in values if _is_valid_row_index(value)}


def derive_child_inventory_from_payload(payload: list[Any]) -> dict[str, Any]:
    """从 index 阶段产物（``index_payload.json`` 记录列表）派生同构 Child 库存。

    输出：``{"tables": {table_id: {"child_ids", "parent_ids", "child_count"}},
    "figure_ids": [...], "formula_ids": [...], "row_indices": {table_id: [...]},
    "orphan_structured_chunks": [...]}``。
    重复 local_id 会抛错（fail-closed）。行覆盖只统计子块，父块的行集合不能
    掩盖缺失的 child 行；仅脚注子块（``footnote_index`` 契约）不含数据行，
    也不贡献行覆盖。无 table_id 的结构化分块被记为孤儿（保留 local_id），
    激活 gate 会据此失败关闭。
    """
    tables: dict[str, dict[str, Any]] = {}
    figures: set[str] = set()
    formulas: set[str] = set()
    row_indices: dict[str, set[int]] = {}
    orphans: set[str] = set()
    seen_local_ids: set[str] = set()
    for record in payload:
        if not isinstance(record, dict):
            raise RuntimeError("Typed inventory requires indexed chunk records.")
        chunk = record.get("chunk")
        if not isinstance(chunk, dict):
            raise RuntimeError("Typed inventory requires a chunk object per record.")
        local_id = chunk.get("local_id")
        if not isinstance(local_id, str) or not local_id:
            raise RuntimeError("Typed inventory requires non-empty chunk local IDs.")
        if local_id in seen_local_ids:
            raise RuntimeError(
                f"Typed inventory contains duplicate chunk local ID {local_id!r}."
            )
        seen_local_ids.add(local_id)
        block_type = chunk.get("block_type")
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        chunk_role = chunk.get("chunk_role")
        if block_type == "table":
            table_id = metadata.get("table_id")
            if not isinstance(table_id, str) or not table_id:
                # 无结构归属的结构化分块是孤儿：保留 local_id，激活时与
                # DB 派生库存比较并失败关闭，而不是静默丢弃。
                orphans.add(local_id)
                continue
            entry = tables.setdefault(
                table_id,
                {"child_ids": set(), "parent_ids": set()},
            )
            if chunk_role == "child":
                entry["child_ids"].add(local_id)
                # 行覆盖只统计真正的数据行子块；脚注子块（现有脚注元数据契约）
                # 携带完整 row_indices 但无数据行，不参与覆盖。
                if not _is_footnote_only_chunk(metadata):
                    rows = metadata.get("row_indices")
                    if isinstance(rows, list):
                        row_indices.setdefault(table_id, set()).update(
                            _normalize_row_indices(rows)
                        )
            elif chunk_role == "parent":
                entry["parent_ids"].add(local_id)
        elif block_type == "figure":
            figure_id = metadata.get("figure_id")
            if isinstance(figure_id, str) and figure_id:
                figures.add(figure_id)
        elif block_type == "formula":
            formula_id = metadata.get("formula_id")
            if isinstance(formula_id, str) and formula_id:
                formulas.add(formula_id)
    return {
        "tables": {
            table_id: {
                "child_ids": sorted(entry["child_ids"]),
                "parent_ids": sorted(entry["parent_ids"]),
                "child_count": len(entry["child_ids"]),
            }
            for table_id, entry in sorted(tables.items())
        },
        "figure_ids": sorted(figures),
        "formula_ids": sorted(formulas),
        "row_indices": {
            table_id: sorted(rows) for table_id, rows in sorted(row_indices.items())
        },
        "orphan_structured_chunks": sorted(orphans),
    }


def derive_db_typed_inventory(
    manifest_inventory: dict[str, Any],
    chunk_rows: list[DocumentChunk],
) -> dict[str, Any]:
    """从 (document_id, parse_version) 的 DocumentChunk 行派生 DB/index 库存。

    表格分块通过 source_span 中的 table_id 标注归属；若缺失，则回退到
    source_block_ids 反查 manifest 中声明的来源 block。找不到归属的结构化
    分块被记为孤儿（保留 chunk id），激活 gate 据此失败关闭；跨版本行会
    被记为 cross-version 污染。行覆盖只统计子块，父块不能掩盖缺失的
    child 行；仅脚注子块（持久化 span metadata 中的 ``footnote_index`` 契约）
    不含数据行，不贡献行覆盖。

    行覆盖来源有两个，取并集：
    1. 逐单元格的 ``source_spans[].row_index``（HTML/DOCX 等解析器）。
    2. span metadata 中的 ``row_indices``（text/Markdown 表格持久化时由
       ``_persist_table_row_indices`` 从 chunk metadata 写入的同一份行覆盖，
       span 本身通常不带逐行 row_index）。
    """
    block_to_table: dict[str, str] = {}
    for record in manifest_inventory.get("tables") or []:
        if not isinstance(record, dict):
            continue
        for block_id in record.get("source_block_ids") or []:
            if isinstance(block_id, str):
                block_to_table.setdefault(block_id, record["table_id"])
    expected_version = manifest_inventory.get("version")
    tables: dict[str, dict[str, Any]] = {}
    figures: set[str] = set()
    formulas: set[str] = set()
    orphans: list[str] = []
    cross_version: list[str] = []
    for row in chunk_rows:
        row_version = getattr(row, "parse_version", None)
        if expected_version is not None and row_version != expected_version:
            cross_version.append(row.id)
        block_type = getattr(row, "block_type", None)
        spans = getattr(row, "source_spans", None)
        if not isinstance(spans, list):
            spans = []
        source_block_ids = getattr(row, "source_block_ids", None) or []
        if block_type == "table":
            table_id = next(
                (
                    value
                    for value in (
                        _span_structured_id(span, "table_id") for span in spans
                    )
                    if value is not None
                ),
                None,
            )
            if table_id is None:
                table_id = next(
                    (block_to_table[block_id] for block_id in source_block_ids if block_id in block_to_table),
                    None,
                )
            if table_id is None:
                # 无结构归属的结构化分块是孤儿：保留 chunk id，激活 gate 拒绝。
                orphans.append(row.id)
                continue
            entry = tables.setdefault(
                table_id,
                {
                    "parent_ids": set(),
                    "child_ids": set(),
                    "source_block_ids": set(),
                    "row_indices": set(),
                },
            )
            entry["source_block_ids"].update(
                block_id for block_id in source_block_ids if isinstance(block_id, str)
            )
            if getattr(row, "chunk_role", None) == "child":
                entry["child_ids"].add(row.id)
                # 行覆盖只统计真正的数据行子块。脚注子块（现有脚注元数据契约：
                # 持久化 span metadata 中的 footnote_index 标记）携带完整
                # row_indices 但无数据行，整体跳过、不贡献任何行下标。
                if any(
                    isinstance(span, dict)
                    and _is_footnote_only_chunk(span.get("metadata"))
                    for span in spans
                ):
                    continue
                # 行覆盖只统计子块：父块的行集合不参与覆盖判定。
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    row_index = span.get("row_index")
                    if _is_valid_row_index(row_index):
                        entry["row_indices"].add(int(row_index))
                    # text/Markdown 表格的 span 通常没有逐行 row_index；
                    # 持久化阶段把 chunk 级 row_indices 写入 span metadata，
                    # 这里读取同一份行覆盖（HTML/DOCX 逐行路径保持不变）。
                    span_metadata = span.get("metadata")
                    if isinstance(span_metadata, dict):
                        persisted_rows = span_metadata.get("row_indices")
                        if isinstance(persisted_rows, list):
                            entry["row_indices"].update(
                                _normalize_row_indices(persisted_rows)
                            )
            elif getattr(row, "chunk_role", None) == "parent":
                entry["parent_ids"].add(row.id)
        elif block_type == "figure":
            figure_id = next(
                (
                    value
                    for value in (
                        _span_structured_id(span, "figure_id") for span in spans
                    )
                    if value is not None
                ),
                None,
            )
            if figure_id is not None:
                figures.add(figure_id)
        elif block_type == "formula":
            formula_id = next(
                (
                    value
                    for value in (
                        _span_structured_id(span, "formula_id") for span in spans
                    )
                    if value is not None
                ),
                None,
            )
            if formula_id is not None:
                formulas.add(formula_id)
    return {
        "tables": {
            table_id: {
                "parent_ids": sorted(entry["parent_ids"]),
                "child_ids": sorted(entry["child_ids"]),
                "source_block_ids": sorted(entry["source_block_ids"]),
                "row_indices": sorted(entry["row_indices"]),
            }
            for table_id, entry in sorted(tables.items())
        },
        "figure_ids": sorted(figures),
        "formula_ids": sorted(formulas),
        "orphan_structured_chunks": sorted(orphans),
        "cross_version_chunk_ids": sorted(cross_version),
    }


def compare_typed_inventory(
    manifest_inventory: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    """逐表比较 manifest 库存与 DB/index 库存，返回可读的不一致描述列表。

    覆盖：table ID 集合、每表 Child ID/child_count、source_block_ids、
    parent/Child 归属、row 覆盖、重复/多余/孤儿记录、figure/formula ID 集合，
    以及跨版本混入。任何非空返回都意味着激活必须被阻止。
    """
    mismatches: list[str] = []
    expected_tables = {
        record["table_id"]: record
        for record in manifest_inventory.get("tables") or []
        if isinstance(record, dict)
    }
    observed_tables = observed.get("tables") or {}

    missing_tables = sorted(set(expected_tables) - set(observed_tables))
    extra_tables = sorted(set(observed_tables) - set(expected_tables))
    if missing_tables:
        mismatches.append("missing tables: " + ", ".join(missing_tables))
    if extra_tables:
        mismatches.append("extra tables: " + ", ".join(extra_tables))

    for table_id in sorted(set(expected_tables) & set(observed_tables)):
        expected = expected_tables[table_id]
        observed_table = observed_tables[table_id]
        expected_child_ids = sorted(set(expected.get("child_ids") or []))
        listed_child_ids = sorted(expected.get("child_ids") or [])
        observed_child_ids = sorted(observed_table.get("child_ids") or [])
        if len(listed_child_ids) != len(set(listed_child_ids)):
            mismatches.append(f"table {table_id} duplicate child entries in expected inventory")
        if expected.get("child_count") != len(listed_child_ids):
            mismatches.append(
                f"table {table_id} child count mismatch: declared "
                f"{expected.get('child_count')}, listed {len(listed_child_ids)}"
            )
        missing_children = sorted(set(expected_child_ids) - set(observed_child_ids))
        extra_children = sorted(set(observed_child_ids) - set(expected_child_ids))
        if missing_children:
            mismatches.append(
                "table " + table_id + " missing children: " + ", ".join(missing_children)
            )
        if extra_children:
            mismatches.append(
                "table " + table_id + " extra children: " + ", ".join(extra_children)
            )
        expected_parents = sorted(set(expected.get("parent_ids") or []))
        observed_parents = sorted(observed_table.get("parent_ids") or [])
        if expected_parents != observed_parents:
            mismatches.append(
                f"table {table_id} parent ownership mismatch: expected "
                f"{expected_parents}, found {observed_parents}"
            )
        expected_blocks = sorted(set(expected.get("source_block_ids") or []))
        observed_blocks = sorted(observed_table.get("source_block_ids") or [])
        if expected_blocks != observed_blocks:
            mismatches.append(
                f"table {table_id} source block mismatch: expected "
                f"{expected_blocks}, found {observed_blocks}"
            )
        expected_rows = expected.get("row_count")
        covered = expected.get("row_indices") or []
        if isinstance(expected_rows, int) and expected_rows > 0:
            expected_coverage = list(range(expected_rows))
            if isinstance(covered, list) and sorted(set(covered)) != expected_coverage:
                mismatches.append(
                    f"table {table_id} row count mismatch: declared {expected_rows}, "
                    f"covered {sorted(set(covered))}"
                )
        observed_rows = sorted(set(observed_table.get("row_indices") or []))
        if isinstance(covered, list) and covered:
            if observed_rows != sorted(set(covered)):
                mismatches.append(
                    f"table {table_id} row coverage mismatch: expected "
                    f"{sorted(set(covered))}, found {observed_rows}"
                )
        elif isinstance(expected_rows, int) and expected_rows > 0 and observed_rows != list(range(expected_rows)):
            mismatches.append(
                f"table {table_id} row count mismatch: declared {expected_rows}, "
                f"found rows {observed_rows}"
            )

    expected_figures = sorted(manifest_inventory.get("figure_ids") or [])
    observed_figures = sorted(observed.get("figure_ids") or [])
    if expected_figures != observed_figures:
        mismatches.append(
            "figure ID mismatch: expected "
            + str(expected_figures)
            + ", found "
            + str(observed_figures)
        )
    expected_formulas = sorted(manifest_inventory.get("formula_ids") or [])
    observed_formulas = sorted(observed.get("formula_ids") or [])
    if expected_formulas != observed_formulas:
        mismatches.append(
            "formula ID mismatch: expected "
            + str(expected_formulas)
            + ", found "
            + str(observed_formulas)
        )

    observed_orphans = sorted(observed.get("orphan_structured_chunks") or [])
    manifest_orphans = sorted(manifest_inventory.get("orphan_structured_chunks") or [])
    if observed_orphans:
        mismatches.append("orphan structured chunks: " + ", ".join(observed_orphans))
    elif manifest_orphans:
        # index 阶段记录过孤儿，但激活时 DB 派生库存不再包含它们，说明数据
        # 不一致，同样失败关闭。
        mismatches.append(
            "orphan structured chunks disappeared: manifest recorded "
            + ", ".join(manifest_orphans)
        )
    cross_version = observed.get("cross_version_chunk_ids") or []
    if cross_version:
        mismatches.append(
            "cross-version chunk contamination: " + ", ".join(cross_version)
        )
    return mismatches


class IngestionPipeline:
    """Ingestion 管线编排器：阶段执行器 + 文档注册 + SAC-KG 知识抽取。

    同时承担两类职责：
    1. canonical 阶段机的各阶段执行器（parse/repair/canonicalize/
       semantic_split/contextualize/embed/index/activate），供
       ``IngestionStageRunner`` 调用。
    2. 文档注册、legacy 同步处理、以及 SAC-KG 抽取（实体/三元组/审阅）。
    """

    def __init__(self, db: Session) -> None:
        """保存会话并初始化检索/生成客户端、验证器与对象存储。

        ``self.ollama`` remains the embedding and local-generation client.  The
        SAC-KG extraction calls go through ``_generate_sac_kg_structured`` so a
        deployment can switch only that generation path to DeepSeek without
        changing Qwen embeddings, MinerU parsing, or the RAG index.
        """
        self.db = db
        self.ollama = OllamaClient()
        self.deepseek: DeepSeekClient | None = None
        self.verifier = ExternalVerifier()
        self.storage = ObjectStorage()

    def _generate_sac_kg_structured(
        self,
        schema,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ):
        """Generate one SAC-KG payload using the configured generation provider.

        This is deliberately a narrow provider boundary.  Embeddings still go
        through ``self.ollama.embed`` (which already selects the hosted Qwen
        embedding API), while only entity/claim extraction and growth decisions
        honor ``GENERATION_PROVIDER``.  In tests and local deployments the
        default remains Ollama; with ``deepseek`` selected, no Ollama generation
        request is made.
        """
        provider = self._sac_kg_generation_provider()
        if provider == "deepseek":
            if self.deepseek is None:
                self.deepseek = DeepSeekClient()
            return self.deepseek.generate_structured(
                schema,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=getattr(settings, "deepseek_model", None),
            )
        if provider not in {"", "ollama"}:
            logger.warning("Unsupported SAC-KG generation provider %r; using Ollama compatibility path.", provider)
        return self.ollama.generate_structured(
            schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )

    @staticmethod
    def _sac_kg_generation_provider() -> str:
        """Return the normalized provider used by SAC-KG structured calls."""
        return str(
            getattr(settings, "generation_provider", "ollama") or "ollama"
        ).strip().lower()

    def ingestion_stage_handlers(self) -> dict[str, object]:
        """Expose stages backed by durable canonical artifact operations.

        暴露"基于持久化 canonical 产物"的各个阶段执行器映射。

        Chunk persistence and the later model/index operations need the formal
        version-scoped implementation. They remain absent so workers fail closed
        at the first unavailable stage.
        分块持久化与后续模型/索引操作需要正式版本作用域实现；
        未提供的阶段缺省为缺失，使 worker 在首个不可用阶段即"失败关闭"。
        """
        return {
            "parse": self._run_canonical_parse_stage,
            "repair": self._run_canonical_repair_gate_stage,
            "canonicalize": self._run_canonical_promotion_stage,
            "semantic_split": self._run_semantic_split_stage,
            "contextualize": self._run_contextualize_stage,
            "embed": self._run_embed_stage,
            "index": self._run_index_stage,
            "activate": self._run_activation_gate_stage,
        }

    def validate_ingestion_identity(
        self,
        _document: Document,
        version: DocumentParseVersion,
        _stage: str,
    ) -> None:
        """阶段前置校验：当前配置必须与版本入队时记录的配置快照一致。

        计算当前 ingestion 配置快照的哈希，与版本 manifest 中记录的
        ``ingestion_config`` / ``ingestion_config_sha256`` 比对；
        不一致则抛错，防止在配置变更后继续执行旧版本管线。
        """
        live_snapshot = build_ingestion_config_snapshot()
        require_matching_ingestion_config(version.manifest_json, live_snapshot)

    def _run_canonical_parse_stage(self, context) -> dict[str, object]:
        """parse 阶段执行器：执行 canonical 解析并把草稿固化到产物目录。

        流程：
        1. ``_parse_canonical_phase`` 解析源文件得到 canonical 文档。
        2. 注入文档 ID 与解析版本键。
        3. 把 canonical JSON 序列化，计算输入指纹；写为
           ``parse.canonical.json`` 草稿（若已存在则校验一致，防冲突）。
        4. 记录清单（artifact_path / input_fingerprint）、质量与解析器名；
           若文档尚无激活版本，回填标题与 canonical_ingestion 元信息。
        5. 返回摘要（artifact_path、输入指纹、质量、块/表数量）。
        """
        canonical = self._parse_canonical_phase(Path(context.document.raw_path))
        canonical = canonical.model_copy(
            update={
                "document_id": context.document.id,
                "parse_version": context.version.version_key,
            }
        )
        document_root = self._stage_artifact_dir(context)
        draft_path = document_root / "parse.canonical.json"
        payload = canonical.model_dump_json().encode("utf-8")
        # 输入指纹：对草稿内容做 SHA-256，供后续阶段校验
        input_fingerprint = hashlib.sha256(payload).hexdigest()
        if draft_path.exists():
            # 草稿已存在：内容必须与本次一致（可重放性）
            if draft_path.read_bytes() != payload:
                raise RuntimeError(
                    f"Canonical parse draft conflicts with checkpoint: {draft_path}"
                )
        else:
            # 原子写入草稿（临时文件 + rename）
            temporary = document_root / (
                f".{context.version.version_key}.{uuid4().hex}.tmp"
            )
            try:
                temporary.write_bytes(payload)
                temporary.replace(draft_path)
            finally:
                temporary.unlink(missing_ok=True)
        quality = canonical.quality.model_dump(mode="json")
        context.version.manifest_json = {
            **dict(context.version.manifest_json or {}),
            "artifact_path": str(draft_path),
            "input_fingerprint": input_fingerprint,
        }
        context.version.quality_json = quality
        context.version.parser_name = canonical.parser_source
        if context.document.active_parse_version in (
            None,
            context.version.version_key,
        ):
            context.document.title = canonical.title or context.document.title
            metadata = dict(context.document.metadata_json or {})
            metadata["canonical_ingestion"] = {
                "version_key": context.version.version_key,
                "input_fingerprint": input_fingerprint,
                "quality": quality,
            }
            # 表格证据写回：检索表格（_search_document_table_contexts）只读
            # document_intelligence.tables；canonical-v4 的表格数据在
            # document_chunks 中但从未写回该字段（2026-08-12 诊断：charmm36
            # 文档该字段为空，R16-R23 表格查询全部拿不到表格证据）。
            # 与旧 DI/mineru 摄取路径的 tables 格式保持一致。
            # 覆盖保护（2026-08-12 code review）：canonical 解析未提取到任何
            # 表格时不得用空列表覆盖已有 tables —— 文档可能由旧 DI/mineru
            # 路径摄取的表格仍在（v4 重新解析失败/表格被过滤的降级场景），
            # 保留旧表格证据比清空更安全。
            canonical_tables = canonical_tables_to_document_intelligence(
                canonical.tables
            )
            if canonical_tables:
                intelligence = metadata.get("document_intelligence")
                if not isinstance(intelligence, dict):
                    intelligence = {}
                intelligence["tables"] = canonical_tables
                metadata["document_intelligence"] = intelligence
                context.document.metadata_json = metadata
        return {
            "artifact_path": str(draft_path),
            "input_fingerprint": input_fingerprint,
            "quality": {
                "status": quality.get("status"),
                "accepted": quality.get("accepted"),
                "score": quality.get("score"),
            },
            "block_count": len(canonical.blocks),
            "table_count": len(canonical.tables),
        }

    def _run_canonical_repair_gate_stage(self, context) -> dict[str, object]:
        """repair 阶段执行器：校验解析草稿指纹、执行修复并写 staging。

        流程：
        1. 读取解析草稿，校验其指纹与检查点中的输入指纹一致。
        2. 用 ``_repair_canonical_phase`` 执行修复（针对 PDF 的质量门
           修复，含文档智能目标页/全页重试）。
        3. 把修复后的 canonical 写入 staging 目录（``write_staging``）。
        4. 读取 manifest 中的质量：不通过则抛错（repair gate 拒绝）。
        5. 记录清单与质量并返回摘要（含表格修复请求数量）。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore
        from app.services.canonical_models import CanonicalDocument

        draft_path = self._checkpoint_canonical_draft(context)
        payload = draft_path.read_bytes()
        expected_fingerprint = (context.input.get("previous_output") or {}).get(
            "input_fingerprint"
        )
        actual_fingerprint = hashlib.sha256(payload).hexdigest()
        # 草稿指纹必须与上一阶段检查点一致
        if actual_fingerprint != expected_fingerprint:
            raise RuntimeError("Canonical parse draft fingerprint does not match checkpoint.")
        canonical = CanonicalDocument.model_validate_json(payload)
        canonical = self._repair_canonical_phase(
            canonical,
            Path(context.document.raw_path),
        )
        store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
        staging = store.write_staging(
            context.document.id,
            context.version.version_key,
            canonical,
        )
        manifest_path = staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        quality = dict(manifest["quality"])
        # 修复门禁：质量必须被接受
        if (
            quality.get("accepted") is not True
            or quality.get("status")
            not in {"accepted", "accepted_with_warnings"}
        ):
            raise RuntimeError(
                f"Canonical repair gate rejected quality {quality.get('status')!r}."
            )
        context.version.manifest_json = {
            **dict(context.version.manifest_json or {}),
            "artifact_path": str(manifest_path),
            "input_fingerprint": manifest["input_fingerprint"],
            "canonical_markdown_sha256": manifest["canonical_markdown_sha256"],
        }
        context.version.quality_json = quality
        return {
            "artifact_path": str(manifest_path),
            "input_fingerprint": manifest["input_fingerprint"],
            "quality": {
                "status": quality.get("status"),
                "accepted": quality.get("accepted"),
                "score": quality.get("score"),
            },
            "repair_requests": list(
                manifest.get("document", {})
                .get("metadata", {})
                .get("table_repair_requests", [])
            ),
        }

    def _parse_canonical_phase(self, path: Path):
        """执行 canonical 解析主流程（含 PDF 多层解析）。

        非 PDF：直接 ``parse_canonical_document``。

        PDF 多层解析策略：
        1. 校验路径并获取页数。
        2. 若启用 MinerU：先尝试 MinerU 深度解析，记录成败于 attempts。
        3. 读取文本层（供审计与兜底）。
        4. 若 MinerU 未产出结果：退回文本层兜底解析。
        5. 挂接 PDF 审计信息（页数、文本层、尝试序列、主解析器）。
        6. 用文本层补充缺失页 / 恢复页遗漏，并记录尝试。
        7. 再次挂接完整审计；运行质量门评估；固化结构化证据与最终审计。

        返回：最终的 canonical 文档对象。
        """
        if path.suffix.lower() != ".pdf":
            from app.services.canonical_adapters import parse_canonical_document

            return parse_canonical_document(path)

        from app.services import parser
        from app.services import canonical_adapters as adapters
        from app.services.canonical_quality import CanonicalQualityGate

        path = adapters._validate_path(path)  # noqa: SLF001
        page_count = parser._validate_pdf_basic(path)
        attempts: list[str] = []  # 解析尝试序列（审计）
        primary = None
        # 第一层：MinerU（若启用）
        if parser.settings.mineru_enabled:
            try:
                primary = adapters.run_mineru(path, page_count)
            except Exception as exc:  # noqa: BLE001
                attempts.append(f"mineru:failed:{type(exc).__name__}:{exc}")
            else:
                attempts.append(
                    "mineru:success" if primary is not None else "mineru:unavailable"
                )
        else:
            attempts.append("mineru:disabled")
        # 读取文本层（审计与兜底共用）
        page_texts, warnings = adapters._read_text_layer_for_audit(  # noqa: SLF001
            path, page_count
        )
        # 第二层：文本层兜底（仅当 MinerU 不可用/无结果）
        if primary is None:
            primary = adapters.run_text_layer_fallback(
                path,
                page_count,
                page_texts,
                text_layer_warnings=warnings,
            )
            attempts.append("pypdf_text_layer:success")
        adapters._attach_pdf_audit(  # noqa: SLF001
            primary,
            page_count=page_count,
            page_texts=page_texts,
            text_layer_warnings=warnings,
            attempts=attempts,
            primary_parser=primary.parser_source,
        )
        # 用文本层补充"缺失页"（MinerU/主解析漏掉的页）
        text_layer_fallback_pages = (
            adapters._supplement_missing_pdf_pages_from_text_layer(  # noqa: SLF001
                primary,
                page_texts,
                page_count,
            )
        )
        # 恢复"页内遗漏"（跳过已补充的页）
        text_layer_recovery_pages = (
            adapters._recover_pdf_page_omissions_from_text_layer(  # noqa: SLF001
                primary,
                page_texts,
                page_count,
                skip_pages=text_layer_fallback_pages,
            )
        )
        if text_layer_fallback_pages:
            attempts.append(
                "pypdf_text_layer:missing_pages:"
                + ",".join(
                    str(page + 1)
                    for page in sorted(text_layer_fallback_pages)
                )
            )
        if text_layer_recovery_pages:
            attempts.append(
                "pypdf_text_layer:page_recovery:"
                + ",".join(
                    str(page + 1)
                    for page in sorted(text_layer_recovery_pages)
                )
            )
        # 重新挂接包含全部尝试的审计
        adapters._attach_pdf_audit(  # noqa: SLF001
            primary,
            page_count=page_count,
            page_texts=page_texts,
            text_layer_warnings=warnings,
            attempts=attempts,
            primary_parser=primary.parser_source,
        )
        # 质量评估 + 结构化证据固化 + 最终审计
        CanonicalQualityGate().evaluate(primary)
        adapters._finalize_structured_evidence(primary)  # noqa: SLF001
        return adapters._finalize_pdf_audit(primary)  # noqa: SLF001

    def _repair_canonical_phase(self, primary, path: Path):
        """对 canonical 文档执行修复（针对 PDF 的质量门修复）。

        非 PDF：直接返回原文档。

        PDF 修复策略：
        1. 从主文档元信息恢复页数、文本层与先前尝试序列；再次补充缺失页。
        2. 运行质量门评估，收集"可修复"问题与修复作用域（页面范围）。
        3. **目标页修复**（无致命问题且有可修复项、有目标页、文档智能
           启用）：对目标页运行文档智能，若覆盖完整且合并后质量无致命
           问题且满足各修复项，则采纳修复候选。
        4. **全页修复**（有致命问题）：对整个文档运行文档智能，若质量
           无致命问题则整体替换。
        5. 挂接含修复作用域的审计、固化结构化证据、返回最终文档。
        """
        if path.suffix.lower() != ".pdf":
            return primary

        from app.services import parser
        from app.services import canonical_adapters as adapters
        from app.services.canonical_quality import CanonicalQualityGate

        # 从主文档元信息恢复上下文
        page_count = int(primary.metadata.get("expected_page_count") or 0)
        page_texts = list(primary.metadata.get("text_layer_pages") or [])
        warnings = list(primary.metadata.get("text_layer_warnings") or [])
        attempts = list(primary.parser_metadata.get("parser_attempts") or [])
        text_layer_fallback_pages = (
            adapters._supplement_missing_pdf_pages_from_text_layer(  # noqa: SLF001
                primary,
                page_texts,
                page_count,
            )
        )
        if text_layer_fallback_pages:
            attempts.append(
                "pypdf_text_layer:missing_pages:"
                + ",".join(
                    str(page + 1)
                    for page in sorted(text_layer_fallback_pages)
                )
            )
        # 排除这些页：它们已由文本层补充，不再作为修复目标
        excluded_fallback_pages = set(text_layer_fallback_pages)
        excluded_fallback_pages.update(
            page
            for page in primary.metadata.get(
                "text_layer_fallback_page_indices", []
            )
            if isinstance(page, int) and 0 <= page < page_count
        )
        # 质量门评估：收集可修复项及其页面作用域
        CanonicalQualityGate().evaluate(primary)
        issues = list(primary.quality.issues)
        repair_issues = [issue for issue in issues if issue.repairable]
        scopes = [
            issue.repair_scope
            for issue in repair_issues
            if issue.repair_scope
        ]
        fatal = any(issue.severity == "fatal" for issue in issues)
        # 目标页集合 = 修复作用域涉及的页 - 已排除的页
        targeted_pages = adapters._repair_page_indices(  # noqa: SLF001
            scopes, page_count
        ) - excluded_fallback_pages
        repaired = None
        # 目标页修复路径
        if (
            not fatal
            and repair_issues
            and targeted_pages
            and parser.settings.document_intelligence_enabled
        ):
            try:
                repair = adapters.run_document_intelligence(
                    path,
                    page_count,
                    page_texts,
                    page_indices=targeted_pages,
                )
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    f"document_intelligence:targeted:failed:{type(exc).__name__}:{exc}"
                )
            else:
                attempts.append(
                    "document_intelligence:targeted:success"
                    if repair is not None
                    else "document_intelligence:targeted:unavailable"
                )
                # 目标页覆盖完整时合并修复页
                if repair is not None and adapters._targeted_repair_has_complete_coverage(  # noqa: SLF001
                    repair, targeted_pages
                ):
                    candidate = adapters._merge_pdf_page_repairs(  # noqa: SLF001
                        primary.model_copy(deep=True),
                        repair,
                        targeted_pages,
                        issues=repair_issues,
                    )
                    report = CanonicalQualityGate().evaluate(candidate)
                    # 合并后无致命问题且各修复项被满足才采纳
                    if (
                        not any(issue.severity == "fatal" for issue in report.issues)
                        and adapters._targeted_repair_satisfies_issues(  # noqa: SLF001
                            primary,
                            repair,
                            candidate,
                            repair_issues,
                            targeted_pages,
                        )
                    ):
                        repaired = candidate
        # 全页修复路径（致命问题）
        if repaired is None and fatal and parser.settings.document_intelligence_enabled:
            try:
                candidate = adapters.run_document_intelligence(
                    path, page_count, page_texts
                )
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    f"document_intelligence:full:failed:{type(exc).__name__}:{exc}"
                )
            else:
                if candidate is not None:
                    report = CanonicalQualityGate().evaluate(candidate)
                    if not any(issue.severity == "fatal" for issue in report.issues):
                        repaired = candidate
                        attempts.append("document_intelligence:full:success")
        result = repaired or primary
        # 挂接含修复作用域的审计并固化
        adapters._attach_pdf_audit(  # noqa: SLF001
            result,
            page_count=page_count,
            page_texts=page_texts,
            text_layer_warnings=warnings,
            attempts=attempts,
            primary_parser=primary.parser_source,
            repair_scopes=scopes,
        )
        adapters._finalize_structured_evidence(result)  # noqa: SLF001
        return adapters._finalize_pdf_audit(result)  # noqa: SLF001

    def _run_canonical_promotion_stage(self, context) -> dict[str, object]:
        """canonicalize 阶段执行器：把 staging 产物提升（promote）为最终版。

        最终目录为 ``artifacts/<document_id>/<version_key>``；若尚未提升，
        先校验上一阶段的 staging 清单路径，再调用 ``store.promote``。
        随后加载 canonical 并返回清单摘要（含块数/表数）。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore

        store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
        final = (
            settings.canonical_artifacts_dir
            / context.document.id
            / context.version.version_key
        )
        if not final.is_dir():
            # 尚未提升：校验 staging 检查点后提升
            self._checkpoint_artifact_path(context)
            final = store.promote(
                context.document.id,
                context.version.version_key,
            )
        canonical = store.load(context.document.id, context.version.version_key)
        manifest_path = final / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "artifact_path": str(manifest_path),
            "input_fingerprint": manifest["input_fingerprint"],
            "canonical_markdown_sha256": manifest["canonical_markdown_sha256"],
            "block_count": len(canonical.blocks),
            "table_count": len(canonical.tables),
        }

    def _run_semantic_split_stage(self, context) -> dict[str, object]:
        """semantic_split 阶段执行器：对 canonical 做语义切分产出分块草稿。

        前置：校验 ingestion 身份（配置快照一致）。加载 canonical 后，
        用 ``SemanticChunker`` 构建分块草稿列表；无草稿则抛错。
        产物写入 ``semantic_chunks.json``。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore
        from app.services.semantic_chunking import SemanticChunker

        self.validate_ingestion_identity(
            context.document, context.version, context.stage
        )
        canonical = CanonicalArtifactStore(settings.canonical_artifacts_dir).load(
            context.document.id,
            context.version.version_key,
        )
        chunker = SemanticChunker(self.ollama)
        drafts = chunker.build(canonical)
        if not drafts:
            raise RuntimeError("Semantic splitting produced no chunk drafts.")
        return self._write_stage_artifact(
            context,
            "semantic_chunks.json",
            [draft.model_dump(mode="json") for draft in drafts],
            extra={
                "chunk_count": len(drafts),
                "source_fidelity_completeness": 1.0,
                "structured_limit_completeness": 1.0,
            },
        )

    def _run_contextualize_stage(self, context) -> dict[str, object]:
        """contextualize 阶段执行器：对需要上下文化的结构化子块做上下文增强。

        流程：
        1. 读取上一阶段产物（分块草稿），区分为 parent / child。
        2. 按块类型区分"需要上下文化"（structured，如表/图/公式）与
           "普通"（plain）子块。
        3. 仅对 structured 子块调用 ``ContextualizationService.contextualize``
           （以文档标题/摘要/大纲为上下文）。
        4. 校验上下文化结果保留完整子块清单（数量与 ID 集合一致）。
        5. 把上下文化结果合并回全部草稿，写入 ``contextualized_chunks.json``。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore
        from app.services.contextualization import (
            ContextualizationService,
            DocumentContext,
        )
        from app.services.semantic_chunking import ChunkDraft

        payload = self._load_previous_stage_artifact(context)
        drafts = [ChunkDraft.model_validate(item) for item in payload]
        parents = {
            draft.local_id: draft for draft in drafts if draft.chunk_role == "parent"
        }
        children = [draft for draft in drafts if draft.chunk_role == "child"]
        canonical = CanonicalArtifactStore(settings.canonical_artifacts_dir).load(
            context.document.id,
            context.version.version_key,
        )
        structured = [
            child for child in children if requires_contextualization(child.block_type)
        ]
        plain = [
            child for child in children if not requires_contextualization(child.block_type)
        ]
        contextualized = (
            ContextualizationService().contextualize(
                document=DocumentContext(
                    title=canonical.title,
                    source_abstract=canonical.abstract,
                    section_outline=self._outline_titles(canonical.outline),
                ),
                children=structured,
                parents=parents,
            )
            if structured
            else []
        )
        contextualized_by_id = {child.local_id: child for child in contextualized}
        expected_ids = {child.local_id for child in structured}
        if len(contextualized_by_id) != len(contextualized) or set(
            contextualized_by_id
        ) != expected_ids:
            raise RuntimeError(
                "Contextualization did not preserve the structured Child inventory."
            )
        combined = [
            (
                contextualized_by_id[draft.local_id].model_dump(mode="json")
                if draft.chunk_role == "child"
                and requires_contextualization(draft.block_type)
                else draft.model_dump(mode="json")
            )
            for draft in drafts
        ]
        if len(combined) != len(drafts):
            raise RuntimeError("Contextualization did not preserve the chunk inventory.")
        return self._write_stage_artifact(
            context,
            "contextualized_chunks.json",
            combined,
            extra={
                "chunk_count": len(combined),
                "child_count": len(children),
                "contextualized_child_count": len(contextualized),
                "plain_child_count": len(plain),
            },
        )

    def _run_embed_stage(self, context) -> dict[str, object]:
        """embed 阶段执行器：对子块做向量嵌入（父块不嵌入）。

        流程：
        1. 读取上一阶段产物并校验：含子块且父/子角色计数正确。
        2. 校验每个子块满足上下文化策略（``valid_contextualized_embedding``
           / ``valid_plain_embedding``）。
        3. 批量调用 Ollama 嵌入；校验向量数量、非有限值、维度一致，
           且维度与配置一致。
        4. 输出记录：子块带嵌入向量，父块 embedding 为 None，
           写入 ``embedded_chunks.json``。
        """
        payload = self._load_previous_stage_artifact(context)
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("Embedding stage requires contextualized chunks.")
        children = [
            item
            for item in payload
            if isinstance(item, dict) and item.get("chunk_role") == "child"
        ]
        parent_count = sum(
            isinstance(item, dict) and item.get("chunk_role") == "parent"
            for item in payload
        )
        if len(children) + parent_count != len(payload):
            raise RuntimeError("Embedding input contains an invalid chunk role.")
        for item in children:
            try:
                valid = (
                    valid_contextualized_embedding(item)
                    if requires_contextualization(str(item.get("block_type")))
                    else valid_plain_embedding(item)
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Embedding input violates the Child contextualization policy."
                ) from exc
            if not valid:
                raise RuntimeError(
                    "Embedding input violates the Child contextualization policy."
                )
        texts = [str(item.get("embedding_text") or "") for item in children]
        if not texts or any(not text.strip() for text in texts):
            raise RuntimeError("Embedding input contains an empty child chunk.")
        embeddings = self.ollama.embed(texts)
        if len(embeddings) != len(children):
            raise RuntimeError("Embedding response count does not match child count.")
        dimensions: int | None = None
        child_embeddings: list[list[float]] = []
        for embedding in embeddings:
            if (
                not isinstance(embedding, list)
                or not embedding
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in embedding
                )
            ):
                raise RuntimeError("Embedding response contains an invalid vector.")
            dimensions = dimensions or len(embedding)
            if len(embedding) != dimensions:
                raise RuntimeError("Embedding response dimensions are inconsistent.")
            child_embeddings.append([float(value) for value in embedding])
        if dimensions != settings.active_embedding_dimensions:
            raise RuntimeError(
                "Embedding response dimensions do not match the configured "
                f"embedding dimensions ({settings.active_embedding_dimensions})."
            )
        embedded = iter(child_embeddings)
        records = [
            {
                "chunk": item,
                "embedding": next(embedded)
                if item["chunk_role"] == "child"
                else None,
            }
            for item in payload
        ]
        return self._write_stage_artifact(
            context,
            "embedded_chunks.json",
            records,
            extra={
                "chunk_count": len(records),
                "child_count": len(children),
                "embedded_count": len(child_embeddings),
                "dimensions": dimensions,
            },
        )

    def _run_index_stage(self, context) -> dict[str, object]:
        """index 阶段执行器：校验嵌入产物并持久化版本化分块到数据库。

        流程：
        1. 读取嵌入产物并与 embed 检查点（数量/维度）比对。
        2. 逐条校验记录：chunk 有非空 local_id、子块带合法嵌入向量、
           父块嵌入为 None、角色合法；local_id 唯一。
        3. 校验子块/嵌入数量与检查点一致。
        4. 调用 ``_persist_versioned_chunks`` 写入 DocumentChunk 表并
           替换向量索引；产物写入 ``index_payload.json``。
        """
        payload = self._load_previous_stage_artifact(context)
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("Index stage requires embedded chunk records.")
        expected = context.input.get("previous_output") or {}
        expected_count = expected.get("chunk_count")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != len(payload)
        ):
            raise RuntimeError("Embedded chunk count does not match embed checkpoint.")
        expected_dimensions = expected.get("dimensions")
        if (
            isinstance(expected_dimensions, bool)
            or not isinstance(expected_dimensions, int)
            or expected_dimensions <= 0
        ):
            raise RuntimeError("Embed checkpoint has invalid dimensions.")
        if expected_dimensions != settings.active_embedding_dimensions:
            raise RuntimeError(
                "Embedding dimensions do not match the configured embedding dimensions."
            )
        local_ids: set[str] = set()
        child_count = 0
        embedded_count = 0
        for record in payload:
            if not isinstance(record, dict) or not isinstance(record.get("chunk"), dict):
                raise RuntimeError("Index stage found an invalid embedded chunk record.")
            chunk = record["chunk"]
            local_id = chunk.get("local_id")
            if not isinstance(local_id, str) or not local_id:
                raise RuntimeError("Embedded chunk requires a non-empty local ID.")
            local_ids.add(local_id)
            chunk_role = chunk.get("chunk_role")
            embedding = record.get("embedding")
            if chunk_role == "child":
                child_count += 1
                if (
                    not isinstance(embedding, list)
                    or not embedding
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in embedding
                    )
                ):
                    raise RuntimeError("Index stage found an invalid child embedding.")
                if len(embedding) != expected_dimensions:
                    raise RuntimeError(
                        "Embedding dimensions do not match the embed checkpoint."
                    )
                embedded_count += 1
            elif chunk_role == "parent":
                if embedding is not None:
                    raise RuntimeError("Parent chunks must not contain embeddings.")
            else:
                raise RuntimeError("Index stage found an invalid chunk role.")
        if len(local_ids) != len(payload):
            raise RuntimeError("Embedded chunk local IDs are not unique.")
        if child_count != expected.get("child_count") or embedded_count != expected.get(
            "embedded_count"
        ):
            raise RuntimeError("Embedded child counts do not match embed checkpoint.")
        self._persist_versioned_chunks(context, payload)
        self._update_manifest_typed_inventory(context, payload)
        return self._write_stage_artifact(
            context,
            "index_payload.json",
            payload,
            extra={
                "row_count": len(payload),
                "child_count": child_count,
                "indexed_count": embedded_count,
                "dimensions": expected_dimensions,
                "parse_version": context.version.version_key,
            },
        )

    def _update_manifest_typed_inventory(
        self,
        context,
        payload: list[dict],
    ) -> None:
        """把 index 阶段持久化的 Child 库存回填进 canonical manifest。

        库存派生自 index 阶段产物（persisted stage data），并绑定到当前
        ``document_id + parse_version``。回填失败（如 row 覆盖不完整）会
        让 index 阶段失败关闭，从而阻止丢行版本进入激活。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore

        child_inventory = derive_child_inventory_from_payload(payload)
        store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
        try:
            store.update_typed_inventory(
                context.document.id,
                context.version.version_key,
                child_inventory,
            )
        except FileNotFoundError:
            # 生产链路在 index 阶段 canonical bundle 必然存在；缺失只出现在
            # 合成/测试路径。bundle 缺失时跳过回填不会造成激活绕过，因为
            # 激活 gate（_verify_typed_inventory_gate）对缺失 bundle 失败关闭。
            return

    def _run_activation_gate_stage(self, context) -> dict[str, object]:
        """activate 阶段执行器（门禁）：激活前做完整性校验与资料准备。

        校验项：
        - index 产物非空且 parse_version 与当前版本一致。
        - 数据库中有非零可检索子块；各块上下文化/普通嵌入完整
          （``valid_*_embedding`` 计数）。
        - 可检索/已嵌入/已索引/有效源跨度/产物子块数量完全一致。
        - 检查点计数与索引现状一致。
        通过后为文档准备 canonical 资料（摘要/来源身份/论文画像）。

        返回：版本键与各项计数。
        """
        payload = self._load_previous_stage_artifact(context)
        if not isinstance(payload, list) or not payload:
            raise ActivationError("Activation artifact validation failed: empty index payload.")
        checkpoint = context.input.get("previous_output") or {}
        version_key = context.version.version_key
        if checkpoint.get("parse_version") != version_key:
            raise ActivationError("Activation artifact validation failed: wrong parse version.")
        children = list(
            context.db.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == context.document.id,
                    DocumentChunk.parse_version == version_key,
                    DocumentChunk.chunk_role == "child",
                )
            ).all()
        )
        retrievable_count = len(children)
        if retrievable_count <= 0:
            raise ActivationError("Activation requires non-zero retrievable Child chunks.")
        try:
            eligible = [
                chunk
                for chunk in children
                if requires_contextualization(chunk.block_type)
            ]
            plain = [
                chunk
                for chunk in children
                if not requires_contextualization(chunk.block_type)
            ]
        except ValueError as exc:
            raise ActivationError(
                "Activation embedding completeness check failed: unknown block type."
            ) from exc
        contextualized_count = sum(
            valid_contextualized_embedding(chunk) for chunk in eligible
        )
        plain_embedding_count = sum(valid_plain_embedding(chunk) for chunk in plain)
        dimensions = settings.active_embedding_dimensions
        embedded_count = sum(
            isinstance(chunk.embedding, list)
            and len(chunk.embedding) == dimensions
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in chunk.embedding
            )
            for chunk in children
        )
        valid_span_count = sum(self._chunk_has_valid_source_spans(chunk) for chunk in children)
        indexed_count = self._indexed_child_count(
            context.document.id,
            version_key,
            fallback=embedded_count,
        )
        artifact_child_count = sum(
            isinstance(item, dict)
            and isinstance(item.get("chunk"), dict)
            and item["chunk"].get("chunk_role") == "child"
            and item["chunk"].get("parse_version") == version_key
            for item in payload
        )
        total_counts = {
            "retrievable": retrievable_count,
            "embedded": embedded_count,
            "indexed": indexed_count,
            "valid_source_spans": valid_span_count,
            "artifact": artifact_child_count,
        }
        policy_complete = (
            contextualized_count == len(eligible)
            and plain_embedding_count == len(plain)
        )
        if not policy_complete or len(set(total_counts.values())) != 1:
            details = {
                **total_counts,
                "contextualization_eligible": len(eligible),
                "contextualized": contextualized_count,
                "plain": len(plain),
                "plain_embedded": plain_embedding_count,
            }
            raise ActivationError(
                "Activation embedding completeness check failed: "
                + ", ".join(f"{name}={value}" for name, value in details.items())
            )
        expected_child_count = checkpoint.get("child_count")
        expected_indexed_count = checkpoint.get("indexed_count")
        if expected_child_count != retrievable_count or expected_indexed_count != indexed_count:
            raise ActivationError(
                "Activation artifact validation failed: checkpoint counts differ from index."
            )
        self._verify_typed_inventory_gate(context, version_key)
        prepare_canonical_profile_source(context.document, children)
        ensure_source_identity(context.document, context.document.title)
        ensure_paper_profile(context.document)
        return {
            "parse_version": version_key,
            **total_counts,
            "contextualization_eligible": len(eligible),
            "contextualized": contextualized_count,
            "plain": len(plain),
            "plain_embedded": plain_embedding_count,
        }

    def _verify_typed_inventory_gate(
        self,
        context,
        version_key: str,
    ) -> None:
        """逐表校验 canonical manifest 库存与 DB/index 库存完全一致。

        比较绑定同一 ``document_id + parse_version``；任何缺失、重复、
        跨版本混入或多余条目都会抛 ``ActivationError``，阻止 active
        pointer 切换。canonical bundle 缺失、typed_inventory 缺失/损坏
        或 legacy bundle 需要迁移时同样失败关闭（不静默跳过）。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore

        store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
        try:
            manifest_inventory = store.load_typed_inventory(
                context.document.id,
                version_key,
            )
        except FileNotFoundError as exc:
            # 激活必须失败关闭：canonical bundle 缺失时没有可比较的 manifest
            # 库存，任何成功返回都会让缺库版本被错误激活。
            raise ActivationError(
                "Activation typed inventory validation failed: canonical bundle "
                f"is missing for {context.document.id} {version_key}; rebuild "
                "is required."
            ) from exc
        except ValueError as exc:
            # typed_inventory 缺失/损坏，或 legacy bundle 需要迁移：一律
            # 失败关闭并透出可操作的诊断信息。
            raise ActivationError(
                "Activation typed inventory validation failed: " + str(exc)
            ) from exc
        chunk_rows = list(
            context.db.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == context.document.id,
                    DocumentChunk.parse_version == version_key,
                )
            ).all()
        )
        observed = derive_db_typed_inventory(manifest_inventory, chunk_rows)
        mismatches = compare_typed_inventory(manifest_inventory, observed)
        if mismatches:
            raise ActivationError(
                "Activation typed inventory validation failed: "
                + "; ".join(mismatches)
            )

    def _persist_versioned_chunks(self, context, payload: list[dict]) -> None:
        """把索引产物持久化为版本化 DocumentChunk 行，并替换向量索引。

        流程：
        1. 先删除该 (document, parse_version) 下的旧分块（版本隔离）。
        2. 父块：校验 parse_version 后写入，建立 local_id -> record 映射。
        3. 子块：按上下文化策略选择模型（ContextualizedChunk / ChunkDraft），
           校验父块存在、解析嵌入向量后写入；同时收集 ChunkVector。
        4. 回填前后分块指针（previous/next）。
        5. 用向量仓库整体替换该文档的分块向量，并校验数量完整。
        """
        from app.services.contextualization import ContextualizedChunk
        from app.services.semantic_chunking import ChunkDraft

        version_key = context.version.version_key
        # 版本化覆盖：先清空该版本的旧分块
        context.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == context.document.id,
            DocumentChunk.parse_version == version_key,
        ).delete(synchronize_session=False)
        records: dict[str, DocumentChunk] = {}
        items_by_role = {
            role: [item for item in payload if item["chunk"]["chunk_role"] == role]
            for role in ("parent", "child")
        }
        for item in items_by_role["parent"]:
            draft = ChunkDraft.model_validate(item["chunk"])
            if draft.parse_version != version_key:
                raise RuntimeError("Chunk parse version does not match the index version.")
            record = self._document_chunk_from_draft(context, draft, embedding=None)
            context.db.add(record)
            records[draft.local_id] = record
        context.db.flush()
        vectors: list[ChunkVector] = []
        child_drafts: list[tuple[dict, ChunkDraft]] = []
        for item in items_by_role["child"]:
            chunk = item["chunk"]
            try:
                if requires_contextualization(str(chunk.get("block_type"))):
                    if not valid_contextualized_embedding(chunk):
                        raise RuntimeError(
                            "Structured Child violates the contextualization policy."
                        )
                    draft = ContextualizedChunk.model_validate(chunk)
                else:
                    if not valid_plain_embedding(chunk):
                        raise RuntimeError(
                            "Plain Child violates the contextualization policy."
                        )
                    draft = ChunkDraft.model_validate(chunk)
            except ValueError as exc:
                raise RuntimeError(
                    "Child violates the contextualization policy."
                ) from exc
            if draft.parse_version != version_key:
                raise RuntimeError("Chunk parse version does not match the index version.")
            parent = records.get(draft.parent_local_id or "")
            if parent is None:
                raise RuntimeError("Contextualized Child references a missing Parent chunk.")
            embedding = [float(value) for value in item["embedding"]]
            record = self._document_chunk_from_draft(
                context,
                draft,
                embedding=embedding,
                parent_chunk_id=parent.id,
            )
            context.db.add(record)
            records[draft.local_id] = record
            child_drafts.append((item, draft))
            vectors.append(
                ChunkVector(
                    chunk_id=record.id,
                    document_id=context.document.id,
                    embedding=embedding,
                    parse_version=version_key,
                )
            )
        context.db.flush()
        for _item, draft in child_drafts:
            record = records[draft.local_id]
            record.previous_chunk_id = draft.previous_child_local_id
            record.next_chunk_id = draft.next_child_local_id
        context.db.flush()
        vector_store = get_vector_store(context.db)
        vector_store.replace_document_chunks(
            context.document.id,
            vectors,
        )
        if vector_store.available() and vector_store.count_document_chunks(
            context.document.id, version_key
        ) != len(vectors):
            raise RuntimeError("Version-scoped vector index is incomplete.")

    @staticmethod
    def _persist_table_row_indices(
        draft,
        source_spans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """把表格子块的行下标持久化到 source_spans 的 metadata（JSON 契约）。

        text/Markdown 解析的表格在 source_spans 上通常只带表级定位，没有逐
        单元格的 ``row_index``；行覆盖由 chunk metadata 的 ``row_indices``
        表达。若持久化阶段不保留它，激活时 ``derive_db_typed_inventory`` 只能
        观察到空行覆盖，DB 库存门会错误地拒绝完整覆盖的表格。

        本方法把 ``draft.metadata["row_indices"]`` 以 ``metadata.row_indices``
        键写入每个持久化 span：不改数据库 schema，也不改动 span 既有的定位
        字段（HTML/DOCX 的逐行 ``row_index`` 保持不变，且 ``derive_db_typed
        _inventory`` 会同时读取两者）。父块不参与行覆盖，因此只处理 child
        角色；无合法行下标时原样返回。

        仅脚注子块（``metadata`` 携带 ``footnote_index`` 契约）不持久化行
        覆盖——它们为引用上下文携带完整 ``row_indices`` 但无数据行——而是把
        脚注契约标记写入 span metadata，使 DB 派生的 typed inventory 能识别
        并整体排除它们。
        """
        if draft.chunk_role != "child" or draft.block_type != "table":
            return source_spans
        metadata = draft.metadata or {}
        if _is_footnote_only_chunk(metadata):
            if not source_spans:
                return source_spans
            return [
                {
                    **span,
                    "metadata": {
                        **dict(span.get("metadata") or {}),
                        "footnote_index": metadata.get("footnote_index"),
                    },
                }
                for span in source_spans
            ]
        row_indices = metadata.get("row_indices")
        if not isinstance(row_indices, list):
            return source_spans
        normalized = sorted(_normalize_row_indices(row_indices))
        if not normalized:
            return source_spans
        return [
            {
                **span,
                "metadata": {
                    **dict(span.get("metadata") or {}),
                    "row_indices": normalized,
                },
            }
            for span in source_spans
        ]

    @staticmethod
    def _document_chunk_from_draft(
        context,
        draft,
        *,
        embedding: list[float] | None,
        parent_chunk_id: str | None = None,
    ) -> DocumentChunk:
        """把分块草稿（ChunkDraft/ContextualizedChunk）转换为 DocumentChunk 记录。

        - 规范化 ``contextualized_at`` 时间戳（字符串转 datetime）。
        - 源跨度转 JSON；页码取首个带 page_label 的跨度。
        - 表格子块的行覆盖（``metadata.row_indices``）写入持久化 span 的
          metadata，保证激活时 DB 派生的 typed inventory 能读到同一份 row
          覆盖（text/Markdown 表格的 span 通常没有逐行 row_index）。
        - 其余字段直接映射（含上下文前缀、上下文化模型/版本、切分器信息、
          语义边界分数、token 计数等）。
        """
        # PostgreSQL text 字段不接受 NUL (0x00) 字节；PDF 解析产物偶含
        # 二进制 NUL，写库前统一剔除，避免 index 阶段 DataError。
        def _strip_nul(value):
            return value.replace("\x00", "") if isinstance(value, str) else value

        contextualized_at = getattr(draft, "contextualized_at", None)
        if isinstance(contextualized_at, str):
            contextualized_at = datetime.fromisoformat(contextualized_at)
        source_spans = [span.model_dump(mode="json") for span in draft.source_spans]
        source_spans = IngestionPipeline._persist_table_row_indices(
            draft, source_spans
        )
        page_label = next(
            (span.get("page_label") for span in source_spans if span.get("page_label")),
            None,
        )
        return DocumentChunk(
            id=draft.local_id,
            document_id=context.document.id,
            parse_version=context.version.version_key,
            parent_chunk_id=parent_chunk_id,
            chunk_role=draft.chunk_role,
            block_type=draft.block_type,
            ordinal=draft.ordinal,
            heading=_strip_nul(draft.section_path[-1] if draft.section_path else None),
            page_label=page_label,
            section_path=list(draft.section_path),
            source_block_ids=list(draft.source_block_ids),
            source_spans=source_spans,
            text=_strip_nul(draft.text),
            contextual_prefix=_strip_nul(getattr(draft, "contextual_prefix", None)),
            embedding_text=_strip_nul(draft.embedding_text),
            contextualization_model=getattr(draft, "contextualization_model", None),
            contextualization_version=getattr(draft, "contextualization_version", None),
            contextualization_prompt_version=getattr(
                draft, "contextualization_prompt_version", None
            ),
            contextualized_at=contextualized_at,
            splitter_name=draft.splitter_name,
            splitter_version=draft.splitter_version,
            splitting_model=draft.splitting_model,
            semantic_boundary_score=draft.semantic_boundary_score,
            token_count=draft.token_count,
            token_estimate=draft.token_count,
            previous_chunk_id=None,
            next_chunk_id=None,
            embedding=embedding,
        )

    @staticmethod
    def _chunk_has_valid_source_spans(chunk: DocumentChunk) -> bool:
        """校验分块的源跨度有效：非空且每个跨度至少含一个定位字段。

        定位字段包括 page_index / source_block_id / paragraph_id /
        table_id / image_relationship_id / xpath / css_selector /
        element_id / line_start / char_start 之一。用于激活完整性检查。
        """
        from app.services.canonical_models import SourceSpan

        spans = chunk.source_spans
        if not isinstance(spans, list) or not spans:
            return False
        locator_fields = (
            "page_index",
            "source_block_id",
            "paragraph_id",
            "table_id",
            "image_relationship_id",
            "xpath",
            "css_selector",
            "element_id",
            "line_start",
            "char_start",
        )
        try:
            validated = [SourceSpan.model_validate(span) for span in spans]
        except Exception:
            return False
        return all(any(getattr(span, field) is not None for field in locator_fields) for span in validated)

    def _indexed_child_count(
        self, document_id: str, parse_version: str, *, fallback: int
    ) -> int:
        """返回向量仓库中该文档版本的已索引子块数；仓库不可用时返回兜底值。"""
        store = get_vector_store(self.db)
        if not store.available():
            return fallback
        return store.count_document_chunks(document_id, parse_version)

    def _write_stage_artifact(
        self,
        context,
        filename: str,
        payload: object,
        *,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """把阶段产物以"带指纹 JSON"原子写盘，返回产物路径与指纹。

        - 序列化使用紧凑、无 NaN、ensure_ascii=False 的 JSON。
        - 已存在时校验内容一致（可重放）；否则以临时文件 + rename 原子写入。
        - 返回 ``{"artifact_path", "artifact_sha256", **extra}``，
          供下一阶段通过 ``_load_previous_stage_artifact`` 读取校验。
        """
        directory = self._stage_artifact_dir(context)
        path = directory / filename
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        if path.exists() and path.read_bytes() != encoded:
            raise RuntimeError(f"Stage artifact conflicts with checkpoint: {path}")
        if not path.exists():
            temporary = directory / f".{filename}.{uuid4().hex}.tmp"
            try:
                temporary.write_bytes(encoded)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "artifact_path": str(path),
            "artifact_sha256": fingerprint,
            **dict(extra or {}),
        }

    def _load_previous_stage_artifact(self, context):
        """读取并校验上一阶段产物：路径必须在版本目录内且指纹一致。

        校验：artifact_path 是字符串且其父目录等于本版本的 stage 目录、
        文件存在、内容 SHA-256 与检查点中的指纹一致。通过后 JSON 解码返回。
        """
        checkpoint = context.input.get("previous_output") or {}
        raw_path = checkpoint.get("artifact_path")
        expected_hash = checkpoint.get("artifact_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"Stage {context.stage!r} requires an artifact reference.")
        path = Path(raw_path).resolve()
        directory = self._stage_artifact_dir(context).resolve()
        # 产物必须位于版本目录内（防路径逃逸）
        if path.parent != directory or not path.is_file():
            raise RuntimeError(f"Stage artifact is missing or outside its version: {path}")
        encoded = path.read_bytes()
        # 指纹校验：产物必须与检查点一致
        if hashlib.sha256(encoded).hexdigest() != expected_hash:
            raise RuntimeError(f"Stage artifact fingerprint mismatch: {path}")
        return json.loads(encoded)

    def _stage_artifact_dir(self, context) -> Path:
        """返回（并在必要时创建）当前版本专属的 stage 产物目录。

        目录为 ``artifacts/<document_id>/<version_key>.pipeline``；创建前
        校验各路径成分安全（无符号链接、不逃逸文档根），并把
        ``context.version.artifact_dir`` 记录为绝对路径。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore

        store = CanonicalArtifactStore(settings.canonical_artifacts_dir)
        store._validate_component(context.document.id)  # noqa: SLF001
        store._validate_component(context.version.version_key)  # noqa: SLF001
        document_root = store._prepare_document_root(  # noqa: SLF001
            context.document.id,
            create=True,
        )
        directory = document_root / f"{context.version.version_key}.pipeline"
        store._validate_component(directory.name)  # noqa: SLF001
        # 拒绝符号链接/重解析点（防目录逃逸攻击）
        if store._is_link_or_reparse_point(directory):  # noqa: SLF001
            raise ValueError(
                f"stage artifact directory cannot be a symbolic link: {directory}"
            )
        if directory.exists():
            if not directory.is_dir():
                raise ValueError(
                    f"stage artifact directory is not a directory: {directory}"
                )
        else:
            directory.mkdir()
        resolved = directory.resolve()
        # 最终解析路径必须仍在文档根之下
        if resolved.parent != document_root.resolve():
            raise ValueError("Stage artifact directory escapes its document root.")
        context.version.artifact_dir = str(resolved)
        return resolved

    @staticmethod
    def _outline_titles(nodes) -> list[str]:
        """递归收集大纲节点的全部标题（含子节点），用于上下文增强。"""
        return [
            title
            for node in nodes
            for title in [node.title, *IngestionPipeline._outline_titles(node.children)]
        ]

    @staticmethod
    def _checkpoint_artifact_path(context) -> Path:
        """校验并返回上一阶段的 canonical 清单（manifest.json）路径。

        校验：文件名必须是 ``manifest.json`` 且存在；解析后的父级父级
        目录必须等于文档根；清单所属版本必须等于当前版本或
        ``<版本>.staging-*``。
        """
        value = context.input.get("previous_output") or {}
        artifact_path = value.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise RuntimeError(
                f"Stage {context.stage!r} requires a canonical artifact checkpoint."
            )
        path = Path(artifact_path)
        if path.name != "manifest.json" or not path.is_file():
            raise RuntimeError(f"Canonical artifact checkpoint is missing: {path}")
        resolved = path.resolve()
        document_root = (
            settings.canonical_artifacts_dir / context.document.id
        ).resolve()
        # 清单必须位于文档根下的 bundle 目录（防逃逸）
        if resolved.parent.parent != document_root:
            raise RuntimeError(
                f"Canonical artifact checkpoint escapes its document root: {path}"
            )
        bundle_name = resolved.parent.name
        version_key = context.version.version_key
        # 版本必须匹配：最终版本目录或 staging 目录
        if bundle_name != version_key and not bundle_name.startswith(
            f"{version_key}.staging-"
        ):
            raise RuntimeError(
                f"Canonical artifact checkpoint has the wrong version: {path}"
            )
        return resolved

    def _checkpoint_canonical_draft(self, context) -> Path:
        """校验并返回解析阶段草稿（parse.canonical.json）的路径。

        校验：文件名必须为 ``parse.canonical.json`` 且存在，其父目录必须
        等于本版本的 stage 目录（防路径逃逸）。
        """
        value = context.input.get("previous_output") or {}
        artifact_path = value.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise RuntimeError("Repair stage requires a canonical parse draft checkpoint.")
        path = Path(artifact_path)
        expected_name = "parse.canonical.json"
        if path.name != expected_name or not path.is_file():
            raise RuntimeError(f"Canonical parse draft checkpoint is missing: {path}")
        resolved = path.resolve()
        document_root = self._stage_artifact_dir(context).resolve()
        if resolved.parent != document_root:
            raise RuntimeError(
                f"Canonical parse draft escapes its document root: {path}"
            )
        return resolved

    def register_document(self, project_slug: str, project_name: str, file_path: Path) -> tuple[Project, Document, PipelineRun]:
        """注册文档（按内容 SHA-256 去重），返回 (project, document, run)。

        流程：
        1. 获取（或创建）项目；计算文件 SHA-256。
        2. 按 (project, sha256) 查重：
           - 已存在且状态 ready：跳过，记一条 completed 的"重复跳过"运行。
           - 已存在但未 ready：更新 raw_path、上传对象存储、置 pending，
             记一条 queued 的"重新入队"运行（记录前状态作为 retry_reason）。
        3. 新文档：上传对象存储，创建 Document（标题取可读文件名），
           记一条 queued 运行。
        """
        project = get_or_create_project(self.db, slug=project_slug, name=project_name)
        sha256 = compute_sha256(file_path)
        existing = self.db.scalar(select(Document).where(Document.project_id == project.id, Document.sha256 == sha256))
        if existing:
            if existing.status == DocumentStatus.ready.value:
                run = PipelineRun(project_id=project.id, document_id=existing.id, run_type=RunType.ingest.value, status=RunStatus.completed.value, notes="Duplicate document skipped.")
                self.db.add(run)
                self.db.commit()
                self.db.refresh(run)
                return project, existing, run

            previous_status = existing.status
            existing.raw_path = str(file_path)
            existing.object_key = self.storage.upload(file_path, f"{project.slug}/{file_path.name}")
            existing.status = DocumentStatus.pending.value
            metadata = dict(existing.metadata_json or {})
            metadata["stored_file_name"] = file_path.name
            metadata["retry_reason"] = f"Duplicate upload requeued from status '{previous_status}'."
            existing.metadata_json = metadata
            run = PipelineRun(
                project_id=project.id,
                document_id=existing.id,
                run_type=RunType.ingest.value,
                status=RunStatus.queued.value,
                notes="Duplicate upload requeued because previous ingest was not ready.",
            )
            self.db.add(run)
            self.db.commit()
            self.db.refresh(existing)
            self.db.refresh(run)
            return project, existing, run

        object_key = self.storage.upload(file_path, f"{project.slug}/{file_path.name}")
        document = Document(
            project_id=project.id,
            title=display_title_from_path(file_path),
            file_name=strip_upload_prefix(file_path.name),
            sha256=sha256,
            raw_path=str(file_path),
            object_key=object_key,
            metadata_json={"stored_file_name": file_path.name},
            status=DocumentStatus.pending.value,
        )
        self.db.add(document)
        self.db.flush()

        run = PipelineRun(project_id=project.id, document_id=document.id, run_type=RunType.ingest.value, status=RunStatus.queued.value)
        self.db.add(run)
        self.db.commit()
        self.db.refresh(document)
        self.db.refresh(run)
        return project, document, run

    def process_document(self, document_id: str) -> PipelineRun:
        """处理文档入口：无 Redis 走 legacy 同步路径，否则入队阶段任务。

        - 无 Redis（legacy 模式）：同步执行 ``_process_document_legacy``，
          完成后激活 legacy 解析版本。
        - 有 Redis：获取/创建解析版本，找首个可执行阶段；若全部完成则
          直接置 ready/active，否则把该阶段入队（JobDispatcher），
          记录 queued 运行。

        返回：本次的 PipelineRun。
        """
        document = self.db.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document {document_id} not found")
        if not settings.redis_url:
            # legacy 同步路径
            run = self._process_document_legacy(document_id)
            if run.status == RunStatus.completed.value:
                self._activate_legacy_parse_version(document_id)
            return run

        # canonical 阶段机路径：获取/创建解析版本
        version = self._get_or_create_parse_version(document)
        self.db.commit()

        run = self.db.scalar(
            select(PipelineRun)
            .where(PipelineRun.document_id == document_id)
            .order_by(PipelineRun.created_at.desc())
        )
        if run is None:
            run = PipelineRun(
                project_id=document.project_id,
                document_id=document.id,
                run_type=RunType.ingest.value,
                status=RunStatus.queued.value,
            )
            self.db.add(run)
            self.db.flush()
        actionable_stage = self._first_actionable_stage(version)
        if actionable_stage is None:
            run.status = RunStatus.completed.value
            document.status = DocumentStatus.ready.value
            if version.status == "active":
                document.active_parse_version = version.version_key
            self._set_progress(
                run,
                100,
                "completed",
                f"Canonical ingestion version {version.version_key} is active.",
                commit=False,
            )
            self.db.commit()
            return run
        run.status = RunStatus.queued.value
        if document.active_parse_version in (None, version.version_key):
            document.status = DocumentStatus.processing.value
        self._set_progress(
            run,
            5,
            "queued",
            f"Canonical ingestion version {version.version_key} queued for {actionable_stage}.",
            commit=False,
        )
        self.db.commit()
        from app.services.queue import JobDispatcher

        JobDispatcher().enqueue_stage(
            document.id,
            version.version_key,
            actionable_stage,
        )
        return run

    def _get_or_create_parse_version(
        self, document: Document
    ) -> DocumentParseVersion:
        """按当前配置计算版本键并获取/创建解析版本（get-or-create）。

        版本键 = ``{pipeline版本}-{源文件哈希前12}-{配置哈希前12}``；
        已存在时校验配置快照一致；新建时把配置快照写入 manifest。
        """
        pipeline_version = str(settings.canonical_pipeline_version)
        self._validate_version_component(
            pipeline_version,
            label="CANONICAL_PIPELINE_VERSION",
        )
        ingestion_config = build_ingestion_config_snapshot()
        ingestion_config_sha256 = canonical_ingestion_config_hash(ingestion_config)
        version_key = build_parse_version_key(
            document.sha256,
            snapshot=ingestion_config,
            config_sha256=ingestion_config_sha256,
            settings=settings,
        )
        self._validate_version_component(version_key, label="parse version key")
        existing = self.db.scalar(
            select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document.id,
                DocumentParseVersion.version_key == version_key,
            )
        )
        if existing is not None:
            require_matching_ingestion_config(
                existing.manifest_json,
                ingestion_config,
            )
            return existing
        artifact_dir = (
            settings.canonical_artifacts_dir
            / document.id
            / f"{version_key}.pipeline"
        )
        version = ParseVersionService(self.db).create(
            document.id,
            version_key,
            str(artifact_dir),
        )
        version.manifest_json = {
            "ingestion_config": ingestion_config,
            "ingestion_config_sha256": ingestion_config_sha256,
        }
        return version

    def _activate_legacy_parse_version(self, document_id: str) -> None:
        """激活 legacy 解析版本（兼容旧路径）：记录 stage_state 并激活。

        要求文档状态为 ready。若无 ``legacy`` 版本则创建；在 stage_state
        中记录 legacy 已完成的分块数；若尚未激活则置 ready_to_activate 并
        经 ``ParseVersionService.activate`` 激活，否则仅回填文档指针。
        """
        document = self.db.get(Document, document_id)
        if document is None or document.status != DocumentStatus.ready.value:
            raise RuntimeError("Legacy activation requires a ready document.")
        version = self.db.scalar(
            select(DocumentParseVersion).where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == "legacy",
            )
        )
        if version is None:
            version = ParseVersionService(self.db).create(
                document_id,
                "legacy",
                "legacy",
                parser_name="legacy",
            )
        version.stage_state = {
            **dict(version.stage_state or {}),
            "legacy": {
                "status": "completed",
                "chunk_count": self.db.scalar(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.document_id == document_id,
                        DocumentChunk.parse_version == "legacy",
                    )
                ),
            },
        }
        if version.status != "active":
            version.status = "ready_to_activate"
            self.db.flush()
            ParseVersionService(self.db).activate(document, version)
        else:
            document.active_parse_version = "legacy"
        self.db.commit()

    @staticmethod
    def _validate_version_component(value: str, *, label: str) -> None:
        """校验版本相关字符串可作为安全路径成分（白名单正则）。"""
        if (
            not value
            or value in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)
        ):
            raise ValueError(f"{label} is not a safe path component: {value!r}.")

    @staticmethod
    def _first_actionable_stage(version: DocumentParseVersion) -> str | None:
        """返回第一个未完成（非 completed）的 ingestion 阶段；全部完成则 None。"""
        from app.services.ingestion_stages import INGESTION_STAGES

        state = version.stage_state or {}
        return next(
            (
                stage
                for stage in INGESTION_STAGES
                if (state.get(stage) or {}).get("status") != "completed"
            ),
            None,
        )

    def _process_document_legacy(self, document_id: str) -> PipelineRun:
        """legacy 同步处理：解析 -> 质量门 -> 分块嵌入 -> SAC-KG 抽取。

        流程：
        1. 标记 run/document 为 processing。
        2. 解析文档；解析质量（canonical）不合格则标记失败并返回。
        3. 替换分块并计算嵌入。
        4. 若未启用 SAC-KG：直接 ready 完成（RAG-only）。
        5. 启用 SAC-KG：抽取实体/三元组、做实体增长决策、创建审阅项，
           最后 ready 完成。
        任何异常：回滚，把文档/运行标记失败后重抛。

        返回：本次 PipelineRun。
        """
        document = self.db.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document {document_id} not found")
        run = self.db.scalar(select(PipelineRun).where(PipelineRun.document_id == document_id).order_by(PipelineRun.created_at.desc()))
        if run is None:
            run = PipelineRun(project_id=document.project_id, document_id=document.id, run_type=RunType.ingest.value, status=RunStatus.queued.value)
            self.db.add(run)
            self.db.flush()

        try:
            run.status = RunStatus.running.value
            document.status = DocumentStatus.processing.value
            self._set_progress(run, 5, "started", "Worker accepted the ingest job.")
            self.db.commit()

            self._set_progress(run, 15, "parsing", "Parsing document and building page/snippet structure.")
            parsed = parse_document(Path(document.raw_path))
            document.title = self._resolve_document_title(parsed, document)
            document.raw_text = parsed.text
            merged_metadata = dict(document.metadata_json or {})
            merged_metadata.update(parsed.metadata)
            document.metadata_json = merged_metadata
            ensure_source_identity(document, document.title)
            ensure_paper_profile(document)
            merged_metadata = dict(document.metadata_json or {})
            canonical_metadata = parsed.metadata.get("canonical", {})
            canonical_quality = canonical_metadata.get("quality", {})
            canonical_quality_status = canonical_quality.get("status")
            canonical_quality_accepted = canonical_quality.get("accepted") is True
            activation_allowed = canonical_metadata.get(
                "table_activation_allowed", True
            )
            quality_accepted = (
                canonical_quality_accepted
                and canonical_quality_status
                in {"accepted", "accepted_with_warnings"}
                and activation_allowed is not False
            )
            quality_rejected = not quality_accepted
            if quality_accepted:
                quality_report_status = "ok"
            elif canonical_quality_status not in {
                "accepted",
                "accepted_with_warnings",
                None,
            }:
                quality_report_status = canonical_quality_status
            else:
                quality_report_status = "validation_failed"
            quality_report = {
                "status": quality_report_status,
                "document_id": document.id,
                "canonical_status": canonical_metadata.get("status"),
                "canonical_quality": canonical_quality,
                "table_activation_allowed": activation_allowed,
                "table_repair_requests": canonical_metadata.get(
                    "table_repair_requests", []
                ),
            }
            merged_metadata["ingest_quality"] = quality_report
            if quality_rejected:
                error = (
                    "Canonical structured evidence validation failed; "
                    "document activation was blocked before chunk persistence."
                )
                merged_metadata["ingest_error"] = error
                document.metadata_json = merged_metadata
                document.status = DocumentStatus.failed.value
                run.status = RunStatus.failed.value
                run.notes = error
                run.provider_report = {
                    **dict(run.provider_report or {}),
                    "ingest_quality": quality_report,
                    "error": error,
                }
                self._set_progress(run, 100, "failed", error)
                self.db.commit()
                return run
            document.metadata_json = merged_metadata
            self._set_progress(run, 30, "chunking", "Replacing document chunks and preparing embeddings.")
            self._replace_chunks(document, parsed.chunks)

            if not settings.sac_kg_enabled:
                self._set_progress(run, 82, "indexing_rag", "Skipping SAC-KG extraction; RAG chunks and vector index are ready.")
                document.status = DocumentStatus.ready.value
                run.status = RunStatus.completed.value
                run.provider_report = {
                    **dict(run.provider_report or {}),
                    "entities": 0,
                    "claims": 0,
                    "review_items": 0,
                    "ingest_quality": quality_report,
                    "sac_kg_enabled": False,
                }
                self._set_progress(run, 100, "completed", "RAG-only ingest completed successfully.")
                self.db.commit()
                return run

            generation_provider = self._sac_kg_generation_provider()
            self._set_progress(
                run,
                45,
                "extracting",
                f"Generating SAC-KG routing facts with {generation_provider}.",
            )
            extraction = self._extract_document(document, parsed.text)
            ensure_source_identity(document, extraction.title or document.title)
            ensure_paper_profile(document)
            self._set_progress(run, 65, "structuring", "Writing entities, claims, verifier metadata, and pruner decisions.")
            entities = self._upsert_entities(document.project_id, extraction)
            claims = self._create_claims(document, extraction)
            entity_decisions = self._decide_entity_growth(document, entities, claims)
            self._apply_claim_growth_decisions(claims, entity_decisions)
            entities = self._ensure_growing_entities(document.project_id, entities, claims, entity_decisions)
            self._set_progress(run, 82, "indexing_rag", "RAG and SAC-KG artifacts are ready.")
            self._set_progress(run, 94, "reviewing", "Creating review items and final provider report.")
            review_count = self._create_review_items(document, extraction, claims)

            document.status = DocumentStatus.ready.value
            run.status = RunStatus.completed.value
            run.provider_report = {
                **dict(run.provider_report or {}),
                "entities": len(entities),
                "claims": len(claims),
                "review_items": review_count,
                "ingest_quality": quality_report,
                "sac_kg_enabled": True,
                "generation_provider": generation_provider,
            }
            self._set_progress(run, 100, "completed", "Ingest completed successfully.")
            self.db.commit()
            return run
        except Exception as exc:  # noqa: BLE001
            logger.exception("Document processing failed")
            document_id = document.id
            run_id = run.id
            self.db.rollback()
            failed_document = self.db.get(Document, document_id)
            failed_run = self.db.get(PipelineRun, run_id)
            if failed_document is not None:
                self._clear_document_chunks(failed_document.id)
                failed_document.status = DocumentStatus.failed.value
            if failed_run is not None:
                failed_run.status = RunStatus.failed.value
                failed_run.notes = str(exc)
                self._set_progress(failed_run, 100, "failed", str(exc))
            self.db.commit()
            raise

    def _resolve_document_title(self, parsed, document: Document) -> str:
        """Choose the best display title for a document.

        Preference order:
        1. A non-empty title supplied by the parser (e.g. MinerU or metadata).
        2. A title found in the parsed metadata.
        3. The existing sanitized filename, unless it is just an internal
           UUID/hash sample name.
        4. A generic fallback.

        Internal UUID/hash/sample identifiers are never promoted to the
        primary display title when a human-readable alternative exists.
        """
        candidates: list[str] = []
        # 候选 1：解析器给出的标题（非内部样本）
        if parsed.title and not looks_like_internal_sample(parsed.title):
            candidates.append(parsed.title.strip())
        # 候选 2：解析元信息中的 title
        metadata_title = (parsed.metadata or {}).get("title")
        if metadata_title and not looks_like_internal_sample(metadata_title):
            candidates.append(str(metadata_title).strip())
        # 候选 3：现有标题 / 可读文件名
        existing_title = document.title or readable_title_from_path(Path(document.raw_path))
        if existing_title and not looks_like_internal_sample(existing_title):
            candidates.append(existing_title.strip())
        if candidates:
            return candidates[0]
        # Final fallback: anything we have, stripped of the upload UUID prefix.
        # 最终兜底：去掉上传 UUID 前缀后的任意可用名称
        fallback = strip_upload_prefix(
            document.title or document.file_name or Path(document.raw_path).stem or ""
        ).strip()
        if fallback and not looks_like_internal_sample(fallback):
            return fallback
        return "Untitled document"

    def _set_progress(
        self,
        run: PipelineRun,
        percent: int,
        stage: str,
        message: str,
        *,
        commit: bool = True,
    ) -> None:
        """更新运行进度（percent 截断到 0-100）并写日志；可选提交。"""
        report = dict(run.provider_report or {})
        report["progress"] = {
            "percent": max(0, min(percent, 100)),
            "stage": stage,
            "message": message,
        }
        run.provider_report = report
        logger.info(
            "[ingest progress] %s %s%% %s document_id=%s run_id=%s - %s",
            self._progress_bar(percent),
            max(0, min(percent, 100)),
            stage,
            run.document_id or "-",
            run.id,
            message,
        )
        if commit:
            self.db.commit()

    @staticmethod
    def _progress_bar(percent: int, width: int = 24) -> str:
        """渲染文本进度条（``[####-----]``）供日志展示。"""
        normalized = max(0, min(percent, 100))
        filled = round(width * normalized / 100)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _replace_chunks(self, document: Document, parsed_chunks) -> None:
        """（legacy）用解析分块替换文档分块并重算嵌入向量。

        先删除该文档全部旧分块，批量嵌入文本（失败时退回全空向量），
        写新分块行后替换向量索引（仅含非空嵌入的块）。
        """
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
        texts = [chunk.text for chunk in parsed_chunks]
        embeddings = safe_model_call(lambda: self.ollama.embed(texts), [[] for _ in texts])
        records: list[DocumentChunk] = []
        for chunk, embedding in zip(parsed_chunks, embeddings, strict=False):
            record = DocumentChunk(
                document_id=document.id,
                ordinal=chunk.ordinal,
                heading=chunk.heading,
                page_label=chunk.page_label,
                text=chunk.text,
                token_estimate=max(1, len(chunk.text) // 4),
                embedding=embedding or None,
            )
            self.db.add(record)
            records.append(record)
        self.db.flush()
        get_vector_store(self.db).replace_document_chunks(
            document.id,
            [
                ChunkVector(chunk_id=record.id, document_id=document.id, embedding=record.embedding or [])
                for record in records
                if record.embedding
            ],
        )

    def _clear_document_chunks(self, document_id: str) -> None:
        """删除文档的全部向量与分块行（失败清理用）。"""
        get_vector_store(self.db).delete_document(document_id)
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

    def _extract_document(self, document: Document, full_text: str) -> DocumentExtraction:
        """执行 SAC-KG 抽取：种子分析 -> 候选 head -> head 级分析 -> 合并。

        流程：
        1. 选生成上下文、构建句条目、做种子文档分析。
        2. 收集候选 head（最多 HEAD_MAX_COUNT 个）。
        3. 对每个 head 检索上下文与示例，生成/校验/纠错其 head 分析。
        4. 合并为最终抽取；无任何 claims 时退回兜底抽取。
        """
        chunks = self.db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.ordinal)).all()
        contexts = self._select_generation_contexts(document, full_text, list(chunks))
        sentence_entries = self._build_sentence_entries(list(chunks))
        corpus_context = ""
        seed_analysis = self._seed_document_analysis(document, full_text, contexts, corpus_context)
        candidate_heads = self._collect_candidate_heads(document, full_text, seed_analysis)
        previous_claims = self._project_verified_claims(document.project_id)

        head_payloads: list[HeadAnalysisPayload] = []
        for head in candidate_heads[:HEAD_MAX_COUNT]:
            head_contexts = self._retrieve_head_contexts(head, sentence_entries)
            examples = self._open_kg_examples(document.project_id, head["name"])
            head_payloads.append(
                self._generate_head_analysis(
                    document=document,
                    head=head,
                    contexts=head_contexts,
                    open_kg_examples=examples,
                    corpus_context=corpus_context,
                    previous_claims=previous_claims,
                )
            )

        extraction = self._merge_document_analysis(document, seed_analysis, head_payloads)
        if extraction.claims:
            return extraction

        fallback = self._analysis_to_extraction(document, self._fallback_analysis(document, full_text, contexts))
        fallback.coverage_notes.append("Head-driven generation produced no verified triples; fallback extraction used.")
        return fallback

    def _seed_document_analysis(
        self,
        document: Document,
        full_text: str,
        contexts: list[dict],
        corpus_context: str,
    ) -> DocumentAnalysisPayload:
        """生成文档级"种子分析"（摘要/关键事实/实体/概念/草稿三元组）。

        调用配置的结构化生成 provider（Ollama 或 DeepSeek）；失败退回
        ``_fallback_analysis``。
        """
        fallback = self._fallback_analysis(document, full_text, contexts)
        prompt = "\n\n".join(
            [
                f"Document title: {document.title}",
                "Task: extract a document overview for RAG routing and SAC-KG-style evidence organization.",
                (
                    "Return a concise summary, key facts, entities, concepts, and draft triples. "
                    "Preserve exact dates, doses, diagnoses, and recommendations."
                ),
                "Current corpus context:\n" + (corpus_context or "No existing corpus context."),
                "Retrieved document contexts:\n" + json.dumps(contexts, ensure_ascii=False),
            ]
        )
        return safe_model_call(
            lambda: self._generate_sac_kg_structured(
                DocumentAnalysisPayload,
                system_prompt=(
                    "You are preparing a structured overview for an internal document knowledge base. "
                    "Focus on candidate heads, key facts, and entity-rich summaries."
                ),
                user_prompt=prompt,
                model=settings.ollama_batch_model,
            ),
            fallback,
        )

    def _collect_candidate_heads(self, document: Document, full_text: str, seed_analysis: DocumentAnalysisPayload) -> list[dict]:
        """收集候选 head 实体列表（去重、清洗、排除瞬时值）。

        来源：文档标题（document 类型）、种子分析中的实体/概念、
        项目已验证主语（且出现在全文中）。候选过少时从关键事实中切分
        片段补充。最终为空时退回以文档标题为唯一候选。
        """
        candidates: list[dict] = []
        seen: set[str] = set()

        def add_candidate(name: str, *, entity_type: str = "concept", aliases: list[str] | None = None, summary: str = "") -> None:
            """清洗并去重后加入候选 head（排除瞬时值/空名）。"""
            cleaned = re.sub(r"\s+", " ", name).strip()
            if not cleaned or self._looks_like_transient_value(cleaned):
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "name": cleaned,
                    "entity_type": entity_type or "concept",
                    "aliases": aliases or [],
                    "summary": summary,
                }
            )

        add_candidate(document.title, entity_type="document")
        for entity in seed_analysis.entities:
            add_candidate(entity.name, entity_type=entity.entity_type, aliases=entity.aliases, summary=entity.summary)
        for concept in seed_analysis.concepts:
            add_candidate(concept, entity_type="concept")
        for subject in self._project_verified_subjects(document.project_id):
            if subject and subject in full_text:
                add_candidate(subject, entity_type="concept")
        if len(candidates) == 1 and seed_analysis.key_facts:
            for fact in seed_analysis.key_facts[:3]:
                for segment in re.findall(r"[\u4e00-\u9fff]{2,12}|[A-Z][a-zA-Z0-9_-]{2,}", fact):
                    add_candidate(segment, entity_type="concept")
        return candidates or [{"name": document.title, "entity_type": "document", "aliases": [], "summary": ""}]

    def _build_sentence_entries(self, chunks: list[DocumentChunk]) -> list[dict]:
        """把分块文本切成句条目，供 head 上下文检索。

        每个条目含 chunk_id/序号/页码/标题/文本/sentence_ref/嵌入向量。
        """
        entries: list[dict] = []
        for chunk in chunks:
            for sentence_index, snippet in enumerate(self._split_text_into_snippets(chunk.text)):
                entries.append(
                    {
                        "chunk_id": chunk.id,
                        "chunk_ordinal": chunk.ordinal,
                        "page_label": chunk.page_label,
                        "heading": chunk.heading,
                        "text": snippet,
                        "score": 0.0,
                        "sentence_ref": f"chunk-{chunk.ordinal}:sentence-{sentence_index}",
                        "embedding": chunk.embedding,
                    }
                )
        return entries

    def _split_text_into_snippets(self, text: str, max_chars: int = 320) -> list[str]:
        """按句末标点/换行切分文本为片段，并在超过 max_chars 时累积切分。"""
        raw_parts = [part.strip() for part in re.split(r"(?<=[。！？!?\.])\s+|\n+", text) if part.strip()]
        snippets: list[str] = []
        buffer = ""
        for part in raw_parts:
            candidate = f"{buffer} {part}".strip() if buffer else part
            if len(candidate) > max_chars and buffer:
                snippets.append(buffer)
                buffer = part
            else:
                buffer = candidate
        if buffer:
            snippets.append(buffer)
        return snippets or [text[:max_chars]]

    def _retrieve_head_contexts(self, head: dict, sentence_entries: list[dict]) -> list[dict]:
        """为某个 head 检索相关句条目（确定性打分 + 语义相似度）。

        打分项：head/别名出现次数、词元重叠、事实标记词、数值单位、
        标题命中、嵌入余弦相似度。取高分条目并按字符预算/条数上限截取。
        没有任何命中时退回前几个条目。
        """
        if not sentence_entries:
            return []
        head_terms = self._text_terms(head["name"])
        alias_terms = set().union(*(self._text_terms(alias) for alias in head.get("aliases", [])))
        query_vector = safe_model_call(lambda: self.ollama.embed([head["name"]])[0], [])
        scored: list[dict] = []
        for entry in sentence_entries:
            text = entry["text"]
            lowered = text.lower()
            exact_score = lowered.count(head["name"].lower()) * 6 if head["name"] else 0
            alias_score = sum(lowered.count(alias.lower()) for alias in head.get("aliases", []) if alias) * 4
            overlap_score = len(head_terms & self._text_terms(text)) * 2
            marker_score = sum(1 for marker in FACT_MARKERS if marker.lower() in lowered)
            value_score = 1 if FACT_VALUE_PATTERN.search(text) else 0
            heading_score = 2 if entry.get("heading") and head_terms & self._text_terms(entry["heading"]) else 0
            semantic_score = 0.0
            if query_vector and entry.get("embedding"):
                semantic_score = max(cosine_similarity(query_vector, entry["embedding"]), 0.0) * 4
            score = exact_score + alias_score + overlap_score + len(alias_terms & self._text_terms(text)) + marker_score + value_score + heading_score + semantic_score
            if score <= 0:
                continue
            scored.append({**entry, "score": round(score, 4)})
        if not scored:
            return [self._prompt_context_entry(entry) for entry in sentence_entries[: min(3, len(sentence_entries))]]
        ordered = sorted(scored, key=lambda item: item["score"], reverse=True)
        selected: list[dict] = []
        total_chars = 0
        for entry in ordered[:HEAD_SNIPPET_LIMIT * 2]:
            if total_chars + len(entry["text"]) > HEAD_CONTEXT_CHAR_BUDGET and selected:
                break
            selected.append(self._prompt_context_entry(entry))
            total_chars += len(entry["text"])
            if len(selected) >= HEAD_SNIPPET_LIMIT:
                break
        return selected

    @staticmethod
    def _prompt_context_entry(entry: dict) -> dict:
        """把句条目压缩为提示词上下文（文本截断 520 字符，剔除空字段）。"""
        compact = {
            "chunk_id": entry.get("chunk_id"),
            "chunk_ordinal": entry.get("chunk_ordinal"),
            "page_label": entry.get("page_label"),
            "heading": entry.get("heading"),
            "sentence_ref": entry.get("sentence_ref"),
            "score": entry.get("score"),
            "text": re.sub(r"\s+", " ", str(entry.get("text") or "")).strip()[:520],
        }
        return {key: value for key, value in compact.items() if value not in (None, "", [])}

    def _open_kg_examples(self, project_id: str, head_name: str, limit: int = 8) -> list[dict]:
        """为 head 检索"开放知识图谱"示例三元组。

        优先该 head 的已验证三元组（精确匹配）；其次按查找 token 模糊
        匹配；最后退回通用示例（GENERIC_TRIPLE_EXAMPLES）。
        """
        verified_claims = self._project_verified_claims(project_id)
        exact = [
            claim
            for claim in verified_claims
            if claim.subject.lower() == head_name.lower()
        ]
        if exact:
            return [self._claim_as_example(claim) for claim in exact[:limit]]

        tokens = [token for token in self._head_lookup_tokens(head_name) if len(token) > 1]
        fuzzy = [
            claim
            for claim in verified_claims
            if any(token in claim.subject.lower() or token in claim.object_text.lower() for token in tokens)
        ]
        if fuzzy:
            return [self._claim_as_example(claim) for claim in fuzzy[:limit]]

        return GENERIC_TRIPLE_EXAMPLES[:limit]

    def _generate_head_analysis(
        self,
        *,
        document: Document,
        head: dict,
        contexts: list[dict],
        open_kg_examples: list[dict],
        corpus_context: str,
        previous_claims: list[Claim],
    ) -> HeadAnalysisPayload:
        """对单个 head 生成分析（三元组优先），并经过验证/纠错。

        生成 base 提示词后调用结构化生成（失败退回兜底 head 分析），
        归一化后经 ``_verify_and_correct_head_analysis`` 校验并按需重试。
        """
        fallback = self._fallback_head_analysis(head, contexts)
        base_prompt = "\n\n".join(
            [
                f"Document title: {document.title}",
                f"Target head entity: {head['name']}",
                "Task: generate triples only for the target head entity.",
                (
                    "Every triple must use the target head as subject. "
                    "Preserve exact evidence, dates, doses, and follow-up guidance. "
                    "Return related entities and concepts only when they are grounded in the snippets."
                ),
                "Current corpus context:\n" + (corpus_context or "No existing corpus context."),
                "Open KG example triples:\n" + json.dumps(open_kg_examples, ensure_ascii=False),
                "Retrieved domain snippets:\n" + json.dumps(contexts, ensure_ascii=False),
            ]
        )
        analysis = safe_model_call(
            lambda: self._generate_sac_kg_structured(
                HeadAnalysisPayload,
                system_prompt=(
                    "You are the Generator in a SAC-KG-inspired document pipeline. "
                    "Return triples-first JSON for a single head entity."
                ),
                user_prompt=base_prompt,
                model=settings.ollama_batch_model,
            ),
            fallback,
        )
        analysis = self._normalize_head_analysis(head["name"], analysis)
        return self._verify_and_correct_head_analysis(
            head_name=head["name"],
            analysis=analysis,
            contexts=contexts,
            open_kg_examples=open_kg_examples,
            previous_claims=previous_claims,
            correction_prompt=base_prompt,
        )

    def _normalize_head_analysis(self, head_name: str, analysis: HeadAnalysisPayload) -> HeadAnalysisPayload:
        """归一化 head 分析：清洗三元组（主语回填 head、去空谓词/宾语）与各文本字段。"""
        normalized_triples = [
            GeneratedTriple(
                subject=triple.subject.strip() or head_name,
                predicate=triple.predicate.strip(),
                object_text=triple.object_text.strip(),
                expected_head=head_name,
                evidence_excerpt=triple.evidence_excerpt.strip(),
                confidence=triple.confidence,
                source_chunk_ordinals=triple.source_chunk_ordinals,
                source_sentence_refs=triple.source_sentence_refs,
                relation_type=triple.relation_type,
            )
            for triple in analysis.triples
            if triple.predicate.strip() and triple.object_text.strip()
        ]
        return HeadAnalysisPayload(
            head_entity=head_name,
            summary=analysis.summary.strip(),
            key_facts=[fact.strip() for fact in analysis.key_facts if fact.strip()],
            triples=normalized_triples,
            related_entities=analysis.related_entities,
            related_concepts=[concept.strip() for concept in analysis.related_concepts if concept.strip()],
            coverage_notes=[note.strip() for note in analysis.coverage_notes if note.strip()],
        )

    def _fallback_head_analysis(self, head: dict, contexts: list[dict]) -> HeadAnalysisPayload:
        """生成 head 分析兜底（结构化生成不可用时）：以"mentions"三元组表达源片段。"""
        facts = [entry["text"][:240] for entry in contexts[:4]]
        triples = [
            GeneratedTriple(
                subject=head["name"],
                predicate="mentions",
                object_text=fact,
                expected_head=head["name"],
                evidence_excerpt=fact,
                confidence=0.62,
                source_chunk_ordinals=[entry["chunk_ordinal"]],
                source_sentence_refs=[entry["sentence_ref"]],
                relation_type="source_fact",
            )
            for entry, fact in zip(contexts[:4], facts, strict=False)
            if fact
        ]
        return HeadAnalysisPayload(
            head_entity=head["name"],
            summary=" ".join(facts[:2]).strip(),
            key_facts=facts[:4],
            triples=triples,
            related_entities=[],
            related_concepts=[],
            coverage_notes=["Fallback head analysis used because structured model generation was unavailable."],
        )

    def _verify_and_correct_head_analysis(
        self,
        *,
        head_name: str,
        analysis: HeadAnalysisPayload,
        contexts: list[dict],
        open_kg_examples: list[dict],
        previous_claims: list[Claim],
        correction_prompt: str,
    ) -> HeadAnalysisPayload:
        """校验并（必要时）纠错 head 分析：错误过多时用验证者纠正提示重试。

        若过滤后错误数低于阈值，直接返回（附错误报告）。
        否则用"验证者纠正 pass"重新生成；纠错结果错误数更少则采纳，
        否则保留原过滤结果并说明未改善。
        """
        filtered, error_report, total_errors = self._filter_head_triples(head_name, analysis, contexts, previous_claims)
        if total_errors < HEAD_REPROMPT_ERROR_THRESHOLD:
            filtered.coverage_notes.extend(error_report)
            return filtered

        correction = safe_model_call(
            lambda: self._generate_sac_kg_structured(
                HeadAnalysisPayload,
                system_prompt=(
                    "You are the Verifier correction pass in a SAC-KG-inspired document pipeline. "
                    "Fix triple count, head-entity mismatches, format issues, contradictions, and missing evidence."
                ),
                user_prompt="\n\n".join(
                    [
                        correction_prompt,
                        "Verifier detected these issues:\n" + "\n".join(error_report),
                        "Please regenerate valid triples for the same head entity only.",
                        "Open KG example triples:\n" + json.dumps(open_kg_examples, ensure_ascii=False),
                    ]
                ),
                model=settings.ollama_batch_model,
            ),
            analysis,
        )
        corrected = self._normalize_head_analysis(head_name, correction)
        corrected_filtered, corrected_report, corrected_errors = self._filter_head_triples(head_name, corrected, contexts, previous_claims)
        if corrected_errors <= total_errors:
            corrected_filtered.coverage_notes.extend(["Verifier correction reprompt executed.", *corrected_report])
            return corrected_filtered
        filtered.coverage_notes.extend(["Verifier correction prompt did not improve the triples.", *error_report])
        return filtered

    def _filter_head_triples(
        self,
        head_name: str,
        analysis: HeadAnalysisPayload,
        contexts: list[dict],
        previous_claims: list[Claim],
    ) -> tuple[HeadAnalysisPayload, list[str], int]:
        """过滤 head 三元组：逐条校验并回填证据，返回 (过滤结果, 报告, 错误数)。

        - 三元组总数 < 3 记 quantity_too_small。
        - 每条调用 ``_generated_triple_errors`` 校验；有错记入报告，
          无错且命中上下文时回填证据摘录/来源序号/句引用后采纳。
        """
        reports: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        accepted: list[GeneratedTriple] = []
        total_errors = 0

        if len(analysis.triples) < 3:
            reports.append(f"[{head_name}] quantity_too_small")
            total_errors += 1

        for index, triple in enumerate(analysis.triples):
            matched_context = self._match_context_for_generated_triple(triple, contexts)
            errors = self._generated_triple_errors(
                head_name=head_name,
                triple=triple,
                matched_context=matched_context,
                previous_claims=previous_claims,
                seen=seen,
            )
            if errors:
                reports.append(f"[{head_name}] triple[{index}] -> {', '.join(errors)}")
                total_errors += len(errors)
            else:
                if matched_context is not None:
                    if not triple.evidence_excerpt:
                        triple.evidence_excerpt = matched_context["text"]
                    if not triple.source_chunk_ordinals:
                        triple.source_chunk_ordinals = [matched_context["chunk_ordinal"]]
                    if not triple.source_sentence_refs:
                        triple.source_sentence_refs = [matched_context["sentence_ref"]]
                accepted.append(triple)

        return (
            HeadAnalysisPayload(
                head_entity=head_name,
                summary=analysis.summary,
                key_facts=analysis.key_facts,
                triples=accepted,
                related_entities=analysis.related_entities,
                related_concepts=analysis.related_concepts,
                coverage_notes=analysis.coverage_notes,
            ),
            reports,
            total_errors,
        )

    def _match_context_for_generated_triple(self, triple: GeneratedTriple, contexts: list[dict]) -> dict | None:
        """为生成的三元组寻找最佳匹配上下文（证据锚定打分）。"""
        evidence = triple.evidence_excerpt.strip()
        object_text = triple.object_text.strip()
        best: dict | None = None
        best_score = -1
        for context in contexts:
            text = context["text"]
            score = 0
            if evidence and evidence[:80] in text:
                score += 5
            if object_text and object_text[:80] in text:
                score += 4
            if triple.subject and triple.subject in text:
                score += 2
            if triple.predicate and triple.predicate.lower() in text.lower():
                score += 1
            if score > best_score:
                best = context
                best_score = score
        return best if best_score > 0 else None

    def _generated_triple_errors(
        self,
        *,
        head_name: str,
        triple: GeneratedTriple,
        matched_context: dict | None,
        previous_claims: list[Claim],
        seen: set[tuple[str, str, str]],
    ) -> list[str]:
        """逐条校验生成的三元组，返回错误列表（去重排序）。

        检查项：格式完整、主语必须等于 head、主语≠宾语（自反矛盾）、
        有匹配证据、重复、与历史已验证三元组潜在冲突。
        """
        errors: list[str] = []
        subject = triple.subject.strip()
        predicate = triple.predicate.strip()
        object_text = triple.object_text.strip()
        if not subject or not predicate or not object_text:
            errors.append("format_error")
            return errors
        if subject.lower() != head_name.lower():
            errors.append("head_entity_error")
        if subject.lower() == object_text.lower():
            errors.append("head_tail_contradiction")
        if matched_context is None:
            errors.append("missing_evidence")
        key = (subject.lower(), predicate.lower(), object_text.lower())
        if key in seen:
            errors.append("duplicate")
        else:
            seen.add(key)
        for previous in previous_claims:
            if previous.subject.lower() == subject.lower() and previous.predicate.lower() == predicate.lower() and previous.object_text.lower() != object_text.lower():
                errors.append("potential_conflict")
                break
        return sorted(set(errors))

    def _merge_document_analysis(
        self,
        document: Document,
        seed_analysis: DocumentAnalysisPayload,
        head_payloads: list[HeadAnalysisPayload],
    ) -> DocumentExtraction:
        """合并种子分析与各 head 分析为最终抽取结果。

        合并关键事实/关键词/实体/概念（保序去重），把各 head 的三元组
        转为 claims；无 claims 但种子分析有三元组时退回种子抽取。
        """
        key_facts = list(dict.fromkeys([*seed_analysis.key_facts, *(fact for payload in head_payloads for fact in payload.key_facts)]))
        flattened_keywords = [
            keyword
            for keyword in [
                *seed_analysis.keywords,
                *(keyword for payload in head_payloads for keyword in self._derive_keywords(payload.summary)),
            ]
            if str(keyword).strip()
        ]
        entities = list(seed_analysis.entities)
        entity_names = {entity.name.lower() for entity in entities}
        concepts = list(seed_analysis.concepts)
        for payload in head_payloads:
            if payload.head_entity.lower() not in entity_names:
                entities.append(ExtractedEntity(name=payload.head_entity, entity_type="concept", summary=payload.summary))
                entity_names.add(payload.head_entity.lower())
            for entity in payload.related_entities:
                if entity.name.lower() not in entity_names:
                    entities.append(entity)
                    entity_names.add(entity.name.lower())
            for concept in payload.related_concepts:
                if concept not in concepts:
                    concepts.append(concept)

        claims = [
            ExtractedClaim(
                subject=triple.subject,
                predicate=triple.predicate,
                object_text=triple.object_text,
                expected_head=triple.expected_head,
                evidence_excerpt=triple.evidence_excerpt,
                confidence=triple.confidence,
                source_chunk_ordinals=triple.source_chunk_ordinals,
                source_sentence_refs=triple.source_sentence_refs,
                relation_type=triple.relation_type,
            )
            for payload in head_payloads
            for triple in payload.triples
        ]
        coverage_notes = list(dict.fromkeys([*seed_analysis.coverage_notes, *(note for payload in head_payloads for note in payload.coverage_notes)]))
        summary_parts = [seed_analysis.summary.strip(), *(payload.summary.strip() for payload in head_payloads if payload.summary.strip())]
        summary = "\n".join(part for part in summary_parts if part).strip() or document.title
        if not claims and seed_analysis.triples:
            return self._analysis_to_extraction(document, seed_analysis)

        return DocumentExtraction(
            title=seed_analysis.title or document.title,
            summary=summary,
            keywords=list(dict.fromkeys(flattened_keywords))[:20],
            entities=entities,
            concepts=concepts,
            claims=claims,
            key_facts=key_facts[:20],
            coverage_notes=coverage_notes,
        )

    def _project_verified_claims(self, project_id: str) -> list[Claim]:
        """返回项目下所有"已验证"的三元组（按创建时间倒序）。"""
        return self.db.scalars(
            select(Claim)
            .where(Claim.project_id == project_id, Claim.verification_status == "verified")
            .order_by(Claim.created_at.desc())
        ).all()

    def _project_verified_subjects(self, project_id: str) -> list[str]:
        """返回项目已验证三元组的去重主语列表（保持顺序）。"""
        return list(dict.fromkeys(claim.subject for claim in self._project_verified_claims(project_id)))

    @staticmethod
    def _claim_as_example(claim: Claim) -> dict:
        """把 Claim 转为提示词示例字典。"""
        return {
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object_text": claim.object_text,
            "evidence_excerpt": (claim.metadata_json or {}).get("evidence_excerpt", ""),
        }

    def _head_lookup_tokens(self, head_name: str) -> list[str]:
        """把 head 名拆为查找 token（无 token 时退回小写原名）。"""
        tokens = list(self._text_terms(head_name))
        if not tokens:
            return [head_name.lower()]
        return [token.lower() for token in tokens]

    def _select_generation_contexts(self, document: Document, full_text: str, chunks: list[DocumentChunk]) -> list[dict]:
        """选择用于抽取生成的上下文分块（打分选优 + 序号保序）。"""
        if not chunks and full_text:
            return [{"chunk_ordinal": 0, "heading": None, "score": 1.0, "text": full_text[:2400]}]

        candidate_terms = self._candidate_terms(document, full_text, chunks)
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            lowered = chunk.text.lower()
            marker_score = sum(1 for marker in FACT_MARKERS if marker.lower() in lowered) * 4
            value_score = 3 if FACT_VALUE_PATTERN.search(chunk.text) else 0
            term_score = sum(1 for term in candidate_terms if term and term in lowered)
            heading_score = 2 if chunk.heading else 0
            early_score = 1.0 / (chunk.ordinal + 1)
            scored.append((marker_score + value_score + term_score + heading_score + early_score, chunk))

        if len(chunks) <= 8:
            selected = sorted(chunks, key=lambda item: item.ordinal)
        else:
            top_chunks = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:8]]
            if chunks[0] not in top_chunks:
                top_chunks[-1] = chunks[0]
            selected = sorted(top_chunks, key=lambda item: item.ordinal)

        return [
            {
                "chunk_ordinal": chunk.ordinal,
                "heading": chunk.heading,
                "page_label": chunk.page_label,
                "score": round(next((score for score, item in scored if item.id == chunk.id), 0.0), 3),
                "text": chunk.text[:2400],
            }
            for chunk in selected
        ]

    def _candidate_terms(self, document: Document, full_text: str, chunks: list[DocumentChunk]) -> set[str]:
        """收集用于上下文打分的候选词元（标题/前几个标题/事实标记词）。"""
        terms = self._text_terms(document.title)
        for chunk in chunks[:5]:
            if chunk.heading:
                terms.update(self._text_terms(chunk.heading))
        for marker in FACT_MARKERS:
            if marker.lower() in full_text.lower():
                terms.add(marker.lower())
        return terms

    def _local_triple_examples(self, project_id: str) -> list[dict]:
        """返回项目最近三元组的前 6 条作为本地示例。"""
        examples: list[dict] = []
        claims = self.db.scalars(select(Claim).where(Claim.project_id == project_id).order_by(Claim.created_at.desc())).all()
        for claim in claims[:6]:
            examples.append(
                {
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object_text": claim.object_text,
                    "evidence_excerpt": (claim.metadata_json or {}).get("evidence_excerpt", ""),
                    "confidence": claim.confidence,
                }
            )
        return examples

    def _fallback_analysis(self, document: Document, full_text: str, contexts: list[dict]) -> DocumentAnalysisPayload:
        """生成文档分析兜底：以关键事实构造 "title states fact" 三元组。"""
        key_facts = self._extract_key_facts(full_text)
        triples = [
            GeneratedTriple(
                subject=document.title,
                predicate="states",
                object_text=fact,
                expected_head=document.title,
                evidence_excerpt=fact,
                confidence=0.65,
                source_chunk_ordinals=self._fact_chunk_ordinals(fact, contexts),
                relation_type="source_fact",
            )
            for fact in key_facts[:8]
        ]
        summary_parts = key_facts[:5] if key_facts else [full_text[:1200]]
        return DocumentAnalysisPayload(
            title=document.title,
            summary="\n".join(part for part in summary_parts if part).strip() or document.title,
            keywords=self._derive_keywords(full_text),
            key_facts=key_facts,
            entities=[],
            concepts=[],
            triples=triples,
            coverage_notes=["Fallback analysis used because structured model generation was unavailable."],
        )

    def _analysis_to_extraction(self, document: Document, analysis: DocumentAnalysisPayload) -> DocumentExtraction:
        """把文档分析（DocumentAnalysisPayload）转换为抽取结果（DocumentExtraction）。"""
        claims = [
            ExtractedClaim(
                subject=triple.subject,
                predicate=triple.predicate,
                object_text=triple.object_text,
                expected_head=triple.expected_head,
                evidence_excerpt=triple.evidence_excerpt,
                confidence=triple.confidence,
                source_chunk_ordinals=triple.source_chunk_ordinals,
                source_sentence_refs=triple.source_sentence_refs,
                relation_type=triple.relation_type,
            )
            for triple in analysis.triples
            if triple.subject.strip() and triple.predicate.strip() and triple.object_text.strip()
        ]
        return DocumentExtraction(
            title=analysis.title or document.title,
            summary=analysis.summary,
            keywords=analysis.keywords,
            entities=analysis.entities,
            concepts=analysis.concepts,
            claims=claims,
            key_facts=analysis.key_facts,
            coverage_notes=analysis.coverage_notes,
        )

    def _extract_key_facts(self, text: str) -> list[str]:
        """启发式提取关键事实句：按句末标点切分，命中事实标记词或数值单位。"""
        pieces = re.split(r"(?<=[。！？!?\.])\s*|\n+", text)
        facts: list[str] = []
        seen: set[str] = set()
        for piece in pieces:
            sentence = re.sub(r"\s+", " ", piece).strip()
            if len(sentence) < 6:
                continue
            lowered = sentence.lower()
            has_marker = any(marker.lower() in lowered for marker in FACT_MARKERS)
            has_value = bool(FACT_VALUE_PATTERN.search(sentence))
            if not has_marker and not has_value:
                continue
            fact = sentence[:600]
            key = fact.lower()
            if key in seen:
                continue
            facts.append(fact)
            seen.add(key)
            if len(facts) >= 12:
                break
        return facts

    def _derive_keywords(self, text: str) -> list[str]:
        """从文本派生关键词：事实标记词 + 英文长词元，至多 12 个。"""
        keywords = [marker for marker in FACT_MARKERS if marker.lower() in text.lower()]
        english_terms = [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()) if word not in keywords]
        for term in english_terms:
            if term not in keywords:
                keywords.append(term)
            if len(keywords) >= 12:
                break
        return keywords[:12]

    @staticmethod
    def _fact_chunk_ordinals(fact: str, contexts: list[dict]) -> list[int]:
        """返回包含该事实的上下文分块序号（至多 3 个）。"""
        return [
            int(context["chunk_ordinal"])
            for context in contexts
            if fact and fact[:80] in str(context.get("text", ""))
        ][:3]

    @staticmethod
    def _text_terms(text: str) -> set[str]:
        """把文本切分为词元集合（英文字母数字词 + CJK 段与其二元组）。"""
        lowered = text.lower()
        terms = {word for word in re.findall(r"[a-z0-9_]+", lowered) if len(word) > 1}
        for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
            if len(segment) <= 2:
                terms.add(segment)
                continue
            terms.add(segment)
            terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
        return {term for term in terms if term}

    def _upsert_entities(self, project_id: str, extraction: DocumentExtraction) -> list[Entity]:
        """upsert 实体：合并概念、去重、更新已存在实体或新建。

        并发冲突（IntegrityError）时回滚并重新按名批量查询返回。
        """
        entities: list[Entity] = []
        items = list(extraction.entities)
        existing_names = {item.name for item in items}
        for concept in extraction.concepts:
            if concept and concept not in existing_names:
                items.append(ExtractedEntity(name=concept, entity_type="concept", summary=""))
                existing_names.add(concept)

        unique_items: list[ExtractedEntity] = []
        seen_item_names: set[str] = set()
        for item in items:
            key = item.name.strip()
            if not key or key in seen_item_names:
                continue
            seen_item_names.add(key)
            unique_items.append(item)

        for item in unique_items:
            existing = self.db.scalar(select(Entity).where(Entity.project_id == project_id, Entity.name == item.name))
            if existing:
                existing.summary = item.summary or existing.summary
                existing.entity_type = item.entity_type or existing.entity_type
                merged_aliases = sorted(set(existing.aliases + item.aliases))
                existing.aliases = merged_aliases
                entities.append(existing)
                continue
            entity = Entity(
                project_id=project_id,
                name=item.name,
                entity_type=item.entity_type,
                aliases=item.aliases,
                summary=item.summary,
            )
            self.db.add(entity)
            entities.append(entity)
        try:
            self.db.commit()
            return entities
        except IntegrityError:
            self.db.rollback()
            return list(self.db.scalars(select(Entity).where(Entity.project_id == project_id, Entity.name.in_([item.name for item in unique_items]))).all())

    def _create_claims(self, document: Document, extraction: DocumentExtraction) -> list[Claim]:
        """创建（替换）文档的三元组记录并做本地验证。

        先删除该文档旧 claims；对每个抽取的 claim：定位证据分块、回填
        证据摘录、执行本地验证（``_verify_claim_locally``），据验证错误
        决定 verification_status（needs-review / verified）。
        """
        self.db.query(Claim).filter(Claim.document_id == document.id).delete()
        chunks = self.db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.ordinal)).all()
        previous_claims = self.db.scalars(select(Claim).where(Claim.project_id == document.project_id)).all()
        seen: set[tuple[str, str, str]] = set()
        claims: list[Claim] = []
        for item in extraction.claims:
            evidence_chunk = self._find_evidence_chunk(item, list(chunks))
            evidence_excerpt = item.evidence_excerpt.strip()
            if not evidence_excerpt and evidence_chunk is not None:
                evidence_excerpt = self._best_evidence_excerpt(item, evidence_chunk.text)
            verification_errors = sorted(
                set(item.verification_errors + self._verify_claim_locally(item, evidence_chunk, previous_claims, seen))
            )
            source_chunk_ids = [evidence_chunk.id] if evidence_chunk is not None else []
            claim = Claim(
                project_id=document.project_id,
                document_id=document.id,
                subject=item.subject,
                predicate=item.predicate,
                object_text=item.object_text,
                evidence_chunk_id=evidence_chunk.id if evidence_chunk is not None else None,
                confidence=item.confidence,
                verification_status="needs-review" if verification_errors else "verified",
                metadata_json={
                    "expected_head": item.expected_head,
                    "evidence_excerpt": evidence_excerpt,
                    "source_chunk_ids": source_chunk_ids,
                    "source_chunk_ordinals": item.source_chunk_ordinals,
                    "source_sentence_refs": item.source_sentence_refs,
                    "verification_errors": verification_errors,
                    "head_growth_decision": item.growth_decision,
                    "tail_growth_decision": "keep",
                    "relation_type": item.relation_type,
                },
            )
            self.db.add(claim)
            claims.append(claim)
        self.db.commit()
        return claims

    def _find_evidence_chunk(self, claim: ExtractedClaim, chunks: list[DocumentChunk]) -> DocumentChunk | None:
        """为 claim 定位证据分块（按句引用 -> 块序号 -> 证据/宾语/主语匹配）。"""
        for sentence_ref in claim.source_sentence_refs:
            match = re.match(r"chunk-(\d+):sentence-\d+", sentence_ref)
            if not match:
                continue
            chunk_ordinal = int(match.group(1))
            for chunk in chunks:
                if chunk.ordinal == chunk_ordinal:
                    return chunk
        for ordinal in claim.source_chunk_ordinals:
            for chunk in chunks:
                if chunk.ordinal == ordinal:
                    return chunk

        evidence = claim.evidence_excerpt.strip()
        if evidence:
            evidence_variants = [evidence, evidence[:160], evidence[:80]]
            for variant in evidence_variants:
                if len(variant) < 8:
                    continue
                for chunk in chunks:
                    if variant in chunk.text:
                        return chunk

        object_text = claim.object_text.strip()
        if object_text and len(object_text) >= 6:
            for chunk in chunks:
                if object_text[:120] in chunk.text or object_text in chunk.text:
                    return chunk

        subject = claim.subject.strip()
        if subject and len(subject) >= 2:
            for chunk in chunks:
                if subject in chunk.text:
                    return chunk
        return None

    def _best_evidence_excerpt(self, claim: ExtractedClaim, text: str, max_chars: int = 360) -> str:
        """从证据文本中截取围绕宾语/主语锚点的最佳摘录（约 max_chars）。"""
        anchors = [claim.object_text.strip(), claim.subject.strip()]
        lowered = text.lower()
        for anchor in anchors:
            if not anchor:
                continue
            position = lowered.find(anchor.lower())
            if position < 0:
                continue
            start = max(0, position - max_chars // 3)
            end = min(len(text), start + max_chars)
            return text[start:end].strip()
        return text[:max_chars].strip()

    def _verify_claim_locally(
        self,
        claim: ExtractedClaim,
        evidence_chunk: DocumentChunk | None,
        previous_claims: list[Claim],
        seen: set[tuple[str, str, str]],
    ) -> list[str]:
        """本地验证 claim，返回错误列表（source_fact 走宽松规则）。

        source_fact（兜底三元组）放宽置信度阈值、豁免证据分块与证据锚定
        检查，使其更易被记为 verified 以支持下游 RAG 打分。
        """
        errors: list[str] = []
        subject = claim.subject.strip()
        predicate = claim.predicate.strip()
        object_text = claim.object_text.strip()
        is_source_fact = claim.relation_type == "source_fact"

        if not subject or not predicate or not object_text:
            errors.append("format_error")

        expected_head = claim.expected_head.strip()
        if expected_head and subject.lower() != expected_head.lower():
            errors.append("head_entity_error")

        if subject.lower() == object_text.lower():
            errors.append("head_tail_contradiction")

        # source_fact claims are fallback triples — relax confidence threshold
        # so they count as verified and improve downstream RAG scoring.
        confidence_threshold = 0.45 if is_source_fact else 0.6
        if claim.confidence < confidence_threshold:
            errors.append("low_confidence")

        # source_fact claims may not map to a single chunk — that's expected.
        if not is_source_fact and evidence_chunk is None:
            errors.append("missing_evidence")

        key = (subject.lower(), predicate.lower(), object_text.lower())
        if key in seen:
            errors.append("duplicate")
        else:
            seen.add(key)

        for previous in previous_claims:
            if previous.subject.lower() != subject.lower() or previous.predicate.lower() != predicate.lower():
                continue
            if previous.object_text.lower() != object_text.lower():
                errors.append("potential_conflict")
                break

        # Skip evidence-anchoring checks for source_fact — they are fallback triples.
        if not is_source_fact and evidence_chunk is not None and subject and len(subject) > 2 and predicate.lower() != "states":
            source_text = evidence_chunk.text
            if subject not in source_text and not self._has_term_overlap(subject, source_text):
                errors.append("subject_not_in_evidence")

        return sorted(set(errors))

    @staticmethod
    def _has_term_overlap(left: str, right: str) -> bool:
        """判断两段文本是否有词元重叠。"""
        left_terms = IngestionPipeline._text_terms(left)
        right_terms = IngestionPipeline._text_terms(right)
        return bool(left_terms & right_terms)

    def _decide_entity_growth(self, document: Document, entities: list[Entity], claims: list[Claim]) -> dict[str, GrowthDecision]:
        """为候选实体/宾语尾项决定增长决策（grow/keep/prune）。

        先按规则（``_rule_growth_decision``）判断；对需要人工/模型判断的
        keep 项，调用 Ollama 修剪器批量决策并归一化。
        """
        verified_claims = [claim for claim in claims if claim.verification_status == "verified"]
        candidates: dict[str, dict] = {}
        for entity in entities:
            candidates.setdefault(
                entity.name,
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type or "concept",
                    "summary": entity.summary or "",
                    "claim_count": sum(1 for claim in verified_claims if claim.subject == entity.name or claim.object_text == entity.name),
                    "item_type": "head",
                },
            )
        for claim in verified_claims:
            tail_name = claim.object_text.strip()
            if not tail_name or len(tail_name) > 80:
                continue
            candidates.setdefault(
                tail_name,
                {
                    "name": tail_name,
                    "entity_type": self._infer_tail_entity_type(tail_name, entities),
                    "summary": f"Referenced by {claim.subject} via {claim.predicate}.",
                    "claim_count": sum(1 for item in verified_claims if item.object_text == tail_name),
                    "item_type": "tail",
                },
            )

        decisions: dict[str, GrowthDecision] = {}
        uncertain: list[dict] = []
        for candidate in candidates.values():
            decision = self._rule_growth_decision(candidate)
            decisions[candidate["name"]] = decision
            if decision.decision == "keep":
                uncertain.append(candidate)

        if uncertain:
            prompt = "\n\n".join(
                [
                    f"Document title: {document.title}",
                    (
                        "Decide whether each candidate should continue growing in the SAC-KG routing graph. "
                        "Use grow for durable head or tail entities, keep for useful but not yet expanded items, "
                        "and prune for dates, doses, isolated numbers, or transient values."
                    ),
                    json.dumps(uncertain[:20], ensure_ascii=False),
                ]
            )
            fallback = GrowthDecisionPayload(decisions=[])
            ai_decisions = safe_model_call(
                lambda: self._generate_sac_kg_structured(
                    GrowthDecisionPayload,
                    system_prompt="You are the Pruner in a SAC-KG-inspired RAG pipeline. Return strict JSON decisions only.",
                    user_prompt=prompt,
                    model=settings.ollama_batch_model,
                ),
                fallback,
            )
            for item in ai_decisions.decisions:
                if item.name not in decisions:
                    continue
                normalized = self._normalize_growth_decision(item.decision)
                decisions[item.name] = GrowthDecision(
                    name=item.name,
                    item_type=item.item_type or "entity",
                    decision=normalized,
                    reason=item.reason or "Configured SAC-KG generation provider decision.",
                )
        return decisions

    def _rule_growth_decision(self, candidate: dict) -> GrowthDecision:
        """基于规则的增长决策：瞬时值/数值类剪枝，持久类型增长，其余需判断。"""
        entity_type = str(candidate.get("entity_type") or "concept").lower()
        name = str(candidate.get("name") or "").strip()
        claim_count = int(candidate.get("claim_count") or 0)
        item_type = str(candidate.get("item_type") or "entity")
        if entity_type in PRUNE_ENTITY_TYPES or self._looks_like_transient_value(name):
            return GrowthDecision(name=name, item_type=item_type, decision="prune", reason="Transient date, dose, or numeric value.")
        if entity_type in GROW_ENTITY_TYPES:
            if item_type == "tail" and claim_count <= 0:
                return GrowthDecision(name=name, item_type=item_type, decision="keep", reason="Durable type but no verified claim supports a standalone tail page yet.")
            return GrowthDecision(name=name, item_type=item_type, decision="grow", reason="Durable entity type suitable for continued growth.")
        if item_type == "tail" and claim_count == 1 and len(name) > 36:
            return GrowthDecision(name=name, item_type=item_type, decision="keep", reason="Tail value is descriptive but may be too broad for its own page.")
        if claim_count > 0:
            return GrowthDecision(name=name, item_type=item_type, decision="grow", reason="Supported by verified claims and suitable for RAG routing expansion.")
        return GrowthDecision(name=name, item_type=item_type, decision="keep", reason="Needs pruner judgment before creating a standalone page.")

    def _infer_tail_entity_type(self, tail_name: str, entities: list[Entity]) -> str:
        """推断宾语尾项实体类型：先查已知实体，再按关键词启发式。"""
        for entity in entities:
            if entity.name.lower() == tail_name.lower():
                return entity.entity_type or "concept"
        lowered = tail_name.lower()
        if FACT_VALUE_PATTERN.search(tail_name):
            return "value"
        if any(keyword in lowered for keyword in ("disease", "diagnosis", "syndrome", "病", "症")):
            return "disease"
        if any(keyword in lowered for keyword in ("drug", "tablet", "capsule", "胍", "药")):
            return "drug"
        if any(keyword in lowered for keyword in ("check", "exam", "scan", "检查", "复查")):
            return "test"
        return "concept"

    @staticmethod
    def _looks_like_transient_value(name: str) -> bool:
        """判断名称是否像瞬时值（数值/单位/纯符号），用于剪枝。"""
        stripped = name.strip()
        if not stripped:
            return True
        if FACT_VALUE_PATTERN.search(stripped):
            return True
        return bool(re.fullmatch(r"[\d\s.,:%/-]+", stripped))

    @staticmethod
    def _normalize_growth_decision(decision: str) -> str:
        """把决策字符串归一化为 grow/keep/prune 之一（未知值退回 keep）。"""
        lowered = decision.lower().strip()
        if lowered in {"grow", "keep", "prune"}:
            return lowered
        return "keep"

    @staticmethod
    def _apply_claim_growth_decisions(claims: list[Claim], entity_decisions: dict[str, GrowthDecision]) -> None:
        """把实体增长决策写入每个 claim 的元信息（head/tail 决策）。"""
        for claim in claims:
            metadata = dict(claim.metadata_json or {})
            head_decision = entity_decisions.get(claim.subject)
            tail_decision = entity_decisions.get(claim.object_text)
            metadata["head_growth_decision"] = head_decision.decision if head_decision else metadata.get("head_growth_decision", "keep")
            metadata["tail_growth_decision"] = tail_decision.decision if tail_decision else metadata.get("tail_growth_decision", "keep")
            metadata["growth_decision"] = metadata["tail_growth_decision"]
            claim.metadata_json = metadata

    def _ensure_growing_entities(
        self,
        project_id: str,
        entities: list[Entity],
        claims: list[Claim],
        entity_decisions: dict[str, GrowthDecision],
    ) -> list[Entity]:
        """确保所有"增长"实体在库中存在（从已有实体或按 claims 新建）。"""
        entity_map = {entity.name.lower(): entity for entity in entities}
        grow_names = [name for name, decision in entity_decisions.items() if decision.decision == "grow"]
        if grow_names:
            existing_entities = self.db.scalars(
                select(Entity).where(Entity.project_id == project_id, Entity.name.in_(grow_names))
            ).all()
            for entity in existing_entities:
                key = entity.name.lower()
                if key not in entity_map:
                    entities.append(entity)
                    entity_map[key] = entity
        for name, decision in entity_decisions.items():
            if decision.decision != "grow" or name.lower() in entity_map:
                continue
            related_claims = [claim for claim in claims if claim.object_text == name]
            if not related_claims:
                continue
            entity = Entity(
                project_id=project_id,
                name=name,
                entity_type=self._infer_tail_entity_type(name, entities),
                aliases=[],
                summary=" ".join(f"{claim.subject} {claim.predicate} {claim.object_text}" for claim in related_claims[:2])[:400],
            )
            self.db.add(entity)
            entities.append(entity)
            entity_map[name.lower()] = entity
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            return list(self.db.scalars(select(Entity).where(Entity.project_id == project_id, Entity.name.in_(grow_names))).all())
        return entities

    def _source_page_metadata(self, extraction: DocumentExtraction, entities: list[Entity], claims: list[Claim]) -> dict:
        """构建源页元信息（摘要/关键术语/来源计数/已验证三元组数/增长决策）。"""
        key_terms = sorted(
            {
                *extraction.keywords,
                *extraction.concepts,
                *(entity.name for entity in entities),
                *(claim.subject for claim in claims if claim.verification_status == "verified"),
                *(
                    claim.object_text
                    for claim in claims
                    if claim.verification_status == "verified" and (claim.metadata_json or {}).get("tail_growth_decision") == "grow"
                ),
            }
        )
        # Ensure we always have at least some key_terms — fall back to
        # extraction title words and key facts so frontmatter is rarely empty.
        if not key_terms:
            fallback_text = " ".join([extraction.title, *extraction.key_facts[:5]])
            key_terms = sorted(
                term
                for term in self._text_terms(fallback_text)
                if term.lower() not in {"the", "and", "for", "with"}
            )
        verified_count = sum(1 for claim in claims if claim.verification_status == "verified")
        return {
            "summary": extraction.summary[:700],
            "key_terms": list(key_terms)[:30],
            "source_count": 1,
            "verified_claim_count": verified_count,
            "growth_decision": "grow",
        }

    @staticmethod
    def _entity_page_metadata(entity_name: str, decision: GrowthDecision, claims: list[Claim]) -> dict:
        """构建实体页元信息（摘要/关键术语/来源计数/增长决策）。"""
        entity_claims = [claim for claim in claims if claim.subject == entity_name or claim.object_text == entity_name]
        return {
            "summary": " ".join(
                claim.object_text if claim.subject == entity_name else f"{claim.subject} {claim.predicate}"
                for claim in entity_claims[:3]
            )[:700],
            "key_terms": sorted({entity_name, *(claim.predicate for claim in entity_claims), *(claim.subject for claim in entity_claims)})[:20],
            "source_count": len({claim.document_id for claim in entity_claims}),
            "verified_claim_count": sum(1 for claim in entity_claims if claim.verification_status == "verified"),
            "growth_decision": decision.decision,
            "growth_reason": decision.reason,
        }

    def _create_review_items(self, document: Document, extraction: DocumentExtraction, claims: list[Claim]) -> int:
        """为抽取结果生成审阅项（含外部验证），返回审阅项数量。

        生成规则：
        - 无 claims 但有正文 → "无三元组"中等级审阅。
        - 每个 head 已验证三元组 < 3 → 低等级数量审阅。
        - 有本地验证错误的 claim → 高/中等级审阅（按错误类型）。
        - 覆盖说明 → 低等级审阅。
        - 外部验证（ExternalVerifier）标记的 claim → 标记其状态并加
          高等级审阅。

        先删除旧的 pending ingest 审阅项再重建。
        """
        existing_items = self.db.scalars(select(ReviewItem).where(ReviewItem.document_id == document.id)).all()
        for item in existing_items:
            if item.status == ReviewStatus.pending.value and (item.payload or {}).get("generated_by") == "ingest":
                self.db.delete(item)
        count = 0
        if not claims and (document.raw_text or "").strip():
            self.db.add(
                ReviewItem(
                    project_id=document.project_id,
                    document_id=document.id,
                    title="Verifier quantity check: no claims extracted",
                    detail="The document contains text, but no structured claims were generated.",
                    severity=ReviewSeverity.medium.value,
                    payload={"generated_by": "ingest", "issue": "empty_claim_set", "check": "quantity"},
                )
            )
            count += 1

        head_groups: dict[str, list[Claim]] = {}
        for claim in claims:
            expected_head = ((claim.metadata_json or {}).get("expected_head") or claim.subject).strip()
            head_groups.setdefault(expected_head, []).append(claim)
        for head_name, head_claims in head_groups.items():
            verified_count = sum(1 for claim in head_claims if claim.verification_status == "verified")
            if verified_count < 3:
                self.db.add(
                    ReviewItem(
                        project_id=document.project_id,
                        document_id=document.id,
                        title=f"Verifier quantity check: {head_name}",
                        detail=f"Head '{head_name}' has only {verified_count} verified triples.",
                        severity=ReviewSeverity.low.value,
                        payload={"generated_by": "ingest", "issue": "quantity_too_small", "head": head_name, "verified_count": verified_count},
                    )
                )
                count += 1

        for claim in claims:
            metadata = claim.metadata_json or {}
            errors = metadata.get("verification_errors", [])
            if errors:
                high_risk_errors = {"format_error", "missing_evidence", "potential_conflict", "head_entity_error", "head_tail_contradiction"}
                self.db.add(
                    ReviewItem(
                        project_id=document.project_id,
                        document_id=document.id,
                        claim_id=claim.id,
                        title=f"Verifier flagged claim: {claim.subject}",
                        detail=f"{claim.subject} {claim.predicate} {claim.object_text}",
                        severity=ReviewSeverity.high.value if high_risk_errors & set(errors) else ReviewSeverity.medium.value,
                        payload={
                            "generated_by": "ingest",
                            "issues": errors,
                            "check": "local_verifier",
                            "confidence": claim.confidence,
                            "evidence_excerpt": metadata.get("evidence_excerpt", ""),
                            "source_chunk_ids": metadata.get("source_chunk_ids", []),
                        },
                    )
                )
                count += 1

        for note in extraction.coverage_notes:
            self.db.add(
                ReviewItem(
                    project_id=document.project_id,
                    document_id=document.id,
                    title="Verifier coverage note",
                    detail=note,
                    severity=ReviewSeverity.low.value,
                    payload={"generated_by": "ingest", "issue": "coverage_note", "note": note},
                )
            )
            count += 1

        verification = self.verifier.verify_claims(extraction.summary, extraction.claims)
        if verification.flagged_claim_indexes:
            for claim_index in verification.flagged_claim_indexes:
                if 0 <= claim_index < len(claims):
                    claims[claim_index].verification_status = verification.verdict
            self.db.add(
                ReviewItem(
                    project_id=document.project_id,
                    document_id=document.id,
                    title="External verification flagged claims",
                    detail=verification.notes,
                    severity=ReviewSeverity.high.value,
                    payload={"generated_by": "ingest", "flagged_claim_indexes": verification.flagged_claim_indexes, "verdict": verification.verdict},
                )
            )
            count += 1
        self.db.commit()
        return count
