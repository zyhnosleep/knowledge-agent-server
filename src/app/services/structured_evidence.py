"""
structured_evidence.py —— 结构化证据（表格/图表/公式）构建与表格修复模块
=======================================================================

职责：
- 把规范化的"结构化块"（表格 CanonicalTable、图表 CanonicalFigure、
  公式 CanonicalFormula）转换为可供结构化检索使用的 Parent/Child 分块
  （``StructuredEvidenceChunk``），保持对源文本的保真（source-faithful）。
- 提供严格的"仅基于源"表格校验（``TableValidator``）：在规范激活前
  判定表格是否可接受，并为不合格表格生成明确的修复请求（vision 修复），
  修复结果需经库存级匹配（``validate_repair_inventory``）与可审计证据
  （``TableRepairProof``）验证后才被采纳。
- 支持跨页表格合并（``merge_cross_page_tables``）：把"延续表"按连续页码
  合并回根表，保留各页单元格的原始行/页元信息。

关键设计：
- **源保真**：Parent/Child 分块的文本来自源内容（caption、表格网格、
  脚注），绝不依赖模型生成文本；模型生成摘要仅进入 ``embedding_text``。
- **Token 预算**：所有子分块通过 ``_lossless_wrapped_windows``（二分
  查找 + 首选边界）切分，保证每个 child 不超过 ``max_tokens``，同时
  不丢失任何源字符（lossless）。
- **稳定 ID**：分块 ID 由内容哈希生成（``_stable_id``），内容不变则
  ID 不变，便于幂等重建与去重。
- **tokenizer 兜底**：tokenizer 加载失败时可退化为 UTF-8 字节计数
  （宁可高估也不静默低估），并由 ``strict_tokenizer`` 控制是否允许兜底。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.services.canonical_provenance import block_is_generated
from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalCell,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
    TableStatus,
)
from app.services.canonical_quality import CanonicalQualityGate
from app.services.canonical_table_identity import (
    table_content_fingerprint,
    table_identity_fingerprint,
)


class TableRepairRequest(BaseModel):
    """A deterministic, page-scoped request for vision table repair.

    表格修复请求：面向视觉模型（vision）的、确定性的、按页作用域的请求。

    字段：
    - ``table_id``：待修复表格 ID。
    - ``reasons``：校验失败原因列表（驱动修复指令）。
    - ``locator``：表格在页面上的定位信息（页码/bbox/region 等）。
    - ``source_fingerprint``：表格身份指纹（用于修复前后一致性校验）。
    - ``instructions``：给视觉模型的修复指令文本。
    """

    model_config = ConfigDict(extra="forbid")

    table_id: str
    reasons: list[str]
    locator: dict[str, Any]
    source_fingerprint: str
    instructions: str


class TableRepairMapping(BaseModel):
    """Orchestrator-validated mapping between parser-specific table objects.

    经过编排器（orchestrator）验证的"解析器特定表格对象"之间的映射：
    记录原始表格与替换表格的 ID 对应关系及所在页。

    字段：
    - ``original_table_id``：原始表格 ID。
    - ``replacement_table_id``：替换（修复后）表格 ID。
    - ``page_index``：两者所在的页序号（>=0）。
    """

    model_config = ConfigDict(extra="forbid")

    original_table_id: str
    replacement_table_id: str
    page_index: int = Field(ge=0)


class TableRepairProof(BaseModel):
    """Parser-independent proof emitted only after inventory-level matching.

    仅在"库存级匹配"完成后才签发的、与解析器无关的修复证明。

    该证明把原始请求、匹配依据（match_basis）与经校验的映射封装在一起，
    供后续 ``TableValidator._validate_inventory_proof`` 做审计校验——
    任何字段不一致都会导致修复被拒绝。

    字段：
    - ``original_request``：最初的修复请求。
    - ``page_index``：表格所在页。
    - ``replacement_content_fingerprint``：替换表格的内容指纹。
    - ``match_basis``：匹配依据（source_region_id / normalized_bbox /
      source_block_id / unique_table_on_page）。
    - ``validated_mapping``：经校验的原始/替换表格映射。
    """

    model_config = ConfigDict(extra="forbid")

    original_request: TableRepairRequest
    page_index: int = Field(ge=0)
    replacement_content_fingerprint: str
    match_basis: Literal[
        "source_region_id",
        "normalized_bbox",
        "source_block_id",
        "unique_table_on_page",
    ]
    validated_mapping: TableRepairMapping


class TableValidationResult(BaseModel):
    """Consumable validation signal used to gate canonical activation.

    用于门控"规范激活"的可消费校验信号：一次性返回校验结果，
    供下游决定是否允许激活该表格。

    字段：
    - ``accepted``：是否通过校验。
    - ``activation_allowed``：是否允许激活。
    - ``status``：表格最终状态（TableStatus 枚举）。
    - ``reasons``：校验失败/警告原因列表。
    - ``table``：校验后的表格对象（可能含修复状态）。
    - ``repair_request``：校验失败时附带的修复请求；通过时为空。
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    activation_allowed: bool
    status: TableStatus
    reasons: list[str] = Field(default_factory=list)
    table: CanonicalTable
    repair_request: TableRepairRequest | None = None


class StructuredEvidenceChunk(BaseModel):
    """Stable source/derived text boundary for structured retrieval evidence.

    结构化检索证据的稳定"源/派生文本边界"：Parent/Child 分块的统一结构。

    字段：
    - ``chunk_id``：稳定分块 ID（由内容哈希派生）。
    - ``parent_chunk_id``：父分块 ID（child 分块必填）。
    - ``chunk_role``：``"parent"``（父）或 ``"child"``（子）。
    - ``block_type``：源块类型（table / figure / formula）。
    - ``text``：分块文本（源保真内容）。
    - ``embedding_text``：用于向量嵌入的文本（可能含模型派生内容）。
    - ``token_count``：token 数（>=0）。
    - ``source_spans``：覆盖的源片段列表。
    - ``metadata``：分块元信息（表 ID、行范围、来源片段、保真信息等）。
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_chunk_id: str | None = None
    chunk_role: Literal["parent", "child"]
    block_type: Literal["table", "figure", "formula"]
    text: str
    embedding_text: str
    token_count: int = Field(ge=0)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableValidator:
    """Strict source-only table validation with explicit repair routing.

    严格的"仅基于源"表格校验器，带明确的修复路由。

    核心思想：
    - 校验完全基于源内容（表格数据本身、caption、source markdown/html、
      跨页信息等），不使用任何模型生成内容。
    - 校验失败时返回 ``TableValidationResult(accepted=False)`` 并附带
      ``TableRepairRequest``，指引后续的视觉修复（vision repair）。
    - 修复结果必须附带证明（``TableRepairProof``）并通过库存级匹配校验，
      才可能被采纳；单表校验通过后才允许激活
      （``activation_allowed=True``）。
    """

    # 匹配 caption 中"table <编号>"模式（如 "Table 3.1"），用于校验
    # caption 编号与表格元数据中的 table_number 是否一致
    _caption_number = re.compile(r"\btable\s*([A-Za-z]?\d+(?:[.-]\d+)*)\b", re.I)

    def validate(
        self,
        table: CanonicalTable,
        repaired_table: CanonicalTable | None = None,
        repair_proof: (
            TableRepairRequest | TableRepairProof | dict[str, Any] | None
        ) = None,
    ) -> TableValidationResult:
        """校验单个表格（可选地针对修复后的表格），返回校验信号。

        参数：
        - ``table``：原始表格（校验基线）。
        - ``repaired_table``：可选的修复后表格；给出时以其为候选并额外
          校验修复证明。
        - ``repair_proof``：修复证明（请求/库存证明/dict 均可），用于
          验证修复来源可信。

        逻辑：
        1. 深拷贝候选（修复后表格或原始表格）作为校验对象。
        2. 计算全部"理由"（``_reasons``）。
        3. 若提供了修复后表格：校验修复证明；且因走的是单表路径而非
           库存级路径，强制追加 ``repair_inventory_missing`` 理由
           （单表修复必须再经库存级验证才会被接受）。
        4. 有理由则判失败：状态 ``validation_failed``、不允许激活、
           附修复请求。
        5. 无理由则判通过：状态按来源取 ``repaired_by_vision`` /
           ``cross_page_merged`` / ``accepted_mineru``，允许激活。
        """
        candidate = (repaired_table or table).model_copy(deep=True)
        reasons = self._reasons(candidate)
        if repaired_table is not None:
            # 校验修复证明的一致性（来源指纹、定位符、映射等）
            self._validate_repair_proof(
                table,
                repaired_table,
                repair_proof,
                reasons,
            )
            # 单表修复缺少"库存级"认证，强制标记，要求走库存级流程
            if "repair_inventory_missing" not in reasons:
                reasons.append("repair_inventory_missing")
        if reasons:
            candidate.status = "validation_failed"
            return TableValidationResult(
                accepted=False,
                activation_allowed=False,
                status="validation_failed",
                reasons=reasons,
                table=candidate,
                # 生成修复请求：指向被校验的候选（修复表存在时指向原始表，
                # 以便以原始表为修复对象）
                repair_request=self._repair_request(
                    table if repaired_table is not None else candidate,
                    reasons,
                ),
            )

        # 通过校验：按来源确定最终状态
        if repaired_table is not None:
            status: TableStatus = "repaired_by_vision"
        elif candidate.status in {"repaired_by_vision", "cross_page_merged"}:
            status = candidate.status
        else:
            status = "accepted_mineru"
        candidate.status = status
        return TableValidationResult(
            accepted=True,
            activation_allowed=True,
            status=status,
            table=candidate,
        )

    def validate_repair_inventory(
        self,
        original_tables: list[CanonicalTable],
        replacement_tables: list[CanonicalTable],
        repair_table_ids: set[str],
        page_index: int,
    ) -> list[
        tuple[
            CanonicalTable,
            CanonicalTable,
            TableValidationResult,
            TableRepairRequest | TableRepairProof,
        ]
    ] | None:
        """Atomically validate a complete page inventory and issue audit proofs.

        原子地校验一页的完整表格库存并签发审计证明。

        前置约束（任一不满足即整体返回 None，表示该页不可修复采纳）：
        - 原始表与替换表数量一致、各自 table_id 无重复、替换表内容指纹
          无重复。
        - 所有表都必须只属于 ``page_index`` 这一页。
        - 待修复表 ID 集合非空且是原始表 ID 的子集。
        - 原始表与替换表存在唯一的一一匹配（``_unique_inventory_matching``）。
        - 替换表自身校验必须无理由（即替换表本身是合格的）。

        对每个匹配的、且属于待修复集合的原始表：
        - 以原始表的校验失败理由构造修复请求，再封装为
          ``TableRepairProof``（含匹配依据与经校验的映射）。
        - 重新校验证明（``_validate_repair_proof``），有理由则整体失败。
        - 生成 ``accepted=True``、状态 ``repaired_by_vision`` 的结果。

        返回：``(original, replacement, result, proof)`` 绑定列表，
        或在任何前置条件失败时返回 None。
        """

        # 前置约束：数量/唯一性/指纹唯一性
        if (
            not original_tables
            or len(original_tables) != len(replacement_tables)
            or len({table.table_id for table in original_tables}) != len(original_tables)
            or len({table.table_id for table in replacement_tables})
            != len(replacement_tables)
            or len(
                {table_content_fingerprint(table) for table in replacement_tables}
            )
            != len(replacement_tables)
        ):
            return None
        # 所有表必须都在同一页
        if any(
            self._table_pages(table) != {page_index}
            for table in [*original_tables, *replacement_tables]
        ):
            return None
        originals_by_id = {table.table_id: table for table in original_tables}
        if not repair_table_ids or not repair_table_ids.issubset(originals_by_id):
            return None

        # 库存级唯一匹配：返回 (original, replacement, match_basis) 列表
        matched = self._unique_inventory_matching(original_tables, replacement_tables)
        if matched is None:
            return None

        bindings = []
        for original, replacement, match_basis in matched:
            # 替换表本身必须是合格的
            replacement_reasons = self._reasons(replacement)
            if replacement_reasons:
                return None
            # 只处理需要修复的原始表
            if original.table_id not in repair_table_ids:
                continue
            # 取原始表校验失败时的修复请求作为证明的原始请求
            request = self.validate(original).repair_request
            if request is None:
                return None
            proof: TableRepairRequest | TableRepairProof = TableRepairProof(
                original_request=request,
                page_index=page_index,
                replacement_content_fingerprint=table_content_fingerprint(replacement),
                match_basis=match_basis,
                validated_mapping=TableRepairMapping(
                    original_table_id=original.table_id,
                    replacement_table_id=replacement.table_id,
                    page_index=page_index,
                ),
            )
            # 用审计逻辑复核证明与替换表的对应关系
            reasons: list[str] = []
            self._validate_repair_proof(original, replacement, proof, reasons)
            if reasons:
                return None
            candidate = replacement.model_copy(deep=True)
            candidate.status = "repaired_by_vision"
            result = TableValidationResult(
                accepted=True,
                activation_allowed=True,
                status="repaired_by_vision",
                table=candidate,
            )
            bindings.append((original, replacement, result, proof))
        return bindings

    @classmethod
    def _unique_inventory_matching(
        cls,
        originals: list[CanonicalTable],
        replacements: list[CanonicalTable],
    ) -> list[tuple[CanonicalTable, CanonicalTable, str]] | None:
        """对原始表与替换表做"唯一双向匹配"，返回 (orig, repl, basis) 列表。

        匹配优先级（每个原始表依次尝试）：
        1. ``source_region_id`` 完全一致（最强定位依据）。
        2. ``normalized_bbox`` 空间重叠（IoU>=0.8 且中心/边界接近）。
        3. ``source_block_id`` 一致（次强依据）。

        步骤：
        1. 为每个原始表构造候选替换表列表（含匹配依据标签）。
           任一原始表无候选则整体失败（None）。
        2. 用 Hopcroft-Karp 增广路径算法（``augment``）求最大二分匹配，
           得到"每个原始表唯一对应一个替换表"的映射；无法全覆盖则失败。
        3. 构建二部图并把"未选中的边"与"选中的边"反向建图，用 DFS 三色
           检测环；若存在环则说明匹配不是唯一的（存在交替环），整体失败。
        4. 汇总结果：为每个原始表取回其选中匹配的替换表与依据标签。
        """
        # 第一步：为每个原始表枚举候选（按优先级只取最强的一档）
        candidates: dict[int, list[tuple[int, str]]] = {}
        for original_index, original in enumerate(originals):
            # 优先级 1：source_region_id 精确匹配
            exact_regions = [
                (replacement_index, "source_region_id")
                for replacement_index, replacement in enumerate(replacements)
                if cls._source_region_id(original) is not None
                and cls._source_region_id(original)
                == cls._source_region_id(replacement)
            ]
            if exact_regions:
                candidates[original_index] = exact_regions
                continue
            # 优先级 2：归一化 bbox 空间匹配
            spatial = [
                (replacement_index, "normalized_bbox")
                for replacement_index, replacement in enumerate(replacements)
                if cls._bbox_locators_match(original, replacement)
            ]
            if spatial:
                candidates[original_index] = spatial
                continue
            # 优先级 3：source_block_id 匹配
            source_block_id = cls._source_block_id(original)
            candidates[original_index] = [
                (replacement_index, "source_block_id")
                for replacement_index, replacement in enumerate(replacements)
                if source_block_id is not None
                and source_block_id == cls._source_block_id(replacement)
            ]
        if any(not edges for edges in candidates.values()):
            return None

        # 第二步：最大二分匹配（增广路径算法）
        matched_right: dict[int, int] = {}  # right -> left

        def augment(left: int, seen: set[int]) -> bool:
            """尝试为 left 找一个未占用的 right；占用则尝试重排（增广）。"""
            for right, _ in candidates[left]:
                if right in seen:
                    continue
                seen.add(right)
                previous = matched_right.get(right)
                if previous is None or augment(previous, seen):
                    matched_right[right] = left
                    return True
            return False

        # 从候选最少的原始表开始贪心增广，减少回溯
        for left in sorted(candidates, key=lambda item: len(candidates[item])):
            if not augment(left, set()):
                return None
        # 必须每个原始表都被匹配到
        if len(matched_right) != len(originals):
            return None
        matched_left = {left: right for right, left in matched_right.items()}

        # 第三步：环检测——若存在"未选边/选中边"构成的交替环，说明
        # 存在两种等价匹配，匹配不唯一，整体失败
        graph: dict[int, list[int]] = {
            node: [] for node in range(len(originals) + len(replacements))
        }
        offset = len(originals)
        for left, edges in candidates.items():
            for right, _ in edges:
                # 选中的边反向建（right -> left）；未选中的正向建（left -> right）
                if matched_left[left] == right:
                    graph[offset + right].append(left)
                else:
                    graph[left].append(offset + right)
        colors: dict[int, int] = {}  # 0=未访问 1=访问中 2=已访问

        def has_cycle(node: int) -> bool:
            """DFS 三色法判环：遇到"访问中"节点即存在环。"""
            colors[node] = 1
            for neighbor in graph[node]:
                if colors.get(neighbor) == 1:
                    return True
                if colors.get(neighbor, 0) == 0 and has_cycle(neighbor):
                    return True
            colors[node] = 2
            return False

        if any(
            colors.get(node, 0) == 0 and has_cycle(node)
            for node in graph
        ):
            return None

        # 第四步：按原始表顺序组装结果，附匹配依据标签
        result = []
        for left, original in enumerate(originals):
            right = matched_left[left]
            basis = next(
                basis for candidate, basis in candidates[left] if candidate == right
            )
            result.append((original, replacements[right], basis))
        return result

    @staticmethod
    def _validate_repair_proof(
        original: CanonicalTable,
        repaired: CanonicalTable,
        proof: TableRepairRequest | TableRepairProof | dict[str, Any] | None,
        reasons: list[str],
    ) -> None:
        """校验修复证明的一致性；任何不符都会追加到 ``reasons``。

        参数：
        - ``original``：原始表格。
        - ``repaired``：修复后表格。
        - ``proof``：修复证明（TableRepairRequest / TableRepairProof /
          dict / None）。
        - ``reasons``：输出参数，校验出的问题会被追加到此列表。

        校验内容（针对 Request/dict 形态）：
        - 来源指纹必须等于原始表的身份指纹。
        - 证明中的定位符（稳定形态）必须等于原始表的稳定定位符。
        - 修复后表格的稳定定位符也必须等于原始表的稳定定位符。

        ``TableRepairProof`` 形态则转交给 ``_validate_inventory_proof``。
        """
        def add(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        if proof is None:
            add("repair_proof_missing")
            return
        if isinstance(proof, TableRepairProof):
            # 库存级证明：走专门的审计校验
            TableValidator._validate_inventory_proof(
                original,
                repaired,
                proof,
                reasons,
            )
            return
        if isinstance(proof, TableRepairRequest):
            fingerprint = proof.source_fingerprint
            locator = proof.locator
        elif isinstance(proof, dict):
            # dict 中带 original_request 说明是"库存证明"的 dict 形态，
            # 不应在此路径出现
            if "original_request" in proof:
                add("repair_proof_invalid")
                return
            fingerprint = proof.get("source_fingerprint")
            locator = proof.get("locator")
        else:
            add("repair_proof_invalid")
            return
        # 来源指纹必须匹配原始表身份指纹
        expected_fingerprint = table_identity_fingerprint(original)
        if fingerprint != expected_fingerprint:
            add("repair_source_fingerprint_mismatch")
        # 定位符：证明与修复表都必须落在原始表的稳定定位符上
        expected_locator = TableValidator._stable_locator(original)
        proof_locator = TableValidator._stable_locator_value(locator)
        repaired_locator = TableValidator._stable_locator(repaired)
        if not expected_locator:
            add("repair_stable_locator_missing")
        elif proof_locator != expected_locator:
            add("repair_proof_locator_mismatch")
        if expected_locator and repaired_locator != expected_locator:
            add("repair_locator_mismatch")

    @staticmethod
    def _validate_inventory_proof(
        original: CanonicalTable,
        repaired: CanonicalTable,
        proof: TableRepairProof,
        reasons: list[str],
    ) -> None:
        """对"库存级证明"（TableRepairProof）做完整审计校验。

        逐项核对（任一不符追加相应 reason）：
        1. 请求中的 table_id 必须等于原始表 ID。
        2. 请求中的来源指纹必须等于原始表身份指纹。
        3. 请求定位符（稳定形态）必须等于原始表稳定定位符。
        4. 修复表必须能提取出稳定定位符。
        5. 匹配依据（match_basis）必须真实成立：
           - source_region_id：两侧 region id 一致。
           - normalized_bbox：两侧 bbox 空间匹配。
           - source_block_id：两侧 block id 一致。
           - unique_table_on_page：本路径不允许（此前沿已排除）。
        6. 原始表与修复表的页码必须恰好是 proof 声明的那一页。
        7. 证明中的替换内容指纹必须等于修复表实测指纹。
        8. 经校验的映射必须与原始表/修复表/页号一致。
        """
        def add(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        request = proof.original_request
        expected_fingerprint = table_identity_fingerprint(original)
        expected_locator = TableValidator._stable_locator(original)
        request_locator = TableValidator._stable_locator_value(request.locator)
        repaired_locator = TableValidator._stable_locator(repaired)
        original_pages = TableValidator._table_pages(original)
        repaired_pages = TableValidator._table_pages(repaired)
        mapping = proof.validated_mapping

        # 1. 表格 ID 一致
        if request.table_id != original.table_id:
            add("repair_original_table_mismatch")
        # 2. 来源指纹一致
        if request.source_fingerprint != expected_fingerprint:
            add("repair_source_fingerprint_mismatch")
        # 3/4. 定位符一致性与存在性
        if not expected_locator or request_locator != expected_locator:
            add("repair_proof_locator_mismatch")
        if not repaired_locator:
            add("repair_replacement_locator_missing")
        # 5. 匹配依据必须真实成立
        if proof.match_basis == "source_region_id" and (
            TableValidator._source_region_id(original) is None
            or TableValidator._source_region_id(original)
            != TableValidator._source_region_id(repaired)
        ):
            add("repair_match_basis_invalid")
        elif proof.match_basis == "normalized_bbox" and not (
            TableValidator._bbox_locators_match(original, repaired)
        ):
            add("repair_match_basis_invalid")
        elif proof.match_basis == "source_block_id" and (
            TableValidator._source_block_id(original) is None
            or TableValidator._source_block_id(original)
            != TableValidator._source_block_id(repaired)
        ):
            add("repair_match_basis_invalid")
        elif proof.match_basis == "unique_table_on_page":
            add("repair_match_basis_invalid")
        # 6. 页码一致
        if original_pages != {proof.page_index} or repaired_pages != {
            proof.page_index
        }:
            add("repair_page_mismatch")
        # 7. 替换内容指纹一致
        if proof.replacement_content_fingerprint != table_content_fingerprint(
            repaired
        ):
            add("repair_replacement_fingerprint_mismatch")
        # 8. 经校验的映射一致
        if (
            mapping.original_table_id != original.table_id
            or mapping.replacement_table_id != repaired.table_id
            or mapping.page_index != proof.page_index
        ):
            add("repair_validated_mapping_mismatch")

    @staticmethod
    def _table_pages(table: CanonicalTable) -> set[int]:
        """返回表格覆盖的页序号集合（来自 source_spans，忽略 None）。"""
        return {
            span.page_index
            for span in table.source_spans
            if span.page_index is not None
        }

    @staticmethod
    def _table_bbox(table: CanonicalTable) -> tuple[float, float, float, float] | None:
        """返回表格首个可用 bbox（优先归一化 bbox，其次原始 bbox）。"""
        for span in table.source_spans:
            box = span.normalized_bbox or span.bbox
            if box is not None:
                return box
        return None

    @staticmethod
    def _normalized_table_bbox(
        table: CanonicalTable,
    ) -> tuple[float, float, float, float] | None:
        """返回表格首个归一化 bbox（不存在则为 None）。"""
        return next(
            (
                span.normalized_bbox
                for span in table.source_spans
                if span.normalized_bbox is not None
            ),
            None,
        )

    @classmethod
    def _bbox_locators_match(
        cls,
        original: CanonicalTable,
        replacement: CanonicalTable,
    ) -> bool:
        """判断两个表格的定位（bbox）是否空间匹配。

        匹配条件：
        - 双方都有归一化 bbox：计算 IoU（交并比）与中心点偏移、边界偏移，
          要求 IoU >= 0.8 且中心偏移 <= 0.02 且边界偏移 <= 0.02（阈值
          均为归一化坐标）。
        - 缺少归一化 bbox：退回比较原始 bbox 是否完全相等。

        用于把"修复后的表格"匹配回"原始表格"，作为空间定位依据。
        """
        original_normalized = cls._normalized_table_bbox(original)
        replacement_normalized = cls._normalized_table_bbox(replacement)
        if original_normalized is None or replacement_normalized is None:
            # 缺少归一化 bbox：要求原始 bbox 完全相等
            original_bbox = cls._table_bbox(original)
            replacement_bbox = cls._table_bbox(replacement)
            return original_bbox is not None and original_bbox == replacement_bbox

        # 归一化坐标下的 IoU 计算
        left, top, right, bottom = original_normalized
        other_left, other_top, other_right, other_bottom = replacement_normalized
        # 交集宽高（两矩形在 x/y 方向重叠部分）
        intersection_width = max(0.0, min(right, other_right) - max(left, other_left))
        intersection_height = max(
            0.0, min(bottom, other_bottom) - max(top, other_top)
        )
        intersection = intersection_width * intersection_height
        original_area = max(0.0, right - left) * max(0.0, bottom - top)
        replacement_area = max(0.0, other_right - other_left) * max(
            0.0, other_bottom - other_top
        )
        union = original_area + replacement_area - intersection
        if union <= 0:
            return False
        iou = intersection / union
        # 中心点最大偏移（x/y 两个方向取大者）
        center_delta = max(
            abs((left + right) / 2 - (other_left + other_right) / 2),
            abs((top + bottom) / 2 - (other_top + other_bottom) / 2),
        )
        # 四条边界各自的最大偏移
        boundary_delta = max(
            abs(left - other_left),
            abs(top - other_top),
            abs(right - other_right),
            abs(bottom - other_bottom),
        )
        # 同时满足 IoU、中心偏移、边界偏移阈值才算匹配
        return iou >= 0.8 and center_delta <= 0.02 and boundary_delta <= 0.02

    @staticmethod
    def _source_block_id(table: CanonicalTable) -> str | None:
        """返回表格首个非空 source_block_id（无则 None）。"""
        for span in table.source_spans:
            if span.source_block_id and span.source_block_id.strip():
                return span.source_block_id.strip()
        return None

    @staticmethod
    def _source_region_id(table: CanonicalTable) -> str | None:
        """从 source_spans 的 metadata 中提取首个非空 source_region_id。"""
        for span in table.source_spans:
            value = span.metadata.get("source_region_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _stable_locator(table: CanonicalTable) -> dict[str, Any]:
        """计算表格的"稳定定位符"（规范化后的定位字典）。

        先取 ``CanonicalQualityGate._table_locator`` 的基础定位，
        若有 source_region_id 则补充进定位符，再经
        ``_stable_locator_value`` 归一化。稳定定位符用于跨
        （原始表 -> 修复表）的一致性与匹配校验。
        """
        locator = CanonicalQualityGate._table_locator(table)
        source_region_id = TableValidator._source_region_id(table)
        if source_region_id is not None:
            locator["source_region_id"] = source_region_id
        return TableValidator._stable_locator_value(locator)

    @staticmethod
    def _stable_locator_value(value: object) -> dict[str, Any]:
        """把定位字典归一化为"稳定形态"：剔除 None 值并只保留一档定位依据。

        优先级（互斥，只保留第一个命中的键集合）：
        1. 有 bbox：只保留 {page_index, bbox}。
        2. 有 source_region_id：只保留 {page_index, source_region_id}。
        3. 有 source_block_id：只保留 {page_index, source_block_id}。
        4. 否则返回空 dict。

        目的：不同解析器产出的定位符字段可能不同，归一化后便于直接比较。
        """
        if not isinstance(value, dict):
            return {}
        page_index = value.get("page_index")
        bbox = value.get("bbox")
        source_region_id = value.get("source_region_id")
        source_block_id = value.get("source_block_id")
        if bbox is not None:
            return {
                key: item
                for key, item in {"page_index": page_index, "bbox": bbox}.items()
                if item is not None
            }
        if source_region_id is not None:
            return {
                key: item
                for key, item in {
                    "page_index": page_index,
                    "source_region_id": source_region_id,
                }.items()
                if item is not None
            }
        if source_block_id is not None:
            return {
                key: item
                for key, item in {
                    "page_index": page_index,
                    "source_block_id": source_block_id,
                }.items()
                if item is not None
            }
        return {}

    @classmethod
    def _reasons(cls, table: CanonicalTable) -> list[str]:
        """收集表格的全部校验理由（仅基于源内容），返回理由列表。

        检查项（由 ``CanonicalQualityGate`` 提供基础项，再补充本模块项）：
        - 源跨度缺失（source_spans_missing）。
        - 表头为空（header_empty）；数据行为空（data_rows_empty）。
        - caption 编号与 table_number 不一致（caption_number_mismatch）。
        - 静默截断（silent_truncation：truncated 标记或行数不一致）。
        - 数值/单位 token 被切碎（numeric_token_fragmented / unit_token_fragmented）。
        - 期望跨页延续但未恢复（cross_page_continuation_missing）。
        - 跨页合并表：源 markdown/html 分段必须能与表内容对得上
          （cross_page_source_markdown_mismatch 等）。
        - 单表：source_html 若能解析，须与表头/数据行/单元格签名一致
          （source_html_invalid / source_html_mismatch /
          source_html_cell_mismatch）。
        """
        # 基础校验项（复用 CanonicalQualityGate 的表格校验器）
        reasons = list(CanonicalQualityGate._invalid_table_reasons(table))

        def add(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        # 源跨度/表头/数据行完整性
        if not table.source_spans:
            add("source_spans_missing")
        if not any(value.strip() for value in table.headers):
            add("header_empty")
        if not any(value.strip() for row in table.rows for value in row):
            add("data_rows_empty")

        # caption 编号一致性：metadata.table_number 与 caption 文本匹配
        expected_number = str(table.metadata.get("table_number") or "").strip()
        if expected_number:
            caption_match = cls._caption_number.search(table.caption or "")
            if caption_match is None or caption_match.group(1).casefold() != expected_number.casefold():
                add("caption_number_mismatch")

        # 静默截断：显式标记或源行数与实际行数不一致
        if table.metadata.get("truncated") is True:
            add("silent_truncation")
        source_row_count = table.metadata.get("source_row_count")
        if isinstance(source_row_count, int) and source_row_count != len(table.rows):
            add("silent_truncation")
        # 数值/单位 token 碎片化
        if table.metadata.get("fragmented_numeric_tokens"):
            add("numeric_token_fragmented")
        if table.metadata.get("fragmented_unit_tokens"):
            add("unit_token_fragmented")
        # 期望跨页延续但未恢复
        if table.metadata.get("continuation_expected") and not table.metadata.get(
            "continuation_recovered"
        ):
            add("cross_page_continuation_missing")
        # 跨页合并表：源 markdown 分段须与最终行内容一致
        source_markdowns = table.metadata.get("source_markdowns")
        if table.status == "cross_page_merged" and isinstance(source_markdowns, list):
            if cls._source_markdown_segments_match(table, source_markdowns):
                # 分段能对上：清除此前可能由 QualityGate 给出的 markdown 异议
                reasons = [
                    reason
                    for reason in reasons
                    if reason
                    not in {"source_markdown_invalid", "source_markdown_mismatch"}
                ]
            else:
                add("cross_page_source_markdown_mismatch")

        # 源 HTML：跨页合并表逐段核对；单表则整体解析核对
        source_htmls = table.metadata.get("source_htmls")
        if (
            table.status == "cross_page_merged"
            and isinstance(source_htmls, list)
            and source_htmls
        ):
            if not cls._source_html_segments_match(table, source_htmls):
                add("cross_page_source_html_mismatch")
        elif table.source_html is not None:
            try:
                from app.services.canonical_artifacts import CanonicalArtifactStore

                # 从 HTML 重建表格三要素：单元格、表头、数据行
                html_cells, html_headers, html_rows = (
                    CanonicalArtifactStore._canonical_table_from_html(
                        table.source_html
                    )
                )
            except (TypeError, ValueError):
                add("source_html_invalid")
            else:
                # HTML 重建的表头/数据行必须与当前表一致
                if (html_headers, html_rows) != (table.headers, table.rows):
                    add("source_html_mismatch")
                # 单元格签名集合必须一致（行/列/跨度/表头/文本）
                html_signatures = sorted(cls._cell_signature(cell) for cell in html_cells)
                cell_signatures = sorted(cls._cell_signature(cell) for cell in table.cells)
                if html_signatures != cell_signatures:
                    add("source_html_cell_mismatch")
        return reasons

    @classmethod
    def _source_markdown_segments_match(
        cls,
        table: CanonicalTable,
        markdowns: list[object],
    ) -> bool:
        """校验"跨页合并"的源 markdown 分段能否拼出最终表格行。

        每个分段都解析为 (headers, rows) 网格；把所有分段的 rows 拼接
        （跳过各段可能重复的表头行），要求拼接结果等于 ``table.rows``。
        任何分段非字符串、解析失败或表头不一致都会返回 False。
        """
        grids: list[tuple[list[str], list[list[str]]]] = []
        for markdown in markdowns:
            if not isinstance(markdown, str):
                return False
            parsed = CanonicalQualityGate._markdown_table_data(markdown)
            if parsed is None:
                return False
            grids.append(parsed)
        return cls._combined_segment_rows(table.headers, grids) == table.rows

    @classmethod
    def _source_html_segments_match(
        cls,
        table: CanonicalTable,
        htmls: list[object],
    ) -> bool:
        """校验"跨页合并"的源 HTML 分段（及对应单元格分段）与最终表格一致。

        每个 HTML 分段重建为 (cells, headers, rows)；同时要求 metadata 中
        ``source_cell_segments`` 与 htmls 一一对应，且重建的单元格签名
        集合与记录的单元格分段签名集合一致。最后把各分段 rows 拼接，
        结果须等于 ``table.rows``。
        """
        from app.services.canonical_artifacts import CanonicalArtifactStore

        grids: list[tuple[list[str], list[list[str]]]] = []
        # 单元格分段必须与 html 分段一一对应
        cell_segments = table.metadata.get("source_cell_segments")
        if not isinstance(cell_segments, list) or len(cell_segments) != len(htmls):
            return False
        for index, source_html in enumerate(htmls):
            if not isinstance(source_html, str):
                return False
            try:
                cells, headers, rows = (
                    CanonicalArtifactStore._canonical_table_from_html(source_html)
                )
                grids.append((headers, rows))
                expected_cells = [
                    CanonicalCell.model_validate(item)
                    for item in cell_segments[index]
                ]
            except (TypeError, ValueError):
                return False
            # 重建单元格与记录的单元格分段必须签名一致
            if sorted(cls._cell_signature(cell) for cell in cells) != sorted(
                cls._cell_signature(cell) for cell in expected_cells
            ):
                return False
        return cls._combined_segment_rows(table.headers, grids) == table.rows

    @staticmethod
    def _combined_segment_rows(
        headers: list[str],
        grids: list[tuple[list[str], list[list[str]]]],
    ) -> list[list[str]] | None:
        """把多个 (headers, rows) 网格拼接为最终行列表。

        - 每段表头必须与目标表头一致，否则返回 None。
        - 若某段首行与表头重复（跨页续表的重复表头），跳过该行。
        - 所有段的剩余行按顺序拼接。

        返回：拼接后的行列表；表头不一致时返回 None。
        """
        combined: list[list[str]] = []
        for segment_headers, segment_rows in grids:
            if segment_headers != headers:
                return None
            rows = list(segment_rows)
            if rows and rows[0] == headers:
                rows = rows[1:]  # 去掉重复表头行
            combined.extend(rows)
        return combined

    @staticmethod
    def _cell_signature(cell: CanonicalCell) -> tuple[object, ...]:
        """计算单元格的签名（行/列/跨度/表头标志/文本），用于集合级比对。"""
        return (
            cell.row_index,
            cell.column_index,
            cell.rowspan,
            cell.colspan,
            cell.is_header,
            cell.text,
        )

    @staticmethod
    def _repair_request(
        table: CanonicalTable,
        reasons: list[str],
    ) -> TableRepairRequest:
        """为校验失败的表格生成"修复请求"。

        请求包含：
        - ``table_id`` / ``locator``：定位待修复的源区域。
        - ``reasons``：校验失败原因（写入指令）。
        - ``source_fingerprint``：原始表身份指纹，修复后用于核对。
        - ``instructions``：指示视觉模型"仅重提取定位的源表格区域，
          保留 caption、完整表头/数据网格、合并单元格跨度、单元格 bbox、
          数值 token、单位、脚注与源 markdown"，并列出失败原因。
        """
        locator = CanonicalQualityGate._table_locator(table)
        reason_text = ", ".join(reasons)
        return TableRepairRequest(
            table_id=table.table_id,
            reasons=reasons,
            locator=locator,
            source_fingerprint=table_identity_fingerprint(table),
            instructions=(
                "Re-extract only the located source table region. Preserve its caption, "
                "complete header/data grid, merged-cell spans, cell bboxes, numeric tokens, "
                f"units, footnotes, and source markdown. Validation failures: {reason_text}."
            ),
        )


class StructuredEvidenceBuilder:
    """Build table Parent/Child chunks and source-faithful visual evidence.

    构建表格 Parent/Child 分块以及源保真的视觉证据（图表/公式）。

    职责：
    - 把 CanonicalTable 拆成"父分块 + 子分块"：父分块覆盖整表，
      子分块按语义行分组或按 token 预算切分，保证不超过 max_tokens。
    - 为过长的表格行生成"列分组/单元格片段"子分块（不丢失源字符）。
    - 跨页表格合并（``merge_cross_page_tables``）。
    - 把图表/公式包装为证据分块，模型生成摘要仅进入 embedding_text。
    - token 计数：注入计数 / transformers tokenizer / UTF-8 字节兜底
      三档模式（``token_count_mode``）。
    """

    # 进程级 tokenizer 缓存（键 = (模型名, 加载器身份)）及并发锁，
    # 避免多线程重复加载大模型文件
    _tokenizer_cache: dict[tuple[str, tuple[str, object]], Any] = {}
    _tokenizer_key_locks: dict[
        tuple[str, tuple[str, object]], threading.Lock
    ] = {}
    _tokenizer_cache_lock = threading.Lock()

    def __init__(
        self,
        *,
        token_counter: Callable[[str], int] | None = None,
        tokenizer: Any | None = None,
        tokenizer_name: str | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
        strict_tokenizer: bool = False,
    ) -> None:
        """初始化构建器，确定 token 计数策略。

        三种计数模式（按优先级）：
        1. ``token_counter`` 注入：``token_count_mode = "injected_counter"``。
        2. ``tokenizer`` 实例注入：``"transformers"``。
        3. 否则用 ``tokenizer_loader``（默认 ``_load_local_tokenizer``，
           读取配置中的本地 tokenizer），并设置缓存键（``tokenizer_name``
           + 加载器身份：默认按修订+路径，自定义按函数 id），
           模式暂记为 ``"transformers"``（加载失败时在
           ``_ensure_tokenizer_loaded`` 中降级为 utf8_bytes_fallback）。

        ``strict_tokenizer`` 为 True 时，tokenizer 加载失败将直接抛错，
        不允许字节计数兜底。
        """
        settings = get_settings()
        self.tokenizer_name = tokenizer_name or settings.semantic_tokenizer_name
        self.tokenizer_revision = settings.semantic_tokenizer_revision
        self.strict_tokenizer = strict_tokenizer
        self.table_aliases: dict[str, str] = {}  # 合并表 ID -> 根表 ID
        self._tokenizer = tokenizer
        self._token_counter = token_counter
        self._tokenizer_loader: Callable[[str], Any] | None = None
        self._tokenizer_cache_key: tuple[str, tuple[str, object]] | None = None
        self._tokenizer_fallback_reason: str | None = None
        if token_counter is not None:
            self.token_count_mode = "injected_counter"
            return
        if tokenizer is not None:
            self.token_count_mode = "transformers"
            return
        loader = tokenizer_loader or self._load_local_tokenizer
        if tokenizer_loader is None:
            # 默认加载器：缓存键基于配置的修订与本地路径
            local_path = settings.semantic_tokenizer_local_path
            resolved_path = (
                str(Path(local_path).expanduser().resolve())
                if local_path is not None
                else None
            )
            loader_identity = (
                "default",
                (settings.semantic_tokenizer_revision, resolved_path),
            )
        else:
            # 自定义加载器：用函数对象 id 作缓存身份
            loader_identity = ("custom", id(tokenizer_loader))
        self._tokenizer_loader = loader
        self._tokenizer_cache_key = (self.tokenizer_name, loader_identity)
        self.token_count_mode = "transformers"

    @staticmethod
    def _load_local_tokenizer(name: str) -> Any:
        """默认 tokenizer 加载器：从本地缓存解析并返回 tokenizer 实例。

        委托给 ``ingestion_identity.resolve_local_tokenizer``，确保离线、
        可复现（带内容哈希校验）地加载固定的 tokenizer。
        """
        from app.services.ingestion_identity import resolve_local_tokenizer

        settings = get_settings()
        return resolve_local_tokenizer(
            name,
            settings.semantic_tokenizer_revision,
            local_path=settings.semantic_tokenizer_local_path,
        ).tokenizer

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数（非负整数）。

        计数来源：
        - 注入的 ``token_counter``：直接调用。
        - transformers tokenizer：``encode(add_special_tokens=False)`` 长度。
        - 兜底（tokenizer 不可用）：按 UTF-8 字节数计。

        # A tokenizer token cannot encode less than one source byte. Counting
        # UTF-8 bytes therefore deliberately overestimates CJK, formulas,
        # punctuation, and long identifiers instead of silently undercounting.
        # 一个 tokenizer token 编码的源字节数不会少于 1，因此用 UTF-8 字节
        # 计数会"故意高估"CJK、公式、标点与长标识符，而不是静默低估——
        # 低估会把超限文本塞进受限窗口，高估只会损失一点容量。
        """
        if self._token_counter is not None:
            count = self._token_counter(text)
        else:
            self._ensure_tokenizer_loaded()
            if self._tokenizer is not None:
                count = len(self._tokenizer.encode(text, add_special_tokens=False))
            else:
                # 兜底：UTF-8 字节计数
                count = len(text.encode("utf-8"))
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("token counter must return a non-negative integer")
        return count

    def _lossless_wrapped_windows(
        self,
        source_text: str,
        *,
        max_tokens: int,
        render: Callable[[str], str],
        preferred_boundaries: re.Pattern[str] | None = None,
    ) -> list[tuple[int, int, str, str]]:
        """把源文本无损地切成若干"渲染后不超 max_tokens"的窗口。

        每个窗口返回 ``(start, end, fragment, rendered)``：
        - ``fragment``：源文本的原始片段（保留一切字符，无损失）。
        - ``rendered``：经 ``render`` 渲染后的文本（如带标题/标记）。

        算法：
        1. 渲染后整段能放下 → 单个窗口。
        2. 空文本渲染后已超限 → 报错（没有给源文本留空间）。
        3. 对每个游标位置，用二分查找确定"渲染后仍不超过 max_tokens"的
           最大前缀长度（``best``）。
        4. 在 ``best`` 内优先在首选边界（``preferred_boundaries`` 正则，
           如公式的换行/分号）切分；否则退回空白字符处切分，避免切断单词。
        5. 若按边界切出的片段渲染后反而超限，退回用 ``best`` 精确切分。
        """
        if not source_text:
            return []
        # 整段放得下：单窗口
        if self.estimate_tokens(render(source_text)) <= max_tokens:
            return [(0, len(source_text), source_text, render(source_text))]
        # 连空内容都放不下：结构性错误
        if self.estimate_tokens(render("")) >= max_tokens:
            raise ValueError("structured Child context leaves no room for source text")
        windows: list[tuple[int, int, str, str]] = []
        cursor = 0
        while cursor < len(source_text):
            # 二分：找渲染后 <= max_tokens 的最大前缀结束位置
            low = cursor + 1
            high = len(source_text)
            best = cursor
            while low <= high:
                middle = (low + high) // 2
                candidate = render(source_text[cursor:middle])
                if self.estimate_tokens(candidate) <= max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best <= cursor:
                raise ValueError("structured source cannot fit within the Child token limit")
            boundary = best
            fragment_scope = source_text[cursor:best]
            # 优先在"首选边界"上切分（不截断结构单元）
            if preferred_boundaries is not None and best < len(source_text):
                matches = list(preferred_boundaries.finditer(fragment_scope))
                if matches:
                    candidate_boundary = cursor + matches[-1].end()
                    if candidate_boundary > cursor:
                        boundary = candidate_boundary
            # 否则退回空白字符边界
            elif best < len(source_text):
                whitespace = max(
                    fragment_scope.rfind(" "),
                    fragment_scope.rfind("\n"),
                    fragment_scope.rfind("\t"),
                )
                if whitespace >= 0:
                    boundary = cursor + whitespace + 1
            fragment = source_text[cursor:boundary]
            rendered = render(fragment)
            # 边界调整后可能超限，退回精确的 best 切分
            if self.estimate_tokens(rendered) > max_tokens:
                boundary = best
                fragment = source_text[cursor:boundary]
                rendered = render(fragment)
            windows.append((cursor, boundary, fragment, rendered))
            cursor = boundary
        return windows

    def _ensure_tokenizer_loaded(self) -> None:
        """确保 tokenizer 已加载（线程安全 + 进程级缓存）。

        并发模型：双层锁——
        1. 先查进程级缓存（无锁读）。
        2. 未命中时，在 ``_tokenizer_cache_lock`` 下取该 key 的专用锁
           （``_tokenizer_key_locks``），不同 key 可并行加载。
        3. 持 key 锁后再查一次缓存（double-check），未命中才真正加载。

        加载失败（strict_tokenizer 为假时）：
        - 清除 key 锁；记录兜底原因；切换 ``token_count_mode`` 为
          ``utf8_bytes_fallback``，后续用 UTF-8 字节计数。
        - ``strict_tokenizer`` 为真时直接抛 ``RuntimeError``，禁止兜底。
        """
        if self._tokenizer is not None or self._tokenizer_loader is None:
            return
        key = self._tokenizer_cache_key
        if key is None:
            return
        # 无锁快速路径
        cached = self._tokenizer_cache.get(key)
        if cached is not None:
            self._tokenizer = cached
            return
        with self._tokenizer_cache_lock:
            cached = self._tokenizer_cache.get(key)
            if cached is not None:
                self._tokenizer = cached
                return
            # 取该 key 的专用锁（首次创建）
            key_lock = self._tokenizer_key_locks.setdefault(key, threading.Lock())
        with key_lock:
            cached = self._tokenizer_cache.get(key)
            if cached is None:
                try:
                    cached = self._tokenizer_loader(self.tokenizer_name)
                except Exception as exc:  # noqa: BLE001 - strict/fallback contract
                    # 加载失败：移除 key 锁，按 strict 配置决定抛错或兜底
                    with self._tokenizer_cache_lock:
                        self._tokenizer_key_locks.pop(key, None)
                    if self.strict_tokenizer:
                        raise RuntimeError(
                            f"Required tokenizer {self.tokenizer_name!r} revision "
                            f"{self.tokenizer_revision!r} is unavailable from the local "
                            f"cache; semantic ingestion cannot use a fallback token count. "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    self._tokenizer_loader = None
                    self._tokenizer_fallback_reason = type(exc).__name__
                    self.token_count_mode = "utf8_bytes_fallback"
                    return
                # 成功：写入缓存并移除 key 锁
                with self._tokenizer_cache_lock:
                    self._tokenizer_cache[key] = cached
                    self._tokenizer_key_locks.pop(key, None)
            self._tokenizer = cached

    def table_chunks(
        self,
        table: CanonicalTable,
        max_tokens: int,
    ) -> tuple[StructuredEvidenceChunk, list[StructuredEvidenceChunk]]:
        """把表格构建为"父分块 + 子分块"。

        参数：``table`` 源表格；``max_tokens`` 子分块 token 上限。

        流程：
        1. 先用 ``TableValidator`` 校验表格；未通过则抛 ValueError
           （构建器拒绝为不合格表格产出证据分块）。
        2. 父分块：覆盖全部行的整表文本，ID 由表格身份指纹派生。
        3. 按语义行分组（``_semantic_row_groups``）产出子分块：
           - 组文本不超限 → 直接作为一组。
           - 超限 → 按 token 预算再拆分（``_split_group_by_token_limit``）。
           - 单行仍超限 → 进入过宽行处理（``_overlong_table_row_chunks``，
             按列分组/单元格片段切分），并以 ``continue`` 跳过常规子分块。
        4. 补充分块：未被任何子分块覆盖的脚注
           （``_missing_table_footnote_chunks``）。

        返回：``(parent, children)``。
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        # 先过表格校验，不合格不产出证据
        validation = TableValidator().validate(table)
        if not validation.accepted:
            raise ValueError(
                f"table {table.table_id!r} failed validation: {validation.reasons}"
            )
        table = validation.table
        # 父分块文本：caption + 全部行 + 脚注
        parent_text = self._table_text(table, list(range(len(table.rows))))
        parent_id = self._stable_id(
            "table-parent",
            table_identity_fingerprint(table),
            parent_text,
        )
        parent = StructuredEvidenceChunk(
            chunk_id=parent_id,
            chunk_role="parent",
            block_type="table",
            text=parent_text,
            embedding_text=parent_text,
            token_count=self.estimate_tokens(parent_text),
            source_spans=self._deduplicate_spans(table.source_spans),
            metadata=self._table_metadata(
                table,
                list(range(len(table.rows))),
                overflow=False,
            ),
        )

        children: list[StructuredEvidenceChunk] = []
        # 按语义行分组产出子分块
        for row_indices in self._semantic_row_groups(table):
            group_text = self._table_text(table, row_indices)
            if self.estimate_tokens(group_text) <= max_tokens:
                child_groups = [row_indices]
            else:
                # 组超限：再按 token 预算拆分
                child_groups = self._split_group_by_token_limit(
                    table,
                    row_indices,
                    max_tokens,
                )
            for child_indices in child_groups:
                child_text = self._table_text(table, child_indices)
                token_count = self.estimate_tokens(child_text)
                overflow = token_count > max_tokens
                if overflow and len(child_indices) == 1:
                    # 单行仍超限：进入过宽行专门切分（列分组/单元格片段），
                    # 并 continue 跳过下面的常规子分块创建
                    children.extend(
                        self._overlong_table_row_chunks(
                            table,
                            row_index=child_indices[0],
                            parent_id=parent_id,
                            max_tokens=max_tokens,
                        )
                    )
                    continue
                child_id = self._stable_id(
                    "table-child",
                    parent_id,
                    json.dumps(child_indices, separators=(",", ":")),
                    child_text,
                )
                children.append(
                    StructuredEvidenceChunk(
                        chunk_id=child_id,
                        parent_chunk_id=parent_id,
                        chunk_role="child",
                        block_type="table",
                        text=child_text,
                        embedding_text=child_text,
                        token_count=token_count,
                        source_spans=self._row_source_spans(table, child_indices),
                        metadata=self._table_metadata(
                            table,
                            child_indices,
                            overflow=overflow,
                        ),
                    )
                )
        # 补充未覆盖的脚注分块
        children.extend(
            self._missing_table_footnote_chunks(
                table,
                parent_id=parent_id,
                existing_children=children,
                max_tokens=max_tokens,
            )
        )
        return parent, children

    def _overlong_table_row_chunks(
        self,
        table: CanonicalTable,
        *,
        row_index: int,
        parent_id: str,
        max_tokens: int,
    ) -> list[StructuredEvidenceChunk]:
        """处理"单行就超过 token 上限"的过宽行，切成若干子分块。

        策略：把行按列分组切分，每组始终携带"标识列"（identity_column，
        默认第 0 列）以保持行身份的上下文；标识列放不下时退化为整行各列。

        实现：
        - ``identity_column=0``：取第 0 列作行标识。
        - 若脚注在"仅标识列 + 脚注"下也放得下，则允许子分块带脚注。
        - 先尝试把若干可变列聚成一组（不超过 max_tokens）→ 生成
          "列分组"子分块；当加入下一列会超限时，把当前组落为子分块。
        - 单列本身超限时：对该列单元格文本做无损窗口切分
          （``_lossless_wrapped_windows``），产出"单元格片段"子分块。
        - 每个子分块附带行身份元信息（row_index、标识列值与哈希）。
        """
        row = table.rows[row_index]
        if not row:
            raise ValueError(f"table {table.table_id!r} contains an empty overlong row")
        identity_column = 0
        # 判断"仅标识列 + 全部脚注"是否放得下，决定子分块能否带脚注
        include_footnotes = bool(table.footnotes) and self.estimate_tokens(
            self._table_column_group_text(
                table,
                row_index,
                [identity_column],
                row_values=[""],
                include_footnotes=True,
            )
        ) < max_tokens
        # 标识列单独是否放得下
        identity_fits = self.estimate_tokens(
            self._table_column_group_text(
                table,
                row_index,
                [identity_column],
                include_footnotes=include_footnotes,
            )
        ) <= max_tokens
        # 需要按列切分的"可变列"范围：标识列放得下则从第 1 列起，
        # 否则退化为整行所有列
        variable_columns = (
            list(range(1, len(row))) or [identity_column]
            if identity_fits
            else list(range(len(row)))
        )
        chunks: list[StructuredEvidenceChunk] = []
        pending_columns: list[int] = []

        def selected_columns(columns: list[int]) -> list[int]:
            if identity_column in columns or not identity_fits:
                return list(dict.fromkeys(columns))
            return [identity_column, *columns]

        def row_identity_metadata() -> dict[str, Any]:
            return {
                "row_index": row_index,
                "identity_column_index": identity_column,
                "identity_value_sha256": hashlib.sha256(
                    row[identity_column].encode("utf-8")
                ).hexdigest(),
            }

        def append_column_group(columns: list[int]) -> None:
            selected = selected_columns(columns)
            text = self._table_column_group_text(
                table,
                row_index,
                selected,
                include_footnotes=include_footnotes,
            )
            if self.estimate_tokens(text) > max_tokens and include_footnotes:
                text = self._table_column_group_text(
                    table,
                    row_index,
                    selected,
                    include_footnotes=False,
                )
            if self.estimate_tokens(text) > max_tokens:
                raise ValueError("table column group exceeds the Child token limit")
            metadata = {
                **self._table_metadata(table, [row_index], overflow=False),
                "column_indices": selected,
                "source_fragment": "".join(row[index] for index in columns),
                "row_identity": row_identity_metadata(),
            }
            chunks.append(
                StructuredEvidenceChunk(
                    chunk_id=self._stable_id(
                        "table-column-child",
                        parent_id,
                        str(row_index),
                        json.dumps(selected, separators=(",", ":")),
                        text,
                    ),
                    parent_chunk_id=parent_id,
                    chunk_role="child",
                    block_type="table",
                    text=text,
                    embedding_text=text,
                    token_count=self.estimate_tokens(text),
                    source_spans=self._row_source_spans(table, [row_index]),
                    metadata=metadata,
                )
            )

        for column_index in variable_columns:
            candidate = [*pending_columns, column_index]
            candidate_text = self._table_column_group_text(
                table,
                row_index,
                selected_columns(candidate),
                include_footnotes=include_footnotes,
            )
            if self.estimate_tokens(candidate_text) <= max_tokens:
                pending_columns = candidate
                continue
            if pending_columns:
                append_column_group(pending_columns)
                pending_columns = []
            selected = selected_columns([column_index])
            empty_values = [row[index] for index in selected]
            empty_values[selected.index(column_index)] = ""
            include_column_footnotes = include_footnotes and self.estimate_tokens(
                self._table_column_group_text(
                    table,
                    row_index,
                    selected,
                    row_values=empty_values,
                    include_footnotes=True,
                )
            ) < max_tokens

            def render(fragment: str, *, current_column: int = column_index) -> str:
                values = [row[index] for index in selected]
                values[selected.index(current_column)] = fragment
                return self._table_column_group_text(
                    table,
                    row_index,
                    selected,
                    row_values=values,
                    include_footnotes=include_column_footnotes,
                )

            windows = self._lossless_wrapped_windows(
                row[column_index],
                max_tokens=max_tokens,
                render=render,
            )
            for start, end, fragment, text in windows:
                metadata = {
                    **self._table_metadata(table, [row_index], overflow=False),
                    "column_indices": selected,
                    "cell_fragment_column_index": column_index,
                    "source_fragment": fragment,
                    "source_fragment_start": start,
                    "source_fragment_end": end,
                    "row_identity": row_identity_metadata(),
                }
                chunks.append(
                    StructuredEvidenceChunk(
                        chunk_id=self._stable_id(
                            "table-cell-child",
                            parent_id,
                            str(row_index),
                            str(column_index),
                            str(start),
                            text,
                        ),
                        parent_chunk_id=parent_id,
                        chunk_role="child",
                        block_type="table",
                        text=text,
                        embedding_text=text,
                        token_count=self.estimate_tokens(text),
                        source_spans=self._row_source_spans(table, [row_index]),
                        metadata=metadata,
                    )
                )
        if pending_columns:
            append_column_group(pending_columns)
        for part_index, chunk in enumerate(chunks):
            chunk.metadata["structured_part_index"] = part_index
            chunk.metadata["structured_part_count"] = len(chunks)
        return chunks

    @staticmethod
    def _table_column_group_text(
        table: CanonicalTable,
        row_index: int,
        column_indices: list[int],
        *,
        row_values: list[str] | None = None,
        include_footnotes: bool = False,
    ) -> str:
        """生成"指定列子集"的 Markdown 网格文本（供过宽行子分块）。

        参数：
        - ``row_index``：目标数据行。
        - ``column_indices``：要包含的列下标。
        - ``row_values``：可选的单元格值覆盖（用于无损窗口渲染占位）。
        - ``include_footnotes``：是否附带脚注行。

        输出：caption（若有） + 表头子集网格 + 单行值网格
        （可选 + 脚注行）。
        """
        headers = [table.headers[index] for index in column_indices]
        values = row_values or [table.rows[row_index][index] for index in column_indices]
        lines: list[str] = []
        if table.caption:
            lines.append(table.caption)
        lines.extend(StructuredEvidenceBuilder._markdown_grid(headers, [values]))
        if include_footnotes:
            lines.extend(f"Footnote: {footnote}" for footnote in table.footnotes)
        return "\n".join(lines)

    def _missing_table_footnote_chunks(
        self,
        table: CanonicalTable,
        *,
        parent_id: str,
        existing_children: list[StructuredEvidenceChunk],
        max_tokens: int,
    ) -> list[StructuredEvidenceChunk]:
        """为未被已有子分块覆盖的脚注生成独立子分块。

        逻辑：拼接所有已有子分块的文本作为 ``existing_text``；凡脚注文本
        已出现在其中则跳过（说明已被覆盖），否则为每个脚注用无损窗口
        切分（``_lossless_wrapped_windows``）生成带
        ``footnote_index/part_index/part_count`` 元信息的分块。
        """
        chunks: list[StructuredEvidenceChunk] = []
        existing_text = "\n".join(child.text for child in existing_children)
        for footnote_index, footnote in enumerate(table.footnotes):
            if footnote in existing_text:
                continue  # 已覆盖，跳过

            def render(fragment: str) -> str:
                return f"Footnote: {fragment}"

            windows = self._lossless_wrapped_windows(
                footnote,
                max_tokens=max_tokens,
                render=render,
            )
            for part_index, (start, end, fragment, text) in enumerate(windows):
                metadata = {
                    **self._table_metadata(
                        table,
                        list(range(len(table.rows))),
                        overflow=False,
                    ),
                    "footnote_index": footnote_index,
                    "footnote_part_index": part_index,
                    "footnote_part_count": len(windows),
                    "source_fragment": fragment,
                    "source_fragment_start": start,
                    "source_fragment_end": end,
                }
                chunks.append(
                    StructuredEvidenceChunk(
                        chunk_id=self._stable_id(
                            "table-footnote-child",
                            parent_id,
                            str(footnote_index),
                            str(part_index),
                            text,
                        ),
                        parent_chunk_id=parent_id,
                        chunk_role="child",
                        block_type="table",
                        text=text,
                        embedding_text=text,
                        token_count=self.estimate_tokens(text),
                        source_spans=self._deduplicate_spans(table.source_spans),
                        metadata=metadata,
                    )
                )
        return chunks

    def merge_cross_page_tables(
        self,
        tables: Iterable[CanonicalTable],
    ) -> list[CanonicalTable]:
        """把"跨页延续表"合并回其根表，返回合并后的根表列表。

        输入：所有相关表（根表 + 延续表）。延续表通过 metadata 中的
        ``continuation_of`` 指向其父表 ID。

        步骤：
        1. 深拷贝输入表；校验 table_id 无重复；记录输入顺序。
        2. 解析延续关系图（parent_by_id / children），校验：
           - 延续父表必须存在于输入中。
           - 每个父表至多一个直接延续子表（无分支冲突）。
           - 图中无环。
        3. 确定根表集合并按"起始页 + 输入顺序 + ID"排序。
        4. 逐根表沿延续链合并：
           - 延续页必须严格递增且相邻（child 页 = parent 页 + 1）。
           - 用 ``_merge_continuation`` 把延续表的数据并入根表；
             记录子表 ID -> 根表 ID 到 ``self.table_aliases``。

        返回：合并后的根表列表（含合并进来的跨页数据）。
        """
        source_tables = [table.model_copy(deep=True) for table in tables]
        self.table_aliases = {}
        by_id: dict[str, CanonicalTable] = {}
        input_order: dict[str, int] = {}
        for index, table in enumerate(source_tables):
            if table.table_id in by_id:
                raise ValueError(f"duplicate continuation table ID: {table.table_id!r}")
            by_id[table.table_id] = table
            input_order[table.table_id] = index

        # 解析延续关系：table_id -> 父表 id，以及 父表 id -> 子表列表
        parent_by_id: dict[str, str] = {}
        children: dict[str, list[str]] = {}
        for table in source_tables:
            parent_id = str(table.metadata.get("continuation_of") or "").strip()
            if not parent_id:
                continue
            if parent_id not in by_id:
                raise ValueError(
                    f"unknown continuation parent {parent_id!r} for {table.table_id!r}"
                )
            parent_by_id[table.table_id] = parent_id
            children.setdefault(parent_id, []).append(table.table_id)
        # 一个父表最多一个直接延续子表
        if any(len(child_ids) > 1 for child_ids in children.values()):
            raise ValueError("continuation graph contains a branch conflict")

        # 检测延续关系图中的环
        for table_id in by_id:
            seen: set[str] = set()
            cursor = table_id
            while cursor in parent_by_id:
                if cursor in seen:
                    raise ValueError("continuation graph contains a cycle")
                seen.add(cursor)
                cursor = parent_by_id[cursor]

        def source_page(table: CanonicalTable) -> int:
            """取表格唯一的源页码（不唯一则报错）。"""
            pages = {
                span.page_index
                for span in table.source_spans
                if span.page_index is not None
            }
            if len(pages) != 1:
                raise ValueError(
                    f"continuation table {table.table_id!r} requires one source page"
                )
            return next(iter(pages))

        root_ids = [table_id for table_id in by_id if table_id not in parent_by_id]
        root_ids.sort(
            key=lambda table_id: (
                min(
                    (
                        span.page_index
                        for span in by_id[table_id].source_spans
                        if span.page_index is not None
                    ),
                    default=10**9,
                ),
                input_order[table_id],
                table_id,
            )
        )
        merged: list[CanonicalTable] = []
        for root_id in root_ids:
            root = by_id[root_id]
            merged.append(root)
            cursor = root_id
            previous_page: int | None = None
            while cursor in children:
                child_id = children[cursor][0]
                parent_page = source_page(by_id[cursor])
                child_page = source_page(by_id[child_id])
                if child_page <= parent_page:
                    raise ValueError("continuation graph has invalid page order")
                if child_page != parent_page + 1:
                    raise ValueError("continuation pages must be adjacent")
                if previous_page is not None and parent_page != previous_page:
                    raise ValueError("continuation graph has invalid page order")
                self._merge_continuation(root, by_id[child_id])
                self.table_aliases[child_id] = root_id
                previous_page = child_page
                cursor = child_id
        return merged

    def figure_chunk(
        self,
        figure: CanonicalFigure,
        nearby_blocks: Iterable[CanonicalBlock],
    ) -> StructuredEvidenceChunk:
        """把图表打包为单个"父证据分块"。

        源保真原则：
        - 文本只由源内容构成：caption、Markdown 图片引用、description、
          邻近叙事文本（``nearby_block_ids`` 指向的正文/附录块）。
        - 模型生成的摘要（``generated_summary``）只并入 ``embedding_text``，
          不进入 ``text``。
        - 元信息记录 figure_id、资产路径、分析状态、警告与溯源
          （``provenance`` 区分"源"与"生成"）。
        """
        nearby = self._nearby_source_text(figure.nearby_block_ids, nearby_blocks)
        source_parts: list[str] = []
        if figure.caption:
            source_parts.append(figure.caption)
        if figure.asset_path:
            source_parts.append(f"![{figure.figure_id}]({figure.asset_path})")
        if figure.description:
            source_parts.append(figure.description)
        source_parts.extend(nearby)
        text = "\n\n".join(self._unique_nonempty(source_parts))
        derived = [figure.generated_summary] if figure.generated_summary else []
        embedding_text = "\n\n".join([text, *derived]).strip()
        warnings = list(figure.warnings)
        if figure.analysis_status == "failed" and not warnings:
            warnings.append("Optional figure analysis failed; source evidence remains accepted.")
        metadata = {
            "figure_id": figure.figure_id,
            "asset_path": figure.asset_path,
            "nearby_block_ids": figure.nearby_block_ids,
            "analysis_status": figure.analysis_status,
            "accepted": True,
            "warnings": warnings,
            "provenance": {
                "source": {"generated": False, "source_spans": self._span_json(figure.source_spans)},
                "generated_summary": {
                    "generated": True,
                    "model": figure.analysis_model,
                    "present": figure.generated_summary is not None,
                },
            },
        }
        return StructuredEvidenceChunk(
            chunk_id=self._stable_id("figure", figure.figure_id, text, embedding_text),
            chunk_role="parent",
            block_type="figure",
            text=text,
            embedding_text=embedding_text,
            token_count=self.estimate_tokens(embedding_text),
            source_spans=figure.source_spans,
            metadata=metadata,
        )

    def figure_chunks(
        self,
        figure: CanonicalFigure,
        nearby_blocks: Iterable[CanonicalBlock],
        *,
        max_tokens: int,
    ) -> tuple[StructuredEvidenceChunk, list[StructuredEvidenceChunk]]:
        """构建图表的"父 + 子"分块。

        以 ``figure.description``（或父文本）为切分源，用无损窗口切分
        成不超过 max_tokens 的子分块；仅当父文本本身放得下时才把邻近
        叙事（nearby）并入子分块渲染（``include_nearby``）。

        返回：``(parent, children)``，子分块元信息含
        ``structured_part_index/count`` 与源片段范围。
        """
        nearby_list = list(nearby_blocks)
        parent = self.figure_chunk(figure, nearby_list)
        nearby = self._nearby_source_text(figure.nearby_block_ids, nearby_list)
        payload = figure.description or parent.text
        # 父文本放得下才把邻近叙事并入子分块，避免子分块被叙事挤爆
        include_nearby = self.estimate_tokens(parent.text) <= max_tokens

        def render(fragment: str) -> str:
            if not figure.description:
                return fragment
            parts: list[str] = []
            if figure.caption:
                parts.append(figure.caption)
            if figure.asset_path:
                parts.append(f"![{figure.figure_id}]({figure.asset_path})")
            parts.append(fragment)
            if include_nearby:
                parts.extend(nearby)
            return "\n\n".join(self._unique_nonempty(parts))

        windows = self._lossless_wrapped_windows(
            payload,
            max_tokens=max_tokens,
            render=render,
        )
        children: list[StructuredEvidenceChunk] = []
        for part_index, (start, end, fragment, text) in enumerate(windows):
            metadata = {
                **parent.metadata,
                "structured_part_index": part_index,
                "structured_part_count": len(windows),
                "source_fragment": fragment,
                "source_fragment_start": start,
                "source_fragment_end": end,
            }
            children.append(
                StructuredEvidenceChunk(
                    chunk_id=self._stable_id(
                        "figure-child",
                        parent.chunk_id,
                        str(part_index),
                        text,
                    ),
                    parent_chunk_id=parent.chunk_id,
                    chunk_role="child",
                    block_type="figure",
                    text=text,
                    embedding_text=text,
                    token_count=self.estimate_tokens(text),
                    source_spans=parent.source_spans,
                    metadata=metadata,
                )
            )
        return parent, children

    def formula_chunk(
        self,
        formula: CanonicalFormula,
        nearby_blocks: Iterable[CanonicalBlock],
    ) -> StructuredEvidenceChunk:
        """把公式打包为单个"父证据分块"。

        源保真原则：
        - 文本由 caption、LaTeX 公式块（``$$...$$``）、description 与
          邻近叙事构成。
        - 模型生成的解释（``generated_explanation``）只并入
          ``embedding_text``。
        - 元信息记录 formula_id、邻近块、分析状态、警告与溯源。
        """
        nearby = self._nearby_source_text(formula.nearby_block_ids, nearby_blocks)
        source_parts: list[str] = []
        if formula.caption:
            source_parts.append(formula.caption)
        source_parts.append(f"$$\n{formula.latex}\n$$")
        if formula.description:
            source_parts.append(formula.description)
        source_parts.extend(nearby)
        text = "\n\n".join(self._unique_nonempty(source_parts))
        derived = [formula.generated_explanation] if formula.generated_explanation else []
        embedding_text = "\n\n".join([text, *derived]).strip()
        warnings = list(formula.warnings)
        if formula.analysis_status == "failed" and not warnings:
            warnings.append("Optional formula analysis failed; source evidence remains accepted.")
        metadata = {
            "formula_id": formula.formula_id,
            "nearby_block_ids": formula.nearby_block_ids,
            "analysis_status": formula.analysis_status,
            "accepted": True,
            "warnings": warnings,
            "provenance": {
                "source": {"generated": False, "source_spans": self._span_json(formula.source_spans)},
                "generated_explanation": {
                    "generated": True,
                    "model": formula.analysis_model,
                    "present": formula.generated_explanation is not None,
                },
            },
        }
        return StructuredEvidenceChunk(
            chunk_id=self._stable_id("formula", formula.formula_id, text, embedding_text),
            chunk_role="parent",
            block_type="formula",
            text=text,
            embedding_text=embedding_text,
            token_count=self.estimate_tokens(embedding_text),
            source_spans=formula.source_spans,
            metadata=metadata,
        )

    def formula_chunks(
        self,
        formula: CanonicalFormula,
        nearby_blocks: Iterable[CanonicalBlock],
        *,
        max_tokens: int,
    ) -> tuple[StructuredEvidenceChunk, list[StructuredEvidenceChunk]]:
        """构建公式的"父 + 子"分块。

        子分块切分：
        - 父文本放得下 → 单窗口（含全部 latex）。
        - 否则把 latex / description / caption 分别按无损窗口切分，
          latex 优先在"反斜杠、换行、分号"边界处断行
          （``preferred_boundaries``），避免切开 LaTeX 命令；
          caption 与 description 依据各自能否放下决定是否并入。
        - 邻近叙事（nearby）在需要窗口切分时仅保留在完整父分块中
          （打上 ``nearby_context_parent_only`` 标记），不塞进子分块。

        返回：``(parent, children)``，子分块元信息含
        ``source_fragment_kind``（latex/description/caption）。
        """
        nearby_list = list(nearby_blocks)
        parent = self.formula_chunk(formula, nearby_list)
        nearby = self._nearby_source_text(formula.nearby_block_ids, nearby_list)
        if self.estimate_tokens(parent.text) <= max_tokens:
            windows = [("latex", 0, len(formula.latex), formula.latex, parent.text)]
        else:
            caption = formula.caption or ""
            caption_fits_latex = bool(caption) and self.estimate_tokens(
                f"{caption}\n\n$$\n\n$$"
            ) < max_tokens
            caption_fits_description = bool(caption) and self.estimate_tokens(
                caption
            ) < max_tokens
            windows: list[tuple[str, int, int, str, str]] = []

            def latex_text(fragment: str) -> str:
                parts = [caption] if caption_fits_latex else []
                parts.append(f"$$\n{fragment}\n$$")
                return "\n\n".join(self._unique_nonempty(parts))

            windows.extend(
                ("latex", start, end, fragment, text)
                for start, end, fragment, text in self._lossless_wrapped_windows(
                    formula.latex,
                    max_tokens=max_tokens,
                    render=latex_text,
                    preferred_boundaries=re.compile(r"(?:\\\\(?:\s|$)|\r?\n|;\s*)"),
                )
            )
            if formula.description:
                def description_text(fragment: str) -> str:
                    parts = [caption] if caption_fits_description else []
                    parts.append(fragment)
                    return "\n\n".join(self._unique_nonempty(parts))

                windows.extend(
                    ("description", start, end, fragment, text)
                    for start, end, fragment, text in self._lossless_wrapped_windows(
                        formula.description,
                        max_tokens=max_tokens,
                        render=description_text,
                    )
                )
            if caption and not (caption_fits_latex or caption_fits_description):
                windows.extend(
                    ("caption", start, end, fragment, text)
                    for start, end, fragment, text in self._lossless_wrapped_windows(
                        caption,
                        max_tokens=max_tokens,
                        render=lambda fragment: fragment,
                    )
                )
            if nearby:
                # Nearby narrative is independently retrievable; retain it only in
                # the complete Parent when the structured source itself needs windows.
                parent.metadata["nearby_context_parent_only"] = True

        children: list[StructuredEvidenceChunk] = []
        for part_index, (kind, start, end, fragment, text) in enumerate(windows):
            metadata = {
                **parent.metadata,
                "structured_part_index": part_index,
                "structured_part_count": len(windows),
                "source_fragment_kind": kind,
                "source_fragment": fragment,
                "source_fragment_start": start,
                "source_fragment_end": end,
            }
            children.append(
                StructuredEvidenceChunk(
                    chunk_id=self._stable_id(
                        "formula-child",
                        parent.chunk_id,
                        str(part_index),
                        text,
                    ),
                    parent_chunk_id=parent.chunk_id,
                    chunk_role="child",
                    block_type="formula",
                    text=text,
                    embedding_text=text,
                    token_count=self.estimate_tokens(text),
                    source_spans=parent.source_spans,
                    metadata=metadata,
                )
            )
        return parent, children

    def _split_group_by_token_limit(
        self,
        table: CanonicalTable,
        indices: list[int],
        max_tokens: int,
    ) -> list[list[int]]:
        """把一组行下标按 token 预算拆分成若干连续分组。

        每组的成本 = 基准成本（空行网格，即表头 + 表头分隔符）+ 各行成本
        之和；逐行累加，超出 ``max_tokens`` 时闭合当前组并新开一组。
        单行本身即超限时也立即闭合该组（交由上层做更细的过宽行切分）。

        返回：行下标分组列表，每个子分组自身可能仍超限（单行场景）。
        """
        # 基准成本：无数据行的网格（caption 若存在已含在 _table_text）
        base_cost = self.estimate_tokens(self._table_text(table, []))
        # 每行的增量成本（仅多一行网格的最后一行行文本）
        row_costs = {
            index: self.estimate_tokens(
                "\n" + self._markdown_grid(table.headers, [table.rows[index]])[-1]
            )
            for index in indices
        }
        groups: list[list[int]] = []
        current: list[int] = []
        current_cost = base_cost
        for index in indices:
            row_cost = row_costs[index]
            if current and current_cost + row_cost > max_tokens:
                # 加这行会超限：闭合当前组，新开一组
                groups.append(current)
                current = [index]
                current_cost = base_cost + row_cost
            else:
                current.append(index)
                current_cost += row_cost
            # 单行组自己就超限：立即闭合（留给过宽行处理）
            if len(current) == 1 and current_cost > max_tokens:
                groups.append(current)
                current = []
                current_cost = base_cost
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _semantic_row_groups(table: CanonicalTable) -> list[list[int]]:
        """对表格行做"语义分组"：返回若干组连续行下标。

        分组依据（优先级）：
        1. 元数据中预配置的 ``semantic_row_groups``（来自上游解析/切分）：
           过滤越界与重复行下标，并把未覆盖的行按单行补齐。
        2. 否则做启发式分组：按每行第 0 列（首列）值连续相等分块——
           首列相同的连续行视为一组，首列变化即新开一组。

        用途：决定子分块的行范围边界，让相关行尽量落在同一分块内。
        """
        configured = table.metadata.get("semantic_row_groups")
        if isinstance(configured, list):
            groups: list[list[int]] = []
            seen: set[int] = set()
            # 校验并过滤配置的分组
            for raw_group in configured:
                if not isinstance(raw_group, list):
                    continue
                group = [
                    index
                    for index in raw_group
                    if isinstance(index, int)
                    and 0 <= index < len(table.rows)
                    and index not in seen
                ]
                if group:
                    groups.append(group)
                    seen.update(group)
            # 未覆盖的行补为单行组
            groups.extend([[index] for index in range(len(table.rows)) if index not in seen])
            if groups:
                return groups

        # 启发式：按首列值分组的兜底逻辑
        groups = []
        current: list[int] = []
        current_key: str | None = None
        for index, row in enumerate(table.rows):
            key = row[0].strip().casefold() if row else ""
            if current and key != current_key:
                groups.append(current)
                current = []
            current.append(index)
            current_key = key
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _table_text(table: CanonicalTable, row_indices: list[int]) -> str:
        """生成指定行集合的整块表格文本（caption + 网格 + 脚注）。"""
        lines: list[str] = []
        if table.caption:
            lines.append(table.caption)
        lines.extend(StructuredEvidenceBuilder._markdown_grid(table.headers, [table.rows[i] for i in row_indices]))
        lines.extend(f"Footnote: {footnote}" for footnote in table.footnotes)
        return "\n".join(lines)

    @staticmethod
    def _markdown_grid(headers: list[str], rows: list[list[str]]) -> list[str]:
        """把表头与数据行渲染为 Markdown 管道网格（转义表内 ``|``、反斜杠与换行）。

        返回的行列表以"表头行 + 分隔行 + 数据行"顺序排列。
        """
        def escape(value: str) -> str:
            # 转义反斜杠、管道符；换行折成 <br> 以保持单行表格
            return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

        lines = [
            "| " + " | ".join(escape(value) for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
        return lines

    def _table_metadata(
        self,
        table: CanonicalTable,
        row_indices: list[int],
        *,
        overflow: bool,
    ) -> dict[str, Any]:
        """组装分块的表格元信息字典（含身份指纹、网格、单元格与保真信息）。

        - ``selected_grid_rows``：把行下标转成 1-based 网格行号。
        - ``cells``：只保留表头行（row_index==0）或与所选行相交
          （含 rowspan 覆盖）的单元格。
        - 输出字段涵盖：table_id / 身份指纹 / 状态 / caption / 表头 /
          所选行 / 行下标 / 单元格 / 脚注 / 源 markdown 与 html /
          normalized_markdown / overflow / token 计数模式等。
        """
        selected_grid_rows = {index + 1 for index in row_indices}
        # 仅选取表头单元格及与所选行（含 rowspan 跨度）相交的单元格
        cells = [
            cell.model_dump(mode="json")
            for cell in table.cells
            if cell.row_index == 0
            or any(
                cell.row_index <= row_index < cell.row_index + cell.rowspan
                for row_index in selected_grid_rows
            )
        ]
        return {
            "table_id": table.table_id,
            "table_identity": table_identity_fingerprint(table),
            "status": table.status,
            "caption": table.caption,
            "headers": table.headers,
            "rows": [table.rows[index] for index in row_indices],
            "row_indices": row_indices,
            "cells": cells,
            "footnotes": table.footnotes,
            "source_markdown": table.source_markdown,
            "source_html": table.source_html,
            "source_markdowns": table.metadata.get("source_markdowns", []),
            "source_htmls": table.metadata.get("source_htmls", []),
            "source_cell_segments": table.metadata.get("source_cell_segments", []),
            "normalized_markdown": StructuredEvidenceBuilder._table_text_without_caption(
                table, row_indices
            ),
            "overflow": overflow,
            "token_count_mode": self.token_count_mode,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_fallback_reason": self._tokenizer_fallback_reason,
        }

    @staticmethod
    def _table_text_without_caption(table: CanonicalTable, row_indices: list[int]) -> str:
        """生成不含 caption 的纯网格文本（normalized_markdown 用）。"""
        return "\n".join(
            StructuredEvidenceBuilder._markdown_grid(
                table.headers,
                [table.rows[index] for index in row_indices],
            )
        )

    @staticmethod
    def _row_source_spans(table: CanonicalTable, row_indices: list[int]) -> list[SourceSpan]:
        """收集与所选行（含 rowspan）相交的单元格的全部源跨度。

        额外处理：
        - 若单元格元信息记录了 ``original_page_index``（跨页合并前所在页）
          且其源跨度中缺失该页，则补一个仅含页码的源跨度占位。
        - 没有任何命中时退回整表的源跨度。
        - 结果经 ``_deduplicate_spans`` 去重。
        """
        selected = {index + 1 for index in row_indices}
        spans: list[SourceSpan] = []
        for cell in table.cells:
            if any(
                cell.row_index <= row_index < cell.row_index + cell.rowspan
                for row_index in selected
            ):
                spans.extend(cell.source_spans)
                # 补回跨页合并前的原始页码
                original_page = cell.metadata.get("original_page_index")
                if isinstance(original_page, int) and not any(
                    span.page_index == original_page for span in cell.source_spans
                ):
                    spans.append(SourceSpan(page_index=original_page))
        if not spans:
            spans = list(table.source_spans)
        return StructuredEvidenceBuilder._deduplicate_spans(spans)

    @staticmethod
    def _deduplicate_spans(spans: Iterable[SourceSpan]) -> list[SourceSpan]:
        """按序列化内容去重源跨度列表（保持首次出现顺序）。

        用排序键的紧凑 JSON 作为去重键，保证跨解析器/版本稳定。
        """
        result: list[SourceSpan] = []
        seen: set[str] = set()
        for span in spans:
            key = json.dumps(span.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            if key not in seen:
                result.append(span)
                seen.add(key)
        return result

    @staticmethod
    def _merge_continuation(root: CanonicalTable, continuation: CanonicalTable) -> None:
        """把延续表（continuation）的数据并入根表（原地修改 ``root``）。

        步骤：
        1. 表头必须完全一致，否则报错。
        2. 累积源产物分段：``source_markdowns`` / ``source_htmls`` /
           ``source_cell_segments``（含根表自身的首段）。
        3. 处理延续表的"重复表头行"：若其首行 == 根表表头，则跳过该行
           （``skip_data_rows = 1``），并把表头行/重复表头行登记进
           ``merged_repeated_header_cells``。
        4. 重排延续表单元格行号：新行号 = 根表数据行数基准 + 原行号偏移，
           并记录原始表 ID / 原始页码 / 原始行号到单元格元信息。
        5. 补齐根表已有单元格的原始来源元信息（original_table_id /
           original_page_index / original_row_index）。
        6. 拼接数据行与源跨度；合并脚注；状态置 ``cross_page_merged``；
           更新合并元信息（merged_table_ids 等）并重算 normalized_markdown。
        """
        if continuation.headers != root.headers:
            raise ValueError(
                f"continuation {continuation.table_id!r} header does not match {root.table_id!r}"
            )
        # 累积源 markdown 分段（根表首段缺失时用根表自身 source_markdown）
        source_markdowns = list(root.metadata.get("source_markdowns") or [])
        if not source_markdowns and root.source_markdown:
            source_markdowns.append(root.source_markdown)
        if continuation.source_markdown:
            source_markdowns.append(continuation.source_markdown)
        # 累积源 html 分段
        source_htmls = list(root.metadata.get("source_htmls") or [])
        if not source_htmls and root.source_html:
            source_htmls.append(root.source_html)
        if continuation.source_html:
            source_htmls.append(continuation.source_html)
        # 累积单元格分段（根表首段缺失时用根表全部单元格）
        source_cell_segments = list(root.metadata.get("source_cell_segments") or [])
        if not source_cell_segments:
            source_cell_segments.append(
                [cell.model_dump(mode="json") for cell in root.cells]
            )
        source_cell_segments.append(
            [cell.model_dump(mode="json") for cell in continuation.cells]
        )

        # 延续表首行若是重复表头行，拼接数据行时跳过
        repeated_data_header = bool(
            continuation.rows and continuation.rows[0] == root.headers
        )
        skip_data_rows = 1 if repeated_data_header else 0
        base_data_count = len(root.rows)
        repeated_cells: list[dict[str, Any]] = list(
            root.metadata.get("merged_repeated_header_cells") or []
        )
        # 延续表所在页（用于记录原始页码）
        continuation_page = next(
            (span.page_index for span in continuation.source_spans if span.page_index is not None),
            None,
        )
        # 合并延续表单元格并重排行号
        for cell in continuation.cells:
            original = cell.model_dump(mode="json")
            if cell.row_index == 0 or (repeated_data_header and cell.row_index == 1):
                # 表头行 / 重复表头行不进入数据区，仅登记
                repeated_cells.append(original)
                continue
            copied = cell.model_copy(deep=True)
            original_row_index = copied.row_index
            # 新行号偏移：跳过表头行(1)与可能的重复表头行(skip)
            copied.row_index = 1 + base_data_count + (copied.row_index - 1 - skip_data_rows)
            copied.metadata = {
                **copied.metadata,
                "original_table_id": continuation.table_id,
                "original_page_index": continuation_page,
                "original_row_index": original_row_index,
            }
            root.cells.append(copied)

        # 根表原有单元格补齐来源元信息
        for cell in root.cells:
            cell.metadata.setdefault("original_table_id", root.table_id)
            if "original_page_index" not in cell.metadata:
                cell.metadata["original_page_index"] = next(
                    (span.page_index for span in root.source_spans if span.page_index is not None),
                    None,
                )
            cell.metadata.setdefault("original_row_index", cell.row_index)
        root.rows.extend(continuation.rows[skip_data_rows:])
        root.source_spans = StructuredEvidenceBuilder._deduplicate_spans(
            [*root.source_spans, *continuation.source_spans]
        )
        root.footnotes = StructuredEvidenceBuilder._unique_nonempty(
            [*root.footnotes, *continuation.footnotes]
        )
        root.status = "cross_page_merged"
        root.metadata.update(
            {
                "continuation_recovered": True,
                "merged_table_ids": [
                    *list(root.metadata.get("merged_table_ids") or [root.table_id]),
                    continuation.table_id,
                ],
                "source_markdowns": source_markdowns,
                "source_htmls": source_htmls,
                "source_cell_segments": source_cell_segments,
                "merged_repeated_header_cells": repeated_cells,
            }
        )
        normalized = "\n".join(
            StructuredEvidenceBuilder._markdown_grid(root.headers, root.rows)
        )
        root.normalized_markdown = normalized

    @staticmethod
    def _nearby_source_text(
        nearby_ids: list[str],
        blocks: Iterable[CanonicalBlock],
    ) -> list[str]:
        """按 nearby_block_ids 提取"邻近叙事"的源文本。

        只收集：
        - block_id 命中 wanted 集合。
        - block_type 属于 narrative / appendix（正文/附录）。
        - 非模型生成（``block_is_generated`` 为假）。
        - 文本非空。

        返回：按输入块顺序排列的源文本列表（不含生成内容）。
        """
        wanted = set(nearby_ids)
        return [
            block.text
            for block in blocks
            if block.block_id in wanted
            and block.block_type in {"narrative", "appendix"}
            and not block_is_generated(block)
            and block.text.strip()
        ]

    @staticmethod
    def _unique_nonempty(values: Iterable[str]) -> list[str]:
        """按序去重并去掉空值（strip 后为空视为空）。"""
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
        return result

    @staticmethod
    def _span_json(spans: list[SourceSpan]) -> list[dict[str, Any]]:
        """把源跨度列表转为 JSON dict 列表（供元信息存储）。"""
        return [span.model_dump(mode="json") for span in spans]

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        """生成稳定 ID：``prefix-<sha256(parts 以 \x1f 连接)前24位>``。

        内容（parts）不变则 ID 不变，支持幂等重建与去重。
        """
        payload = "\x1f".join(parts).encode("utf-8")
        return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"
