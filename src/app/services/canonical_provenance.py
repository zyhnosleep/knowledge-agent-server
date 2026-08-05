"""来源与生成内容溯源（Provenance）工具。

在 Canonical 文档中，某些 block 可能不是来自原始文档，而是由 AI / 后处理
生成的（例如表格修复、摘要补全、公式解释）。本模块提供识别与清理这些
生成内容的能力，确保落盘和检索可以只使用 "source-only" 的真实来源内容。

主要入口：
- metadata_is_generated(metadata): 判断元数据是否标记为生成内容。
- block_is_generated(block): 判断一个 block 是否来自生成内容。
- source_only_document(document): 返回移除所有生成 block 及其关联引用后的文档副本。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalDocument,
    SectionNode,
)


def metadata_is_generated(metadata: Mapping[str, Any]) -> bool:
    """识别两种支持的生成来源编码格式。

    支持的形式：
    1. metadata["generated"] == True
    2. metadata["provenance"]["generated"] == True
    """
    provenance = metadata.get("provenance")
    return bool(
        metadata.get("generated") is True
        or (
            isinstance(provenance, Mapping)
            and provenance.get("generated") is True
        )
    )


def block_is_generated(block: CanonicalBlock) -> bool:
    """判断一个 CanonicalBlock 是否被标记为生成内容。"""
    return metadata_is_generated(block.metadata)


def source_only_document(document: CanonicalDocument) -> CanonicalDocument:
    """返回移除所有生成 block 及其关联后的 CanonicalDocument 副本。

    清理范围：
    1. 从 blocks 列表中移除生成 block。
    2. 从 figure.nearby_block_ids 和 formula.nearby_block_ids 中移除生成 block。
    3. 从 quality.issues 的 block_ids 中移除生成 block。
    4. 从 outline 中移除生成 block 对应的节点，并把其子节点提升到父级。
    """
    result = document.model_copy(deep=True)
    generated_ids = {
        block.block_id for block in result.blocks if block_is_generated(block)
    }
    if not generated_ids:
        return result
    result.blocks = [
        block for block in result.blocks if block.block_id not in generated_ids
    ]
    for figure in result.figures:
        figure.nearby_block_ids = [
            block_id
            for block_id in figure.nearby_block_ids
            if block_id not in generated_ids
        ]
    for formula in result.formulas:
        formula.nearby_block_ids = [
            block_id
            for block_id in formula.nearby_block_ids
            if block_id not in generated_ids
        ]
    for issue in result.quality.issues:
        issue.block_ids = [
            block_id for block_id in issue.block_ids if block_id not in generated_ids
        ]

    def clean_outline(nodes: list[SectionNode]) -> list[SectionNode]:
        """递归清理大纲：生成 block 对应的节点被移除，子节点上移。"""
        cleaned: list[SectionNode] = []
        for node in nodes:
            children = clean_outline(node.children)
            if node.block_id in generated_ids:
                cleaned.extend(children)
                continue
            node.children = children
            cleaned.append(node)
        return cleaned

    result.outline = clean_outline(result.outline)
    return result
