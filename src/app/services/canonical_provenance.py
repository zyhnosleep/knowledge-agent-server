from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalDocument,
    SectionNode,
)


def metadata_is_generated(metadata: Mapping[str, Any]) -> bool:
    """Recognize the two supported generated-provenance encodings."""

    provenance = metadata.get("provenance")
    return bool(
        metadata.get("generated") is True
        or (
            isinstance(provenance, Mapping)
            and provenance.get("generated") is True
        )
    )


def block_is_generated(block: CanonicalBlock) -> bool:
    return metadata_is_generated(block.metadata)


def source_only_document(document: CanonicalDocument) -> CanonicalDocument:
    """Return a canonical copy with generated blocks and their links removed."""

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
