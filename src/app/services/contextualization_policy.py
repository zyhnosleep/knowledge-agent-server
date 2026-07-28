from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CONTEXTUALIZED_BLOCK_TYPES = frozenset({"table", "figure", "formula"})
PLAIN_EMBEDDING_BLOCK_TYPES = frozenset({"narrative", "caption", "appendix"})
RETRIEVABLE_BLOCK_TYPES = CONTEXTUALIZED_BLOCK_TYPES | PLAIN_EMBEDDING_BLOCK_TYPES

CONTEXTUALIZATION_FIELDS = (
    "contextual_prefix",
    "contextualization_model",
    "contextualization_version",
    "contextualization_prompt_version",
    "contextualized_at",
)


def _field(chunk: object, name: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(name)
    return getattr(chunk, name, None)


def requires_contextualization(block_type: str) -> bool:
    if block_type not in RETRIEVABLE_BLOCK_TYPES:
        raise ValueError(f"unsupported retrievable block type: {block_type}")
    return block_type in CONTEXTUALIZED_BLOCK_TYPES


def valid_contextualized_embedding(chunk: object) -> bool:
    block_type = str(_field(chunk, "block_type"))
    if not requires_contextualization(block_type):
        return False
    prefix = _field(chunk, "contextual_prefix")
    text = _field(chunk, "text")
    return bool(
        prefix
        and _field(chunk, "contextualization_model")
        and _field(chunk, "contextualization_version")
        and _field(chunk, "contextualization_prompt_version")
        and _field(chunk, "contextualized_at") is not None
        and _field(chunk, "embedding_text") == f"{prefix}\n\n{text}"
    )


def valid_plain_embedding(chunk: object) -> bool:
    block_type = str(_field(chunk, "block_type"))
    if requires_contextualization(block_type):
        return False
    return bool(
        all(_field(chunk, name) is None for name in CONTEXTUALIZATION_FIELDS)
        and _field(chunk, "embedding_text") == _field(chunk, "text")
    )
