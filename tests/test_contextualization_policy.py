from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.services.contextualization_policy import (
    CONTEXTUALIZED_BLOCK_TYPES,
    PLAIN_EMBEDDING_BLOCK_TYPES,
    requires_contextualization,
    valid_contextualized_embedding,
    valid_plain_embedding,
)


def _child(
    *,
    block_type: str,
    text: str = "Raw child",
    embedding_text: str | None = None,
) -> dict[str, object]:
    return {
        "block_type": block_type,
        "text": text,
        "embedding_text": text if embedding_text is None else embedding_text,
    }


def _contextualized_child(*, block_type: str = "figure") -> dict[str, object]:
    child = _child(block_type=block_type, text="| A | B |")
    child.update(
        contextual_prefix="This table reports the main comparison.",
        embedding_text="This table reports the main comparison.\n\n| A | B |",
        contextualization_model="qwen3.5:9b",
        contextualization_version="contextualization-v1",
        contextualization_prompt_version="context-v1",
        contextualized_at=datetime(2026, 7, 28, 12, 0),
    )
    return child


def test_no_retrievable_child_requires_llm_context() -> None:
    assert CONTEXTUALIZED_BLOCK_TYPES == frozenset()
    assert PLAIN_EMBEDDING_BLOCK_TYPES == frozenset(
        {"narrative", "caption", "appendix", "table", "figure", "formula"}
    )
    for block_type in CONTEXTUALIZED_BLOCK_TYPES:
        assert requires_contextualization(block_type) is True
    for block_type in PLAIN_EMBEDDING_BLOCK_TYPES:
        assert requires_contextualization(block_type) is False


def test_unknown_retrievable_block_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported retrievable block type"):
        requires_contextualization("reference")


def test_plain_child_requires_exact_raw_embedding_text_and_no_context_fields() -> None:
    child = _child(block_type="table")

    assert valid_plain_embedding(child) is True

    modified = deepcopy(child)
    modified["embedding_text"] = "Section context\n\nRaw child"
    assert valid_plain_embedding(modified) is False

    contextualized = deepcopy(child)
    contextualized["contextual_prefix"] = "Unexpected context."
    assert valid_plain_embedding(contextualized) is False


def test_no_child_type_accepts_contextualized_embedding() -> None:
    child = _contextualized_child()

    assert valid_contextualized_embedding(child) is False


def test_policy_validation_accepts_attribute_objects() -> None:
    child = SimpleNamespace(**_child(block_type="figure"))

    assert valid_plain_embedding(child) is True
