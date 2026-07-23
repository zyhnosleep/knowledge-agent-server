from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from app.services.contextualization import (
    ContextualizationFailed,
    ContextualizationService,
    ContextualPrefixBatch,
    ContextualPrefixItem,
    DocumentContext,
)
from app.services.semantic_chunking import ChunkDraft
from app.services.canonical_models import SourceSpan


class FakeContextualizationClient:
    def __init__(
        self,
        responder: Callable[[dict, str], ContextualPrefixBatch],
        *,
        prompt_version: str = "context-test-v1",
        model: str = "context-test-model",
    ) -> None:
        self.responder = responder
        self.prompt_version = prompt_version
        self.model = model
        self.calls: list[dict] = []

    def generate_contextualization(
        self,
        schema,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> ContextualPrefixBatch:
        assert schema is ContextualPrefixBatch
        decoder = json.JSONDecoder()
        payload = None
        for index, character in enumerate(user_prompt):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(user_prompt[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "document" in candidate:
                payload = candidate
                break
        assert payload is not None
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "payload": payload,
            }
        )
        return self.responder(payload, user_prompt)


def _document() -> DocumentContext:
    return DocumentContext(
        title="GraphFormer: 图结构预测",
        source_abstract="原始 Abstract 介绍 GraphFormer 在 Cora 数据集上的方法。",
        section_outline=["1 引言", "2 GraphFormer 方法", "3 实验"],
    )


def _chunk(
    local_id: str,
    *,
    role: str,
    block_type: str = "narrative",
    text: str | None = None,
    parent_local_id: str | None = None,
) -> ChunkDraft:
    original = text or f"{local_id} 的原始内容用于检索。"
    return ChunkDraft(
        local_id=local_id,
        parse_version="canonical-v1",
        chunk_role=role,
        block_type=block_type,
        text=original,
        embedding_text=original,
        token_count=10,
        parent_local_id=parent_local_id,
        source_block_ids=[f"source-{local_id}"],
        source_spans=[SourceSpan(page_index=1, source_block_id=f"source-{local_id}")],
        section_path=["2 GraphFormer 方法", f"2.1 {block_type}"],
        ordinal=0,
        splitter_name="test",
        splitter_version="test-v1",
        splitting_model="embed-test",
        metadata={"preserved": True},
    )


def _parent(local_id: str = "parent-1") -> ChunkDraft:
    return _chunk(
        local_id,
        role="parent",
        text="该 Parent 原文讨论 GraphFormer 的训练目标和 Cora 实验。",
    )


def _children(count: int, *, parent_id: str = "parent-1") -> list[ChunkDraft]:
    return [
        _chunk(f"child-{index}", role="child", parent_local_id=parent_id)
        for index in range(count)
    ]


def _valid_response(payload: dict, _prompt: str) -> ContextualPrefixBatch:
    return ContextualPrefixBatch(
        items=[
            ContextualPrefixItem(
                child_id=item["child_id"],
                prefix=f"该部分说明 GraphFormer 方法中与 {item['block_type']} 内容的关系。",
            )
            for item in payload["children"]
        ]
    )


def test_prompt_contains_complete_source_context_and_contextualizes_every_structure() -> None:
    block_types = ["narrative", "table", "figure", "formula", "caption", "appendix"]
    parent = _parent()
    children = [
        _chunk(
            f"child-{block_type}",
            role="child",
            block_type=block_type,
            parent_local_id=parent.local_id,
            text=f"{block_type} 检索单元的原始文本。",
        )
        for block_type in block_types
    ]
    originals = {item.local_id: item.model_copy(deep=True) for item in children}
    client = FakeContextualizationClient(_valid_response)
    checkpoints: list[list[ChunkDraft]] = []

    result = ContextualizationService(client=client).contextualize(
        document=_document(),
        children=children,
        parents={parent.local_id: parent},
        checkpoint=lambda batch: checkpoints.append(batch),
    )

    assert [item.block_type for item in result] == block_types
    payload = client.calls[0]["payload"]
    assert payload["document"] == {
        "title": "GraphFormer: 图结构预测",
        "source_abstract": "原始 Abstract 介绍 GraphFormer 在 Cora 数据集上的方法。",
        "section_outline": ["1 引言", "2 GraphFormer 方法", "3 实验"],
    }
    assert payload["children"][0]["section_path"] == children[0].section_path
    assert payload["children"][0]["parent_text"] == parent.text
    assert payload["children"][0]["child_id"] == children[0].local_id
    assert payload["children"][0]["original_text"] == children[0].text
    assert "summary" not in payload["document"]
    assert "context-test-v1" in client.calls[0]["system_prompt"]
    assert len(checkpoints) == 1
    for contextualized in result:
        original = originals[contextualized.local_id]
        prefix = contextualized.metadata["contextual_prefix"]
        assert contextualized.embedding_text == f"{prefix['text']}\n\n{original.text}"
        assert contextualized.text == original.text
        assert contextualized.source_spans == original.source_spans
        assert contextualized.source_block_ids == original.source_block_ids
        assert prefix["generated"] is True
        assert prefix["citation_eligible"] is False
        assert prefix["source_spans"] == []
        assert prefix["source_block_ids"] == []


def test_returns_first_class_contextualized_chunks_for_direct_persistence() -> None:
    parent = _parent()
    child = _children(1)[0]
    checkpointed = []
    fixed_time = datetime(2026, 7, 23, 8, 30, tzinfo=timezone.utc)
    client = FakeContextualizationClient(
        _valid_response,
        model="context-model-v2",
        prompt_version="context-prompt-v3",
    )

    result = ContextualizationService(
        client=client,
        contextualization_version="contextualizer-v4",
        clock=lambda: fixed_time,
    ).contextualize(
        document=_document(),
        children=[child],
        parents={parent.local_id: parent},
        checkpoint=lambda batch: checkpointed.extend(batch),
    )

    contextualized = result[0]
    assert contextualized.contextual_prefix.startswith("该部分说明 GraphFormer")
    assert contextualized.contextualization_model == "context-model-v2"
    assert contextualized.contextualization_version == "contextualizer-v4"
    assert contextualized.contextualization_prompt_version == "context-prompt-v3"
    assert contextualized.contextualized_at == fixed_time
    assert contextualized.contextualized_at.tzinfo is timezone.utc
    assert checkpointed == [contextualized]
    assert checkpointed[0] is contextualized


def test_batches_at_twelve_without_reprocessing_successful_children() -> None:
    parent = _parent()
    client = FakeContextualizationClient(_valid_response)

    result = ContextualizationService(client=client, batch_size=12).contextualize(
        document=_document(),
        children=_children(25),
        parents={parent.local_id: parent},
    )

    assert [len(call["payload"]["children"]) for call in client.calls] == [12, 12, 1]
    assert len(result) == 25


@pytest.mark.parametrize(
    ("invalid_kind", "expected_failed", "expected_checkpoint"),
    [
        ("missing", {"child-1"}, [["child-0"]]),
        ("duplicate", {"child-0", "child-1"}, []),
        ("extra", {"child-0", "child-1"}, []),
    ],
)
def test_rejects_non_exact_response_child_id_sets(
    invalid_kind: str,
    expected_failed: set[str],
    expected_checkpoint: list[list[str]],
) -> None:
    parent = _parent()
    children = _children(2)

    def invalid_response(payload: dict, _prompt: str) -> ContextualPrefixBatch:
        ids = [item["child_id"] for item in payload["children"]]
        if invalid_kind == "missing":
            ids = ids[:-1]
        elif invalid_kind == "duplicate":
            ids = [ids[0], ids[0]]
        else:
            ids.append("unexpected-child")
        return ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=item, prefix="该部分说明研究方法的关系。") for item in ids]
        )

    checkpoints: list[list[str]] = []
    with pytest.raises(ContextualizationFailed) as exc_info:
        ContextualizationService(
            client=FakeContextualizationClient(invalid_response),
            max_retries=0,
        ).contextualize(
            document=_document(),
            children=children,
            parents={parent.local_id: parent},
            checkpoint=lambda batch: checkpoints.append([item.local_id for item in batch]),
        )

    assert exc_info.value.failed_child_ids == expected_failed
    assert checkpoints == expected_checkpoint
    assert invalid_kind in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("prefix", "message"),
    [
        ("   ", "empty"),
        ("该部分" * 100, "length"),
        ("该部分介绍方法。该部分介绍实验。该部分总结结果。", "sentences"),
        ("This describes the relation.", "chinese"),
    ],
)
def test_rejects_empty_overlong_too_many_sentences_or_non_chinese_prefixes(
    prefix: str,
    message: str,
) -> None:
    parent = _parent()
    child = _children(1)
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=payload["children"][0]["child_id"], prefix=prefix)]
        )
    )

    with pytest.raises(ContextualizationFailed, match=message):
        ContextualizationService(client=client, max_retries=0, max_prefix_chars=120).contextualize(
            document=_document(),
            children=child,
            parents={parent.local_id: parent},
        )


@pytest.mark.parametrize(
    ("prefix", "message"),
    [
        ("该部分使用 TensorFlow 解释图结构预测。", "entity"),
        ("模型实现稳定收敛。", "copies"),
        ("该部分报告准确率达到 99.9%。", "numeric"),
    ],
)
def test_rejects_new_english_entities_child_copying_and_invented_numbers(
    prefix: str,
    message: str,
) -> None:
    parent = _parent()
    child = _chunk(
        "child-1",
        role="child",
        parent_local_id=parent.local_id,
        text="模型实现稳定收敛",
    )
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=payload["children"][0]["child_id"], prefix=prefix)]
        )
    )

    with pytest.raises(ContextualizationFailed, match=message):
        ContextualizationService(client=client, max_retries=0).contextualize(
            document=_document(),
            children=[child],
            parents={parent.local_id: parent},
        )


def test_rejects_grounded_numbers_when_they_are_not_needed_for_a_metric_or_structure() -> None:
    parent = _parent()
    child = _chunk(
        "child-1",
        role="child",
        parent_local_id=parent.local_id,
        text="消融实验共执行 10 个独立步骤。",
    )
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[
                ContextualPrefixItem(
                    child_id=payload["children"][0]["child_id"],
                    prefix="该部分说明消融实验的 10 个步骤与研究流程的关系。",
                )
            ]
        )
    )

    with pytest.raises(ContextualizationFailed, match="numeric"):
        ContextualizationService(client=client, max_retries=0).contextualize(
            document=_document(),
            children=[child],
            parents={parent.local_id: parent},
        )


def test_allows_grounded_metric_numbers_and_common_english_terms() -> None:
    parent = _parent()
    child = _chunk(
        "child-table",
        role="child",
        block_type="table",
        parent_local_id=parent.local_id,
        text="Accuracy 指标为 92%，其余列给出不同实验条件。",
    )
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[
                ContextualPrefixItem(
                    child_id=payload["children"][0]["child_id"],
                    prefix="该表说明 AI 方法的 Accuracy 指标为 92%，并对应实验结果。",
                )
            ]
        )
    )

    result = ContextualizationService(client=client, max_retries=0).contextualize(
        document=_document(),
        children=[child],
        parents={parent.local_id: parent},
    )

    assert result[0].embedding_text.startswith("该表说明 AI 方法")


@pytest.mark.parametrize(
    ("block_type", "parent_text", "child_text", "prefix"),
    [
        (
            "figure",
            "该 Parent 原文介绍研究时间线。",
            "研究工作在 2024 年进入验证阶段。",
            "该图说明 2024 年与研究时间线的关系。",
        ),
        (
            "formula",
            "该 Parent 原文介绍目标函数。",
            "常量 3 用作偏移项。",
            "该公式说明常量 3 与目标函数的关系。",
        ),
        (
            "narrative",
            "该 Parent 原文仅泛称模型性能表现。",
            "训练过程执行 10 个步骤。",
            "该部分说明 10 个步骤与实验流程的关系。",
        ),
        (
            "narrative",
            "该 Parent 原文讨论 mapping 操作。",
            "mapping 操作包含 10 个阶段。",
            "该部分说明 mapping 10 与研究流程的关系。",
        ),
    ],
)
def test_rejects_grounded_numbers_without_explicit_table_or_named_metric_necessity(
    block_type: str,
    parent_text: str,
    child_text: str,
    prefix: str,
) -> None:
    parent = _chunk("parent-1", role="parent", text=parent_text)
    child = _chunk(
        "child-1",
        role="child",
        block_type=block_type,
        parent_local_id=parent.local_id,
        text=child_text,
    )
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=payload["children"][0]["child_id"], prefix=prefix)]
        )
    )

    with pytest.raises(ContextualizationFailed, match="numeric"):
        ContextualizationService(client=client, max_retries=0).contextualize(
            document=_document(),
            children=[child],
            parents={parent.local_id: parent},
        )


def test_allows_grounded_table_identifier_when_both_source_and_prefix_identify_it() -> None:
    parent = _parent()
    child = _chunk(
        "child-table",
        role="child",
        block_type="table",
        parent_local_id=parent.local_id,
        text="表2汇总不同方法的实验结果。",
    )
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[
                ContextualPrefixItem(
                    child_id=payload["children"][0]["child_id"],
                    prefix="该部分对应表2，并说明其与实验章节的关系。",
                )
            ]
        )
    )

    result = ContextualizationService(client=client, max_retries=0).contextualize(
        document=_document(),
        children=[child],
        parents={parent.local_id: parent},
    )

    assert result[0].contextual_prefix.startswith("该部分对应表2")


def test_rejects_chinese_text_that_is_not_a_contextual_relationship() -> None:
    parent = _parent()
    child = _children(1)
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[
                ContextualPrefixItem(
                    child_id=payload["children"][0]["child_id"],
                    prefix="今天天气很好。",
                )
            ]
        )
    )

    with pytest.raises(ContextualizationFailed, match="relation"):
        ContextualizationService(client=client, max_retries=0).contextualize(
            document=_document(),
            children=child,
            parents={parent.local_id: parent},
        )


def test_rejects_new_lower_leading_internal_capital_entity() -> None:
    parent = _parent()
    child = _children(1)
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[
                ContextualPrefixItem(
                    child_id=payload["children"][0]["child_id"],
                    prefix="该部分说明 eBPF 与研究方法的关系。",
                )
            ]
        )
    )

    with pytest.raises(ContextualizationFailed, match="entity"):
        ContextualizationService(client=client, max_retries=0).contextualize(
            document=_document(),
            children=child,
            parents={parent.local_id: parent},
        )


@pytest.mark.parametrize("abbreviation", ["U.S. Patent", "e.g.", "i.e.", "et al."])
def test_english_abbreviations_do_not_create_false_sentence_boundaries(
    abbreviation: str,
) -> None:
    parent = _parent()
    child = _chunk(
        "child-1",
        role="child",
        parent_local_id=parent.local_id,
        text=f"来源使用 {abbreviation} 作为已有英文表达，并继续描述上下文。",
    )
    prefix = f"该部分说明 {abbreviation} 与研究方法的关系。"
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=payload["children"][0]["child_id"], prefix=prefix)]
        )
    )

    result = ContextualizationService(client=client, max_retries=0).contextualize(
        document=_document(),
        children=[child],
        parents={parent.local_id: parent},
    )

    assert result[0].contextual_prefix == prefix


def test_rejects_prefix_that_copies_most_of_child_and_only_rewrites_the_tail() -> None:
    parent = _parent()
    child = _chunk(
        "child-1",
        role="child",
        parent_local_id=parent.local_id,
        text="GraphFormer通过分层注意力机制聚合局部邻居并生成稳定的图节点表示。",
    )
    prefix = "该部分说明GraphFormer通过分层注意力机制聚合局部邻居并生成可靠表示与研究方法的关系。"
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=payload["children"][0]["child_id"], prefix=prefix)]
        )
    )

    with pytest.raises(ContextualizationFailed, match="copies most"):
        ContextualizationService(client=client, max_retries=0).contextualize(
            document=_document(),
            children=[child],
            parents={parent.local_id: parent},
        )


def test_allows_repeating_short_grounded_names_without_copying_child_prose() -> None:
    parent = _parent()
    child = _chunk(
        "child-1",
        role="child",
        parent_local_id=parent.local_id,
        text="GraphFormer 使用 Cora 数据集评估节点分类，并分析不同设置。",
    )
    prefix = "该部分说明 GraphFormer 与 Cora 数据集在研究实验中的关系。"
    client = FakeContextualizationClient(
        lambda payload, _prompt: ContextualPrefixBatch(
            items=[ContextualPrefixItem(child_id=payload["children"][0]["child_id"], prefix=prefix)]
        )
    )

    result = ContextualizationService(client=client, max_retries=0).contextualize(
        document=_document(),
        children=[child],
        parents={parent.local_id: parent},
    )

    assert result[0].contextual_prefix == prefix


def test_retries_with_two_second_correction_then_eight_second_split() -> None:
    parent = _parent()
    attempts = 0

    def responder(payload: dict, _prompt: str) -> ContextualPrefixBatch:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ValueError(f"invalid batch {attempts}")
        return _valid_response(payload, _prompt)

    client = FakeContextualizationClient(responder)
    sleeps: list[float] = []
    checkpoints: list[list[str]] = []
    result = ContextualizationService(
        client=client,
        sleep=sleeps.append,
        batch_size=12,
        max_retries=2,
    ).contextualize(
        document=_document(),
        children=_children(4),
        parents={parent.local_id: parent},
        checkpoint=lambda batch: checkpoints.append([item.local_id for item in batch]),
    )

    assert len(result) == 4
    assert sleeps == [2.0, 8.0]
    assert [len(call["payload"]["children"]) for call in client.calls] == [4, 4, 2, 2]
    assert "invalid batch" not in client.calls[0]["user_prompt"]
    assert "invalid batch 1" in client.calls[1]["user_prompt"]
    assert "invalid batch 2" in client.calls[2]["user_prompt"]
    assert checkpoints == [["child-0", "child-1"], ["child-2", "child-3"]]


def test_oom_recursively_shrinks_to_single_children() -> None:
    parent = _parent()
    client = FakeContextualizationClient(
        lambda payload, prompt: (
            (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
            if len(payload["children"]) > 1
            else _valid_response(payload, prompt)
        )
    )
    checkpoints: list[list[str]] = []

    result = ContextualizationService(client=client, sleep=lambda _seconds: None).contextualize(
        document=_document(),
        children=_children(4),
        parents={parent.local_id: parent},
        checkpoint=lambda batch: checkpoints.append([item.local_id for item in batch]),
    )

    assert len(result) == 4
    assert [len(call["payload"]["children"]) for call in client.calls] == [4, 2, 1, 1, 2, 1, 1]
    assert checkpoints == [["child-0"], ["child-1"], ["child-2"], ["child-3"]]


def test_partial_split_success_is_checkpointed_but_never_returned_as_complete() -> None:
    parent = _parent()

    def responder(payload: dict, prompt: str) -> ContextualPrefixBatch:
        ids = [item["child_id"] for item in payload["children"]]
        if ids == ["child-0", "child-1"]:
            return _valid_response(payload, prompt)
        raise ValueError("permanent contextualization failure")

    client = FakeContextualizationClient(responder)
    checkpoints: list[list[str]] = []

    with pytest.raises(ContextualizationFailed) as exc_info:
        ContextualizationService(client=client, sleep=lambda _seconds: None).contextualize(
            document=_document(),
            children=_children(4),
            parents={parent.local_id: parent},
            checkpoint=lambda batch: checkpoints.append([item.local_id for item in batch]),
        )

    assert exc_info.value.failed_child_ids == {"child-2", "child-3"}
    assert "permanent contextualization failure" in exc_info.value.errors["child-2"]
    assert checkpoints == [["child-0", "child-1"]]
    assert [len(call["payload"]["children"]) for call in client.calls] == [4, 4, 2, 2]


def test_partial_prefix_validation_checkpoints_only_valid_children_with_precise_failure() -> None:
    parent = _parent()
    call_ids: list[list[str]] = []

    def responder(payload: dict, prompt: str) -> ContextualPrefixBatch:
        ids = [item["child_id"] for item in payload["children"]]
        call_ids.append(ids)
        if len(call_ids) <= 2:
            raise ValueError(f"whole batch failure {len(call_ids)}")
        if ids == ["child-0", "child-1"]:
            return ContextualPrefixBatch(
                items=[
                    ContextualPrefixItem(child_id="child-0", prefix=""),
                    ContextualPrefixItem(
                        child_id="child-1",
                        prefix="该部分说明 GraphFormer 方法与研究章节的关系。",
                    ),
                ]
            )
        return _valid_response(payload, prompt)

    checkpoints: list[list[str]] = []

    with pytest.raises(ContextualizationFailed) as exc_info:
        ContextualizationService(
            client=FakeContextualizationClient(responder),
            sleep=lambda _seconds: None,
        ).contextualize(
            document=_document(),
            children=_children(4),
            parents={parent.local_id: parent},
            checkpoint=lambda batch: checkpoints.append([item.local_id for item in batch]),
        )

    assert exc_info.value.failed_child_ids == {"child-0"}
    assert checkpoints == [["child-1"], ["child-2", "child-3"]]
    assert sum("child-1" in ids for ids in call_ids) == 3
    assert call_ids == [
        ["child-0", "child-1", "child-2", "child-3"],
        ["child-0", "child-1", "child-2", "child-3"],
        ["child-0", "child-1"],
        ["child-2", "child-3"],
    ]


@pytest.mark.parametrize("failure", ["parent-role", "missing-parent", "unsupported-type"])
def test_uncontextualizable_inputs_fail_explicitly_before_ai(failure: str) -> None:
    parent = _parent()
    if failure == "parent-role":
        children = [parent]
    elif failure == "missing-parent":
        children = [_chunk("child-1", role="child", parent_local_id="absent")]
    else:
        invalid = _chunk("child-1", role="child", parent_local_id=parent.local_id)
        invalid.block_type = "reference"  # type: ignore[assignment]
        children = [invalid]
    client = FakeContextualizationClient(_valid_response)

    with pytest.raises(ContextualizationFailed) as exc_info:
        ContextualizationService(client=client).contextualize(
            document=_document(),
            children=children,
            parents={parent.local_id: parent},
        )

    assert exc_info.value.failed_child_ids == {children[0].local_id}
    assert not client.calls
