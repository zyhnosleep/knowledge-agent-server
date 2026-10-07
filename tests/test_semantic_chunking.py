from __future__ import annotations

import math
import re

import pytest

from app.core.config import get_settings
from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
)
from app.services.semantic_chunking import (
    ChunkDraft,
    SemanticChunker,
    SourceFidelityError,
)
from app.services.structured_evidence import StructuredEvidenceBuilder


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


class RecordingEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.vectors is not None:
            assert len(self.vectors) == len(texts)
            return self.vectors
        return [[1.0, float(index % 2)] for index, _text in enumerate(texts)]


class NamedEmbedder(RecordingEmbedder):
    model_name = "test/semantic-embedder"


def span(block_id: str, page: int = 0, start: int = 0, end: int = 10) -> SourceSpan:
    return SourceSpan(
        page_index=page,
        source_block_id=block_id,
        char_start=start,
        char_end=end,
    )


def block(
    block_id: str,
    text: str,
    order: int,
    *,
    block_type: str = "narrative",
    section_path: list[str] | None = None,
    table_id: str | None = None,
    figure_id: str | None = None,
    formula_id: str | None = None,
) -> CanonicalBlock:
    return CanonicalBlock(
        block_id=block_id,
        block_type=block_type,
        text=text,
        section_path=section_path or ["Methods"],
        reading_order=order,
        source_spans=[span(block_id, page=order)],
        parser_source="test",
        table_id=table_id,
        figure_id=figure_id,
        formula_id=formula_id,
    )


def document(*blocks: CanonicalBlock, parse_version: str = "parse-v1") -> CanonicalDocument:
    return CanonicalDocument(
        document_id="doc-1",
        parser_source="test",
        parse_version=parse_version,
        blocks=list(blocks),
    )


def make_chunker(embedder: object | None = None, **overrides: int) -> SemanticChunker:
    values = {
        "parent_min_tokens": 4,
        "parent_target_tokens": 9,
        "parent_max_tokens": 15,
        "child_min_tokens": 3,
        "child_target_tokens": 6,
        "child_max_tokens": 9,
        "overlap_tokens": 2,
        "break_percentile": 20,
    }
    values.update(overrides)
    return SemanticChunker(embedder or RecordingEmbedder(), word_count, **values)


def sentence_sequence(count: int, prefix: str = "Sentence") -> str:
    return " ".join(f"{prefix}{index} has evidence." for index in range(1, count + 1))


def test_identical_source_in_different_documents_has_disjoint_chunk_ids_and_local_links():
    source=document(block('shared',sentence_sequence(8),0))
    other=source.model_copy(update={'document_id':'doc-2'})
    first=make_chunker().build(source)
    second=make_chunker().build(other)
    first_ids={chunk.local_id for chunk in first}
    second_ids={chunk.local_id for chunk in second}
    assert first_ids.isdisjoint(second_ids)
    assert [chunk.local_id for chunk in make_chunker().build(source)]==[chunk.local_id for chunk in first]
    for chunks,allowed in ((first,first_ids),(second,second_ids)):
        children=[chunk for chunk in chunks if chunk.chunk_role=='child']
        assert len(children)>1
        for chunk in children:
            assert chunk.parent_local_id in allowed
            assert chunk.previous_child_local_id is None or chunk.previous_child_local_id in allowed
            assert chunk.next_child_local_id is None or chunk.next_child_local_id in allowed
    assert [chunk.text for chunk in first]==[chunk.text for chunk in second]


def test_chunk_draft_is_serializable_and_defaults_come_from_settings() -> None:
    chunker = SemanticChunker(RecordingEmbedder(), word_count)

    assert chunker.parent_token_limits == (500, 1200, 1800)
    assert chunker.child_token_limits == (180, 400, 600)
    assert chunker.overlap_tokens == 50
    assert chunker.break_percentile == 20

    draft = ChunkDraft(
        local_id="local-1",
        parse_version="parse-v1",
        chunk_role="parent",
        block_type="narrative",
        text="source text",
        embedding_text="source text",
        token_count=2,
        source_block_ids=["b1"],
        source_spans=[span("b1")],
        section_path=["Methods"],
        ordinal=0,
        splitter_name="section_aware_semantic",
        splitter_version="semantic-v1",
        splitting_model="test/model",
        semantic_boundary_score=None,
    )
    assert draft.model_dump(mode="json")["source_spans"][0]["source_block_id"] == "b1"


def test_default_token_counter_uses_configured_cached_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is False
            return text.split()

    def load(name: str) -> FakeTokenizer:
        loaded.append(name)
        return FakeTokenizer()

    tokenizer_name = "test/qwen-semantic-tokenizer"
    monkeypatch.setattr(get_settings(), "semantic_tokenizer_name", tokenizer_name)
    monkeypatch.setattr(StructuredEvidenceBuilder, "_tokenizer_cache", {})
    monkeypatch.setattr(StructuredEvidenceBuilder, "_tokenizer_key_locks", {})
    monkeypatch.setattr(
        StructuredEvidenceBuilder,
        "_load_local_tokenizer",
        staticmethod(load),
    )

    first = SemanticChunker(RecordingEmbedder())
    second = SemanticChunker(RecordingEmbedder())

    assert loaded == []
    assert first._count_tokens("alpha beta") == 2
    assert second._count_tokens("gamma delta") == 2
    assert loaded == [tokenizer_name]


def test_default_offset_tokenizer_uses_evidenced_near_linear_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = " ".join(f"D{index}." for index in range(1000))
    loaded: list[str] = []

    class FakeTokenizer:
        def __init__(self) -> None:
            self.scanned_characters = 0

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            self.scanned_characters += len(text)
            return list(range(word_count(text)))

        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            return_offsets_mapping: bool,
        ) -> dict[str, list[tuple[int, int]]]:
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            self.scanned_characters += len(text)
            return {
                "offset_mapping": [
                    match.span() for match in re.finditer(r"\S+", text)
                ]
            }

    tokenizer = FakeTokenizer()

    def load(name: str) -> FakeTokenizer:
        loaded.append(name)
        return tokenizer

    tokenizer_name = "test/qwen-offset-tokenizer"
    monkeypatch.setattr(get_settings(), "semantic_tokenizer_name", tokenizer_name)
    monkeypatch.setattr(StructuredEvidenceBuilder, "_tokenizer_cache", {})
    monkeypatch.setattr(StructuredEvidenceBuilder, "_tokenizer_key_locks", {})
    monkeypatch.setattr(
        StructuredEvidenceBuilder,
        "_load_local_tokenizer",
        staticmethod(load),
    )

    chunks = SemanticChunker(
        RecordingEmbedder(),
        parent_min_tokens=1,
        parent_target_tokens=2000,
        parent_max_tokens=2000,
        child_min_tokens=1,
        child_target_tokens=2000,
        child_max_tokens=2000,
        overlap_tokens=0,
        break_percentile=20,
    ).build(document(block("default-complexity", source, 0)))

    assert {item.chunk_role for item in chunks} == {"parent", "child"}
    assert loaded == [tokenizer_name]
    assert tokenizer.scanned_characters <= len(source) * 64


def test_every_draft_has_roundtrippable_splitter_audit_and_boundary_reason() -> None:
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    embedder = NamedEmbedder(vectors)
    chunks = make_chunker(
        embedder,
        parent_min_tokens=1,
        parent_target_tokens=4,
        parent_max_tokens=4,
        child_min_tokens=1,
        child_target_tokens=4,
        child_max_tokens=4,
        overlap_tokens=0,
    ).build(
        document(
            block(
                "audit",
                "S1 evidence. S2 evidence. S3 evidence. S4 evidence.",
                0,
            )
        )
    )

    assert chunks
    assert all(item.splitter_name == "section_aware_semantic" for item in chunks)
    assert all(item.splitter_version == "semantic-v1" for item in chunks)
    assert all(item.splitting_model == "test/semantic-embedder" for item in chunks)
    assert all(
        ChunkDraft.model_validate(item.model_dump(mode="json")) == item
        for item in chunks
    )
    semantic_parent = next(
        item
        for item in chunks
        if item.chunk_role == "parent"
        and item.metadata["boundary_reason"] == "semantic_percentile"
    )
    assert semantic_parent.semantic_boundary_score == pytest.approx(
        semantic_parent.metadata["semantic_boundary_score"]
    )
    assert any(
        item.metadata["boundary_reason"] in {"max_tokens", "section_end"}
        for item in chunks
    )


def test_embeddings_are_batched_once_and_low_cosine_boundary_is_preferred() -> None:
    text = " ".join(f"S{index} evidence." for index in range(1, 7))
    vectors = [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    embedder = RecordingEmbedder(vectors)
    chunks = make_chunker(
        embedder,
        parent_min_tokens=2,
        parent_target_tokens=6,
        parent_max_tokens=20,
        child_min_tokens=1,
        child_target_tokens=20,
        child_max_tokens=20,
        overlap_tokens=0,
    ).build(document(block("b1", text, 0)))

    parents = [item for item in chunks if item.chunk_role == "parent"]
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == [
        *(f"S{index} evidence. " for index in range(1, 6)),
        "S6 evidence.",
    ]
    assert len(parents) == 2
    assert parents[0].text == "S1 evidence. S2 evidence. S3 evidence. "


def test_parent_semantic_percentile_is_recomputed_for_each_current_chunk() -> None:
    angles = [0.0, 0.1, 1.6, 1.7, 2.7, 2.8]
    vectors = [[math.cos(angle), math.sin(angle)] for angle in angles]
    chunks = make_chunker(
        RecordingEmbedder(vectors),
        parent_min_tokens=1,
        parent_target_tokens=4,
        parent_max_tokens=6,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
        break_percentile=10,
    ).build(
        document(
            block(
                "local-percentile",
                " ".join(f"S{index} evidence." for index in range(1, 7)),
                0,
            )
        )
    )
    parents = [item for item in chunks if item.chunk_role == "parent"]

    assert [item.token_count for item in parents] == [4, 4, 4]
    assert len({item.parent_local_id for item in chunks if item.chunk_role == "child"}) == 3


def test_children_use_semantic_boundary_after_target_instead_of_cutting_at_target() -> None:
    angles = [0.0, 0.1, 0.2, 1.7, 1.8, 2.8]
    vectors = [[math.cos(angle), math.sin(angle)] for angle in angles]
    chunks = make_chunker(
        RecordingEmbedder(vectors),
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=4,
        child_max_tokens=6,
        overlap_tokens=0,
        break_percentile=20,
    ).build(
        document(
            block(
                "child-semantic",
                " ".join(f"S{index} evidence." for index in range(1, 7)),
                0,
            )
        )
    )
    children = [item for item in chunks if item.chunk_role == "child"]

    assert children[0].token_count == 6


def test_english_and_cjk_sentence_boundaries_and_section_boundary_are_hard() -> None:
    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
        document(
            block("intro", "First claim. Second claim!", 0, section_path=["Intro"]),
            block("cn", "第一句。第二句！第三句？", 1, section_path=["方法"]),
        )
    )
    parents = [item for item in chunks if item.chunk_role == "parent"]

    assert len(parents) == 2
    assert parents[0].section_path == ["Intro"]
    assert parents[1].section_path == ["方法"]
    assert parents[1].text == "第一句。第二句！第三句？"


def test_sentence_units_preserve_exact_source_and_do_not_split_abbreviations_decimals_or_urls() -> None:
    source = (
        "Dr. Smith measured 3.14 units.  "
        "Visit https://example.com/a.b?q=1.\n\n"
        "Next line。中文句！"
    )
    source_block = block("fidelity", source, 0)
    source_block.source_spans = [
        SourceSpan(
            page_index=0,
            source_block_id="fidelity",
            char_start=100,
            char_end=100 + len(source),
        )
    ]
    embedder = RecordingEmbedder()
    chunks = make_chunker(
        embedder,
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(source_block))
    parent = next(item for item in chunks if item.chunk_role == "parent")
    child = next(item for item in chunks if item.chunk_role == "child")

    assert parent.text == source
    assert child.text == source
    assert embedder.calls == [
        [
            "Dr. Smith measured 3.14 units.  ",
            "Visit https://example.com/a.b?q=1.\n\n",
            "Next line。",
            "中文句！",
        ]
    ]


def test_narrative_merge_inserts_audited_separator_between_source_blocks() -> None:
    first = block("first-source", "First paragraph.", 0)
    second = block("second-source", "Second paragraph.", 1)
    first.source_spans = [
        SourceSpan(source_block_id="first-source", char_start=0, char_end=len(first.text))
    ]
    second.source_spans = [
        SourceSpan(source_block_id="second-source", char_start=20, char_end=20 + len(second.text))
    ]
    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(first, second))
    parent = next(item for item in chunks if item.chunk_role == "parent")
    child = next(item for item in chunks if item.chunk_role == "child")

    expected = first.text + "\n\n" + second.text
    assert parent.text == expected
    assert parent.embedding_text == expected
    assert parent.token_count == 4
    assert child.text == expected
    assert {span.source_block_id for span in parent.source_spans} == {
        "first-source",
        "second-source",
    }
    assert parent.metadata["structural_separators"] == [
        {
            "text": "\n\n",
            "source_backed": False,
            "between_block_ids": ["first-source", "second-source"],
        }
    ]


def test_quoted_question_and_exclamation_marks_are_sentence_boundaries() -> None:
    source = 'He asked "Ready?"  She shouted (Go!)\nDone.'
    embedder = RecordingEmbedder()
    chunks = make_chunker(
        embedder,
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(block("quoted", source, 0)))

    assert embedder.calls == [
        ['He asked "Ready?"  ', "She shouted (Go!)\n", "Done."]
    ]
    assert next(item for item in chunks if item.chunk_role == "parent").text == source


def test_academic_abbreviations_and_initialisms_do_not_create_false_boundaries() -> None:
    source = (
        "Fig. 2 shows Eq. 3 in Sec. 4. "
        "Smith et al. agree. "
        "A Ph.D. study in the U.S. confirms it."
    )
    embedder = RecordingEmbedder()
    chunks = make_chunker(
        embedder,
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(block("academic-abbrev", source, 0)))

    assert embedder.calls == [
        [
            "Fig. 2 shows Eq. 3 in Sec. 4. ",
            "Smith et al. agree. ",
            "A Ph.D. study in the U.S. confirms it.",
        ]
    ]
    assert next(item for item in chunks if item.chunk_role == "parent").text == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Prior work by Smith et al. This extends it.",
            ["Prior work by Smith et al. ", "This extends it."],
        ),
        (
            "The method is used in the U.S. Results improve.",
            ["The method is used in the U.S. ", "Results improve."],
        ),
        (
            "Several variants exist, etc. 下一句。",
            ["Several variants exist, etc. ", "下一句。"],
        ),
    ],
)
def test_terminal_capable_abbreviation_splits_before_clear_new_sentence(
    source: str,
    expected: list[str],
) -> None:
    embedder = RecordingEmbedder()
    make_chunker(
        embedder,
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(block("terminal-abbrev", source, 0)))

    assert embedder.calls == [expected]


def test_nonterminal_abbreviations_stay_attached_to_their_continuation() -> None:
    source = "Dr. Smith used Fig. 2 and Eq. (3) with a U.S. model."
    embedder = RecordingEmbedder()
    make_chunker(
        embedder,
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(block("nonterminal-abbrev", source, 0)))

    assert embedder.calls == [[source]]


def test_parent_max_is_hard_and_overlong_sentence_uses_lossless_token_windows() -> None:
    source = " ".join(f"token{index}" for index in range(23)) + "."
    chunks = make_chunker(
        parent_min_tokens=3,
        parent_target_tokens=7,
        parent_max_tokens=10,
        child_min_tokens=1,
        child_target_tokens=30,
        child_max_tokens=30,
        overlap_tokens=0,
    ).build(document(block("long", source, 0)))
    parents = [item for item in chunks if item.chunk_role == "parent"]

    assert len(parents) == 3
    assert all(item.token_count <= 10 for item in parents)
    assert all(item.metadata["token_window_split"] is True for item in parents)
    assert "".join(item.text.replace(" ", "") for item in parents) == source.replace(" ", "")


def test_parent_and_child_max_count_joined_sentence_separators() -> None:
    chunks = SemanticChunker(
        RecordingEmbedder(),
        len,
        parent_min_tokens=1,
        parent_target_tokens=6,
        parent_max_tokens=6,
        child_min_tokens=1,
        child_target_tokens=6,
        child_max_tokens=6,
        overlap_tokens=0,
        break_percentile=20,
    ).build(document(block("joined", "aa. bb. cc.", 0)))

    assert all(item.token_count <= 6 for item in chunks)


def test_children_use_whole_sentence_overlap_and_do_not_lose_source() -> None:
    source = sentence_sequence(7)
    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=3,
        child_target_tokens=9,
        child_max_tokens=12,
        overlap_tokens=4,
    ).build(document(block("b1", source, 0)))
    parent = next(item for item in chunks if item.chunk_role == "parent")
    children = [item for item in chunks if item.chunk_role == "child"]

    assert len(children) >= 2
    assert "Sentence3 has evidence." in children[0].text
    assert children[1].text.startswith("Sentence3 has evidence.")
    assert all(not item.text.startswith("has evidence") for item in children)
    recovered: list[str] = []
    for child in children:
        for sentence in re.findall(r"Sentence\d+ has evidence\.", child.text):
            if sentence not in recovered:
                recovered.append(sentence)
    assert recovered == re.findall(r"Sentence\d+ has evidence\.", parent.text)


def test_child_overlap_is_empty_when_no_complete_sentence_fits_budget() -> None:
    source = (
        "One two three four five six. "
        "Seven eight nine ten eleven twelve."
    )
    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=6,
        child_max_tokens=12,
        overlap_tokens=4,
    ).build(document(block("overlap-budget", source, 0)))
    children = [item for item in chunks if item.chunk_role == "child"]

    assert len(children) == 2
    assert children[1].text == "Seven eight nine ten eleven twelve."
    assert children[1].metadata["overlap_sentence_count"] == 0
    assert children[1].metadata["overlap_tokens"] == 0


def test_child_tail_is_merged_and_splittable_long_sentence_respects_child_max() -> None:
    source = sentence_sequence(5)
    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=7,
        child_target_tokens=9,
        child_max_tokens=18,
        overlap_tokens=0,
    ).build(document(block("tail", source, 0)))
    children = [item for item in chunks if item.chunk_role == "child"]
    assert len(children) == 1
    assert children[0].token_count == 15

    long_sentence = " ".join(f"wide{index}" for index in range(11)) + "."
    overflow_chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=2,
        child_target_tokens=4,
        child_max_tokens=6,
        overlap_tokens=0,
    ).build(document(block("overflow", long_sentence, 0)))
    overflow_children = [
        item for item in overflow_chunks if item.chunk_role == "child"
    ]
    assert all(item.token_count <= 6 for item in overflow_children)
    assert "".join(item.text for item in overflow_children) == long_sentence
    assert all(
        item.metadata["single_sentence_overflow"] is False
        for item in overflow_children
    )


def test_parent_tail_rebalances_complete_units_when_direct_merge_would_exceed_max() -> None:
    source = " ".join(f"P{index} unit." for index in range(1, 6))
    chunks = make_chunker(
        parent_min_tokens=4,
        parent_target_tokens=8,
        parent_max_tokens=8,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(block("parent-rebalance", source, 0)))
    parents = [item for item in chunks if item.chunk_role == "parent"]

    assert [item.token_count for item in parents] == [6, 4]
    assert all(item.metadata.get("undersized_reason") is None for item in parents)
    assert "".join(item.text for item in parents) == source


def test_child_tail_rebalances_complete_units_without_loss_or_reordering() -> None:
    source = " ".join(f"C{index} unit." for index in range(1, 6))
    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=4,
        child_target_tokens=8,
        child_max_tokens=8,
        overlap_tokens=0,
    ).build(document(block("child-rebalance", source, 0)))
    children = [item for item in chunks if item.chunk_role == "child"]

    assert [item.token_count for item in children] == [6, 4]
    assert all(item.metadata.get("undersized_reason") is None for item in children)
    assert "".join(item.text for item in children) == source


def test_indivisible_source_character_overflow_is_rejected_by_final_audit() -> None:
    def indivisible_counter(text: str) -> int:
        return len(text) * 10

    chunker = SemanticChunker(
        RecordingEmbedder(),
        indivisible_counter,
        parent_min_tokens=1,
        parent_target_tokens=4,
        parent_max_tokens=6,
        child_min_tokens=1,
        child_target_tokens=4,
        child_max_tokens=6,
        overlap_tokens=0,
        break_percentile=20,
    )

    with pytest.raises(SourceFidelityError) as raised:
        chunker.build(document(block("one-char", "界", 0)))

    assert raised.value.block_type == "narrative"
    assert raised.value.source_id == "one-char"
    assert raised.value.excerpt == "界"


def test_child_neighbors_cross_parent_within_structure_but_stop_at_section_boundary() -> None:
    chunks = make_chunker(
        parent_min_tokens=2,
        parent_target_tokens=6,
        parent_max_tokens=6,
        child_min_tokens=1,
        child_target_tokens=3,
        child_max_tokens=4,
        overlap_tokens=0,
    ).build(document(block("b1", sentence_sequence(6), 0)))
    parents = [item for item in chunks if item.chunk_role == "parent"]
    assert len(parents) == 3

    children = [item for item in chunks if item.chunk_role == "child"]
    for index, child in enumerate(children):
        assert child.previous_child_local_id == (
            children[index - 1].local_id if index else None
        )
        assert child.next_child_local_id == (
            children[index + 1].local_id if index + 1 < len(children) else None
        )
    parent_by_child = {item.local_id: item.parent_local_id for item in children}
    assert any(
        child.next_child_local_id is not None
        and parent_by_child[child.next_child_local_id] != child.parent_local_id
        for child in children
    )

    section_chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=3,
        child_max_tokens=6,
        overlap_tokens=0,
    ).build(
        document(
            block("section-a", sentence_sequence(3, "A"), 0, section_path=["A"]),
            block("section-b", sentence_sequence(3, "B"), 1, section_path=["B"]),
        )
    )
    section_a = [
        item
        for item in section_chunks
        if item.chunk_role == "child" and item.section_path == ["A"]
    ]
    section_b = [
        item
        for item in section_chunks
        if item.chunk_role == "child" and item.section_path == ["B"]
    ]
    assert section_a[-1].next_child_local_id is None
    assert section_b[0].previous_child_local_id is None


def test_references_are_excluded_but_appendix_and_caption_are_searchable() -> None:
    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
        document(
            block("main", "Main source sentence.", 0),
            block(
                "reference-section",
                "Hidden citation sentence.",
                1,
                section_path=["References"],
            ),
            block("reference", "Also hidden.", 2, block_type="reference"),
            block(
                "numbered-reference-section",
                "Numbered hidden citation.",
                3,
                section_path=["7. References"],
            ),
            block("caption", "Figure caption evidence.", 4, block_type="caption"),
            block("appendix", "Appendix evidence sentence.", 5, block_type="appendix"),
        )
    )
    combined = "\n".join(item.text for item in chunks)

    assert "Hidden citation" not in combined
    assert "Also hidden" not in combined
    assert "Numbered hidden citation" not in combined
    assert "Figure caption evidence" in combined
    assert "Appendix evidence sentence" in combined


def test_reference_structures_are_excluded_but_appendix_structures_are_searchable() -> None:
    reference_table = valid_table().model_copy(deep=True)
    reference_table.table_id = "reference-table"
    appendix_table = valid_table().model_copy(deep=True)
    appendix_table.table_id = "appendix-table"
    reference_block = block(
        "reference-table-block",
        "",
        0,
        block_type="table",
        section_path=["References"],
        table_id=reference_table.table_id,
    )
    reference_block.retrievable = False
    doc = document(
        reference_block,
        block(
            "appendix-table-block",
            "",
            1,
            block_type="table",
            section_path=["Appendix A"],
            table_id=appendix_table.table_id,
        ),
    )
    doc.tables = [reference_table, appendix_table]

    chunks = make_chunker(child_max_tokens=100).build(doc)

    assert all(item.source_block_ids != ["reference-table-block"] for item in chunks)
    assert any(item.source_block_ids == ["appendix-table-block"] for item in chunks)


@pytest.mark.parametrize(
    "section_title",
    [
        "Literature Cited",
        "Sources",
        "Cited Literature",
        "8. Literature Cited",
        "文献",
        "参考资料",
        "引用资料",
    ],
)
def test_common_reference_section_variants_are_excluded(section_title: str) -> None:
    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
        document(
            block(
                "variant-reference",
                "This citation must stay out of ordinary RAG.",
                0,
                section_path=[section_title],
            ),
            block(
                "appendix-kept",
                "Appendix source remains searchable.",
                1,
                block_type="appendix",
                section_path=["Appendix B"],
            ),
        )
    )
    combined = "".join(item.text for item in chunks)

    assert "citation must stay out" not in combined
    assert "Appendix source remains searchable" in combined


def test_numbered_cited_literature_structured_block_is_excluded() -> None:
    table = valid_table().model_copy(deep=True)
    table.table_id = "cited-table"
    doc = document(
        block(
            "cited-table-block",
            "",
            0,
            block_type="table",
            section_path=["4. Cited Literature"],
            table_id=table.table_id,
        )
    )
    doc.tables = [table]

    assert make_chunker().build(doc) == []


@pytest.mark.parametrize("format_kind", ["code", "pre"])
def test_preformatted_blocks_are_hard_boundaries_and_preserve_source(
    format_kind: str,
) -> None:
    code_text = "  if x:\n    print('a.b')\nreturn x\n\n"
    code_block = block("code", code_text, 1)
    code_block.metadata["kind"] = format_kind

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
        document(
            block("before", "Prose before code.", 0),
            code_block,
            block("after", "Prose after code.", 2),
        )
    )
    code_parent = next(
        item
        for item in chunks
        if item.chunk_role == "parent" and item.source_block_ids == ["code"]
    )

    assert code_parent.text == code_text
    assert "Prose before" not in code_parent.text
    assert "Prose after" not in code_parent.text


def test_adjacent_preformatted_blocks_stay_separate_and_oversized_text_is_lossless() -> None:
    first = block("code-1", "alpha = 1\n", 0)
    second_text = "  beta = 2222222222\n\n"
    second = block("code-2", second_text, 1)
    first.metadata["kind"] = "code"
    second.metadata["kind"] = "code"
    chunks = SemanticChunker(
        RecordingEmbedder(),
        len,
        parent_min_tokens=1,
        parent_target_tokens=12,
        parent_max_tokens=12,
        child_min_tokens=1,
        child_target_tokens=6,
        child_max_tokens=6,
        overlap_tokens=0,
        break_percentile=20,
    ).build(document(first, second))
    first_parents = [
        item
        for item in chunks
        if item.chunk_role == "parent" and item.source_block_ids == ["code-1"]
    ]
    second_parents = [
        item
        for item in chunks
        if item.chunk_role == "parent" and item.source_block_ids == ["code-2"]
    ]

    assert first_parents
    assert second_parents
    assert "".join(item.text for item in first_parents) == first.text
    assert "".join(item.text for item in second_parents) == second_text
    assert all(item.token_count <= 12 for item in [*first_parents, *second_parents])
    assert all(item.token_count <= 6 for item in chunks if item.chunk_role == "child")


def test_ids_and_spans_are_stable_deduplicated_and_source_sensitive() -> None:
    repeated_span = span("stable", start=0, end=50)
    source_block = block("stable", "Stable source. Another sentence.", 0)
    source_block.source_spans = [repeated_span, repeated_span]
    chunker = make_chunker(parent_max_tokens=100, child_max_tokens=100)

    first = chunker.build(document(source_block))
    second = chunker.build(document(source_block))
    changed = chunker.build(
        document(block("stable", "Changed source. Another sentence.", 0))
    )

    assert [item.local_id for item in first] == [item.local_id for item in second]
    assert [item.local_id for item in first] != [item.local_id for item in changed]
    assert all(len(item.source_spans) == 1 for item in first)
    assert all(item.embedding_text == item.text for item in first)


def test_child_spans_use_precise_block_relative_character_ranges() -> None:
    source = " ".join(f"S{index} evidence." for index in range(1, 7))
    source_block = block("precise", source, 0)
    source_block.source_spans = [
        SourceSpan(
            page_index=3,
            source_block_id="precise",
            char_start=100,
            char_end=100 + len(source),
        )
    ]
    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=2,
        child_max_tokens=2,
        overlap_tokens=0,
    ).build(document(source_block))
    children = [item for item in chunks if item.chunk_role == "child"]
    ranges = [
        (item.source_spans[0].char_start, item.source_spans[0].char_end)
        for item in children
    ]

    assert len(children) == 6
    assert len(set(ranges)) == 6
    assert ranges[0][0] == 100
    assert ranges[-1][1] == 100 + len(source)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert "".join(item.text for item in children) == source
    assert all(item.metadata["source_span_mapping"] == "exact" for item in children)


def test_source_span_mapping_is_approximate_when_all_units_lack_spans() -> None:
    source_block = block("no-spans", "First sentence. Second sentence.", 0)
    source_block.source_spans = []

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
        document(source_block)
    )

    assert chunks
    assert all(not item.source_spans for item in chunks)
    assert all(item.metadata["source_span_mapping"] == "approximate" for item in chunks)


def test_source_span_mapping_is_approximate_when_any_unit_lacks_spans() -> None:
    mapped = block("mapped", "Mapped sentence.", 0)
    mapped.source_spans = [
        SourceSpan(
            source_block_id="mapped",
            char_start=0,
            char_end=len(mapped.text),
        )
    ]
    unmapped = block("unmapped", "Unmapped sentence.", 1)
    unmapped.source_spans = []

    chunks = make_chunker(
        parent_min_tokens=1,
        parent_target_tokens=100,
        parent_max_tokens=100,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
    ).build(document(mapped, unmapped))
    parent = next(item for item in chunks if item.chunk_role == "parent")
    child = next(item for item in chunks if item.chunk_role == "child")

    assert parent.source_block_ids == ["mapped", "unmapped"]
    assert parent.metadata["source_span_mapping"] == "approximate"
    assert child.metadata["source_span_mapping"] == "approximate"


def test_locator_without_character_range_is_preserved_and_marked_approximate() -> None:
    source_block = block("locator", "First sentence. Second sentence.", 0)
    source_block.source_spans = [
        SourceSpan(
            page_index=4,
            source_block_id="locator",
            bbox=(1.0, 2.0, 3.0, 4.0),
        )
    ]
    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
        document(source_block)
    )

    assert all(item.source_spans[0].bbox == (1.0, 2.0, 3.0, 4.0) for item in chunks)
    assert all(item.metadata["source_span_mapping"] == "approximate" for item in chunks)


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[1.0, 0.0], [1.0]], "dimension"),
        ([[1.0, 0.0], [math.nan, 1.0]], "finite"),
    ],
)
def test_invalid_embedding_batches_fail_deterministically(
    vectors: list[list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_chunker(RecordingEmbedder(vectors)).build(
            document(block("b1", "First sentence. Second sentence.", 0))
        )


def test_zero_vectors_have_deterministic_fallback() -> None:
    chunker = make_chunker(
        RecordingEmbedder([[0.0, 0.0], [0.0, 0.0]]),
        parent_min_tokens=1,
        parent_target_tokens=2,
        parent_max_tokens=20,
        child_max_tokens=20,
    )
    chunks = chunker.build(document(block("b1", "First sentence. Second sentence.", 0)))
    parent = next(item for item in chunks if item.chunk_role == "parent")
    assert parent.metadata["semantic_zero_vector_fallback"] is True


def valid_table() -> CanonicalTable:
    table_span = span("source-table", page=1)
    headers = ["Name", "Score"]
    rows = [["Alpha", "10"], ["Beta", "20"]]
    cells = [
        CanonicalCell(
            text=value,
            row_index=row_index,
            column_index=column_index,
            is_header=row_index == 0,
            source_spans=[table_span],
        )
        for row_index, row in enumerate([headers, *rows])
        for column_index, value in enumerate(row)
    ]
    markdown = "\n".join(
        [
            "| Name | Score |",
            "| --- | --- |",
            "| Alpha | 10 |",
            "| Beta | 20 |",
        ]
    )
    return CanonicalTable(
        table_id="t1",
        caption="Table 1. Scores",
        headers=headers,
        rows=rows,
        cells=cells,
        source_markdown=markdown,
        normalized_markdown=markdown,
        source_spans=[table_span],
        metadata={"table_number": "1", "semantic_row_groups": [[0], [1]]},
    )


def _replace_draft_text(draft: ChunkDraft, old: str, new: str = "") -> None:
    draft.text = draft.text.replace(old, new)
    draft.embedding_text = draft.embedding_text.replace(old, new)
    draft.token_count = word_count(draft.embedding_text)


def test_source_fidelity_audit_rejects_omitted_narrative_unit() -> None:
    doc = document(block("narrative-source", "First fact. Second fact.", 3))
    chunker = make_chunker(parent_max_tokens=100, child_max_tokens=100)
    drafts = chunker.build(doc)
    child = next(item for item in drafts if item.chunk_role == "child")
    _replace_draft_text(child, "Second fact.")

    with pytest.raises(SourceFidelityError) as raised:
        chunker._audit_source_fidelity(doc, drafts)

    assert raised.value.document_id == doc.document_id
    assert raised.value.page == 3
    assert raised.value.block_type == "narrative"
    assert raised.value.source_id == "narrative-source"
    assert raised.value.excerpt == "Second fact."
    assert len(raised.value.excerpt) <= 200


def test_source_fidelity_audit_rejects_omitted_table_field() -> None:
    table = valid_table()
    table.footnotes = ["Scores are source measurements."]
    doc = document(block("table-source", "", 1, block_type="table", table_id=table.table_id))
    doc.tables = [table]
    chunker = make_chunker(parent_max_tokens=200, child_max_tokens=100)
    drafts = chunker.build(doc)
    for draft in drafts:
        draft.metadata["footnotes"] = []
        _replace_draft_text(draft, "Footnote: Scores are source measurements.")

    with pytest.raises(SourceFidelityError) as raised:
        chunker._audit_source_fidelity(doc, drafts)

    assert raised.value.block_type == "table"
    assert raised.value.source_id == table.table_id
    assert raised.value.excerpt == table.footnotes[0]


def test_source_fidelity_audit_rejects_omitted_formula_field() -> None:
    formula = CanonicalFormula(
        formula_id="formula-audit",
        latex="x = y + 1",
        caption="Equation 1.",
        description="Complete formula description.",
        source_spans=[span("formula-object", page=4)],
    )
    doc = document(
        block("formula-source", "", 4, block_type="formula", formula_id=formula.formula_id)
    )
    doc.formulas = [formula]
    chunker = make_chunker(parent_max_tokens=200, child_max_tokens=100)
    drafts = chunker.build(doc)
    for draft in drafts:
        _replace_draft_text(draft, formula.description or "")

    with pytest.raises(SourceFidelityError) as raised:
        chunker._audit_source_fidelity(doc, drafts)

    assert raised.value.page == 4
    assert raised.value.block_type == "formula"
    assert raised.value.source_id == formula.formula_id
    assert raised.value.excerpt == formula.description


def test_source_fidelity_audit_rejects_omitted_figure_asset_identity() -> None:
    figure = CanonicalFigure(
        figure_id="figure-audit",
        caption="Figure 1. Source trend",
        description="Complete source description.",
        asset_path="assets/figure-audit.png",
        source_spans=[span("figure-object", page=5)],
    )
    doc = document(
        block("figure-source", "", 5, block_type="figure", figure_id=figure.figure_id)
    )
    doc.figures = [figure]
    doc.assets = [
        CanonicalAsset(
            asset_id="asset-figure-audit",
            path=figure.asset_path or "",
            media_type="image/png",
        )
    ]
    chunker = make_chunker(parent_max_tokens=200, child_max_tokens=100)
    drafts = chunker.build(doc)
    for draft in drafts:
        draft.metadata["asset_path"] = None
        for source_span in draft.source_spans:
            source_span.metadata.pop("asset_id", None)
        _replace_draft_text(draft, figure.asset_path or "")

    with pytest.raises(SourceFidelityError) as raised:
        chunker._audit_source_fidelity(doc, drafts)

    assert raised.value.page == 5
    assert raised.value.block_type == "figure"
    assert raised.value.source_id == figure.figure_id
    assert raised.value.excerpt == figure.asset_path


def test_source_fidelity_audit_rejects_non_direct_child_embedding() -> None:
    doc = document(block("embedding-source", "Direct source evidence.", 0))
    chunker = make_chunker(parent_max_tokens=100, child_max_tokens=100)
    drafts = chunker.build(doc)
    child = next(item for item in drafts if item.chunk_role == "child")
    child.embedding_text = f"Context prefix\n\n{child.text}"

    with pytest.raises(SourceFidelityError) as raised:
        chunker._audit_source_fidelity(doc, drafts)

    assert raised.value.source_id == "embedding-source"
    assert raised.value.excerpt == child.embedding_text[:200]


def test_source_fidelity_audit_rejects_child_over_token_limit() -> None:
    doc = document(block("limit-source", "Bounded source evidence.", 0))
    chunker = make_chunker(parent_max_tokens=100, child_max_tokens=100)
    drafts = chunker.build(doc)
    child = next(item for item in drafts if item.chunk_role == "child")
    child.token_count = 101

    with pytest.raises(SourceFidelityError) as raised:
        chunker._audit_source_fidelity(doc, drafts)

    assert raised.value.source_id == "limit-source"
    assert raised.value.excerpt == child.text


@pytest.mark.parametrize(
    ("block_type", "identity_field"),
    [
        ("table", "table_id"),
        ("figure", "figure_id"),
        ("formula", "formula_id"),
    ],
)
def test_missing_structured_source_raises_bounded_fidelity_error(
    block_type: str,
    identity_field: str,
) -> None:
    source_id = f"missing-{block_type}"
    source_block = block(
        f"{block_type}-block",
        "",
        6,
        block_type=block_type,
        **{identity_field: source_id},
    )

    with pytest.raises(SourceFidelityError) as raised:
        make_chunker(parent_max_tokens=100, child_max_tokens=100).build(
            document(source_block)
        )

    assert raised.value.document_id == "doc-1"
    assert raised.value.page == 6
    assert raised.value.block_type == block_type
    assert raised.value.source_id == source_id
    assert len(raised.value.excerpt) <= 200


@pytest.mark.parametrize("field_name", ["caption", "description"])
def test_formula_field_can_reconstruct_from_its_own_fragments(
    field_name: str,
) -> None:
    source = " ".join(f"{field_name}-{index}" for index in range(80))
    formula = CanonicalFormula(
        formula_id=f"formula-fragmented-{field_name}",
        latex="x = 1",
        caption=source if field_name == "caption" else None,
        description=source if field_name == "description" else None,
        source_spans=[span("formula-fragment-source", page=7)],
    )
    doc = document(
        block(
            "formula-fragment-block",
            "",
            7,
            block_type="formula",
            formula_id=formula.formula_id,
        )
    )
    doc.formulas = [formula]
    chunker = make_chunker(parent_max_tokens=300, child_max_tokens=12)
    drafts = chunker.build(doc)
    parent = next(item for item in drafts if item.chunk_role == "parent")
    _replace_draft_text(parent, source)

    chunker._audit_source_fidelity(doc, drafts)

    children = [item for item in drafts if item.chunk_role == "child"]
    assert "".join(
        item.metadata["source_fragment"]
        for item in children
        if item.metadata.get("source_fragment_kind") == field_name
    ) == source


def test_source_fidelity_audit_accepts_valid_mixed_document() -> None:
    table = valid_table()
    table.footnotes = ["Source footnote."]
    figure = CanonicalFigure(
        figure_id="mixed-figure",
        caption="Figure 2. Mixed evidence",
        description="Figure description.",
        asset_path="assets/mixed.png",
        source_spans=[span("mixed-figure-object", page=2)],
    )
    formula = CanonicalFormula(
        formula_id="mixed-formula",
        latex="a^2 + b^2 = c^2",
        caption="Equation 2.",
        description="Formula description.",
        source_spans=[span("mixed-formula-object", page=3)],
    )
    doc = document(
        block("mixed-narrative", "Narrative source evidence.", 0),
        block("mixed-table", "", 1, block_type="table", table_id=table.table_id),
        block("mixed-figure", "", 2, block_type="figure", figure_id=figure.figure_id),
        block("mixed-formula", "", 3, block_type="formula", formula_id=formula.formula_id),
    )
    doc.tables = [table]
    doc.figures = [figure]
    doc.formulas = [formula]

    chunks = make_chunker(parent_max_tokens=300, child_max_tokens=100).build(doc)

    assert {item.block_type for item in chunks if item.chunk_role == "child"} == {
        "narrative",
        "table",
        "figure",
        "formula",
    }


def test_structured_blocks_are_isolated_and_table_children_keep_precise_spans() -> None:
    doc = document(
        block("before", "ordinary paragraph before.", 0),
        block("table-block", "", 1, block_type="table", table_id="t1"),
        block("after", "ordinary paragraph after.", 2),
    )
    doc.tables = [valid_table()]

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=20).build(doc)
    table_parent = next(
        item for item in chunks if item.block_type == "table" and item.chunk_role == "parent"
    )
    table_children = [
        item for item in chunks if item.parent_local_id == table_parent.local_id
    ]

    assert table_children
    assert all("ordinary paragraph" not in item.text for item in table_children)
    assert all(item.source_block_ids == ["table-block"] for item in table_children)
    assert all(item.source_spans for item in table_children)
    assert [item.metadata["row_indices"] for item in table_children] == [[0], [1]]
    assert table_parent.metadata["boundary_reason"] == "structured_boundary"
    assert all(
        item.metadata["boundary_reason"] == "structured_boundary"
        for item in table_children
    )
    assert table_parent.semantic_boundary_score is None
    assert {item.source_block_id for item in table_parent.source_spans} == {
        "source-table",
        "table-block",
    }
    assert all(
        {span.source_block_id for span in item.source_spans} == {"source-table"}
        for item in table_children
    )


def test_figure_and_formula_have_source_faithful_children_and_generated_provenance() -> None:
    doc = document(
        block("figure-block", "", 0, block_type="figure", figure_id="f1"),
        block("formula-block", "", 1, block_type="formula", formula_id="q1"),
        block("figure-nearby", "Nearby figure source discussion.", 7),
        block("formula-nearby", "Nearby formula source discussion.", 8),
    )
    doc.figures = [
        CanonicalFigure(
            figure_id="f1",
            caption="Figure 1. Source curve",
            description="Source description.",
            source_spans=[span("figure-object")],
            nearby_block_ids=["figure-nearby"],
            analysis_status="complete",
            generated_summary="AI generated trend.",
            analysis_model="vision-test",
        )
    ]
    doc.formulas = [
        CanonicalFormula(
            formula_id="q1",
            latex="x = y + 1",
            caption="Equation 1.",
            source_spans=[span("formula-object")],
            nearby_block_ids=["formula-nearby"],
            analysis_status="complete",
            generated_explanation="AI generated explanation.",
            analysis_model="formula-test",
        )
    ]

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(doc)
    for block_type, generated, nearby_id, nearby_page, object_id, block_id in (
        (
            "figure",
            "AI generated trend.",
            "figure-nearby",
            7,
            "figure-object",
            "figure-block",
        ),
        (
            "formula",
            "AI generated explanation.",
            "formula-nearby",
            8,
            "formula-object",
            "formula-block",
        ),
    ):
        parent = next(
            item
            for item in chunks
            if item.block_type == block_type and item.chunk_role == "parent"
        )
        child = next(item for item in chunks if item.parent_local_id == parent.local_id)
        assert generated not in parent.text
        assert generated in parent.embedding_text
        assert child.text == parent.text
        assert child.source_spans == parent.source_spans
        assert child.metadata["source_faithful_single_child"] is True
        assert parent.metadata["boundary_reason"] == "structured_boundary"
        assert child.metadata["boundary_reason"] == "structured_boundary"
        assert nearby_id in parent.source_block_ids
        assert nearby_id in child.source_block_ids
        assert nearby_page in {item.page_index for item in parent.source_spans}
        assert object_id in {item.source_block_id for item in parent.source_spans}
        assert block_id in {item.source_block_id for item in parent.source_spans}


def test_overlong_formula_is_losslessly_split_into_direct_embedding_children() -> None:
    latex_rows = [f"x_{{{index}}} = y_{{{index}}} + z_{{{index}}}" for index in range(18)]
    formula = CanonicalFormula(
        formula_id="q-long",
        latex=r" \\ ".join(latex_rows),
        caption="Equation 9. Coupled system",
        description="Complete source formula.",
        source_spans=[span("formula-long-source")],
    )
    doc = document(
        block("formula-long", "", 0, block_type="formula", formula_id=formula.formula_id)
    )
    doc.formulas = [formula]
    chunker = make_chunker(parent_max_tokens=200, child_max_tokens=18)

    chunks = chunker.build(doc)

    parent = next(item for item in chunks if item.chunk_role == "parent")
    children = [item for item in chunks if item.parent_local_id == parent.local_id]
    assert all(row in parent.text for row in latex_rows)
    assert len(children) > 1
    assert all(word_count(item.text) <= 18 for item in children)
    assert all(item.token_count == word_count(item.text) for item in children)
    assert all(item.embedding_text == item.text for item in children)
    assert [item.metadata["structured_part_index"] for item in children] == list(
        range(len(children))
    )
    assert all(
        item.metadata["structured_part_count"] == len(children) for item in children
    )
    assert all(item.metadata["formula_id"] == "q-long" for item in children)
    assert "".join(
        item.metadata["source_fragment"]
        for item in children
        if item.metadata["source_fragment_kind"] == "latex"
    ) == formula.latex
    assert "".join(
        item.metadata["source_fragment"]
        for item in children
        if item.metadata["source_fragment_kind"] == "description"
    ) == formula.description


def test_overlong_formula_description_passes_final_fidelity_audit() -> None:
    description = " ".join(f"definition-{index}" for index in range(80))
    formula = CanonicalFormula(
        formula_id="q-long-description",
        latex="x = y + 1",
        caption="Equation 10.",
        description=description,
        source_spans=[span("formula-description-source")],
    )
    doc = document(
        block(
            "formula-description",
            "",
            0,
            block_type="formula",
            formula_id=formula.formula_id,
        )
    )
    doc.formulas = [formula]

    children = [
        item
        for item in make_chunker(parent_max_tokens=200, child_max_tokens=18).build(doc)
        if item.chunk_role == "child"
    ]

    assert all(item.token_count <= 18 for item in children)
    assert "".join(
        item.metadata["source_fragment"]
        for item in children
        if item.metadata["source_fragment_kind"] == "description"
    ) == description


def test_overlong_table_cell_is_split_without_overlimit_child() -> None:
    long_cell = " ".join(f"measurement-{index}" for index in range(80))
    table = CanonicalTable(
        table_id="table-long-cell",
        caption="Table 4. Complete measurements",
        headers=["Method", "Measurements"],
        rows=[["SAC-KG", long_cell]],
        cells=[
            CanonicalCell(row_index=0, column_index=0, text="Method", is_header=True),
            CanonicalCell(row_index=0, column_index=1, text="Measurements", is_header=True),
            CanonicalCell(row_index=1, column_index=0, text="SAC-KG"),
            CanonicalCell(row_index=1, column_index=1, text=long_cell),
        ],
        source_spans=[span("table-long-source")],
        status="accepted_mineru",
    )
    doc = document(
        block("table-long", "", 0, block_type="table", table_id=table.table_id)
    )
    doc.tables = [table]
    chunker = make_chunker(parent_max_tokens=300, child_max_tokens=20)

    chunks = chunker.build(doc)

    parent = next(item for item in chunks if item.chunk_role == "parent")
    children = [item for item in chunks if item.parent_local_id == parent.local_id]
    assert long_cell in parent.text
    assert len(children) > 1
    assert all(word_count(item.text) <= 20 for item in children)
    assert all(item.token_count == word_count(item.text) for item in children)
    assert all(item.embedding_text == item.text for item in children)
    assert all(item.metadata["table_id"] == table.table_id for item in children)
    assert all(item.metadata["row_indices"] == [0] for item in children)
    assert all("Table 4" in item.text and "Method" in item.text for item in children)
    assert "".join(item.metadata["source_fragment"] for item in children) == long_cell


def test_overlong_figure_source_description_is_losslessly_bounded() -> None:
    description = " ".join(f"trend-{index}" for index in range(75))
    figure = CanonicalFigure(
        figure_id="figure-long",
        caption="Figure 7. Long source description",
        description=description,
        asset_path="assets/figure-7.png",
        source_spans=[span("figure-long-source")],
    )
    doc = document(
        block("figure-long", "", 0, block_type="figure", figure_id=figure.figure_id)
    )
    doc.figures = [figure]
    chunker = make_chunker(parent_max_tokens=300, child_max_tokens=18)

    chunks = chunker.build(doc)

    parent = next(item for item in chunks if item.chunk_role == "parent")
    children = [item for item in chunks if item.parent_local_id == parent.local_id]
    assert description in parent.text
    assert len(children) > 1
    assert all(word_count(item.text) <= 18 for item in children)
    assert all(item.token_count == word_count(item.text) for item in children)
    assert all(item.embedding_text == item.text for item in children)
    assert all(item.metadata["figure_id"] == figure.figure_id for item in children)
    assert all("Figure 7" in item.text for item in children)
    assert "".join(item.metadata["source_fragment"] for item in children) == description


@pytest.mark.parametrize("block_type", ["figure", "formula"])
def test_structured_nearby_sources_share_retrievable_nonreference_allowlist(
    block_type: str,
) -> None:
    structure_id = "f-filter" if block_type == "figure" else "q-filter"
    structure = block(
        f"{block_type}-block",
        "",
        0,
        block_type=block_type,
        figure_id=structure_id if block_type == "figure" else None,
        formula_id=structure_id if block_type == "formula" else None,
    )
    allowed_narrative = block("allowed-narrative", "Allowed nearby narrative.", 1)
    reference_narrative = block(
        "reference-narrative",
        "LEAKED REFERENCE PAYLOAD.",
        2,
        section_path=["References"],
    )
    hidden_narrative = block(
        "hidden-narrative",
        "LEAKED NONRETRIEVABLE PAYLOAD.",
        3,
    )
    hidden_narrative.retrievable = False
    nearby_ids = [
        allowed_narrative.block_id,
        reference_narrative.block_id,
        hidden_narrative.block_id,
    ]
    doc = document(
        structure,
        allowed_narrative,
        reference_narrative,
        hidden_narrative,
    )
    if block_type == "figure":
        doc.figures = [
            CanonicalFigure(
                figure_id=structure_id,
                caption="Figure source caption.",
                source_spans=[span("figure-source")],
                nearby_block_ids=nearby_ids,
                generated_summary="Generated figure analysis.",
            )
        ]
    else:
        doc.formulas = [
            CanonicalFormula(
                formula_id=structure_id,
                latex="x = 1",
                caption="Formula source caption.",
                source_spans=[span("formula-source")],
                nearby_block_ids=nearby_ids,
                generated_explanation="Generated formula analysis.",
            )
        ]

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(doc)
    structured = [item for item in chunks if item.block_type == block_type]

    assert {item.chunk_role for item in structured} == {"parent", "child"}
    for item in structured:
        expected_caption = (
            "Figure source caption."
            if block_type == "figure"
            else "Formula source caption."
        )
        assert expected_caption in item.text
        assert "Allowed nearby narrative." in item.text
        assert "LEAKED REFERENCE PAYLOAD." not in item.text
        assert "LEAKED REFERENCE PAYLOAD." not in item.embedding_text
        assert "LEAKED NONRETRIEVABLE PAYLOAD." not in item.text
        assert "LEAKED NONRETRIEVABLE PAYLOAD." not in item.embedding_text
        assert "reference-narrative" not in item.source_block_ids
        assert "hidden-narrative" not in item.source_block_ids
        assert "allowed-narrative" in item.source_block_ids
        span_ids = {source_span.source_block_id for source_span in item.source_spans}
        assert "reference-narrative" not in span_ids
        assert "hidden-narrative" not in span_ids


@pytest.mark.parametrize("block_type", ["figure", "formula"])
def test_structured_nearby_sources_inherit_effective_reference_heading(
    block_type: str,
) -> None:
    structure_id = "f-heading" if block_type == "figure" else "q-heading"
    structure = block(
        f"{block_type}-heading-block",
        "",
        0,
        block_type=block_type,
        figure_id=structure_id if block_type == "figure" else None,
        formula_id=structure_id if block_type == "formula" else None,
    )
    heading = block(
        "references-heading",
        "References",
        1,
        block_type="heading",
        section_path=["References"],
    )
    inherited_reference = block(
        "inherited-reference",
        "LEAKED INHERITED REFERENCE PAYLOAD.",
        2,
    )
    inherited_reference.section_path = []
    doc = document(structure, heading, inherited_reference)
    if block_type == "figure":
        doc.figures = [
            CanonicalFigure(
                figure_id=structure_id,
                caption="Figure heading source.",
                nearby_block_ids=[inherited_reference.block_id],
            )
        ]
    else:
        doc.formulas = [
            CanonicalFormula(
                formula_id=structure_id,
                latex="y = 2",
                caption="Formula heading source.",
                nearby_block_ids=[inherited_reference.block_id],
            )
        ]

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(doc)
    structured = [item for item in chunks if item.block_type == block_type]

    assert structured
    for item in structured:
        assert "LEAKED INHERITED REFERENCE PAYLOAD." not in item.text
        assert "LEAKED INHERITED REFERENCE PAYLOAD." not in item.embedding_text
        assert inherited_reference.block_id not in item.source_block_ids
        assert inherited_reference.block_id not in {
            source_span.source_block_id for source_span in item.source_spans
        }


def test_nearby_caption_is_not_recorded_when_builder_does_not_consume_it() -> None:
    structure = block(
        "figure-caption-block",
        "",
        0,
        block_type="figure",
        figure_id="f-caption-filter",
    )
    nearby_caption = block(
        "nearby-caption",
        "Independent caption block.",
        1,
        block_type="caption",
    )
    doc = document(structure, nearby_caption)
    doc.figures = [
        CanonicalFigure(
            figure_id="f-caption-filter",
            caption="Canonical figure caption.",
            nearby_block_ids=[nearby_caption.block_id],
        )
    ]

    chunks = make_chunker(parent_max_tokens=100, child_max_tokens=100).build(doc)
    structured = [item for item in chunks if item.block_type == "figure"]

    assert structured
    for item in structured:
        assert "Canonical figure caption." in item.text
        assert "Independent caption block." not in item.text
        assert nearby_caption.block_id not in item.source_block_ids
        assert nearby_caption.block_id not in {
            source_span.source_block_id for source_span in item.source_spans
        }


def test_unmarked_nonmonotonic_counter_preserves_semantic_boundary() -> None:
    prefix_counts = {1: 1, 2: 3, 3: 2, 4: 3, 5: 3}

    def nonmonotonic_counter(text: str) -> int:
        unit_count = len(re.findall(r"S\d+\.", text))
        return prefix_counts.get(unit_count, unit_count)

    chunks = SemanticChunker(
        RecordingEmbedder(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        nonmonotonic_counter,
        parent_min_tokens=1,
        parent_target_tokens=3,
        parent_max_tokens=3,
        child_min_tokens=1,
        child_target_tokens=100,
        child_max_tokens=100,
        overlap_tokens=0,
        break_percentile=20,
    ).build(document(block("nonmonotonic", "S1. S2. S3. S4. S5.", 0)))
    parents = [item for item in chunks if item.chunk_role == "parent"]

    assert parents[0].text == "S1. S2. "
    assert parents[0].metadata["boundary_reason"] == "semantic_percentile"


def test_semantic_grouping_token_counter_work_is_near_linear() -> None:
    source = " ".join(f"S{index}." for index in range(1000))

    class InstrumentedMonotonicCounter:
        monotonic_prefix_counts = True

        def __init__(self) -> None:
            self.scanned_characters = 0

        def __call__(self, text: str) -> int:
            self.scanned_characters += len(text)
            return word_count(text)

    counter = InstrumentedMonotonicCounter()

    chunks = SemanticChunker(
        RecordingEmbedder(),
        counter,
        parent_min_tokens=1,
        parent_target_tokens=2000,
        parent_max_tokens=2000,
        child_min_tokens=1,
        child_target_tokens=2000,
        child_max_tokens=2000,
        overlap_tokens=0,
        break_percentile=20,
    ).build(document(block("complexity", source, 0)))

    assert {item.chunk_role for item in chunks} == {"parent", "child"}
    near_linear_scan_budget = len(source) * 64
    assert counter.scanned_characters <= near_linear_scan_budget
