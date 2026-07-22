from __future__ import annotations

import math
import re

import pytest

from app.services.canonical_models import (
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SourceSpan,
)
from app.services.semantic_chunking import ChunkDraft, SemanticChunker


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
    )
    assert draft.model_dump(mode="json")["source_spans"][0]["source_block_id"] == "b1"


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
    assert embedder.calls[0] == [f"S{index} evidence." for index in range(1, 7)]
    assert len(parents) == 2
    assert parents[0].text == "S1 evidence. S2 evidence. S3 evidence."


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
    assert parents[1].text == "第一句。 第二句！ 第三句？"


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


def test_child_tail_is_merged_when_it_fits_and_single_sentence_overflow_is_marked() -> None:
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
    overflow_child = next(item for item in overflow_chunks if item.chunk_role == "child")
    assert overflow_child.token_count == 11
    assert overflow_child.metadata["single_sentence_overflow"] is True
    assert overflow_child.text == long_sentence


def test_child_neighbors_are_scoped_to_their_parent() -> None:
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

    for parent in parents:
        children = [item for item in chunks if item.parent_local_id == parent.local_id]
        assert children
        assert children[0].previous_child_local_id is None
        assert children[-1].next_child_local_id is None
        allowed = {item.local_id for item in children}
        assert all(
            item.previous_child_local_id is None or item.previous_child_local_id in allowed
            for item in children
        )
        assert all(
            item.next_child_local_id is None or item.next_child_local_id in allowed
            for item in children
        )


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

    chunks = make_chunker().build(doc)

    assert all(item.source_block_ids != ["reference-table-block"] for item in chunks)
    assert any(item.source_block_ids == ["appendix-table-block"] for item in chunks)


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
        assert nearby_id in parent.source_block_ids
        assert nearby_id in child.source_block_ids
        assert nearby_page in {item.page_index for item in parent.source_spans}
        assert object_id in {item.source_block_id for item in parent.source_spans}
        assert block_id in {item.source_block_id for item in parent.source_spans}
