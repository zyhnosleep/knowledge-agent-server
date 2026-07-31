from __future__ import annotations

from app.services.table_evidence import (
    CanonicalTableChunk,
    assemble_table_context,
    extract_table_facts,
)


def test_assembly_keeps_rows_after_old_4000_character_boundary() -> None:
    rows = [f"| Model-{i} | {i}.123 |" for i in range(500)]
    chunks = [
        CanonicalTableChunk(
            chunk_id=f"row-{i}",
            document_id="doc-1",
            parse_version="canonical-v4",
            table_id="table-9",
            ordinal=i,
            text=("Table 9\n| Model | Value |\n| --- | --- |\n" + row),
        )
        for i, row in enumerate(rows)
    ]

    table = assemble_table_context(chunks)

    assert "Model-499" in table.markdown
    assert table.row_count == 500


def test_extract_table_facts_supports_arbitrary_columns_and_exact_values() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="row-1",
                document_id="doc-1",
                parse_version="canonical-v4",
                table_id="table-7",
                ordinal=1,
                text=(
                    "Table 7\n| Model | Asp | C6 | exptl |\n"
                    "| --- | --- | --- | --- |\n"
                    "| OPLS5 | 21.0 | 8.95 | 95.3 |"
                ),
            ),
        ]
    )

    facts = extract_table_facts(
        "What are the OPLS5 Asp, C6, and exptl values in Table 7?",
        table,
    )

    assert {fact.column for fact in facts} == {"Asp", "C6", "exptl"}
    assert {fact.value for fact in facts} == {"21.0", "8.95", "95.3"}
    assert all(fact.table_id == "table-7" for fact in facts)


def test_assembly_deduplicates_repeated_headers_but_preserves_source_rows() -> None:
    chunks = [
        CanonicalTableChunk(
            chunk_id="header-a",
            document_id="doc-1",
            parse_version="v1",
            table_id="table-1",
            ordinal=1,
            text="Table 1\n| Model | Score |\n| --- | --- |\n| A | 1.0 |",
        ),
        CanonicalTableChunk(
            chunk_id="header-b",
            document_id="doc-1",
            parse_version="v1",
            table_id="table-1",
            ordinal=2,
            text="| Model | Score |\n| --- | --- |\n| B | 2.0 |",
        ),
    ]

    table = assemble_table_context(chunks)

    assert table.markdown.count("| Model | Score |") == 1
    assert "| A | 1.0 |" in table.markdown
    assert "| B | 2.0 |" in table.markdown
