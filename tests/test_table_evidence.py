from __future__ import annotations

from app.services.table_evidence import (
    CanonicalTableChunk,
    _is_numeric_cell,
    assemble_table_context,
    extract_table_facts,
)
from app.services.table_extraction import structure_table_markdown


def test_extract_table_facts_forward_fills_repeated_hierarchical_labels() -> None:
    """Blank continuation cells in a hierarchical row label inherit the parent.

    A table rendered with merged System cells repeats only the Simulation
    child (``C36m``) on continuation rows.  Facts must keep the parent so
    repeated child labels stay distinct instead of collapsing into one row.
    """
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 1: alpha L conformational sampling.\n"
                    "| System | Simulation | alpha_u probability (%) | alpha_L propensity Max. (%) | alpha_L propensity Max. length |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| HEWL19 peptide | C36 | 11 ± 7 | 12 ± 2 | 8 aa |\n"
                    "|  | C36m | 3 ± 2 | 5.6 ± 0.5 | 4 aa |\n"
                    "| FG-nucleoporin peptide | C36 | 32 ± 6 | 22 ± 2 | 14 aa |\n"
                    "|  | C36m | 1.1 ± 0.3 | 6.2 ± 0.2 | 5 aa |"
                ),
            )
        ]
    )

    facts = extract_table_facts("HEWL19 和 FG-nucleoporin 的 alpha L probability 数值？", table)

    value_by_row_column = {(fact.row_label, fact.column): fact.value for fact in facts}
    assert value_by_row_column[("HEWL19 peptide / C36m", "alpha_u probability (%)")] == "3 ± 2"
    assert value_by_row_column[("FG-nucleoporin peptide / C36m", "alpha_u probability (%)")] == "1.1 ± 0.3"
    assert value_by_row_column[("FG-nucleoporin peptide / C36", "alpha_u probability (%)")] == "32 ± 6"


def test_extract_table_facts_selects_requested_property_and_tail_row() -> None:
    """Generic table facts select requested property rows and a long-table tail row.

    The first column carries the row label but has a blank header (a common
    published-table layout).  The question asks for C6 and surface tension;
    the extraction must keep those property rows (including the tail row)
    without relying on a character excerpt, and only report the requested
    model columns.
    """
    lines = [
        "Table 1. Parameters of water models.",
        "|  | Expt | TIP3P | TIP4P-D |",
        "| --- | --- | --- | --- |",
    ]
    for i in range(40):
        lines.append(f"| Row {i} | {i}.0 | {i}.5 | {i}.9 |")
    lines.append("| C6 (kcal mol^-1 A^6) | 622 | 595 | 900 |")
    lines.append("| surface tension (mN m^-1) | 71.7 | 47.8 | 71.2 |")

    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text="\n".join(lines),
            )
        ]
    )

    facts = extract_table_facts("TIP4P-D 的 C6 偶极矩和表面张力数值？", table)

    labels = {fact.row_label for fact in facts}
    assert any("C6" in label for label in labels)
    assert any("surface tension" in label for label in labels)
    assert {fact.column for fact in facts} <= {"TIP3P", "TIP4P-D"}
    assert "900" in {fact.value for fact in facts}
    assert "71.2" in {fact.value for fact in facts}


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


def test_extract_table_facts_composes_grouped_opls_headers() -> None:
    chunks = [
        CanonicalTableChunk(
            chunk_id="binding-a",
            document_id="opls5",
            parse_version="canonical-v4",
            table_id="binding-rmse",
            ordinal=1,
            text=(
                "Table 9. Binding RMSE by system.\n"
                "| System | No.cmpds | RMSE (kcal/mol) |  |\n"
                "| --- | --- | --- | --- |\n"
                "|  |  | OPLS4 | OPLS5 |\n"
                "| HeterocycleFocused | 200 | 1.18 | 1.12 |"
            ),
        ),
        CanonicalTableChunk(
            chunk_id="binding-b",
            document_id="opls5",
            parse_version="canonical-v4",
            table_id="binding-rmse",
            ordinal=2,
            text=(
                "Table 9. Binding RMSE by system.\n"
                "| System | No.cmpds | RMSE (kcal/mol) |  |\n"
                "| --- | --- | --- | --- |\n"
                "|  |  | OPLS4 | OPLS5 |\n"
                "| WaterDisplacement | 65 | 1.19 | 1.13 |"
            ),
        ),
    ]

    table = assemble_table_context(chunks)
    facts = extract_table_facts(
        "OPLS5 的 binding RMSE 相比 OPLS4 有哪些数值改善？",
        table,
    )

    assert table.headers == (
        "System",
        "No.cmpds",
        "RMSE (kcal/mol) OPLS4",
        "RMSE (kcal/mol) OPLS5",
    )
    assert {(fact.column, fact.value) for fact in facts} >= {
        ("RMSE (kcal/mol) OPLS4", "1.18"),
        ("RMSE (kcal/mol) OPLS5", "1.12"),
        ("RMSE (kcal/mol) OPLS4", "1.19"),
        ("RMSE (kcal/mol) OPLS5", "1.13"),
    }
    assert {fact.source_chunk_ids for fact in facts} >= {
        ("binding-a",),
        ("binding-b",),
    }


def test_assembly_excludes_semantic_repeated_header_row() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 7\n"
                    "| Group | OPLS4 | OPLS5 |\n"
                    "| --- | --- | --- |\n"
                    "| Group | Edgewise | Pairwise |\n"
                    "| R-group | 0.93 | 1.06 |"
                ),
            )
        ]
    )

    assert table.rows == ({"Group": "R-group", "OPLS4": "0.93", "OPLS5": "1.06"},)
    assert "header_like_data_row" in table.quality_flags


def test_assembly_composes_multilevel_headers_columnwise_forward_fill() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 7\n"
                    "|  |  |  | Delta H vap | Delta H vap |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| liquid | T | E inter | calcd | exptl |\n"
                    "| methanol | 25.00 | 8.51 | 8.95 | 8.95c |"
                ),
            )
        ]
    )

    assert "Delta H vap calcd" in table.headers
    assert "Delta H vap exptl" in table.headers
    assert table.row_count == 1
    assert table.rows[0].get("Delta H vap calcd") == "8.95"
    assert table.rows[0].get("Delta H vap exptl") == "8.95c"


def test_assembly_composes_three_level_headers_forward_fill() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 7\n"
                    "|  |  |  |  |  |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "|  |  |  | Delta H vap | Delta H vap |\n"
                    "| liquid | T | E inter | calcd | exptl |\n"
                    "| methanol | 25.00 | 8.51 | 8.95 | 8.95c |"
                ),
            )
        ]
    )

    assert "Delta H vap calcd" in table.headers
    assert "Delta H vap exptl" in table.headers
    assert table.row_count == 1
    assert table.rows[0].get("Delta H vap calcd") == "8.95"
    assert table.rows[0].get("Delta H vap exptl") == "8.95c"


def test_extract_table_facts_preserves_hierarchical_row_labels() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 1: alpha L conformational sampling.\n"
                    "| System | Simulation | alpha L probability (%) | alpha L propensity Max. | alpha L length |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| RS peptide | C36 | 80 ± 2 | 41 ± 1 | 17 aa |\n"
                    "| RS peptide | C36m | 1.8 ± 0.5 | 5.5 ± 0.2 | 5 aa |\n"
                    "| FG-nucleoporin peptide | C36m | 1.1 ± 0.3 | 6.2 ± 0.2 | 5 aa |\n"
                    "| HEWL19 peptide | C36m | 0.5 ± 0.4 | 6.1 ± 0.7 | 3 aa |"
                ),
            )
        ]
    )

    facts = extract_table_facts(
        "RS peptide 和 FG-nucleoporin peptide 以及 HEWL19 的 alpha L probability 数值？",
        table,
    )

    value_by_label = {fact.row_label: fact.value for fact in facts}
    assert value_by_label["RS peptide / C36"] == "80 ± 2"
    assert value_by_label["RS peptide / C36m"] == "1.8 ± 0.5"
    assert value_by_label["FG-nucleoporin peptide / C36m"] == "1.1 ± 0.3"
    assert value_by_label["HEWL19 peptide / C36m"] == "0.5 ± 0.4"


def test_extract_table_facts_prefers_model_label_for_ranked_table() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="rank-1",
                document_id="doc-1",
                parse_version="canonical-v4",
                table_id="table-11",
                ordinal=0,
                text=(
                    "Table 11: benchmark accuracy by model.\n"
                    "| Rank | Model | Accuracy |\n"
                    "| --- | --- | --- |\n"
                    "| 1 | GPT-4 | 92.3 |\n"
                    "| 2 | Claude-3 | 95.1 |"
                ),
            )
        ]
    )

    facts = extract_table_facts(
        "What is the accuracy of Claude-3 in Table 11?",
        table,
    )

    assert {fact.row_label for fact in facts} == {"Claude-3"}
    assert {fact.value for fact in facts} == {"95.1"}
    assert all(fact.row_label != "2" for fact in facts)


def test_assembly_keeps_fully_text_rows_repeating_leading_label() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 1\n"
                    "| Model | A | B |\n"
                    "| --- | --- | --- |\n"
                    "| Model | Alpha | Beta |\n"
                    "| Model | Gamma | Delta |"
                ),
            )
        ]
    )

    assert table.row_count == 2
    assert table.rows == (
        {"Model": "Model", "A": "Alpha", "B": "Beta"},
        {"Model": "Model", "A": "Gamma", "B": "Delta"},
    )
    assert "header_like_data_row" not in table.quality_flags


def test_structure_table_markdown_composes_semantic_secondary_header() -> None:
    structured = structure_table_markdown(
        "Table 7. OPLS-AA Energetic Results.\n"
        "|  |  |  | Delta H vap | Delta H vap |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| liquid | T | E inter | calcd | exptl |\n"
        "| methanol | 25.00 | 8.51 | 8.95 | 8.95c |"
    )

    assert "Delta H vap calcd" in structured.headers
    assert "Delta H vap exptl" in structured.headers
    assert structured.rows == [
        {
            "liquid": "methanol",
            "T": "25.00",
            "E inter": "8.51",
            "Delta H vap calcd": "8.95",
            "Delta H vap exptl": "8.95c",
        }
    ]
    assert "calcd" not in {value for row in structured.rows for value in row.values()}


def test_structure_table_markdown_composes_three_level_headers() -> None:
    structured = structure_table_markdown(
        "Table 7. OPLS-AA Energetic Results.\n"
        "|  |  |  | Delta H vap | Delta H vap |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| liquid | T | E inter |  |  |\n"
        "|  |  |  | calcd | exptl |\n"
        "| methanol | 25.00 | 8.51 | 8.95 | 8.95c |"
    )

    assert "Delta H vap calcd" in structured.headers
    assert "Delta H vap exptl" in structured.headers
    assert structured.rows == [
        {
            "liquid": "methanol",
            "T": "25.00",
            "E inter": "8.51",
            "Delta H vap calcd": "8.95",
            "Delta H vap exptl": "8.95c",
        }
    ]
    assert "calcd" not in {value for row in structured.rows for value in row.values()}
    assert "header_like_data_row" in structured.quality_flags


def test_structure_table_markdown_composes_metric_row() -> None:
    structured = structure_table_markdown(
        "Table 9: Dataset metrics.\n"
        "| Model | NYT | WEB |\n"
        "| --- | --- | --- |\n"
        "| Model | F1 | AUC |\n"
        "| OPLS5 | 21.0 | 8.95 |"
    )

    assert structured.headers == ["Model", "NYT F1", "WEB AUC"]
    assert structured.rows == [{"Model": "OPLS5", "NYT F1": "21.0", "WEB AUC": "8.95"}]
    assert "header_like_data_row" not in structured.quality_flags


def test_structure_table_markdown_drops_semantic_repeated_header() -> None:
    structured = structure_table_markdown(
        "Table 7.\n"
        "| Group | OPLS4 | OPLS5 |\n"
        "| --- | --- | --- |\n"
        "| Group | Edgewise | Pairwise |\n"
        "| R-group | 0.93 | 1.06 |"
    )

    assert structured.headers == ["Group", "OPLS4", "OPLS5"]
    assert structured.rows == [{"Group": "R-group", "OPLS4": "0.93", "OPLS5": "1.06"}]
    assert "header_like_data_row" in structured.quality_flags


def test_assembly_keeps_text_only_fills_row() -> None:
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 1\n"
                    "| Group |  |  |\n"
                    "| --- | --- | --- |\n"
                    "| Group | Alpha | Beta |\n"
                    "| Group | Gamma | Delta |"
                ),
            )
        ]
    )

    assert table.row_count == 2
    assert "header_like_data_row" not in table.quality_flags


def test_is_numeric_cell_accepts_flattened_error_forms() -> None:
    """OCR/LaTeX-flattened numeric cells are still numeric.

    MinerU can drop the ``±`` marker (``2.0 0.2``) or split a sign from its
    number (``+ 0.9 0.2``); these must stay value cells so their columns are
    never absorbed into the row label.
    """
    for value in (
        "2.0 0.2",
        "+ 0.9 0.2",
        "- 1.7 0.3",
        "0.19 5 - 0.84 6",
        "582 000",
        "1.8±0.2",
    ):
        assert _is_numeric_cell(value), value


def test_is_numeric_cell_keeps_plain_signed_percent_plusminus_forms() -> None:
    """Plain numbers, signed values, percentages, and ``±`` pairs stay numeric."""
    for value in (
        "21.0",
        "-3.2",
        "+1.5",
        "92.1%",
        "1.8 ± 0.2",
        "5.6 ± 0.5",
        "0.0",
        "1.08",
        "0.40",
    ):
        assert _is_numeric_cell(value), value
    for value in (
        "C6",
        "OPLS4",
        "RMS error",
        "alpha L probability (%)",
        "5 aa",
        "Ile",
        "N - C",
        "gamma 2",
        "",
    ):
        assert not _is_numeric_cell(value), value


def test_opls4_table5_flattened_rows_are_data_rows_and_extract_facts() -> None:
    """OPLS4 Table 5-style rows with flattened errors stay data rows.

    Every value cell is LaTeX-flattened (``2.0 0.2``, ``+ 0.9 0.2``).  The
    model column must remain the row label and the numeric facts must remain
    extractable; otherwise the value columns collapse into the row label and
    no facts are produced.
    """
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t5",
                ordinal=0,
                text=(
                    "Table 5. Errors (kcal/mol) over the asp/glu pKa set.\n"
                    "| model | RMS error | MSE | MSE (chi1 = 60) | MSE (chi1 = 180) | MSE (chi1 = 300) |\n"
                    "| --- | --- | --- | --- | --- | --- |\n"
                    "| OPLS4* | 1.8 0.2 | + 0.9 0.2 | + 0.8 0.3 | + 1.7 0.3 | + 0.5 0.2 |\n"
                    "| OPLS3e | 2.0 0.2 | + 0.8 0.2 | + 0.5 0.4 | + 1.4 0.3 | + 0.5 0.2 |\n"
                    "| OPLS4 | 1.2 0.1 | + 0.2 0.1 | + 0.3 0.2 | + 0.1 0.2 | + 0.2 0.2 |"
                ),
            )
        ]
    )

    assert {row["model"] for row in table.rows} == {"OPLS4*", "OPLS3e", "OPLS4"}
    facts = extract_table_facts(
        "OPLS4 的 pKa 表格中，OPLS3e 到 OPLS4 的 RMS error 和 MSE 误差改善是多少？",
        table,
    )
    value_by_cell = {(fact.row_label, fact.column): fact.value for fact in facts}
    assert value_by_cell[("OPLS3e", "RMS error")] == "2.0 0.2"
    assert value_by_cell[("OPLS3e", "MSE")] == "+ 0.8 0.2"
    assert value_by_cell[("OPLS4", "RMS error")] == "1.2 0.1"
    assert value_by_cell[("OPLS4", "MSE")] == "+ 0.2 0.1"


def test_ff99sb_water_model_keeps_flattened_property_row_and_aliases() -> None:
    """ff99SB water-model property rows survive flattened cells and aliases.

    The C12 row carries space-flattened thousands values (``582 000``) that
    must not be classified as a semantic header row and dropped.  The
    surface-tension property must still resolve through the Chinese alias
    ``表面张力`` (gamma) with TIP3P/TIP4P-D columns intact.
    """
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table 1. Parameters and Physical Properties of Water Models.\n"
                    "|  | Expt | TIP3P | SPC/E | TIP4P-EW | TIP4P/2005 | TIP4P-D |\n"
                    "| --- | --- | --- | --- | --- | --- | --- |\n"
                    "| µ(D) | 1.85 | 2.35 | 2.27 | 2.32 | 2.30 | 2.403 |\n"
                    "| C6 (kcal mol^-1 A^6) | 622 | 595 | 625 | 653 | 736 | 900 |\n"
                    "| C12 (kcal mol^-1 A^12) |  | 582 000 | 629 482 | 656 138 | 731 380 | 904 657 |\n"
                    "| γ (mN m^-1) | 71.7 | 47.8 | 58.4 | 59.2 | 63.3 | 71.2 |"
                ),
            )
        ]
    )

    assert any("C12" in str(row.get("") or "") for row in table.rows)
    facts = extract_table_facts(
        "TIP4P-D 在水模型参数表中相对 TIP3P 的 C6、偶极矩和表面张力数值是什么？",
        table,
    )
    value_by_cell = {(fact.row_label, fact.column): fact.value for fact in facts}
    assert value_by_cell[("γ (mN m^-1)", "TIP3P")] == "47.8"
    assert value_by_cell[("γ (mN m^-1)", "TIP4P-D")] == "71.2"
    assert value_by_cell[("C6 (kcal mol^-1 A^6)", "TIP3P")] == "595"
    assert value_by_cell[("C6 (kcal mol^-1 A^6)", "TIP4P-D")] == "900"


def test_ff99sb_ildn_table_i_preserves_hierarchical_labels_and_aliases() -> None:
    """ff99SB-ILDN Table I-style rows keep hierarchical labels and aliases.

    The ``Res. / Angle`` columns form a hierarchical row label and the theta
    alias resolves to the ``theta0`` column while the flattened k values stay
    extractable.
    """
    table = assemble_table_context(
        [
            CanonicalTableChunk(
                chunk_id="c1",
                document_id="d1",
                parse_version="v1",
                table_id="t1",
                ordinal=0,
                text=(
                    "Table I List of Modified Parameters for the χ1 and χ2 Torsion Potentials.\n"
                    "| Res. | Angle | theta0 | k2 | k3 |\n"
                    "| --- | --- | --- | --- | --- |\n"
                    "| Ile | N - C^alpha - C^beta - C^gamma 2 | 0.0 | 0.19 5 - 0.84 6 |  |\n"
                    "| Leu | C - C^alpha - C^beta - C^gamma | 0.0 | 0.57 1 - 0.35 8 | 0.135 |\n"
                ),
            )
        ]
    )

    facts = extract_table_facts(
        "ff99SB-ILDN 的 Table I 中 Ile 和 Leu 的 theta0 和 k2 数值？",
        table,
    )
    value_by_cell = {(fact.row_label, fact.column): fact.value for fact in facts}
    assert value_by_cell["Ile / N - C^alpha - C^beta - C^gamma 2", "theta0"] == "0.0"
    assert value_by_cell["Ile / N - C^alpha - C^beta - C^gamma 2", "k2"] == "0.19 5 - 0.84 6"
    assert value_by_cell["Leu / C - C^alpha - C^beta - C^gamma", "theta0"] == "0.0"
