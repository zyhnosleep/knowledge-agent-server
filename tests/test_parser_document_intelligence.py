import json
from pathlib import Path
import subprocess

import pytest

from app.services.ai import DocumentPagePayload
from app.services import parser


def test_classify_text_layer_quality_distinguishes_high_and_low() -> None:
    high_quality = "患者诊断为高血压，医生建议三个月后复查，并继续当前治疗方案。" * 8
    low_quality = "� � �"

    assert parser._classify_text_layer_quality(high_quality) == "high"
    assert parser._classify_text_layer_quality(low_quality) == "low"


def test_fuse_pdf_page_content_preserves_tables_and_formulas() -> None:
    analysis = DocumentPagePayload(
        page_label="1",
        page_summary="这是第一页摘要。",
        sections=["患者诊断为高血压。", "医生建议三个月后复查。"],
        tables=["| 项目 | 值 |\n| --- | --- |\n| 血压 | 140/90 |"],
        formulas=["HbA1c = 7.2%"],
        figures=["图1显示随访趋势。"],
        evidence_spans=["患者诊断为高血压。"],
    )

    fused = parser._fuse_pdf_page_content(
        page_label="1",
        raw_text="患者诊断为高血压。医生建议三个月后复查。",
        text_quality="high",
        analysis=analysis,
    )

    assert "### Tables" in fused["page_markdown"]
    assert "HbA1c = 7.2%" in fused["page_markdown"]
    assert any("table" in (heading or "") for _, heading in fused["chunk_blocks"])


def test_parse_pdf_with_document_intelligence_uses_page_outputs(monkeypatch) -> None:
    monkeypatch.setattr(parser, "_render_pdf_pages", lambda path, dpi: [b"fake-page"])
    monkeypatch.setattr(
        parser,
        "_analyze_pdf_page",
        lambda **kwargs: DocumentPagePayload(
            page_label="1",
            page_summary="第一页摘要",
            sections=["第一节：主要诊断", "第二节：治疗建议"],
            tables=["| 项目 | 值 |\n| --- | --- |\n| 复查时间 | 3个月 |"],
            formulas=[],
            figures=[],
            evidence_spans=["医生建议患者在3个月后进行复查。"],
        ),
    )

    parsed = parser._parse_pdf_with_document_intelligence(Path("dummy.pdf"), ["患者诊断为高血压。"], 1)

    assert parsed is not None
    assert parsed.metadata["parser_mode"] == "pdf_document_intelligence"
    assert parsed.metadata["document_intelligence"]["enabled"] is True
    assert parsed.chunks
    assert any("复查时间" in chunk.text for chunk in parsed.chunks)


def test_document_intelligence_merges_small_sections_on_the_same_page(monkeypatch) -> None:
    monkeypatch.setattr(parser, "_render_pdf_pages", lambda path, dpi: [b"fake-page"])
    sections = [f"Sentence {index} " + "evidence " * 8 for index in range(12)]
    monkeypatch.setattr(
        parser,
        "_analyze_pdf_page",
        lambda **kwargs: DocumentPagePayload(
            page_label="1",
            page_summary="Page summary",
            sections=sections,
            tables=[],
            formulas=[],
            figures=[],
            evidence_spans=sections,
        ),
    )

    parsed = parser._parse_pdf_with_document_intelligence(
        Path("dummy.pdf"), [""], 1
    )

    assert parsed is not None
    assert len(parsed.chunks) < 8
    assert "Sentence 0" in parsed.chunks[0].text
    assert "Sentence 1" in parsed.chunks[0].text


def test_parse_pdf_with_document_intelligence_returns_none_when_rendering_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(parser, "_render_pdf_pages", lambda path, dpi: [])

    parsed = parser._parse_pdf_with_document_intelligence(Path("dummy.pdf"), ["患者诊断为高血压。"], 1)

    assert parsed is None


def test_mineru_content_to_parsed_doc_maps_structured_blocks(tmp_path) -> None:
    content_list = [
        {"type": "title", "text": "SAC-KG", "page_idx": 0},
        {"type": "text", "text": "SAC-KG introduces a generator, verifier, and pruner.", "page_idx": 0},
        {
            "type": "table",
            "table_caption": ["Table 1: Main results"],
            "table_body": "<table><tr><th>Model</th><th>F1</th></tr><tr><td>OpenIE6</td><td>42.05</td></tr></table>",
            "page_idx": 0,
        },
        {"type": "interline_equation", "text": "$F_1 = 2PR / (P + R)$", "page_idx": 0},
        {
            "type": "image",
            "image_caption": ["Figure 1 shows the SAC-KG workflow."],
            "image_note": "Generator, Verifier, and Pruner are connected.",
            "img_path": "images/figure-1.png",
            "page_idx": 1,
        },
    ]

    parsed = parser._mineru_content_to_parsed_doc(
        path=tmp_path / "knowledge-graph.pdf",
        content_list=content_list,
        page_count=2,
    )

    intelligence = parsed.metadata["document_intelligence"]
    assert parsed.metadata["parser_mode"] == "pdf_mineru"
    assert intelligence["engine"] == "mineru"
    assert "| Model | F1 |" in intelligence["tables"][0]["markdown"]
    assert intelligence["formulas"][0]["text"] == "$F_1 = 2PR / (P + R)$"
    assert intelligence["figures"][0]["page_label"] == "2"
    assert intelligence["figures"][0]["caption"] == "Figure 1 shows the SAC-KG workflow."
    assert intelligence["figures"][0]["note"] == "Generator, Verifier, and Pruner are connected."
    assert intelligence["figures"][0]["image_path"] == "images/figure-1.png"
    assert intelligence["figures"][0]["path"] == "images/figure-1.png"
    assert any(chunk.page_label == "1" and "OpenIE6" in chunk.text for chunk in parsed.chunks)
    assert any(
        chunk.page_label == "2"
        and "Figure evidence" in chunk.text
        and "Generator, Verifier, and Pruner" in chunk.text
        and "images/figure-1.png" in chunk.text
        for chunk in parsed.chunks
    )


def test_mineru_merges_small_adjacent_text_blocks_on_the_same_page(tmp_path) -> None:
    content_list = [
        {"type": "paragraph", "text": f"Sentence {index} " + "evidence " * 8, "page_idx": 0}
        for index in range(12)
    ]

    parsed = parser._mineru_content_to_parsed_doc(
        path=tmp_path / "fragmented.pdf",
        content_list=content_list,
        page_count=1,
    )

    assert len(parsed.chunks) < len(content_list)
    assert all(chunk.page_label == "1" for chunk in parsed.chunks)
    assert "Sentence 0" in parsed.chunks[0].text
    assert "Sentence 1" in parsed.chunks[0].text
    assert sum(len(chunk.text) for chunk in parsed.chunks) >= 900


def test_mineru_real_content_list_v2_fixture_recovers_tables_and_images() -> None:
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "mineru" / "knowledge_graph_auto_subset"
    content_list_path = fixture_dir / "content_list_v2.json"
    markdown_path = parser._find_mineru_markdown(fixture_dir)
    if not content_list_path.exists():
        pytest.skip("MinerU content_list_v2 fixture is not present.")
    if markdown_path is None:
        pytest.skip("MinerU markdown fixture is not present.")

    payload = json.loads(content_list_path.read_text(encoding="utf-8"))
    content_list = parser._normalize_mineru_content_list(payload)
    page_count = len(payload) if isinstance(payload, list) else 0
    parsed = parser._mineru_content_to_parsed_doc(
        path=fixture_dir / "Knowledge graph.pdf",
        content_list=content_list,
        page_count=page_count,
        output_dir=fixture_dir,
        content_list_path=content_list_path,
    )
    parser._augment_mineru_parsed_doc_from_markdown(parsed, markdown_path)

    intelligence = parsed.metadata["document_intelligence"]
    tables = intelligence["tables"]
    figures = intelligence["figures"]
    figure_1 = next((figure for figure in figures if "Figure 1" in str(figure.get("caption") or "")), None)
    assert parsed.metadata["parser_mode"] == "pdf_mineru"
    assert tables
    assert figures
    assert figure_1 is not None

    image_path = figure_1["image_path"]
    assert image_path.startswith("images/")
    assert image_path.endswith(".jpg")
    assert (fixture_dir / image_path).is_file()

    combined_table_text = "\n".join(str(table.get("markdown") or "") for table in tables)
    combined_structured_text = json.dumps(intelligence["structured_tables"], ensure_ascii=False)
    combined_chunk_text = "\n".join(chunk.text for chunk in parsed.chunks)
    combined_text = "\n".join([combined_table_text, combined_structured_text, combined_chunk_text, parsed.text])
    assert "Table 5" in combined_text or "SAC-KG ChatGPT" in combined_text


def test_mineru_figure_paths_are_kept_within_output_root(tmp_path) -> None:
    output_dir = tmp_path / "mineru-output"
    content_dir = output_dir / "auto"
    content_dir.mkdir(parents=True)
    content_list_path = content_dir / "content_list.json"
    content_list_path.write_text("[]", encoding="utf-8")

    safe = parser._mineru_content_to_parsed_doc(
        path=tmp_path / "safe.pdf",
        content_list=[
            {
                "type": "image",
                "image_caption": "Safe figure",
                "img_path": "images/safe.png",
                "page_idx": 0,
            }
        ],
        page_count=1,
        output_dir=output_dir,
        content_list_path=content_list_path,
    )
    safe_figure = safe.metadata["document_intelligence"]["figures"][0]
    assert safe_figure["image_path"] == "images/safe.png"
    assert safe_figure["path"] == "images/safe.png"

    unsafe = parser._mineru_content_to_parsed_doc(
        path=tmp_path / "unsafe.pdf",
        content_list=[
            {"type": "image", "image_caption": "Traversal figure", "img_path": "../../secret.png", "page_idx": 0},
            {"type": "image", "image_caption": "Absolute figure", "img_path": str(tmp_path.parent / "secret.png"), "page_idx": 0},
            {"type": "image", "image_caption": "Remote figure", "img_path": "https://example.test/figure.png", "page_idx": 0},
        ],
        page_count=1,
        output_dir=output_dir,
        content_list_path=content_list_path,
    )
    unsafe_figures = unsafe.metadata["document_intelligence"]["figures"]

    assert all("image_path" not in figure for figure in unsafe_figures)
    assert all("path" not in figure for figure in unsafe_figures)
    assert "../../secret" not in unsafe.text
    assert "https://example.test" not in unsafe.text


def test_html_table_to_markdown_expands_colspan_and_rowspan() -> None:
    html = """
    <table>
      <tr><th rowspan="2">Model</th><th colspan="2">OIE2016</th><th colspan="2">NYT</th></tr>
      <tr><th>F1</th><th>AUC</th><th>F1</th><th>AUC</th></tr>
      <tr><td>SAC-KG ChatGPT</td><td>74.7</td><td>73.2</td><td>88.8</td><td>87.3</td></tr>
    </table>
    """

    markdown = parser._html_table_to_markdown(html)

    assert "| Model | OIE2016 | OIE2016 | NYT | NYT |" in markdown
    assert "| Model | F1 | AUC | F1 | AUC |" in markdown
    assert "| SAC-KG ChatGPT | 74.7 | 73.2 | 88.8 | 87.3 |" in markdown


def test_mineru_content_to_parsed_doc_normalizes_latex_table_cells(tmp_path) -> None:
    content_list = [
        {
            "type": "table",
            "table_caption": ["Table 5: Results"],
            "table_body": (
                "| Model | OIE2016 | WEB | NYT | PENN |  |  |  |  |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
                "|  | F1 | AUC | F1 | AUC | F1 | AUC | F1 | AUC |\n"
                "| $\\mathbf { S } \\mathbf { A } \\mathbf { C } \\mathbf { - } \\mathbf { K } \\mathbf { G } _ { \\mathrm { C h a t G P T } }$ | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |"
            ),
            "page_idx": 7,
        }
    ]

    parsed = parser._mineru_content_to_parsed_doc(
        path=tmp_path / "knowledge-graph.pdf",
        content_list=content_list,
        page_count=8,
    )

    table = parsed.metadata["document_intelligence"]["tables"][0]["markdown"]
    structured = parsed.metadata["document_intelligence"]["structured_tables"][0]
    assert "SAC-KG ChatGPT" in table
    assert "| Model | OIE2016 | OIE2016 | WEB | WEB | NYT | NYT | PENN | PENN |" in table
    assert "| SAC-KG ChatGPT | 74.7 | 73.2 | 96.6 | 95.7 | 88.8 | 87.3 | 91.1 | 90.1 |" in table
    assert structured["rows"][0]["NYT F1"] == "88.8"


def test_parse_pdf_with_mineru_returns_none_when_cli_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(parser.settings, "mineru_bin", "missing-mineru")
    monkeypatch.setattr(parser, "_resolve_mineru_binary", lambda value: None)

    parsed = parser._parse_pdf_with_mineru(tmp_path / "dummy.pdf", page_count=1)

    assert parsed is None


def test_build_mineru_command_normalizes_hybrid_backend_and_extra_args(tmp_path) -> None:
    command = parser._build_mineru_command(
        mineru_bin="mineru",
        source_path=tmp_path / "paper.pdf",
        output_dir=tmp_path / "mineru-output",
        backend="hybrid",
        extra_args='--effort high --start-page 0 --lang "ch, en"',
    )

    assert command[:5] == ["mineru", "-p", str(tmp_path / "paper.pdf"), "-o", str(tmp_path / "mineru-output")]
    assert command[5:7] == ["-b", "hybrid-engine"]
    assert command[7:] == ["--effort", "high", "--start-page", "0", "--lang", "ch, en"]


def test_build_mineru_command_keeps_pipeline_backend(tmp_path) -> None:
    command = parser._build_mineru_command(
        mineru_bin="mineru",
        source_path=tmp_path / "paper.pdf",
        output_dir=tmp_path / "mineru-output",
        backend="pipeline",
        extra_args="",
    )

    assert command[5:7] == ["-b", "pipeline"]


def test_parse_pdf_with_mineru_reads_content_list(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(parser.settings, "mineru_bin", "mineru")
    monkeypatch.setattr(parser.settings, "mineru_backend", "pipeline")
    monkeypatch.setattr(parser.settings, "mineru_model_source", "modelscope")
    monkeypatch.setattr(parser.settings, "mineru_output_dir", tmp_path / "mineru-cache")
    monkeypatch.setattr(parser.settings, "mineru_timeout", 30)
    monkeypatch.setattr(parser.settings, "mineru_extra_args", "")
    monkeypatch.setattr(parser, "_resolve_mineru_binary", lambda value: "mineru")

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    def fake_run(command, cwd, env, capture_output, text, encoding, errors, timeout, check):
        pdf_arg = Path(command[command.index("-p") + 1])
        output_dir = Path(command[command.index("-o") + 1])
        assert pdf_arg.is_absolute()
        assert output_dir.is_absolute()
        assert cwd == str(pdf_path.parent.resolve())
        assert command[command.index("-b") + 1] == "pipeline"
        result_dir = output_dir / "paper" / "auto"
        result_dir.mkdir(parents=True)
        (result_dir / "paper_content_list_v2.json").write_text(
            '[{"type":"text","text":"MinerU extracted this page.","page_idx":0}]',
            encoding="utf-8",
        )
        assert env["MINERU_MODEL_SOURCE"] == "modelscope"
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(parser.subprocess, "run", fake_run)

    parsed = parser._parse_pdf_with_mineru(pdf_path, page_count=1)

    assert parsed is not None
    assert parsed.metadata["parser_mode"] == "pdf_mineru"
    assert "MinerU extracted this page." in parsed.text


def test_normalize_mineru_content_list_flattens_page_blocks() -> None:
    payload = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {"type": "title", "text": "Nested title"},
                    {"type": "text", "text": "Nested paragraph"},
                ],
            }
        ]
    }

    flattened = parser._normalize_mineru_content_list(payload)

    assert len(flattened) == 2
    assert flattened[0]["page_idx"] == 0
    assert flattened[1]["text"] == "Nested paragraph"


def test_normalize_mineru_content_list_supports_v2_page_lists() -> None:
    payload = [
        [
            {"type": "title", "content": {"title_content": "SAC-KG"}},
            {"type": "paragraph", "content": {"paragraph_content": "The generator creates triples."}},
        ],
        [
            {"type": "table", "content": {"table_content": "| Model | F1 |\n| --- | --- |\n| OpenIE6 | 42.05 |"}},
            {"type": "equation", "content": {"math_content": "$F_1$"}},
        ],
    ]

    flattened = parser._normalize_mineru_content_list(payload)
    parsed = parser._mineru_content_to_parsed_doc(
        path=Path("paper.pdf"),
        content_list=flattened,
        page_count=2,
    )

    intelligence = parsed.metadata["document_intelligence"]
    assert len(flattened) == 4
    assert flattened[0]["page_idx"] == 0
    assert flattened[2]["page_idx"] == 1
    assert "SAC-KG" in parsed.text
    assert "generator creates triples" in parsed.text
    assert intelligence["tables"][0]["page_label"] == "2"
    assert intelligence["formulas"][0]["text"] == "$F_1$"


def test_augment_mineru_parsed_doc_from_markdown_recovers_tables(tmp_path) -> None:
    parsed = parser._mineru_content_to_parsed_doc(
        path=tmp_path / "paper.pdf",
        content_list=[{"type": "paragraph", "content": {"paragraph_content": "Ablation study is discussed."}, "page_idx": 4}],
        page_count=5,
    )
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        "\n".join(
            [
                "## Page 5",
                "Table 2: Ablation results",
                "| Variant | F1 | AUC |",
                "| --- | --- | --- |",
                "| SAC-KG w/o verifier | 68.1 | 70.2 |",
            ]
        ),
        encoding="utf-8",
    )

    parser._augment_mineru_parsed_doc_from_markdown(parsed, markdown_path)

    tables = parsed.metadata["document_intelligence"]["tables"]
    assert any("Table 2" in table["markdown"] for table in tables)
    assert any("68.1" in chunk.text for chunk in parsed.chunks)
