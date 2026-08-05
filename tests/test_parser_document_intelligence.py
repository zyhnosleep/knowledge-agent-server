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


def test_pdf_basic_validation_does_not_extract_page_text(monkeypatch) -> None:
    class Page:
        def extract_text(self):
            raise AssertionError("basic validation must not extract text")

    class Reader:
        is_encrypted = False
        pages = [Page(), Page()]

    monkeypatch.setattr(parser, "PdfReader", lambda _path: Reader())

    assert parser._validate_pdf_basic(Path("dummy.pdf")) == 2


def test_pdf_text_layer_is_best_effort_per_page(monkeypatch) -> None:
    class Page:
        def __init__(self, value=None, error=None):
            self.value = value
            self.error = error

        def extract_text(self):
            if self.error:
                raise self.error
            return self.value

    class Reader:
        is_encrypted = False
        pages = [Page("First page"), Page(error=RuntimeError("bad stream")), Page("Third page")]

    monkeypatch.setattr(parser, "PdfReader", lambda _path: Reader())

    texts, page_count, warnings = parser._extract_pdf_text_layer_best_effort(Path("dummy.pdf"))

    assert texts == ["First page", "", "Third page"]
    assert page_count == 3
    assert warnings == ["Unable to extract text from page 2: bad stream"]
    assert parser._extract_pdf_text_layer(Path("dummy.pdf")) == (texts, page_count)


def test_targeted_document_intelligence_preserves_real_page_labels(monkeypatch) -> None:
    monkeypatch.setattr(parser, "_render_pdf_pages", lambda path, dpi: [b"one", b"two", b"three"])
    analyzed_labels = []

    def analyze(**kwargs):
        analyzed_labels.append(kwargs["page_label"])
        return DocumentPagePayload(
            page_label=kwargs["page_label"],
            page_summary="Target page",
            sections=["Only selected page"],
            tables=[],
            formulas=[],
            figures=[],
            evidence_spans=["Only selected page"],
        )

    monkeypatch.setattr(parser, "_analyze_pdf_page", analyze)

    parsed = parser._parse_pdf_with_document_intelligence(
        Path("dummy.pdf"), ["one", "two", "three"], 3, page_indices={1}
    )

    assert parsed is not None
    assert analyzed_labels == ["2"]
    assert {chunk.page_label for chunk in parsed.chunks} == {"2"}
    assert parsed.metadata["pages"] == 3
    assert parsed.metadata["document_intelligence"]["page_indices"] == [1]


def test_targeted_document_intelligence_renders_only_requested_pages(monkeypatch) -> None:
    render_calls = []

    def render(path, dpi, page_indices=None):
        render_calls.append(None if page_indices is None else set(page_indices))
        return {1: b"page-two"}

    monkeypatch.setattr(parser, "_render_pdf_pages", render)
    monkeypatch.setattr(
        parser,
        "_analyze_pdf_page",
        lambda **kwargs: DocumentPagePayload(
            page_label=kwargs["page_label"],
            page_summary="Target page",
            sections=["Only target page"],
            evidence_spans=["Only target page"],
        ),
    )

    parsed = parser._parse_pdf_with_document_intelligence(
        Path("dummy.pdf"), ["one", "two", "three"], 3, page_indices={1}
    )

    assert parsed is not None
    assert render_calls == [{1}]
    assert [chunk.page_label for chunk in parsed.chunks] == ["2"]


def test_document_intelligence_chunk_is_not_truncated(monkeypatch) -> None:
    long_section = "evidence " * 700
    monkeypatch.setattr(parser, "_render_pdf_pages", lambda path, dpi: [b"page"])
    monkeypatch.setattr(
        parser,
        "_analyze_pdf_page",
        lambda **kwargs: DocumentPagePayload(
            page_label="1",
            page_summary="Long section",
            sections=[long_section],
            tables=[],
            formulas=[],
            figures=[],
            evidence_spans=[],
        ),
    )

    parsed = parser._parse_pdf_with_document_intelligence(Path("dummy.pdf"), [""], 1)

    assert parsed is not None
    assert parsed.chunks[0].text == long_section.strip()


def test_analyze_pdf_page_overrides_model_forged_provenance() -> None:
    class ForgingClient:
        def generate_structured_with_images(self, *_args, **_kwargs):
            return DocumentPagePayload.model_validate(
                {
                    "page_label": "1",
                    "analysis_source": "pypdf_text_layer_fallback",
                    "sections": ["Vision section"],
                }
            )

    analysis = parser._analyze_pdf_page(
        client=ForgingClient(),
        path=Path("paper.pdf"),
        page_label="1",
        image_bytes=b"page",
        raw_text="Source text",
        text_quality="high",
    )

    assert analysis.analysis_source == "document_intelligence"
    assert analysis.sections == ["Vision section"]


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


def test_html_table_to_markdown_preserves_formula_pipes_inside_one_cell() -> None:
    from app.services.canonical_quality import CanonicalQualityGate

    html = r"""
    <table>
      <tr><td>Setting</td><td>$\| \nabla E \| / E_h$</td></tr>
      <tr><td>Crude</td><td>$1 \times 10^{-2}$</td></tr>
    </table>
    """

    markdown = parser._html_table_to_markdown(html)

    assert CanonicalQualityGate._markdown_table_data(markdown) == (
        ["Setting", r"$\| \nabla E \| / E_h$"],
        [["Crude", r"$1 \times 10^{-2}$"]],
    )


def test_mineru_html_table_is_retained_and_canonicalized_from_source_grid(
    tmp_path: Path,
) -> None:
    from app.services import canonical_adapters
    from app.services.structured_evidence import TableValidator

    source_html = r"""<table>
      <tr><td>Constraint</td><td>Potential</td></tr>
      <tr><td>Angle</td><td>$V(\theta)=\frac{1}{2}k\left|r\right|^2$</td></tr>
        </table>"""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {
                "type": "table",
                "table_caption": ["TABLE III. Constraint potentials."],
                "table_body": source_html,
                "page_idx": 0,
            }
        ],
        page_count=1,
    )

    parsed_table = parsed.metadata["document_intelligence"]["tables"][0]
    assert parsed_table["source_html"] == source_html

    canonical = canonical_adapters._parsed_pdf_to_canonical(
        pdf_path,
        parsed,
        "mineru",
        1,
    )
    table = canonical.tables[0]

    assert table.caption == "TABLE III. Constraint potentials."
    assert table.headers == ["Constraint", "Potential"]
    assert table.rows == [["Angle", r"$V(\theta)=\frac{1}{2}k\left|r\right|^2$"]]
    assert table.source_html == source_html
    assert all(cell.source_spans for cell in table.cells)
    assert TableValidator().validate(table).accepted is True


def test_mineru_html_table_repairs_fragmented_numbers_without_losing_formula(
    tmp_path: Path,
) -> None:
    from app.services import canonical_adapters

    source_html = r"""<table>
      <tr><td>System</td><td>Exp.</td><td>Expression</td></tr>
      <tr><td>Aβ40</td><td>$1 2 . 0 \pm 1 . 3$</td><td>$V(\theta)=\frac{1}{2}k$</td></tr>
    </table>"""
    pdf_path = tmp_path / "fragmented-numbers.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {"type": "table", "table_body": source_html, "page_idx": 0}
        ],
        page_count=1,
    )

    canonical = canonical_adapters._parsed_pdf_to_canonical(
        pdf_path,
        parsed,
        "mineru",
        1,
    )

    assert canonical.tables[0].rows == [
        ["Aβ40", r"$12.0 \pm 1.3$", r"$V(\theta)=\frac{1}{2}k$"],
    ]


def test_mineru_structured_appendix_after_references_is_retrievable(
    tmp_path: Path,
) -> None:
    from app.services import canonical_adapters
    from app.services.semantic_chunking import SemanticChunker

    pdf_path = tmp_path / "author-manuscript.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {"type": "title", "text": "References", "page_idx": 0},
            {
                "type": "paragraph",
                "text": "1. Example A and Example B, Journal 2024, 1, 1-10.",
                "page_idx": 0,
            },
            {
                "type": "table",
                "table_caption": ["Table 2. Supporting measurements."],
                "table_body": (
                    "<table><tr><td>System</td><td>Value</td></tr>"
                    "<tr><td>ACTR</td><td>13.07</td></tr></table>"
                ),
                "page_idx": 1,
            },
        ],
        page_count=2,
    )
    table_markdown = parsed.metadata["document_intelligence"]["tables"][0][
        "markdown"
    ]
    parsed.chunks = [
        parser.ParsedChunk(
            ordinal=0,
            text="### References",
            heading="mineru-page-1-title",
            page_label="1",
        ),
        parser.ParsedChunk(
            ordinal=1,
            text="1. Example A and Example B, Journal 2024, 1, 1-10.",
            heading="mineru-page-1-paragraph",
            page_label="1",
        ),
        parser.ParsedChunk(
            ordinal=2,
            text=table_markdown,
            heading="mineru-page-2-table",
            page_label="2",
        ),
    ]

    canonical = canonical_adapters._parsed_pdf_to_canonical(
        pdf_path,
        parsed,
        "mineru",
        2,
    )

    reference = next(
        block for block in canonical.blocks if "Example A" in block.text
    )
    table = next(block for block in canonical.blocks if block.block_type == "table")
    assert reference.retrievable is False
    assert table.retrievable is True
    assert table.section_path != ["References"]

    chunks = SemanticChunker(
        lambda texts: [[1.0, 0.0] for _text in texts],
        lambda text: len(text.split()),
    ).build(canonical)
    assert any(
        chunk.chunk_role == "child" and chunk.block_type == "table"
        for chunk in chunks
    )


def test_mineru_v2_nested_table_content_survives_canonical_validation(
    tmp_path: Path,
) -> None:
    from app.services import canonical_adapters
    from app.services.structured_evidence import TableValidator

    table_iii_html = r"""<table>
      <tr><td>Constraint</td><td>Potential</td></tr>
      <tr><td>Angle</td><td>$V(\theta)=\frac{1}{2}k\left|r\right|^2$</td></tr>
    </table>"""
    table_iv_html = r"""<table>
      <tr><td>Setting</td><td>$\Delta E_{conv}/E_h$</td><td>$\|\nabla E\|/E_h a^{-1}$</td><td>Max. Cycles</td></tr>
      <tr><td>Crude</td><td>$5\times10^{-4}$</td><td>$1\times10^{-2}$</td><td>$N_{at}$</td></tr>
      <tr><td>Extreme</td><td>$5\times10^{-8}$</td><td>$5\times10^{-5}$</td><td>$20N_{at}$</td></tr>
    </table>"""
    payload = [
        [],
        [
            {
                "type": "table",
                "content": {
                    "table_caption": [
                        {"type": "text", "content": "TABLE III. Constraint "},
                        {"type": "equation_inline", "content": "V(r)"},
                        {"type": "text", "content": " potentials."},
                    ],
                    "table_footnote": [
                        {"type": "text", "content": "Distances are in angstrom."}
                    ],
                    "html": table_iii_html,
                    "image_source": {"path": "images/table-iii.jpg"},
                },
                "bbox": [69, 119, 486, 279],
            },
            {
                "type": "table",
                "content": {
                    "table_caption": [
                        {"type": "text", "content": "TABLE IV. Optimization using "},
                        {"type": "equation_inline", "content": "E _ { \\mathfrak { h } }"},
                        {"type": "text", "content": " thresholds."},
                    ],
                    "table_footnote": [],
                    "html": table_iv_html,
                    "image_source": {"path": "images/table-iv.jpg"},
                },
                "bbox": [510, 738, 929, 871],
            },
        ],
    ]
    content_list = parser._normalize_mineru_content_list(payload)
    pdf_path = tmp_path / "nested-v2.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=content_list,
        page_count=2,
        output_dir=tmp_path,
        content_list_path=tmp_path / "paper_content_list_v2.json",
    )
    parsed_tables = parsed.metadata["document_intelligence"]["tables"]

    assert [table["source_html"] for table in parsed_tables] == [
        table_iii_html,
        table_iv_html,
    ]
    assert parsed_tables[0]["markdown"].startswith(
        "TABLE III. Constraint V(r) potentials."
    )
    assert parsed_tables[0]["footnotes"] == ["Distances are in angstrom."]
    assert parsed_tables[0]["image_path"] == "images/table-iii.jpg"
    assert parsed_tables[0]["bbox"] == [69, 119, 486, 279]
    assert all(table["page_label"] == "2" for table in parsed_tables)

    canonical = canonical_adapters._parsed_pdf_to_canonical(
        pdf_path, parsed, "mineru", 2
    )
    table_iii, table_iv = canonical.tables

    assert table_iii.caption == "TABLE III. Constraint V(r) potentials."
    assert table_iii.footnotes == ["Distances are in angstrom."]
    assert table_iii.headers == ["Constraint", "Potential"]
    assert table_iii.rows == [
        ["Angle", r"$V(\theta)=\frac{1}{2}k\left|r\right|^2$"]
    ]
    assert table_iii.source_spans[0].bbox == (69.0, 119.0, 486.0, 279.0)
    assert table_iv.headers == [
        "Setting",
        r"$\Delta E_{conv}/E_h$",
        r"$\|\nabla E\|/E_h a^{-1}$",
        "Max. Cycles",
    ]
    assert table_iv.rows[-1] == [
        "Extreme",
        r"$5\times10^{-8}$",
        r"$5\times10^{-5}$",
        r"$20N_{at}$",
    ]
    assert all(TableValidator().validate(table).accepted for table in canonical.tables)


def test_mineru_html_table_spans_use_one_canonical_markdown_grid(
    tmp_path: Path,
) -> None:
    from app.services import canonical_adapters
    from app.services.canonical_quality import CanonicalQualityGate
    from app.services.structured_evidence import TableValidator

    source_html = """<table>
      <tr><td>Method</td><td colspan="2">Scores</td></tr>
      <tr><td rowspan="2">Base</td><td>80</td><td>90</td></tr>
      <tr><td>81</td><td>91</td></tr>
    </table>"""
    pdf_path = tmp_path / "spans.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {
                "type": "table",
                "table_caption": ["TABLE IV. Span-aware results."],
                "table_body": source_html,
                "page_idx": 0,
            }
        ],
        page_count=1,
    )

    canonical = canonical_adapters._parsed_pdf_to_canonical(
        pdf_path, parsed, "mineru", 1
    )
    table = canonical.tables[0]

    assert table.headers == ["Method", "Scores", ""]
    assert table.rows == [["Base", "80", "90"], ["", "81", "91"]]
    assert CanonicalQualityGate._markdown_table_data(table.source_markdown or "") == (
        table.headers,
        table.rows,
    )
    assert table.metadata["parser_source_markdown"] != table.source_markdown
    assert "rowspan" in (table.source_html or "")
    assert TableValidator().validate(table).accepted is True
    assert CanonicalQualityGate._invalid_table_reasons(table) == []


def test_mineru_html_table_always_uses_first_logical_row_as_header(
    tmp_path: Path,
) -> None:
    from app.services import canonical_adapters
    from app.services.structured_evidence import TableValidator

    source_html = """<table>
      <tr><td>Dataset</td><td>Score</td></tr>
      <tr><th>OIE2016</th><th>74.7</th></tr>
    </table>"""
    pdf_path = tmp_path / "mixed-header.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    parsed = parser._mineru_content_to_parsed_doc(
        path=pdf_path,
        content_list=[
            {"type": "table", "table_body": source_html, "page_idx": 0}
        ],
        page_count=1,
    )

    canonical = canonical_adapters._parsed_pdf_to_canonical(
        pdf_path, parsed, "mineru", 1
    )
    table = canonical.tables[0]

    assert table.headers == ["Dataset", "Score"]
    assert table.rows == [["OIE2016", "74.7"]]
    assert all(cell.is_header for cell in table.cells if cell.row_index == 0)
    assert TableValidator().validate(table).accepted is True


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
