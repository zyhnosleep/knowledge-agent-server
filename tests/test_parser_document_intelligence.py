from pathlib import Path
import subprocess

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
        {"type": "image", "image_caption": ["Figure 1 shows the SAC-KG workflow."], "page_idx": 1},
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
    assert any(chunk.page_label == "1" and "OpenIE6" in chunk.text for chunk in parsed.chunks)


def test_parse_pdf_with_mineru_returns_none_when_cli_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(parser.settings, "mineru_bin", "missing-mineru")
    monkeypatch.setattr(parser, "_resolve_mineru_binary", lambda value: None)

    parsed = parser._parse_pdf_with_mineru(tmp_path / "dummy.pdf", page_count=1)

    assert parsed is None


def test_parse_pdf_with_mineru_reads_content_list(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(parser.settings, "mineru_bin", "mineru")
    monkeypatch.setattr(parser.settings, "mineru_backend", "pipeline")
    monkeypatch.setattr(parser.settings, "mineru_model_source", "modelscope")
    monkeypatch.setattr(parser.settings, "mineru_output_dir", tmp_path / "mineru-cache")
    monkeypatch.setattr(parser.settings, "mineru_timeout", 30)
    monkeypatch.setattr(parser.settings, "mineru_extra_args", "")
    monkeypatch.setattr(parser, "_resolve_mineru_binary", lambda value: "mineru")

    def fake_run(command, cwd, env, capture_output, text, timeout, check):
        output_dir = Path(command[command.index("-o") + 1])
        result_dir = output_dir / "paper" / "auto"
        result_dir.mkdir(parents=True)
        (result_dir / "paper_content_list_v2.json").write_text(
            '[{"type":"text","text":"MinerU extracted this page.","page_idx":0}]',
            encoding="utf-8",
        )
        assert env["MINERU_MODEL_SOURCE"] == "modelscope"
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(parser.subprocess, "run", fake_run)

    parsed = parser._parse_pdf_with_mineru(tmp_path / "paper.pdf", page_count=1)

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
