from pathlib import Path

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
