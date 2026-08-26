from __future__ import annotations

import pytest

from app.services.parser import DocumentParseError, _extract_pdf_text_layer, parse_document


def test_parse_document_wraps_malformed_pdf_errors(tmp_path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"not a real pdf")

    with pytest.raises(DocumentParseError) as exc_info:
        parse_document(pdf_path)

    assert "broken.pdf" in str(exc_info.value)
    assert exc_info.value.path == pdf_path


def _pdf_with_nul_text() -> bytes:
    """构造文本层含 NUL (0x00) 的最小单页 PDF。

    复刻线上故障：pypdf 提取出的页面文本含 NUL 字节，直接入库时
    PostgreSQL 拒绝（text 字段不能含 0x00）→ 附件上传 500。
    """
    content = b"BT /F1 12 Tf 72 720 Td (alpha\x00beta) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(body))
        body += b"%d 0 obj\n" % index + obj + b"\nendobj\n"
    xref_offset = len(body)
    xref = b"xref\n0 %d\n" % (len(objects) + 1)
    xref += b"0000000000 65535 f \n"
    for off in offsets:
        xref += b"%010d 00000 n \n" % off
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    return body + xref + trailer + b"startxref\n%d\n%%%%EOF\n" % xref_offset


def test_extract_pdf_text_layer_strips_nul_bytes(tmp_path) -> None:
    """PDF 文本层里的 NUL (0x00) 必须被清洗，否则 PG 入库即炸。"""
    pdf_path = tmp_path / "nul.pdf"
    pdf_path.write_bytes(_pdf_with_nul_text())

    from app.services.parser import _extract_pdf_text_layer_best_effort

    page_texts, page_count, warnings = _extract_pdf_text_layer_best_effort(pdf_path)

    assert page_count == 1
    assert not warnings
    assert "\x00" not in "".join(page_texts)
    assert "alphabeta" in "".join(page_texts)


def test_pdf_text_layer_reports_encrypted_pdf(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.write_bytes(b"%PDF")

    class FakeEncryptedReader:
        is_encrypted = True

        def __init__(self, path: str) -> None:
            self.pages = []

        def decrypt(self, password: str) -> int:
            return 0

    from app.services import parser

    monkeypatch.setattr(parser, "PdfReader", FakeEncryptedReader)

    with pytest.raises(DocumentParseError, match="Encrypted PDF"):
        _extract_pdf_text_layer(pdf_path)
