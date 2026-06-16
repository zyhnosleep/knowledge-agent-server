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
