from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import mineru_parser_smoke


def test_mineru_parser_smoke_uses_project_parser_path(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "mineru-output"
    calls: list[str] = []
    settings = mineru_parser_smoke.document_parser.settings
    original_extra_args = settings.mineru_extra_args
    original_bin = settings.mineru_bin

    def fake_extract(path):
        calls.append("_extract_pdf_text_layer")
        assert path == pdf_path.resolve()
        return ["page text"], 2

    def fake_parse(path, page_count):
        calls.append("_parse_pdf_with_mineru")
        assert path == pdf_path.resolve()
        assert page_count == 2
        assert settings.mineru_bin == "mineru"
        assert settings.mineru_model_source == "modelscope"
        assert settings.mineru_output_dir == output_dir.resolve()
        assert settings.mineru_timeout == 90
        assert "--start 1 --end 2" in settings.mineru_extra_args
        return mineru_parser_smoke.document_parser.ParsedDocument(
            title="paper",
            text="parsed",
            chunks=[
                mineru_parser_smoke.document_parser.ParsedChunk(ordinal=0, text="chunk 1"),
                mineru_parser_smoke.document_parser.ParsedChunk(ordinal=1, text="chunk 2"),
            ],
            metadata={
                "parser_mode": "pdf_mineru",
                "document_intelligence": {
                    "page_outputs": [{"page_label": "1"}],
                    "tables": [{"markdown": "| A |"}],
                    "figures": [{"image_path": "images/fig.jpg"}],
                    "formulas": [{"text": "x"}],
                    "content_list_path": str(output_dir / "auto" / "content_list_v2.json"),
                    "output_dir": str(output_dir / "auto"),
                },
            },
        )

    monkeypatch.setattr(mineru_parser_smoke.document_parser, "_extract_pdf_text_layer", fake_extract)
    monkeypatch.setattr(mineru_parser_smoke.document_parser, "_parse_pdf_with_mineru", fake_parse)

    args = mineru_parser_smoke.parse_args(
        [
            "--pdf",
            str(pdf_path),
            "--output",
            str(tmp_path / "summary.json"),
            "--mineru-bin",
            "mineru",
            "--mineru-output-dir",
            str(output_dir),
            "--mineru-model-source",
            "modelscope",
            "--start",
            "1",
            "--end",
            "2",
            "--timeout",
            "90",
        ]
    )
    summary = mineru_parser_smoke.run_smoke(args)

    assert calls == ["_extract_pdf_text_layer", "_parse_pdf_with_mineru"]
    assert summary["status"] == "passed"
    assert summary["parser_mode"] == "pdf_mineru"
    assert summary["chunks"] == 2
    assert summary["page_outputs"] == 1
    assert summary["tables"] == 1
    assert summary["figures"] == 1
    assert summary["formulas"] == 1
    assert summary["content_list_path"].endswith("content_list_v2.json")
    assert settings.mineru_extra_args == original_extra_args
    assert settings.mineru_bin == original_bin


def test_mineru_parser_smoke_writes_failed_summary_for_missing_pdf(tmp_path) -> None:
    output_path = tmp_path / "summary.json"

    exit_code = mineru_parser_smoke.main(
        [
            "--pdf",
            str(tmp_path / "missing.pdf"),
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert summary["status"] == "failed"
    assert summary["parser_mode"] is None
    assert "PDF does not exist" in summary["error"]


def test_mineru_parser_smoke_requires_chunks_page_outputs_and_content_list(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(mineru_parser_smoke.document_parser, "_extract_pdf_text_layer", lambda path: (["text"], 1))
    monkeypatch.setattr(
        mineru_parser_smoke.document_parser,
        "_parse_pdf_with_mineru",
        lambda path, page_count: mineru_parser_smoke.document_parser.ParsedDocument(
            title="paper",
            text="",
            chunks=[],
            metadata={"parser_mode": "pdf_mineru", "document_intelligence": {"page_outputs": [], "content_list_path": None}},
        ),
    )

    args = mineru_parser_smoke.parse_args(["--pdf", str(pdf_path), "--output", str(tmp_path / "summary.json")])

    summary = mineru_parser_smoke.run_smoke(args)

    assert summary["status"] == "failed"
    assert "chunks>0" in summary["error"]


def test_merge_extra_args_replaces_existing_page_bounds() -> None:
    merged = mineru_parser_smoke._merge_extra_args("--foo bar --start 0 --end 9", start=1, end=2)

    assert merged == "--foo bar --start 1 --end 2"
