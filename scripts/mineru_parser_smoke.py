from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.services import parser as document_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the project parser's real MinerU PDF path.")
    parser.add_argument("--pdf", type=Path, required=True, help="PDF to parse through app.services.parser._parse_pdf_with_mineru.")
    parser.add_argument("--output", type=Path, required=True, help="JSON summary output path.")
    parser.add_argument("--mineru-bin", default=None)
    parser.add_argument("--mineru-output-dir", type=Path, default=None)
    parser.add_argument("--mineru-model-source", default=None)
    parser.add_argument("--start", type=int, default=None, help="Optional MinerU start page, passed as --start.")
    parser.add_argument("--end", type=int, default=None, help="Optional MinerU end page, passed as --end.")
    parser.add_argument("--timeout", type=int, default=None, help="MinerU subprocess timeout in seconds.")
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    pdf_path = args.pdf.expanduser().resolve()
    summary: dict[str, Any] = {
        "status": "failed",
        "pdf": str(pdf_path),
        "parser_mode": None,
        "chunks": 0,
        "page_outputs": 0,
        "tables": 0,
        "figures": 0,
        "formulas": 0,
        "content_list_path": None,
        "output_dir": None,
    }
    if not pdf_path.is_file():
        summary["error"] = f"PDF does not exist: {pdf_path}"
        return summary

    original = _capture_mineru_settings()
    try:
        _apply_mineru_settings(args)
        _, page_count = document_parser._extract_pdf_text_layer(pdf_path)
        parsed = document_parser._parse_pdf_with_mineru(pdf_path, page_count)
        if parsed is None:
            summary["error"] = "MinerU parser returned no parsed document."
            return summary

        intelligence = parsed.metadata.get("document_intelligence", {})
        page_outputs = intelligence.get("page_outputs") if isinstance(intelligence, dict) else []
        tables = intelligence.get("tables") if isinstance(intelligence, dict) else []
        figures = intelligence.get("figures") if isinstance(intelligence, dict) else []
        formulas = intelligence.get("formulas") if isinstance(intelligence, dict) else []
        content_list_path = intelligence.get("content_list_path") if isinstance(intelligence, dict) else None
        output_dir = intelligence.get("output_dir") if isinstance(intelligence, dict) else None
        chunk_count = len(parsed.chunks)
        page_output_count = len(page_outputs) if isinstance(page_outputs, list) else 0
        passed = (
            parsed.metadata.get("parser_mode") == "pdf_mineru"
            and chunk_count > 0
            and page_output_count > 0
            and bool(content_list_path)
        )
        summary.update(
            {
                "status": "passed" if passed else "failed",
                "parser_mode": parsed.metadata.get("parser_mode"),
                "chunks": chunk_count,
                "page_outputs": page_output_count,
                "tables": len(tables) if isinstance(tables, list) else 0,
                "figures": len(figures) if isinstance(figures, list) else 0,
                "formulas": len(formulas) if isinstance(formulas, list) else 0,
                "content_list_path": content_list_path,
                "output_dir": output_dir,
            }
        )
        if summary["status"] != "passed":
            summary["error"] = "Parser did not produce required MinerU evidence: parser_mode=pdf_mineru, chunks>0, page_outputs>0, and content_list_path."
        return summary
    except Exception as exc:  # noqa: BLE001
        summary["error"] = str(exc)
        return summary
    finally:
        _restore_mineru_settings(original)


def _capture_mineru_settings() -> dict[str, Any]:
    settings = document_parser.settings
    return {
        "mineru_bin": settings.mineru_bin,
        "mineru_model_source": settings.mineru_model_source,
        "mineru_output_dir": settings.mineru_output_dir,
        "mineru_timeout": settings.mineru_timeout,
        "mineru_extra_args": settings.mineru_extra_args,
    }


def _apply_mineru_settings(args: argparse.Namespace) -> None:
    settings = document_parser.settings
    if args.mineru_bin:
        settings.mineru_bin = args.mineru_bin
    if args.mineru_model_source:
        settings.mineru_model_source = args.mineru_model_source
    if args.mineru_output_dir is not None:
        settings.mineru_output_dir = args.mineru_output_dir.expanduser().resolve()
    if args.timeout is not None:
        settings.mineru_timeout = args.timeout
    settings.mineru_extra_args = _merge_extra_args(settings.mineru_extra_args, start=args.start, end=args.end)


def _restore_mineru_settings(values: dict[str, Any]) -> None:
    settings = document_parser.settings
    for key, value in values.items():
        setattr(settings, key, value)


def _merge_extra_args(existing: str, *, start: int | None, end: int | None) -> str:
    parts = shlex.split(existing) if existing else []
    if start is not None:
        parts = _remove_option_with_value(parts, "--start")
    if end is not None:
        parts = _remove_option_with_value(parts, "--end")
    if start is not None:
        parts.extend(["--start", str(start)])
    if end is not None:
        parts.extend(["--end", str(end)])
    return " ".join(shlex.quote(part) for part in parts)


def _remove_option_with_value(parts: list[str], option: str) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(parts):
        if parts[index] == option:
            index += 2
            continue
        cleaned.append(parts[index])
        index += 1
    return cleaned


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_smoke(args)
    write_summary(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
