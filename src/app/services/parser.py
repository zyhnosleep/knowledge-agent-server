from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import trafilatura
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.core.config import get_settings
from app.services.ai import DocumentPagePayload, OllamaClient, safe_model_call
from app.services.filesystem import display_title_from_path, slugify
from app.services.table_extraction import extract_structured_tables
from app.services.table_normalization import normalize_table_text

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentParseError(RuntimeError):
    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(f"{path.name}: {message}")


@dataclass
class ParsedChunk:
    ordinal: int
    text: str
    heading: str | None = None
    page_label: str | None = None


@dataclass
class ParsedDocument:
    title: str
    text: str
    chunks: list[ParsedChunk]
    metadata: dict


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    from app.services.canonical_adapters import parse_canonical_document

    canonical = parse_canonical_document(path)
    parsed = _canonical_to_parsed_document(canonical)
    if canonical.parser_source == "text":
        try:
            with Path(path).open("r", encoding="utf-8", newline="") as source_file:
                source_text = source_file.read()
        except (OSError, UnicodeError) as exc:
            raise DocumentParseError(Path(path), f"Unable to read text: {exc}") from exc
        parsed.text = source_text
        if not parsed.chunks:
            parsed.chunks = [ParsedChunk(ordinal=0, text=source_text)]
    return parsed


def _canonical_to_parsed_document(document) -> ParsedDocument:
    chunks = [
        ParsedChunk(
            ordinal=block.reading_order,
            text=block.text,
            heading=block.section_path[-1] if block.section_path else None,
            page_label=next(
                (span.page_label for span in block.source_spans if span.page_label is not None),
                None,
            ),
        )
        for block in document.blocks
    ]
    metadata = dict(document.parser_metadata)
    metadata.update(
        {
            "canonical": {
                "document_id": document.document_id,
                "parse_version": document.parse_version,
                "parser_source": document.parser_source,
                "source_path": document.source_path,
                "source_media_type": document.source_media_type,
                "warnings": list(document.warnings),
            },
            "parser_metadata": dict(document.parser_metadata),
            "source_metadata": dict(document.source_metadata),
        }
    )
    return ParsedDocument(
        title=document.title,
        text="\n\n".join(block.text for block in document.blocks),
        chunks=chunks,
        metadata=metadata,
    )


def _parse_pdf(path: Path) -> ParsedDocument:
    from app.services.canonical_adapters import PDFCanonicalAdapter

    return _canonical_to_parsed_document(PDFCanonicalAdapter().parse(path))


def _open_pdf_pages(path: Path) -> list[object]:
    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise DocumentParseError(path, "Encrypted PDF could not be decrypted.") from exc
            if not decrypted:
                raise DocumentParseError(path, "Encrypted PDF is not supported.")
        return list(reader.pages)
    except DocumentParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentParseError(path, f"Unable to read PDF: {exc}") from exc


def _validate_pdf_basic(path: Path) -> int:
    """Validate the PDF container and return page count without extracting text."""
    return len(_open_pdf_pages(path))


def _extract_pdf_text_layer_best_effort(path: Path) -> tuple[list[str], int, list[str]]:
    pages = _open_pdf_pages(path)

    page_texts: list[str] = []
    warnings: list[str] = []
    for index, page in enumerate(pages):
        try:
            page_texts.append((page.extract_text() or "").strip())
        except Exception as exc:  # noqa: BLE001
            page_texts.append("")
            warnings.append(f"Unable to extract text from page {index + 1}: {exc}")
    return page_texts, len(pages), warnings


def _extract_pdf_text_layer(
    path: Path,
    *,
    warnings_out: list[str] | None = None,
) -> tuple[list[str], int]:
    """Compatibility API returning best-effort page text and the page count."""
    page_texts, page_count, warnings = _extract_pdf_text_layer_best_effort(path)
    if warnings_out is not None:
        warnings_out.extend(warnings)
    return page_texts, page_count


def _parse_pdf_with_mineru(path: Path, page_count: int) -> ParsedDocument | None:
    mineru_bin = _resolve_mineru_binary(settings.mineru_bin)
    if mineru_bin is None:
        logger.info("MinerU is enabled but the CLI was not found: %s", settings.mineru_bin)
        return None

    source_path = path.expanduser().resolve()
    if not source_path.exists():
        logger.warning("MinerU source PDF does not exist: %s", source_path)
        return None

    output_root = (settings.mineru_output_dir or settings.cache_dir / "mineru").expanduser().resolve()
    run_dir = output_root / f"{slugify(source_path.stem) or 'document'}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    command = _build_mineru_command(
        mineru_bin=mineru_bin,
        source_path=source_path,
        output_dir=run_dir,
        backend=settings.mineru_backend,
        extra_args=settings.mineru_extra_args,
    )

    env = os.environ.copy()
    if settings.mineru_model_source:
        env["MINERU_MODEL_SOURCE"] = settings.mineru_model_source

    logger.info("Running MinerU PDF parser: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=str(source_path.parent),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.mineru_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("MinerU timed out after %s seconds; falling back.", settings.mineru_timeout)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("MinerU failed to run; falling back: %s", exc)
        return None

    if completed.returncode != 0:
        logger.warning("MinerU exited with code %s: %s", completed.returncode, completed.stderr[-1200:])
        return None

    content_paths = _find_mineru_content_lists(run_dir)
    if not content_paths:
        logger.warning("MinerU completed but no content_list JSON was found under %s.", run_dir)
        return None

    content_path: Path | None = None
    content_list: list[dict] = []
    for candidate_path in content_paths:
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read MinerU output %s: %s", candidate_path, exc)
            continue
        candidate_content = _normalize_mineru_content_list(payload)
        if candidate_content:
            content_path = candidate_path
            content_list = candidate_content
            break
        logger.warning("MinerU output had no supported blocks: %s (%s)", candidate_path, _describe_mineru_payload(payload))

    if content_path is None or not content_list:
        logger.warning("MinerU produced no usable structured content under %s.", run_dir)
        return None

    markdown_path = _find_mineru_markdown(content_path.parent)
    parsed = _mineru_content_to_parsed_doc(
        path=source_path,
        content_list=content_list,
        page_count=page_count,
        output_dir=content_path.parent,
        content_list_path=content_path,
    )
    if markdown_path is not None:
        _augment_mineru_parsed_doc_from_markdown(parsed, markdown_path)
    if not parsed.text.strip():
        logger.warning("MinerU output did not contain usable text; falling back.")
        return None
    return parsed


def _resolve_mineru_binary(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved:
        return resolved
    if candidate.name != value:
        return shutil.which(candidate.name)
    return None


def _build_mineru_command(
    *,
    mineru_bin: str,
    source_path: Path,
    output_dir: Path,
    backend: str | None,
    extra_args: str,
) -> list[str]:
    command = [mineru_bin, "-p", str(source_path), "-o", str(output_dir)]
    normalized_backend = _normalize_mineru_backend(backend)
    if normalized_backend:
        command.extend(["-b", normalized_backend])
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


def _normalize_mineru_backend(value: str | None) -> str | None:
    backend = (value or "").strip()
    if not backend:
        return None
    aliases = {
        "hybrid": "hybrid-engine",
        "hybrid_engine": "hybrid-engine",
        "vlm": "vlm-engine",
        "vlm_engine": "vlm-engine",
        "vlm_http_client": "vlm-http-client",
        "hybrid_http_client": "hybrid-http-client",
    }
    return aliases.get(backend.lower(), backend)


def _find_mineru_content_list(output_dir: Path) -> Path | None:
    candidates = _find_mineru_content_lists(output_dir)
    return candidates[0] if candidates else None


def _find_mineru_content_lists(output_dir: Path) -> list[Path]:
    patterns = ("*content_list_v2.json", "content_list_v2.json", "*content_list.json", "content_list.json")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(output_dir.rglob(pattern))
    unique_candidates = list(dict.fromkeys(candidates))
    return sorted(
        unique_candidates,
        key=lambda item: (0 if item.name.endswith("content_list_v2.json") else 1, -item.stat().st_mtime),
    )


def _find_mineru_markdown(output_dir: Path) -> Path | None:
    candidates = [
        path
        for path in output_dir.rglob("*.md")
        if not path.name.lower().endswith(("_origin.md", "_layout.md"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _normalize_mineru_content_list(payload: object) -> list[dict]:
    raw_items: list[dict] = []
    if isinstance(payload, list):
        for page_index, item in enumerate(payload):
            if isinstance(item, dict):
                raw_items.append(item)
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        merged = {"page_idx": page_index, **child}
                        raw_items.append(merged)
    elif isinstance(payload, dict):
        for key in ("content_list", "content", "pages", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                for page_index, item in enumerate(value):
                    if isinstance(item, dict):
                        raw_items.append(item)
                    elif isinstance(item, list):
                        for child in item:
                            if isinstance(child, dict):
                                merged = {"page_idx": page_index, **child}
                                raw_items.append(merged)
                break

    flattened: list[dict] = []
    for item in raw_items:
        nested = _mineru_nested_items(item)
        if not nested:
            flattened.append(item)
            continue
        inherited_page = {key: item[key] for key in ("page_idx", "page_id", "page", "page_no", "page_number") if key in item}
        for child in nested:
            merged = {**inherited_page, **child}
            flattened.append(merged)
    return flattened


def _mineru_nested_items(item: dict) -> list[dict]:
    for key in ("blocks", "items", "content_list"):
        value = item.get(key)
        if isinstance(value, list):
            return [child for child in value if isinstance(child, dict)]
    value = item.get("content")
    if isinstance(value, list):
        return [child for child in value if isinstance(child, dict)]
    return []


def _describe_mineru_payload(payload: object) -> str:
    if isinstance(payload, list):
        if not payload:
            return "list(len=0)"
        first = payload[0]
        if isinstance(first, list):
            return f"list(len={len(payload)}, first=list(len={len(first)}))"
        if isinstance(first, dict):
            return f"list(len={len(payload)}, first_keys={list(first)[:8]})"
        return f"list(len={len(payload)}, first_type={type(first).__name__})"
    if isinstance(payload, dict):
        return f"dict(keys={list(payload)[:12]})"
    return type(payload).__name__


def _mineru_content_to_parsed_doc(
    *,
    path: Path,
    content_list: list[dict],
    page_count: int,
    output_dir: Path | None = None,
    content_list_path: Path | None = None,
) -> ParsedDocument:
    page_blocks: dict[str, list[str]] = {}
    page_stats: dict[str, dict[str, int]] = {}
    tables: list[dict] = []
    formulas: list[dict] = []
    figures: list[dict] = []
    chunks: list[ParsedChunk] = []

    for item in content_list:
        content_type = str(item.get("type") or item.get("category") or "text").lower()
        page_label = _mineru_page_label(item)
        page_blocks.setdefault(page_label, [])
        stats = page_stats.setdefault(page_label, {"sections": 0, "tables": 0, "formulas": 0, "figures": 0})

        block_text = ""
        heading = f"mineru-page-{page_label}-{content_type}"
        if content_type in {"title", "heading"}:
            block_text = _mineru_first_text(item, "text", "title_content", "content", "md_content")
            if block_text:
                stats["sections"] += 1
                block_text = f"### {block_text}"
        elif content_type == "paragraph":
            block_text = _mineru_first_text(item, "text", "paragraph_content", "content", "md_content")
            if block_text:
                stats["sections"] += 1
        elif "table" in content_type:
            caption = _mineru_caption_text(item, "table_caption", "caption")
            table_body = _mineru_first_text(item, "table_body", "table_html", "table_content", "html", "text", "content", "md_content")
            table_markdown = _html_table_to_markdown(table_body) if "<table" in table_body.lower() else table_body
            table_markdown = normalize_table_text(table_markdown)
            block_text = "\n\n".join(part for part in (caption, table_markdown) if part)
            if block_text:
                stats["tables"] += 1
                tables.append({"page_label": page_label, "markdown": block_text})
        elif "equation" in content_type or "formula" in content_type:
            formula = _mineru_first_text(item, "text", "latex", "math_content", "content", "md_content")
            if formula:
                stats["formulas"] += 1
                formulas.append({"page_label": page_label, "text": formula})
                block_text = formula
        elif content_type in {"image", "figure"} or "image" in content_type or "figure" in content_type:
            figure = _mineru_figure_metadata(
                item,
                page_label=page_label,
                output_dir=output_dir,
                content_list_path=content_list_path,
            )
            block_text = _mineru_figure_chunk_text(figure)
            if block_text:
                stats["figures"] += 1
                figures.append(figure)
        else:
            block_text = _mineru_first_text(item, "text", "paragraph_content", "content", "md_content")
            if block_text:
                stats["sections"] += 1

        block_text = block_text.strip()
        if not block_text:
            continue
        page_blocks[page_label].append(block_text)
        chunks.append(
            ParsedChunk(
                ordinal=len(chunks),
                text=block_text,
                heading=heading,
                page_label=page_label,
            )
        )

    page_outputs: list[dict] = []
    full_pages: list[str] = []
    ordered_labels = sorted(page_blocks, key=_page_label_sort_key)
    for page_label in ordered_labels:
        blocks = page_blocks[page_label]
        stats = page_stats.get(page_label, {})
        page_markdown = "\n\n".join([f"## Page {page_label}", *blocks])
        summary = _summarize_blocks(blocks)
        page_outputs.append(
            {
                "page_label": page_label,
                "text_quality": "mineru",
                "page_summary": summary,
                "section_count": stats.get("sections", 0),
                "table_count": stats.get("tables", 0),
                "formula_count": stats.get("formulas", 0),
                "figure_count": stats.get("figures", 0),
                "page_markdown": page_markdown,
                "coverage_notes": [],
            }
        )
        full_pages.append(page_markdown)

    full_text = "\n\n".join(full_pages)
    metadata = {
        "pages": page_count or len(ordered_labels),
        "parser_mode": "pdf_mineru",
        "document_intelligence": {
            "enabled": True,
            "engine": "mineru",
            "backend": _normalize_mineru_backend(settings.mineru_backend),
            "output_dir": str(output_dir) if output_dir else None,
            "content_list_path": str(content_list_path) if content_list_path else None,
            "page_outputs": page_outputs,
            "tables": tables,
            "structured_tables": extract_structured_tables(tables),
            "formulas": formulas,
            "figures": figures,
        },
    }
    return ParsedDocument(
        title=display_title_from_path(path),
        text=full_text,
        chunks=_coalesce_parsed_chunks(chunks) if chunks else _fallback_chunks(full_text),
        metadata=metadata,
    )


def _coalesce_parsed_chunks(
    chunks: list[ParsedChunk], target_size: int = 1200
) -> list[ParsedChunk]:
    """Merge adjacent text fragments while keeping structured evidence isolated."""
    merged: list[ParsedChunk] = []
    buffer: ParsedChunk | None = None
    structured_types = ("-table", "-equation", "-formula", "-image", "-figure")

    def flush() -> None:
        nonlocal buffer
        if buffer is not None:
            buffer.ordinal = len(merged)
            merged.append(buffer)
            buffer = None

    for chunk in chunks:
        heading = chunk.heading or ""
        is_structured = any(marker in heading for marker in structured_types)
        if is_structured:
            flush()
            chunk.ordinal = len(merged)
            merged.append(chunk)
            continue
        if buffer is None:
            buffer = ParsedChunk(
                ordinal=0,
                text=chunk.text,
                heading=chunk.heading,
                page_label=chunk.page_label,
            )
            continue
        candidate = f"{buffer.text}\n\n{chunk.text}"
        if buffer.page_label == chunk.page_label and len(candidate) <= target_size:
            buffer.text = candidate
        else:
            flush()
            buffer = ParsedChunk(
                ordinal=0,
                text=chunk.text,
                heading=chunk.heading,
                page_label=chunk.page_label,
            )
    flush()
    return merged


def _augment_mineru_parsed_doc_from_markdown(parsed: ParsedDocument, markdown_path: Path) -> None:
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read MinerU markdown output %s: %s", markdown_path, exc)
        return
    tables = _extract_tables_from_mineru_markdown(markdown)
    intelligence = parsed.metadata.setdefault("document_intelligence", {})
    intelligence["markdown_path"] = str(markdown_path)
    existing_tables = intelligence.setdefault("tables", [])
    existing_texts = {str(table.get("markdown") or "").strip() for table in existing_tables if isinstance(table, dict)}
    for table in tables:
        markdown_text = str(table.get("markdown") or "").strip()
        markdown_text = normalize_table_text(markdown_text)
        if markdown_text and markdown_text not in existing_texts:
            table["markdown"] = markdown_text
            existing_tables.append(table)
            existing_texts.add(markdown_text)
            parsed.chunks.append(
                ParsedChunk(
                    ordinal=len(parsed.chunks),
                    text=markdown_text,
                    heading="mineru-markdown-table",
                    page_label=table.get("page_label"),
                )
            )
    intelligence["structured_tables"] = extract_structured_tables(existing_tables)
    if markdown.strip() and markdown.strip() not in parsed.text:
        parsed.text = (parsed.text + "\n\n## MinerU Markdown\n\n" + markdown).strip()


def _extract_tables_from_mineru_markdown(markdown: str) -> list[dict]:
    tables: list[dict] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if "|" not in line:
            index += 1
            continue
        if index + 1 >= len(lines) or not re.search(r"\|\s*:?-{3,}:?\s*(\||$)", lines[index + 1]):
            index += 1
            continue
        start = index
        while start > 0 and lines[start - 1].strip() and not lines[start - 1].startswith("#"):
            if "|" in lines[start - 1]:
                break
            start -= 1
        end = index + 2
        while end < len(lines) and "|" in lines[end]:
            end += 1
        block = "\n".join(lines[start:end]).strip()
        if block:
            tables.append({"page_label": _page_label_from_markdown_context(lines[:start]), "markdown": block})
        index = end
    return tables


def _page_label_from_markdown_context(lines: list[str]) -> str | None:
    for line in reversed(lines[-20:]):
        match = re.search(r"(?:Page|page)\s*(\d+)", line)
        if match:
            return match.group(1)
    return None


def _mineru_page_label(item: dict) -> str:
    for key in ("page_idx", "page_id", "page", "page_no", "page_number"):
        value = item.get(key)
        if isinstance(value, int):
            return str(value + 1 if key in {"page_idx", "page_id"} else value)
        if isinstance(value, str) and value.strip():
            if value.strip().isdigit() and key in {"page_idx", "page_id"}:
                return str(int(value.strip()) + 1)
            return value.strip()
    return "?"


def _mineru_first_text(item: dict, *keys: str) -> str:
    for key in keys:
        value = _mineru_lookup_value(item, key)
        text = _stringify_mineru_value(value)
        if text:
            return text
    return ""


def _mineru_caption_text(item: dict, *keys: str) -> str:
    parts: list[str] = []
    for key in keys:
        text = _stringify_mineru_value(_mineru_lookup_value(item, key))
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _mineru_figure_metadata(
    item: dict,
    *,
    page_label: str,
    output_dir: Path | None,
    content_list_path: Path | None,
) -> dict:
    caption = _mineru_caption_text(item, "image_caption", "chart_caption", "figure_caption", "caption")
    note = _mineru_caption_text(item, "note", "image_note", "figure_note", "description", "alt_text", "text")
    asset_path = _safe_mineru_asset_path(
        _mineru_first_text(item, "img_path", "image_path", "path", "image_url"),
        output_dir=output_dir,
        content_list_path=content_list_path,
    )
    metadata = {
        "page_label": page_label,
        "caption": caption,
        "note": note or caption or asset_path,
        "image_path": asset_path,
        "path": asset_path,
    }
    return {key: value for key, value in metadata.items() if value}


def _mineru_figure_chunk_text(figure: dict) -> str:
    parts = ["Figure evidence"]
    page_label = str(figure.get("page_label") or "").strip()
    caption = str(figure.get("caption") or "").strip()
    note = str(figure.get("note") or "").strip()
    image_path = str(figure.get("image_path") or "").strip()
    path = str(figure.get("path") or "").strip()
    if page_label and page_label != "?":
        parts.append(f"Page: {page_label}")
    if caption:
        parts.append(f"Caption: {caption}")
    if note and note != caption:
        parts.append(f"Note: {note}")
    if image_path:
        parts.append(f"Image path: {image_path}")
    if path and path != image_path:
        parts.append(f"Path: {path}")
    return "\n".join(parts).strip() if len(parts) > 1 else ""


def _safe_mineru_asset_path(
    image_path: str,
    *,
    output_dir: Path | None,
    content_list_path: Path | None,
) -> str:
    value = image_path.strip()
    if not value or "://" in value:
        return ""
    path = Path(value)
    bases = [base.resolve() for base in (content_list_path.parent if content_list_path else None, output_dir) if base is not None]
    if path.is_absolute():
        for base in bases:
            try:
                return path.resolve().relative_to(base).as_posix()
            except ValueError:
                continue
        return ""
    if not bases:
        return "" if ".." in path.parts else path.as_posix()
    for base in bases:
        candidate = (base / path).resolve()
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return ""


def _mineru_lookup_value(item: dict, key: str) -> object:
    if key in item:
        return item.get(key)
    content = item.get("content")
    if isinstance(content, dict) and key in content:
        return content.get(key)
    if isinstance(content, dict):
        for source_key in ("image_source", "img_source", "image"):
            source = content.get(source_key)
            if isinstance(source, dict) and key in source:
                return source.get(key)
    return None


def _stringify_mineru_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_stringify_mineru_value(item) for item in value).strip()
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "path",
            "image_path",
            "img_path",
            "title_content",
            "paragraph_content",
            "math_content",
            "table_content",
            "table_body",
            "image_caption",
            "chart_caption",
            "caption",
        ):
            text = _stringify_mineru_value(value.get(key))
            if text:
                return text
        children = value.get("children")
        if isinstance(children, list):
            return _stringify_mineru_value(children)
    return ""


def _html_table_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[list[str]] = []
    rowspans: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(soup.find_all("tr")):
        cells: list[str] = []
        column = 0

        def apply_pending_spans() -> None:
            nonlocal column
            while (row_index, column) in rowspans:
                cells.append(rowspans.pop((row_index, column)))
                column += 1

        apply_pending_spans()
        for cell in row.find_all(["th", "td"]):
            apply_pending_spans()
            text = cell.get_text(" ", strip=True)
            colspan = _html_span_value(cell.get("colspan"))
            rowspan = _html_span_value(cell.get("rowspan"))
            for offset in range(colspan):
                cells.append(text)
                if rowspan > 1:
                    for span_row in range(1, rowspan):
                        rowspans[(row_index + span_row, column + offset)] = text
            column += colspan
        apply_pending_spans()
        if cells:
            rows.append(cells)
    if not rows:
        return html.strip()

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _html_span_value(value: object) -> int:
    try:
        parsed = int(str(value or "1"))
    except ValueError:
        return 1
    return max(parsed, 1)


def _page_label_sort_key(label: str) -> tuple[int, str]:
    return (int(label), "") if label.isdigit() else (10**9, label)


def _summarize_blocks(blocks: list[str]) -> str:
    for block in blocks:
        clean = re.sub(r"\s+", " ", block).strip("#- ")
        if clean:
            return clean[:300]
    return "No summary available."


def _parse_pdf_with_document_intelligence(
    path: Path,
    page_texts: list[str],
    page_count: int,
    page_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> ParsedDocument | None:
    requested_indices = (
        None
        if page_indices is None
        else sorted(
            {
                index
                for index in page_indices
                if isinstance(index, int) and 0 <= index < page_count
            }
        )
    )
    try:
        rendered_payload = _render_pdf_pages(
            path,
            dpi=settings.pdf_render_dpi,
            page_indices=requested_indices,
        )
    except TypeError:
        # Preserve compatibility with integrations that still expose the
        # historical two-argument renderer while production uses page scopes.
        rendered_payload = _render_pdf_pages(path, dpi=settings.pdf_render_dpi)

    rendered_pages: dict[int, bytes] = {}
    if isinstance(rendered_payload, dict):
        rendered_pages = {
            index: image
            for index, image in rendered_payload.items()
            if isinstance(index, int) and isinstance(image, bytes)
        }
    elif isinstance(rendered_payload, list):
        if rendered_payload and all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], bytes)
            for item in rendered_payload
        ):
            rendered_pages = dict(rendered_payload)
        else:
            raw_images = [item for item in rendered_payload if isinstance(item, bytes)]
            if requested_indices is not None and len(raw_images) == len(requested_indices):
                rendered_pages = dict(zip(requested_indices, raw_images))
            else:
                rendered_pages = dict(enumerate(raw_images))
    if not rendered_pages:
        logger.info("PDF document intelligence skipped because page rendering was unavailable.")
        return None

    page_outputs: list[dict] = []
    all_chunks: list[ParsedChunk] = []
    full_pages: list[str] = []
    tables: list[dict] = []
    formulas: list[dict] = []
    figures: list[dict] = []
    client = OllamaClient()

    selected_indices = requested_indices
    selected_set = None if selected_indices is None else set(selected_indices)

    for page_index, image_bytes in sorted(rendered_pages.items()):
        if selected_set is not None and page_index not in selected_set:
            continue
        page_label = str(page_index + 1)
        raw_text = page_texts[page_index] if page_index < len(page_texts) else ""
        quality = _classify_text_layer_quality(raw_text)
        analysis = _analyze_pdf_page(
            client=client,
            path=path,
            page_label=page_label,
            image_bytes=image_bytes,
            raw_text=raw_text,
            text_quality=quality,
        )
        if analysis.analysis_source != "document_intelligence":
            logger.info(
                "PDF document intelligence page %s used %s; treating the parser attempt as unavailable.",
                page_label,
                analysis.analysis_source,
            )
            return None
        fused = _fuse_pdf_page_content(page_label=page_label, raw_text=raw_text, text_quality=quality, analysis=analysis)
        full_pages.append(fused["page_text"])
        page_outputs.append(
            {
                "page_label": page_label,
                "text_quality": quality,
                "page_summary": analysis.page_summary,
                "sections": list(analysis.sections),
                "evidence_spans": list(analysis.evidence_spans),
                "section_count": len(analysis.sections),
                "table_count": len(analysis.tables),
                "formula_count": len(analysis.formulas),
                "figure_count": len(analysis.figures),
                "page_markdown": fused["page_markdown"],
                "coverage_notes": analysis.coverage_notes,
            }
        )
        for table_markdown in analysis.tables:
            tables.append({"page_label": page_label, "markdown": table_markdown})
        for formula_text in analysis.formulas:
            formulas.append({"page_label": page_label, "text": formula_text})
        for figure_note in analysis.figures:
            figures.append({"page_label": page_label, "note": figure_note})
        for chunk_text, heading in fused["chunk_blocks"]:
            if not chunk_text.strip():
                continue
            all_chunks.append(
                ParsedChunk(
                    ordinal=len(all_chunks),
                    text=chunk_text,
                    heading=heading,
                    page_label=page_label,
                )
            )

    full_text = "\n\n".join(page for page in full_pages if page.strip())
    metadata = {
        "pages": page_count,
        "parser_mode": "pdf_document_intelligence",
        "document_intelligence": {
            "enabled": True,
            "vision_model": settings.ollama_vision_model or settings.ollama_generation_model,
            "render_dpi": settings.pdf_render_dpi,
            "ocr_fallback_enabled": settings.ocr_fallback_enabled,
            "page_indices": selected_indices,
            "page_outputs": page_outputs,
            "tables": tables,
            "structured_tables": extract_structured_tables(tables),
            "formulas": formulas,
            "figures": figures,
        },
    }
    return ParsedDocument(
        title=display_title_from_path(path),
        text=full_text,
        chunks=_coalesce_parsed_chunks(all_chunks) if all_chunks else _fallback_chunks(full_text),
        metadata=metadata,
    )


def _render_pdf_pages(
    path: Path,
    dpi: int,
    page_indices: list[int] | set[int] | tuple[int, ...] | None = None,
) -> dict[int, bytes]:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.info("PyMuPDF not available for PDF rendering: %s", exc)
        return []

    zoom = max(dpi, 72) / 72
    document = fitz.open(str(path))
    selected = (
        set(range(len(document)))
        if page_indices is None
        else {
            index
            for index in page_indices
            if isinstance(index, int) and 0 <= index < len(document)
        }
    )
    images: dict[int, bytes] = {}
    try:
        for page_index in sorted(selected):
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            images[page_index] = pixmap.tobytes("png")
    finally:
        document.close()
    return images


def _analyze_pdf_page(
    *,
    client: OllamaClient,
    path: Path,
    page_label: str,
    image_bytes: bytes,
    raw_text: str,
    text_quality: str,
) -> DocumentPagePayload:
    ocr_text = ""
    if settings.ocr_fallback_enabled and text_quality == "low":
        ocr_text = _ocr_image_bytes(image_bytes)
    fallback = _fallback_page_analysis(page_label=page_label, raw_text=raw_text or ocr_text, text_quality=text_quality)
    prompt = "\n\n".join(
        [
            f"Document title: {display_title_from_path(path)}",
            f"Page label: {page_label}",
            f"Detected text layer quality: {text_quality}",
            (
                "Analyze this PDF page and return structured JSON. "
                "Preserve layout semantics, tables, formulas, captions, headings, and clinical or technical facts."
            ),
            "Text layer (if available):\n" + (raw_text[:3500] or "No reliable text layer available."),
            "OCR fallback text (if available):\n" + (ocr_text[:3500] or "No OCR fallback text available."),
        ]
    )
    analysis, analysis_source = safe_model_call(
        lambda: (
            client.generate_structured_with_images(
                DocumentPagePayload,
                system_prompt=(
                    "You are a multimodal document intelligence parser. "
                    "Read the PDF page image, align it with any provided text layer, and output structured page Markdown."
                ),
                user_prompt=prompt,
                images=[image_bytes],
                model=settings.ollama_vision_model or settings.ollama_generation_model,
            ),
            "document_intelligence",
        ),
        (fallback, "pypdf_text_layer_fallback"),
    )
    analysis._analysis_source = analysis_source
    return analysis


def _fallback_page_analysis(*, page_label: str, raw_text: str, text_quality: str) -> DocumentPagePayload:
    sections = _split_into_sections(raw_text)
    summary = sections[0] if sections else raw_text[:240]
    notes = []
    if text_quality != "high":
        notes.append("Text layer quality was limited; multimodal fallback used.")
    analysis = DocumentPagePayload(
        page_label=page_label,
        page_summary=summary[:400],
        page_markdown="\n\n".join(sections) if sections else raw_text,
        sections=sections,
        tables=[],
        figures=[],
        formulas=[],
        key_facts=sections,
        entities=[],
        evidence_spans=sections,
        coverage_notes=notes,
    )
    analysis._analysis_source = "pypdf_text_layer_fallback"
    return analysis


def _classify_text_layer_quality(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 40:
        return "low"
    meaningful = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned))
    suspicious = cleaned.count("�")
    density = meaningful / max(len(cleaned), 1)
    if suspicious > 0 or density < 0.28:
        return "low"
    if len(cleaned) > 120 and density >= 0.4:
        return "high"
    return "medium"


def _fuse_pdf_page_content(*, page_label: str, raw_text: str, text_quality: str, analysis: DocumentPagePayload) -> dict:
    summary = analysis.page_summary.strip() or raw_text[:240].strip() or f"Page {page_label}"
    sections = [section.strip() for section in analysis.sections if section.strip()]#删除空白字符
    if text_quality == "high" and raw_text:
        raw_sections = _split_into_sections(raw_text)
        if raw_sections:
            sections = _dedupe_preserve_order(raw_sections + sections)
    tables = [table.strip() for table in analysis.tables if table.strip()]
    formulas = [formula.strip() for formula in analysis.formulas if formula.strip()]
    figures = [figure.strip() for figure in analysis.figures if figure.strip()]
    evidence = [span.strip() for span in analysis.evidence_spans if span.strip()]
    page_markdown_lines = [
        f"## Page {page_label}",
        "",
        "### Summary",
        summary,
        "",
        "### Narrative Blocks",
    ]
    page_markdown_lines.extend(f"- {section}" for section in sections or ["No narrative blocks identified."])
    if tables:
        page_markdown_lines.extend(["", "### Tables", *tables])#用生成器表达式把所有公式转换为markdown列表项，先换行，再添加标题，最后添加表格内容
    if formulas:
        page_markdown_lines.extend(["", "### Formulas", *(f"- {item}" for item in formulas)])
    if figures:
        page_markdown_lines.extend(["", "### Figures", *(f"- {item}" for item in figures)])
    if evidence:
        page_markdown_lines.extend(["", "### Evidence Spans", *(f"- {item}" for item in evidence)])

    chunk_blocks: list[tuple[str, str | None]] = []#章节，表格，公式，图片
    for index, section in enumerate(sections):
        chunk_blocks.append((section, f"page-{page_label}-section-{index + 1}"))
    for index, table in enumerate(tables):
        chunk_blocks.append((table, f"page-{page_label}-table-{index + 1}"))
    for index, formula in enumerate(formulas):
        chunk_blocks.append((formula, f"page-{page_label}-formula-{index + 1}"))
    for index, figure in enumerate(figures):
        chunk_blocks.append((figure, f"page-{page_label}-figure-{index + 1}"))
    if not chunk_blocks:#如果文本没有这些分块，就给markdowm或原文
        fallback_text = analysis.page_markdown.strip() or raw_text.strip()
        if fallback_text:
            chunk_blocks.append((fallback_text, f"page-{page_label}-content"))

    return {
        "page_markdown": "\n".join(page_markdown_lines).strip(),
        "page_text": "\n".join(page_markdown_lines).strip(),
        "chunk_blocks": chunk_blocks,
    }


def _split_into_sections(text: str) -> list[str]:
    sections = [part.strip() for part in re.split(r"(?<=[。！？!?\.])\s+|\n{2,}", text) if part.strip()]
    if not sections and text.strip():
        return [text.strip()]
    return sections


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered# 返回最终的去重、有序、清洗后的列表


def _ocr_image_bytes(image_bytes: bytes) -> str:
    try:
        from io import BytesIO

        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.info("OCR fallback unavailable: %s", exc)
        return ""

    try:
        image = Image.open(BytesIO(image_bytes))
        return pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR fallback failed: %s", exc)
        return ""


def _parse_docx(path: Path) -> ParsedDocument:
    doc = DocxDocument(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    full_text = "\n".join(paragraphs)
    return ParsedDocument(title=display_title_from_path(path), text=full_text, chunks=_fallback_chunks(full_text), metadata={"paragraphs": len(paragraphs)})


def _parse_html(path: Path) -> ParsedDocument:
    html = path.read_text(encoding="utf-8", errors="ignore")
    extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
    if extracted:
        text = extracted
    else:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
    return ParsedDocument(title=display_title_from_path(path), text=text, chunks=_fallback_chunks(text), metadata={"format": "html"})


def _parse_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ParsedDocument(title=display_title_from_path(path), text=text, chunks=_fallback_chunks(text), metadata={"format": "text"})


def _fallback_chunks(text: str, target_size: int = 1200) -> list[ParsedChunk]:
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[ParsedChunk] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}\n\n{part}".strip() if buffer else part
        if len(candidate) > target_size and buffer:
            chunks.append(ParsedChunk(ordinal=len(chunks), text=buffer))
            buffer = part
        else:
            buffer = candidate
    if buffer:
        chunks.append(ParsedChunk(ordinal=len(chunks), text=buffer))
    return chunks or [ParsedChunk(ordinal=0, text=text[:target_size])]
