from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.core.config import get_settings
from app.services.ai import DocumentPagePayload, OllamaClient, safe_model_call
from app.services.filesystem import display_title_from_path

logger = logging.getLogger(__name__)
settings = get_settings()


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
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in {".html", ".htm"}:
        return _parse_html(path)
    return _parse_text(path)


def _parse_pdf(path: Path) -> ParsedDocument:
    page_texts, page_count = _extract_pdf_text_layer(path)
    if settings.document_intelligence_enabled:
        enriched = _parse_pdf_with_document_intelligence(path, page_texts, page_count)
        if enriched is not None:
            return enriched

    pages = [text for text in page_texts if text.strip()]
    chunks = [
        ParsedChunk(ordinal=index, text=text[:4000], page_label=str(index + 1))
        for index, text in enumerate(page_texts)
        if text.strip()
    ]
    full_text = "\n\n".join(pages)
    metadata = {
        "pages": page_count,
        "parser_mode": "pdf_text_layer",
        "document_intelligence": {"enabled": False},
    }
    return ParsedDocument(title=display_title_from_path(path), text=full_text, chunks=chunks or _fallback_chunks(full_text), metadata=metadata)


def _extract_pdf_text_layer(path: Path) -> tuple[list[str], int]:
    reader = PdfReader(str(path))
    page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    return page_texts, len(reader.pages)


def _parse_pdf_with_document_intelligence(path: Path, page_texts: list[str], page_count: int) -> ParsedDocument | None:
    rendered_pages = _render_pdf_pages(path, dpi=settings.pdf_render_dpi)
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

    for page_index, image_bytes in enumerate(rendered_pages):
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
        fused = _fuse_pdf_page_content(page_label=page_label, raw_text=raw_text, text_quality=quality, analysis=analysis)
        full_pages.append(fused["page_text"])
        page_outputs.append(
            {
                "page_label": page_label,
                "text_quality": quality,
                "page_summary": analysis.page_summary,
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
                    text=chunk_text[:4000],
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
            "page_outputs": page_outputs,
            "tables": tables,
            "formulas": formulas,
            "figures": figures,
        },
    }
    return ParsedDocument(
        title=display_title_from_path(path),
        text=full_text,
        chunks=all_chunks or _fallback_chunks(full_text),
        metadata=metadata,
    )


def _render_pdf_pages(path: Path, dpi: int) -> list[bytes]:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.info("PyMuPDF not available for PDF rendering: %s", exc)
        return []

    zoom = max(dpi, 72) / 72
    document = fitz.open(str(path))
    images: list[bytes] = []
    try:
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            images.append(pixmap.tobytes("png"))
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
    return safe_model_call(
        lambda: client.generate_structured_with_images(
            DocumentPagePayload,
            system_prompt=(
                "You are a multimodal document intelligence parser. "
                "Read the PDF page image, align it with any provided text layer, and output structured page Markdown."
            ),
            user_prompt=prompt,
            images=[image_bytes],
            model=settings.ollama_vision_model or settings.ollama_generation_model,
        ),
        fallback,
    )


def _fallback_page_analysis(*, page_label: str, raw_text: str, text_quality: str) -> DocumentPagePayload:
    sections = _split_into_sections(raw_text)
    summary = sections[0] if sections else raw_text[:240]
    notes = []
    if text_quality != "high":
        notes.append("Text layer quality was limited; multimodal fallback used.")
    return DocumentPagePayload(
        page_label=page_label,
        page_summary=summary[:400],
        page_markdown="\n\n".join(sections[:6]) if sections else raw_text[:2000],
        sections=sections[:8],
        tables=[],
        figures=[],
        formulas=[],
        key_facts=sections[:5],
        entities=[],
        evidence_spans=sections[:5],
        coverage_notes=notes,
    )


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
            sections = _dedupe_preserve_order(raw_sections[:8] + sections)
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
    page_markdown_lines.extend(f"- {section}" for section in sections[:8] or ["No narrative blocks identified."])#取前8个章节，超过部分省略
    if tables:
        page_markdown_lines.extend(["", "### Tables", *tables])#用生成器表达式把所有公式转换为markdown列表项，先换行，再添加标题，最后添加表格内容
    if formulas:
        page_markdown_lines.extend(["", "### Formulas", *(f"- {item}" for item in formulas)])
    if figures:
        page_markdown_lines.extend(["", "### Figures", *(f"- {item}" for item in figures)])
    if evidence:
        page_markdown_lines.extend(["", "### Evidence Spans", *(f"- {item}" for item in evidence[:8])])

    chunk_blocks: list[tuple[str, str | None]] = []#章节，表格，公式，图片
    for index, section in enumerate(sections[:8]):
        chunk_blocks.append((section, f"page-{page_label}-section-{index + 1}"))
    for index, table in enumerate(tables[:4]):
        chunk_blocks.append((table, f"page-{page_label}-table-{index + 1}"))
    for index, formula in enumerate(formulas[:4]):
        chunk_blocks.append((formula, f"page-{page_label}-formula-{index + 1}"))
    for index, figure in enumerate(figures[:4]):
        chunk_blocks.append((figure, f"page-{page_label}-figure-{index + 1}"))
    if not chunk_blocks:#如果文本没有这些分块，就给markdowm或原文
        fallback_text = analysis.page_markdown.strip() or raw_text[:2000].strip()
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
