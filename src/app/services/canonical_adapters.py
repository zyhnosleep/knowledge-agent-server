from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bs4 import BeautifulSoup, Tag
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree

from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalTable,
    SectionNode,
    SourceSpan,
)
from app.services.filesystem import display_title_from_path


class CanonicalAdapter(Protocol):
    def parse(self, path: Path) -> CanonicalDocument: ...


@dataclass(frozen=True)
class _Line:
    number: int
    start: int
    content_end: int
    end: int
    text: str


def _parse_error(path: Path, message: str, exc: Exception | None = None) -> Exception:
    # Lazy import keeps parser.parse_document() free to import the dispatcher lazily.
    from app.services.parser import DocumentParseError

    error = DocumentParseError(path, message)
    if exc is not None:
        error.__cause__ = exc
    return error


def _validate_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise _parse_error(path, "Source document does not exist.")
    if not path.is_file():
        raise _parse_error(path, "Source document is not a readable file.")
    if not os.access(path, os.R_OK):
        raise _parse_error(path, "Source document is not readable.")
    return path.resolve()


def _read_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source_file:
            return source_file.read()
    except (OSError, UnicodeError) as exc:
        raise _parse_error(path, f"Unable to read text: {exc}", exc)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _new_document(path: Path, parser_source: str, media_type: str) -> CanonicalDocument:
    try:
        stat = path.stat()
        with path.open("rb") as source_file:
            source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    except OSError as exc:
        raise _parse_error(path, f"Unable to read source document: {exc}", exc)
    return CanonicalDocument(
        document_id=_stable_id("doc", str(path), source_hash),
        source_path=str(path),
        source_media_type=media_type,
        parser_source=parser_source,
        parse_version="canonical-v1",
        title=display_title_from_path(path),
        source_metadata={
            "filename": path.name,
            "size_bytes": stat.st_size,
            "sha256": source_hash,
        },
        parser_metadata={
            "adapter": parser_source,
            "format": parser_source,
            "canonical_schema": "v1",
        },
        metadata={"source_suffix": path.suffix.lower()},
    )


class _DocumentBuilder:
    def __init__(self, document: CanonicalDocument) -> None:
        self.document = document
        self._headings: list[tuple[int, str]] = []
        self._outline_stack: list[tuple[int, SectionNode]] = []
        self._reference_level: int | None = None
        self._appendix_level: int | None = None

    @property
    def heading_path(self) -> list[str]:
        return [title for _, title in self._headings]

    def _span(self, span: SourceSpan) -> SourceSpan:
        return span.model_copy(update={"heading_path": self.heading_path})

    def _block_id(self, block_type: str, span: SourceSpan, text: str) -> str:
        locator = json.dumps(span.model_dump(mode="json"), sort_keys=True, ensure_ascii=True)
        return _stable_id(
            "block",
            self.document.parser_source,
            len(self.document.blocks),
            block_type,
            locator,
            text,
        )

    def add_heading(self, title: str, level: int, span: SourceSpan, **metadata: object) -> CanonicalBlock:
        if self._reference_level is not None and level <= self._reference_level:
            self._reference_level = None
        if self._appendix_level is not None and level <= self._appendix_level:
            self._appendix_level = None

        while self._headings and self._headings[-1][0] >= level:
            self._headings.pop()
        self._headings.append((level, title))

        normalized = title.strip().lower().rstrip(":")
        if normalized in {"references", "reference", "bibliography"}:
            self._reference_level = level
            self._appendix_level = None
        elif normalized.startswith("appendix"):
            self._appendix_level = level
            self._reference_level = None

        updated_span = self._span(span)
        block = self._append_block(
            block_type="heading",
            text=title,
            span=updated_span,
            metadata={"heading_level": level, **metadata},
        )

        node = SectionNode(title=title, level=level, block_id=block.block_id)
        while self._outline_stack and self._outline_stack[-1][0] >= level:
            self._outline_stack.pop()
        if self._outline_stack:
            self._outline_stack[-1][1].children.append(node)
        else:
            self.document.outline.append(node)
        self._outline_stack.append((level, node))
        return block

    def add_content(
        self,
        text: str,
        span: SourceSpan,
        *,
        block_type: str = "narrative",
        metadata: dict[str, object] | None = None,
        table_id: str | None = None,
        figure_id: str | None = None,
        formula_id: str | None = None,
    ) -> CanonicalBlock:
        if block_type == "narrative":
            if self._reference_level is not None:
                block_type = "reference"
            elif self._appendix_level is not None:
                block_type = "appendix"
        return self._append_block(
            block_type=block_type,
            text=text,
            span=self._span(span),
            metadata=metadata or {},
            table_id=table_id,
            figure_id=figure_id,
            formula_id=formula_id,
        )

    def _append_block(
        self,
        *,
        block_type: str,
        text: str,
        span: SourceSpan,
        metadata: dict[str, object],
        table_id: str | None = None,
        figure_id: str | None = None,
        formula_id: str | None = None,
    ) -> CanonicalBlock:
        block = CanonicalBlock(
            block_id=self._block_id(block_type, span, text),
            block_type=block_type,
            text=text,
            section_path=self.heading_path,
            reading_order=len(self.document.blocks),
            source_spans=[span],
            parser_source=self.document.parser_source,
            table_id=table_id,
            figure_id=figure_id,
            formula_id=formula_id,
            metadata=metadata,
        )
        self.document.blocks.append(block)
        return block


def _lines(source: str) -> list[_Line]:
    records: list[_Line] = []
    offset = 0
    for number, raw in enumerate(source.splitlines(keepends=True)):
        content = raw.rstrip("\r\n")
        records.append(
            _Line(
                number=number,
                start=offset,
                content_end=offset + len(content),
                end=offset + len(raw),
                text=content,
            )
        )
        offset += len(raw)
    if source and not records:
        records.append(_Line(0, 0, len(source), len(source), source))
    return records


def _text_span(lines: list[_Line], start: int, end: int) -> SourceSpan:
    return SourceSpan(
        line_start=lines[start].number,
        line_end=lines[end].number,
        char_start=lines[start].start,
        char_end=lines[end].content_end,
    )


def _table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    width = max([len(headers), *(len(row) for row in rows)], default=0)
    if width == 0:
        return ""
    padded_headers = (headers + [""] * width)[:width]
    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in padded_headers) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in rows:
        padded = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in padded) + " |")
    return "\n".join(lines)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _split_pipe_row(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith("\\|"):
        value = value[:-1]
    parts = re.split(r"(?<!\\)\|", value)
    return [part.strip().replace("\\|", "|") for part in parts]


def _is_table_separator(value: str) -> bool:
    cells = _split_pipe_row(value)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


class TextCanonicalAdapter:
    parser_source = "text"

    def parse(self, path: Path) -> CanonicalDocument:
        path = _validate_path(path)
        source = _read_text(path)
        media_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        document = _new_document(path, self.parser_source, media_type)
        builder = _DocumentBuilder(document)
        lines = _lines(source)

        index = 0
        while index < len(lines):
            if not lines[index].text.strip():
                index += 1
                continue
            start = index
            while index + 1 < len(lines) and lines[index + 1].text.strip():
                index += 1
            end = index
            span = _text_span(lines, start, end)
            builder.add_content(source[span.char_start : span.char_end], span)
            index += 1

        if not document.blocks:
            document.warnings.append("Source text is empty; canonical document contains no blocks.")
        return document


class MarkdownCanonicalAdapter:
    parser_source = "markdown"
    _heading_re = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
    _fence_re = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
    _image_re = re.compile(
        r"^[ \t]*!\[(?P<alt>[^]]*)\]\((?P<target>[^\s)]+)(?:\s+[\"'](?P<title>.*?)[\"'])?\)[ \t]*$"
    )

    def parse(self, path: Path) -> CanonicalDocument:
        path = _validate_path(path)
        source = _read_text(path)
        document = _new_document(path, self.parser_source, "text/markdown")
        builder = _DocumentBuilder(document)
        lines = _lines(source)
        index = 0

        while index < len(lines):
            if not lines[index].text.strip():
                index += 1
                continue

            heading_match = self._heading_re.match(lines[index].text)
            if heading_match:
                title = heading_match.group(2).strip()
                builder.add_heading(title, len(heading_match.group(1)), _text_span(lines, index, index))
                if not document.title or document.title == display_title_from_path(path):
                    if len(heading_match.group(1)) == 1:
                        document.title = title
                index += 1
                continue

            fence_match = self._fence_re.match(lines[index].text)
            if fence_match:
                fence = fence_match.group(1)
                end = index + 1
                close_re = re.compile(rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
                while end < len(lines) and not close_re.match(lines[end].text):
                    end += 1
                closed = end < len(lines)
                last = end if closed else len(lines) - 1
                body_start = lines[index].end
                body_end = lines[end].start if closed else lines[last].content_end
                body = source[body_start:body_end].rstrip("\r\n")
                builder.add_content(
                    body,
                    _text_span(lines, index, last),
                    metadata={
                        "kind": "code",
                        "language": fence_match.group(2).strip(),
                        "fence": fence,
                        "closed": closed,
                    },
                )
                if not closed:
                    document.warnings.append(f"Unclosed Markdown code fence at line {index + 1}.")
                index = last + 1
                continue

            formula = self._formula_at(source, lines, index)
            if formula is not None:
                latex, end, delimiter = formula
                span = _text_span(lines, index, end)
                formula_id = _stable_id("formula", str(path), span.char_start, latex)
                canonical_formula = CanonicalFormula(
                    formula_id=formula_id,
                    latex=latex,
                    source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                    metadata={"delimiter": delimiter, "source_format": "latex"},
                )
                document.formulas.append(canonical_formula)
                builder.add_content(
                    latex,
                    span,
                    block_type="formula",
                    formula_id=formula_id,
                    metadata={"delimiter": delimiter},
                )
                index = end + 1
                continue

            if index + 1 < len(lines) and "|" in lines[index].text and _is_table_separator(lines[index + 1].text):
                end = index + 2
                while end < len(lines) and lines[end].text.strip() and "|" in lines[end].text:
                    end += 1
                last = end - 1
                headers = _split_pipe_row(lines[index].text)
                rows = [_split_pipe_row(lines[row].text) for row in range(index + 2, end)]
                span = _text_span(lines, index, last)
                source_markdown = source[span.char_start : span.char_end]
                table_id = _stable_id("table", str(path), span.char_start, source_markdown)
                cells = [
                    CanonicalCell(text=value, row_index=0, column_index=column, is_header=True)
                    for column, value in enumerate(headers)
                ]
                cells.extend(
                    CanonicalCell(text=value, row_index=row_index, column_index=column)
                    for row_index, row in enumerate(rows, start=1)
                    for column, value in enumerate(row)
                )
                table = CanonicalTable(
                    table_id=table_id,
                    headers=headers,
                    rows=rows,
                    cells=cells,
                    source_markdown=source_markdown,
                    normalized_markdown=_table_markdown(headers, rows),
                    source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                )
                document.tables.append(table)
                builder.add_content(
                    table.normalized_markdown or source_markdown,
                    span,
                    block_type="table",
                    table_id=table_id,
                )
                index = end
                continue

            image_match = self._image_re.match(lines[index].text)
            if image_match:
                span = _text_span(lines, index, index)
                raw = source[span.char_start : span.char_end]
                target = image_match.group("target")
                caption = image_match.group("alt") or image_match.group("title") or None
                figure_id = _stable_id("figure", str(path), span.char_start, target)
                figure = CanonicalFigure(
                    figure_id=figure_id,
                    caption=caption,
                    asset_path=target,
                    source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                    metadata={
                        "target": target,
                        "alt": image_match.group("alt"),
                        "title": image_match.group("title"),
                        "source_markdown": raw,
                    },
                )
                document.figures.append(figure)
                builder.add_content(raw, span, block_type="figure", figure_id=figure_id)
                index += 1
                continue

            start = index
            index += 1
            while index < len(lines) and lines[index].text.strip() and not self._is_special(lines, index):
                index += 1
            end = index - 1
            span = _text_span(lines, start, end)
            text = source[span.char_start : span.char_end]
            block = builder.add_content(text, span)
            if builder.heading_path and builder.heading_path[-1].strip().lower().rstrip(":") == "abstract":
                document.abstract = document.abstract or block.text

        if not document.blocks:
            document.warnings.append("Source Markdown is empty; canonical document contains no blocks.")
        return document

    def _is_special(self, lines: list[_Line], index: int) -> bool:
        value = lines[index].text
        return bool(
            self._heading_re.match(value)
            or self._fence_re.match(value)
            or self._image_re.match(value)
            or value.strip().startswith(("$$", "\\["))
            or (
                index + 1 < len(lines)
                and "|" in value
                and _is_table_separator(lines[index + 1].text)
            )
        )

    @staticmethod
    def _formula_at(source: str, lines: list[_Line], index: int) -> tuple[str, int, str] | None:
        stripped = lines[index].text.strip()
        if stripped.startswith("$$"):
            delimiter, closing = "$$", "$$"
        elif stripped.startswith("\\["):
            delimiter, closing = "\\[", "\\]"
        else:
            return None

        first = stripped[len(delimiter) :]
        if first.endswith(closing) and len(first) >= len(closing):
            return first[: -len(closing)].strip(), index, delimiter

        end = index + 1
        while end < len(lines) and not lines[end].text.strip().endswith(closing):
            end += 1
        if end >= len(lines):
            end = len(lines) - 1
            latex_end = lines[end].content_end
        else:
            latex_end = lines[end].content_end - len(closing)
        latex_start = lines[index].start + lines[index].text.find(delimiter) + len(delimiter)
        return source[latex_start:latex_end].strip(), end, delimiter


def _html_selector(element: Tag) -> str:
    if element.get("id"):
        return f"#{element['id']}"
    parts: list[str] = []
    current: Tag | None = element
    while current is not None and current.name != "[document]":
        siblings = (
            current.parent.find_all(current.name, recursive=False) if current.parent else []
        )
        position = siblings.index(current) + 1 if current in siblings else 1
        parts.append(f"{current.name}:nth-of-type({position})")
        current = current.parent if isinstance(current.parent, Tag) else None
    return " > ".join(reversed(parts))


def _html_xpath(element: Tag) -> str:
    parts: list[str] = []
    current: Tag | None = element
    while current is not None and current.name != "[document]":
        siblings = (
            current.parent.find_all(current.name, recursive=False) if current.parent else []
        )
        position = siblings.index(current) + 1 if current in siblings else 1
        parts.append(f"{current.name}[{position}]")
        current = current.parent if isinstance(current.parent, Tag) else None
    return "/" + "/".join(reversed(parts))


def _html_span(element: Tag) -> SourceSpan:
    return SourceSpan(
        xpath=_html_xpath(element),
        css_selector=_html_selector(element),
        element_id=element.get("id"),
    )


def _has_ancestor(element: Tag, names: set[str]) -> bool:
    return any(isinstance(parent, Tag) and parent.name in names for parent in element.parents)


def _is_math_element(element: Tag) -> bool:
    classes = " ".join(element.get("class", [])).lower()
    return element.name == "math" or any(token in classes for token in ("math", "latex", "equation"))


class HtmlCanonicalAdapter:
    parser_source = "html"

    def parse(self, path: Path) -> CanonicalDocument:
        path = _validate_path(path)
        source = _read_text(path)
        soup = BeautifulSoup(source, "html.parser")
        html_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for unwanted in soup.find_all(["script", "style", "nav", "noscript", "template"]):
            unwanted.decompose()

        document = _new_document(path, self.parser_source, "text/html")
        builder = _DocumentBuilder(document)
        first_h1 = soup.find("h1")
        document.title = html_title or (
            first_h1.get_text(" ", strip=True) if first_h1 else display_title_from_path(path)
        )

        for element in list(soup.find_all(True)):
            if element.parent is None or element.name in {
                "html",
                "head",
                "body",
                "title",
                "section",
                "article",
                "main",
            }:
                continue
            if any(isinstance(parent, Tag) and _is_math_element(parent) for parent in element.parents):
                continue
            if re.fullmatch(r"h[1-6]", element.name):
                builder.add_heading(
                    element.get_text(" ", strip=True), int(element.name[1]), _html_span(element)
                )
                continue
            if element.name == "table":
                if _has_ancestor(element, {"table"}):
                    continue
                self._add_table(document, builder, element, path)
                continue
            if element.name == "figure":
                if _has_ancestor(element, {"figure"}):
                    continue
                self._add_figure(document, builder, element, path)
                continue
            if element.name == "pre":
                builder.add_content(
                    element.get_text("", strip=False).strip("\r\n"),
                    _html_span(element),
                    metadata={
                        "kind": "code",
                        "language": self._code_language(element.find("code")),
                    },
                )
                continue
            if element.name == "code" and not _has_ancestor(element, {"pre"}):
                builder.add_content(
                    element.get_text("", strip=False),
                    _html_span(element),
                    metadata={"kind": "code", "language": self._code_language(element)},
                )
                continue
            if _is_math_element(element):
                if any(isinstance(parent, Tag) and _is_math_element(parent) for parent in element.parents):
                    continue
                self._add_formula(document, builder, element, path)
                continue
            if element.name == "img" and not _has_ancestor(element, {"figure"}):
                self._add_image(document, builder, element, path)
                continue
            if element.name == "p" and not _has_ancestor(element, {"table", "figure", "pre"}):
                text = element.get_text(" ", strip=True)
                if text:
                    builder.add_content(text, _html_span(element))

        if not document.blocks:
            document.warnings.append("HTML contains no supported content blocks.")
        return document

    @staticmethod
    def _code_language(element: Tag | None) -> str:
        if element is None:
            return ""
        for class_name in element.get("class", []):
            if class_name.startswith("language-"):
                return class_name.removeprefix("language-")
        return ""

    @staticmethod
    def _add_table(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag, path: Path) -> None:
        span = _html_span(element)
        table_id = _stable_id("table", str(path), span.xpath, str(element))
        cells: list[CanonicalCell] = []
        headers: list[str] = []
        rows: list[list[str]] = []
        for row_index, row in enumerate(element.find_all("tr")):
            row_values: list[str] = []
            row_has_data = False
            for column_index, cell in enumerate(row.find_all(["th", "td"], recursive=False)):
                value = cell.get_text(" ", strip=True)
                is_header = cell.name == "th"
                if is_header:
                    headers.append(value)
                else:
                    row_has_data = True
                    row_values.append(value)
                cells.append(
                    CanonicalCell(
                        text=value,
                        row_index=row_index,
                        column_index=column_index,
                        rowspan=max(1, int(cell.get("rowspan", 1))),
                        colspan=max(1, int(cell.get("colspan", 1))),
                        is_header=is_header,
                        source_spans=[
                            SourceSpan(
                                xpath=_html_xpath(cell),
                                css_selector=_html_selector(cell),
                                element_id=cell.get("id"),
                                table_id=table_id,
                                row_index=row_index,
                                column_index=column_index,
                                heading_path=builder.heading_path,
                            )
                        ],
                    )
                )
            if row_has_data:
                rows.append(row_values)
        caption_element = element.find("caption")
        caption = caption_element.get_text(" ", strip=True) if caption_element else None
        normalized = _table_markdown(headers, rows)
        table = CanonicalTable(
            table_id=table_id,
            caption=caption,
            headers=headers,
            rows=rows,
            cells=cells,
            source_html=str(element),
            normalized_markdown=normalized,
            source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
        )
        document.tables.append(table)
        builder.add_content(normalized, span, block_type="table", table_id=table_id)

    @staticmethod
    def _add_figure(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag, path: Path) -> None:
        span = _html_span(element)
        image = element.find("img")
        caption_element = element.find("figcaption")
        caption = caption_element.get_text(" ", strip=True) if caption_element else None
        src = image.get("src") if image else None
        alt = image.get("alt") if image else None
        figure_id = _stable_id("figure", str(path), span.xpath, src, caption)
        document.figures.append(
            CanonicalFigure(
                figure_id=figure_id,
                caption=caption or alt,
                asset_path=src,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={"src": src, "alt": alt, "source_html": str(element)},
            )
        )
        builder.add_content(caption or alt or src or "Figure", span, block_type="figure", figure_id=figure_id)

    @staticmethod
    def _add_image(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag, path: Path) -> None:
        span = _html_span(element)
        src = element.get("src")
        alt = element.get("alt")
        figure_id = _stable_id("figure", str(path), span.xpath, src)
        document.figures.append(
            CanonicalFigure(
                figure_id=figure_id,
                caption=alt,
                asset_path=src,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={"src": src, "alt": alt, "source_html": str(element)},
            )
        )
        builder.add_content(alt or src or "Image", span, block_type="figure", figure_id=figure_id)

    @staticmethod
    def _add_formula(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag, path: Path) -> None:
        span = _html_span(element)
        annotation = element.find("annotation", attrs={"encoding": re.compile("tex", re.I)})
        latex = element.get("data-latex") or (
            annotation.get_text("", strip=True) if annotation else element.get_text(" ", strip=True)
        )
        formula_id = _stable_id("formula", str(path), span.xpath, latex)
        document.formulas.append(
            CanonicalFormula(
                formula_id=formula_id,
                latex=latex,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={"source_format": "html", "source_html": str(element)},
            )
        )
        builder.add_content(latex, span, block_type="formula", formula_id=formula_id)


class DocxCanonicalAdapter:
    parser_source = "docx"

    def parse(self, path: Path) -> CanonicalDocument:
        path = _validate_path(path)
        try:
            docx = DocxDocument(str(path))
        except Exception as exc:
            raise _parse_error(path, f"Unable to read DOCX: {exc}", exc)
        document = _new_document(
            path,
            self.parser_source,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        document.title = docx.core_properties.title or display_title_from_path(path)
        builder = _DocumentBuilder(document)
        paragraph_index = 0
        table_index = 0
        seen_assets: set[str] = set()

        for child in docx.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, docx)
                paragraph_id = child.get(qn("w14:paraId")) or f"paragraph-{paragraph_index}"
                span = SourceSpan(paragraph_id=paragraph_id)
                style_name = paragraph.style.name if paragraph.style is not None else ""
                heading_match = re.match(r"Heading\s+(\d+)", style_name, re.I)
                text = paragraph.text
                if heading_match and text.strip():
                    builder.add_heading(text, int(heading_match.group(1)), span, style=style_name)
                elif style_name.lower() == "title" and text.strip():
                    document.title = text.strip()
                    builder.add_heading(text, 1, span, style=style_name)
                elif text.strip():
                    builder.add_content(text, span, metadata={"style": style_name})

                self._add_paragraph_images(
                    document, builder, docx, paragraph, span, path, seen_assets
                )
                self._add_paragraph_formulas(document, builder, paragraph, span, path)
                paragraph_index += 1
            elif isinstance(child, CT_Tbl):
                table = Table(child, docx)
                self._add_table(document, builder, table, table_index, path)
                table_index += 1

        document.parser_metadata.update(
            {
                "paragraph_count": paragraph_index,
                "paragraphs": paragraph_index,
                "table_count": table_index,
            }
        )
        if not document.blocks:
            document.warnings.append("DOCX contains no supported content blocks.")
        return document

    @staticmethod
    def _add_table(
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        table: Table,
        table_index: int,
        path: Path,
    ) -> None:
        table_id = _stable_id("table", str(path), "docx", table_index)
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        headers = rows[0] if rows else []
        data_rows = rows[1:] if rows else []
        cells: list[CanonicalCell] = []
        for row_index, row in enumerate(table.rows):
            for column_index, cell in enumerate(row.cells):
                grid_span = cell._tc.tcPr.gridSpan
                colspan = int(grid_span.val) if grid_span is not None else 1
                cell_span = SourceSpan(
                    table_id=table_id,
                    row_index=row_index,
                    column_index=column_index,
                    heading_path=builder.heading_path,
                )
                cells.append(
                    CanonicalCell(
                        text=cell.text,
                        row_index=row_index,
                        column_index=column_index,
                        colspan=max(1, colspan),
                        is_header=row_index == 0,
                        source_spans=[cell_span],
                    )
                )
        span = SourceSpan(table_id=table_id)
        normalized = _table_markdown(headers, data_rows)
        document.tables.append(
            CanonicalTable(
                table_id=table_id,
                headers=headers,
                rows=data_rows,
                cells=cells,
                normalized_markdown=normalized,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={"body_table_index": table_index},
            )
        )
        builder.add_content(normalized, span, block_type="table", table_id=table_id)

    @staticmethod
    def _add_paragraph_images(
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        docx,
        paragraph: Paragraph,
        paragraph_span: SourceSpan,
        path: Path,
        seen_assets: set[str],
    ) -> None:
        for image_index, blip in enumerate(paragraph._p.xpath(".//a:blip")):
            relationship_id = blip.get(qn("r:embed"))
            if not relationship_id or relationship_id not in docx.part.rels:
                document.warnings.append("DOCX image has no readable relationship target.")
                continue
            relationship = docx.part.rels[relationship_id]
            target_part = relationship.target_part
            target_name = Path(str(target_part.partname)).name
            asset_path = f"embedded/{target_name}"
            media_type = getattr(target_part, "content_type", None) or (
                mimetypes.guess_type(target_name)[0] or "application/octet-stream"
            )
            source_target = str(relationship.target_ref).replace("\\", "/")
            image_span = paragraph_span.model_copy(
                update={
                    "image_relationship_id": relationship_id,
                    "heading_path": builder.heading_path,
                    "metadata": {"relationship_target": source_target},
                }
            )
            asset_id = _stable_id("asset", str(path), relationship_id, target_name)
            if asset_id not in seen_assets:
                blob = target_part.blob
                document.assets.append(
                    CanonicalAsset(
                        asset_id=asset_id,
                        path=asset_path,
                        media_type=media_type,
                        sha256=hashlib.sha256(blob).hexdigest(),
                        source_path=source_target,
                        source_spans=[image_span],
                        metadata={
                            "relationship_id": relationship_id,
                            "embedded": True,
                            "size_bytes": len(blob),
                        },
                    )
                )
                seen_assets.add(asset_id)
            figure_id = _stable_id(
                "figure", str(path), paragraph_span.paragraph_id, relationship_id, image_index
            )
            caption = paragraph.text.strip() or None
            document.figures.append(
                CanonicalFigure(
                    figure_id=figure_id,
                    caption=caption,
                    asset_path=asset_path,
                    source_spans=[image_span],
                    metadata={
                        "asset_id": asset_id,
                        "relationship_id": relationship_id,
                        "relationship_target": source_target,
                    },
                )
            )
            builder.add_content(
                caption or target_name,
                image_span,
                block_type="figure",
                figure_id=figure_id,
                metadata={"relationship_id": relationship_id},
            )

    @staticmethod
    def _add_paragraph_formulas(
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        paragraph: Paragraph,
        paragraph_span: SourceSpan,
        path: Path,
    ) -> None:
        for formula_index, node in enumerate(paragraph._p.xpath(".//m:oMath")):
            omml = etree.tostring(node, encoding="unicode")
            source_text = "".join(text_node.text or "" for text_node in node.iter(qn("m:t"))).strip()
            formula_id = _stable_id(
                "formula", str(path), paragraph_span.paragraph_id, formula_index, omml
            )
            formula_span = paragraph_span.model_copy(
                update={
                    "heading_path": builder.heading_path,
                    "metadata": {"formula_index": formula_index},
                }
            )
            document.formulas.append(
                CanonicalFormula(
                    formula_id=formula_id,
                    latex=source_text or "[OMML formula]",
                    source_spans=[formula_span],
                    warnings=["OMML source text is preserved but is not normalized LaTeX."],
                    metadata={"source_format": "omml", "omml": omml},
                )
            )
            builder.add_content(
                source_text or "[OMML formula]",
                formula_span,
                block_type="formula",
                formula_id=formula_id,
                metadata={"source_format": "omml"},
            )


class PDFCanonicalAdapter:
    parser_source = "pdf_legacy"

    def parse(self, path: Path) -> CanonicalDocument:
        path = _validate_path(path)
        # Task 5 replaces this compatibility bridge with MinerU-first canonical orchestration.
        from app.services.parser import _parse_pdf

        parsed = _parse_pdf(path)
        document = _new_document(path, self.parser_source, "application/pdf")
        document.title = parsed.title
        document.metadata.update(parsed.metadata)
        document.parser_metadata.update(
            {"legacy_parser_metadata": parsed.metadata, "compatibility_bridge": True}
        )
        builder = _DocumentBuilder(document)
        chunks = parsed.chunks
        if not chunks and parsed.text:
            from app.services.parser import ParsedChunk

            chunks = [ParsedChunk(ordinal=0, text=parsed.text)]
        for chunk in chunks:
            page_index = None
            if chunk.page_label and str(chunk.page_label).isdigit():
                page_index = max(0, int(chunk.page_label) - 1)
            span = SourceSpan(
                page_index=page_index,
                page_label=chunk.page_label,
                source_block_id=f"legacy-chunk-{chunk.ordinal}",
            )
            block = builder.add_content(
                chunk.text,
                span,
                metadata={"legacy_ordinal": chunk.ordinal},
            )
            if chunk.heading:
                block.section_path = [chunk.heading]
                block.source_spans[0].heading_path = [chunk.heading]
        if not document.blocks:
            document.warnings.append("PDF parser returned no content blocks.")
        return document


def parse_canonical_document(path: Path) -> CanonicalDocument:
    path = _validate_path(Path(path))
    adapter: CanonicalAdapter = {
        ".pdf": PDFCanonicalAdapter(),
        ".docx": DocxCanonicalAdapter(),
        ".html": HtmlCanonicalAdapter(),
        ".htm": HtmlCanonicalAdapter(),
        ".md": MarkdownCanonicalAdapter(),
        ".markdown": MarkdownCanonicalAdapter(),
        ".txt": TextCanonicalAdapter(),
    }.get(path.suffix.lower(), TextCanonicalAdapter())
    return adapter.parse(path)
