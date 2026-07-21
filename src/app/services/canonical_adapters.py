from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from lxml import etree

from app.core.config import get_settings
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
        with path.open("r", encoding="utf-8", newline="") as source_file:
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
        document_id=_stable_id("doc", parser_source, source_hash),
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
            self.document.document_id,
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
            if self._appendix_level is not None:
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
            retrievable=block_type != "heading" and self._reference_level is None,
        )
        self.document.blocks.append(block)
        return block


def _lines(source: str) -> list[_Line]:
    records: list[_Line] = []
    offset = 0
    for number, raw in enumerate(source.splitlines(keepends=True)):
        raw_content = raw.rstrip("\r\n")
        has_bom = number == 0 and raw_content.startswith("\ufeff")
        content = raw_content[1:] if has_bom else raw_content
        records.append(
            _Line(
                number=number,
                start=offset + (1 if has_bom else 0),
                content_end=offset + len(raw_content),
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


def _span_for_chars(lines: list[_Line], char_start: int, char_end: int) -> SourceSpan:
    start_line = next(
        line for line in lines if line.start <= char_start <= max(line.content_end, line.start)
    )
    end_position = max(char_start, char_end - 1)
    end_line = next(
        line for line in lines if line.start <= end_position <= max(line.content_end, line.start)
    )
    return SourceSpan(
        line_start=start_line.number,
        line_end=end_line.number,
        char_start=char_start,
        char_end=char_end,
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
        r"!\[(?P<alt>[^]]*)\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<bare>[^\s)]+))"
        r"(?:\s+(?:\"(?P<double_title>[^\"]*)\"|'(?P<single_title>[^']*)'))?\s*\)"
    )
    _setext_re = re.compile(r"^[ \t]*(?P<underline>=+|-+)[ \t]*$")

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

            setext_level = self._setext_level(lines, index)
            if setext_level is not None:
                title = lines[index].text.strip()
                builder.add_heading(title, setext_level, _text_span(lines, index, index + 1))
                if setext_level == 1 and document.title == display_title_from_path(path):
                    document.title = title
                index += 2
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
                body_end = lines[end].start if closed else len(source)
                body = source[body_start:body_end]
                code_span = _text_span(lines, index, last)
                if not closed:
                    code_span = code_span.model_copy(update={"char_end": len(source)})
                builder.add_content(
                    body,
                    code_span,
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
                formula_id = _stable_id("formula", document.document_id, span.char_start, latex)
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
                table_id = _stable_id(
                    "table", document.document_id, span.char_start, source_markdown
                )
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

            image_match = self._image_re.fullmatch(lines[index].text.strip())
            if image_match:
                span = _text_span(lines, index, index)
                raw = source[span.char_start : span.char_end]
                self._add_image(document, builder, span, raw, image_match)
                index += 1
                continue

            start = index
            index += 1
            while index < len(lines) and lines[index].text.strip() and not self._is_special(lines, index):
                index += 1
            end = index - 1
            span = _text_span(lines, start, end)
            text = source[span.char_start : span.char_end]
            blocks = self._add_paragraph(document, builder, lines, text, span)
            if builder.heading_path and builder.heading_path[-1].strip().lower().rstrip(":") == "abstract":
                abstract_text = "".join(
                    block.text for block in blocks if block.block_type == "narrative"
                )
                document.abstract = document.abstract or abstract_text or None

        if not document.blocks:
            document.warnings.append("Source Markdown is empty; canonical document contains no blocks.")
        return document

    def _is_special(self, lines: list[_Line], index: int) -> bool:
        value = lines[index].text
        return bool(
            self._heading_re.match(value)
            or self._fence_re.match(value)
            or self._image_re.fullmatch(value.strip())
            or self._setext_level(lines, index) is not None
            or value.strip().startswith(("$$", "\\["))
            or (
                index + 1 < len(lines)
                and "|" in value
                and _is_table_separator(lines[index + 1].text)
            )
        )

    def _setext_level(self, lines: list[_Line], index: int) -> int | None:
        if index + 1 >= len(lines) or not lines[index].text.strip():
            return None
        match = self._setext_re.fullmatch(lines[index + 1].text)
        if match is None:
            return None
        return 1 if match.group("underline").startswith("=") else 2

    def _add_paragraph(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        lines: list[_Line],
        text: str,
        span: SourceSpan,
    ) -> list[CanonicalBlock]:
        blocks: list[CanonicalBlock] = []
        absolute_start = span.char_start or 0
        cursor = 0
        for match in self._image_re.finditer(text):
            if match.start() > cursor:
                before = text[cursor : match.start()]
                if before.strip():
                    before_start = absolute_start + cursor
                    blocks.append(
                        builder.add_content(
                            before,
                            _span_for_chars(lines, before_start, absolute_start + match.start()),
                        )
                    )
            raw = match.group(0)
            image_start = absolute_start + match.start()
            blocks.append(
                self._add_image(
                    document,
                    builder,
                    _span_for_chars(lines, image_start, image_start + len(raw)),
                    raw,
                    match,
                )
            )
            cursor = match.end()
        if cursor < len(text):
            after = text[cursor:]
            if after.strip():
                after_start = absolute_start + cursor
                blocks.append(
                    builder.add_content(
                        after,
                        _span_for_chars(lines, after_start, absolute_start + len(text)),
                    )
                )
        if not blocks and text.strip():
            blocks.append(builder.add_content(text, span))
        return blocks

    @staticmethod
    def _add_image(
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        span: SourceSpan,
        raw: str,
        match: re.Match[str],
    ) -> CanonicalBlock:
        target = match.group("angle") or match.group("bare")
        title = match.group("double_title") or match.group("single_title")
        caption = match.group("alt") or title or None
        figure_id = _stable_id(
            "figure", document.document_id, span.char_start, target, caption
        )
        document.figures.append(
            CanonicalFigure(
                figure_id=figure_id,
                caption=caption,
                asset_path=target,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={
                    "target": target,
                    "alt": match.group("alt"),
                    "title": title,
                    "source_markdown": raw,
                },
            )
        )
        return builder.add_content(
            caption or target,
            span,
            block_type="figure",
            figure_id=figure_id,
            metadata={"source_markdown": raw, "target": target, "title": title},
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

        root = soup.body or soup
        for child in list(root.children):
            self._walk_node(document, builder, child)

        if not document.blocks:
            document.warnings.append("HTML contains no supported content blocks.")
        return document

    def _walk_node(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        node: Tag | NavigableString,
    ) -> None:
        if isinstance(node, NavigableString):
            if str(node).strip() and isinstance(node.parent, Tag):
                builder.add_content(str(node), _html_span(node.parent))
            return
        if not isinstance(node, Tag):
            return
        if re.fullmatch(r"h[1-6]", node.name):
            builder.add_heading(node.get_text(" ", strip=True), int(node.name[1]), _html_span(node))
            return
        if node.name == "table":
            self._walk_table_tree(document, builder, node)
            return
        if node.name == "figure":
            self._add_figure(document, builder, node)
            return
        if node.name == "p":
            self._walk_inline_container(document, builder, node)
            return
        if node.name == "pre":
            builder.add_content(
                node.get_text("", strip=False).strip("\r\n"),
                _html_span(node),
                metadata={"kind": "code", "language": self._code_language(node.find("code"))},
            )
            return
        if node.name == "code":
            builder.add_content(
                node.get_text("", strip=False),
                _html_span(node),
                metadata={"kind": "code", "language": self._code_language(node)},
            )
            return
        if _is_math_element(node):
            self._add_formula(document, builder, node)
            return
        if node.name == "img":
            self._add_image(document, builder, node)
            return
        for child in list(node.children):
            self._walk_node(document, builder, child)

    def _walk_table_tree(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        table: Tag,
    ) -> None:
        self._add_table(document, builder, table)
        for nested in table.find_all("table"):
            if nested.find_parent("table") is table:
                self._walk_table_tree(document, builder, nested)

    def _walk_inline_container(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        container: Tag,
    ) -> None:
        buffer: list[str] = []
        text_index = 0

        def flush() -> None:
            nonlocal text_index
            text = "".join(buffer)
            buffer.clear()
            if not text.strip():
                return
            text_index += 1
            parent_span = _html_span(container)
            builder.add_content(
                text,
                parent_span.model_copy(
                    update={
                        "xpath": f"{parent_span.xpath}/text()[{text_index}]",
                        "metadata": {"text_segment_index": text_index},
                    }
                ),
            )

        def visit(node: Tag | NavigableString) -> None:
            if isinstance(node, NavigableString):
                buffer.append(str(node))
                return
            if not isinstance(node, Tag):
                return
            if node.name == "br":
                buffer.append("\n")
                return
            if node.name == "code":
                flush()
                builder.add_content(
                    node.get_text("", strip=False),
                    _html_span(node),
                    metadata={"kind": "code", "language": self._code_language(node)},
                )
                return
            if _is_math_element(node):
                flush()
                self._add_formula(document, builder, node)
                return
            if node.name == "img":
                flush()
                self._add_image(document, builder, node)
                return
            for child in list(node.children):
                visit(child)

        for child in list(container.children):
            visit(child)
        flush()

    @staticmethod
    def _code_language(element: Tag | None) -> str:
        if element is None:
            return ""
        for class_name in element.get("class", []):
            if class_name.startswith("language-"):
                return class_name.removeprefix("language-")
        return ""

    @staticmethod
    def _add_table(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag) -> None:
        span = _html_span(element)
        table_id = _stable_id("table", document.document_id, span.xpath, str(element))
        cells: list[CanonicalCell] = []
        grid: list[list[str]] = []
        rowspan_until: dict[int, int] = {}
        direct_rows = [row for row in element.find_all("tr") if row.find_parent("table") is element]
        header_row = False
        for row_index, row in enumerate(direct_rows):
            occupied = {
                column for column, last_row in rowspan_until.items() if last_row >= row_index
            }
            row_values: list[str] = [""] * (max(occupied, default=-1) + 1)
            column_index = 0
            direct_cells = [
                cell
                for cell in row.find_all(["th", "td"], recursive=False)
                if cell.find_parent("tr") is row
            ]
            if row_index == 0:
                header_row = any(cell.name == "th" for cell in direct_cells)
            for cell in direct_cells:
                rowspan = HtmlCanonicalAdapter._safe_span_value(
                    document, cell, "rowspan"
                )
                colspan = HtmlCanonicalAdapter._safe_span_value(
                    document, cell, "colspan"
                )
                while any(
                    logical_column in occupied
                    for logical_column in range(column_index, column_index + colspan)
                ):
                    column_index += 1
                value = HtmlCanonicalAdapter._cell_text(element, cell)
                is_header = cell.name == "th"
                required = column_index + colspan
                if len(row_values) < required:
                    row_values.extend([""] * (required - len(row_values)))
                row_values[column_index] = value
                cells.append(
                    CanonicalCell(
                        text=value,
                        row_index=row_index,
                        column_index=column_index,
                        rowspan=rowspan,
                        colspan=colspan,
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
                for occupied_column in range(column_index, column_index + colspan):
                    occupied.add(occupied_column)
                    if rowspan > 1:
                        rowspan_until[occupied_column] = row_index + rowspan - 1
                column_index += colspan
            grid.append(row_values)
        width = max((len(row) for row in grid), default=0)
        grid = [(row + [""] * width)[:width] for row in grid]
        headers = grid[0] if header_row and grid else []
        rows = grid[1:] if header_row else grid
        caption_element = element.find("caption")
        if caption_element is not None and caption_element.find_parent("table") is not element:
            caption_element = None
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
        if caption_element is not None and caption:
            builder.add_content(
                caption,
                _html_span(caption_element),
                block_type="caption",
                table_id=table_id,
            )
        builder.add_content(normalized, span, block_type="table", table_id=table_id)

    @staticmethod
    def _safe_span_value(document: CanonicalDocument, cell: Tag, attribute: str) -> int:
        raw = cell.get(attribute, 1)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            document.warnings.append(
                f"Invalid HTML {attribute}={raw!r}; using 1 at {_html_xpath(cell)}."
            )
            return 1
        if value < 1:
            document.warnings.append(
                f"Invalid HTML {attribute}={raw!r}; using 1 at {_html_xpath(cell)}."
            )
            return 1
        return value

    @staticmethod
    def _cell_text(table: Tag, cell: Tag) -> str:
        pieces = [
            str(item).strip()
            for item in cell.find_all(string=True)
            if item.find_parent("table") is table and str(item).strip()
        ]
        return " ".join(pieces)

    @staticmethod
    def _add_figure(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag) -> None:
        span = _html_span(element)
        image = element.find("img")
        caption_element = element.find("figcaption")
        caption = caption_element.get_text(" ", strip=True) if caption_element else None
        src = image.get("src") if image else None
        alt = image.get("alt") if image else None
        figure_id = _stable_id("figure", document.document_id, span.xpath, src, caption)
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
        if caption_element is not None and caption:
            builder.add_content(
                caption,
                _html_span(caption_element),
                block_type="caption",
                figure_id=figure_id,
            )

    @staticmethod
    def _add_image(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag) -> None:
        span = _html_span(element)
        src = element.get("src")
        alt = element.get("alt")
        figure_id = _stable_id("figure", document.document_id, span.xpath, src)
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
    def _add_formula(document: CanonicalDocument, builder: _DocumentBuilder, element: Tag) -> None:
        span = _html_span(element)
        annotation = element.find("annotation", attrs={"encoding": re.compile("tex", re.I)})
        latex = element.get("data-latex") or (
            annotation.get_text("", strip=True) if annotation else element.get_text(" ", strip=True)
        )
        formula_id = _stable_id("formula", document.document_id, span.xpath, latex)
        document.formulas.append(
            CanonicalFormula(
                formula_id=formula_id,
                latex=latex,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={"source_format": "html", "source_html": str(element)},
            )
        )
        builder.add_content(latex, span, block_type="formula", formula_id=formula_id)


@dataclass(frozen=True)
class _DocxEvent:
    kind: str
    text: str = ""
    element: object | None = None
    relationship_id: str | None = None
    metadata: dict[str, str] | None = None


class DocxCanonicalAdapter:
    parser_source = "docx"

    def __init__(self, asset_cache_root: Path | None = None) -> None:
        self.asset_cache_root = Path(asset_cache_root) if asset_cache_root is not None else None

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

        for child in docx.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, docx)
                paragraph_id = child.get(qn("w14:paraId")) or f"paragraph-{paragraph_index}"
                self._emit_paragraph(
                    document,
                    builder,
                    docx,
                    paragraph,
                    SourceSpan(paragraph_id=paragraph_id),
                )
                paragraph_index += 1
            elif isinstance(child, CT_Tbl):
                self._add_table(
                    document, builder, docx, Table(child, docx), table_index
                )
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

    def _emit_paragraph(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        docx,
        paragraph: Paragraph,
        span: SourceSpan,
        *,
        structures_only: bool = False,
    ) -> None:
        events = list(self._paragraph_events(paragraph))
        style_name = paragraph.style.name if paragraph.style is not None else ""
        heading_match = re.match(r"Heading\s+(\d+)", style_name, re.I)
        heading_text = "".join(event.text for event in events if event.kind == "text")
        if not structures_only and heading_match and heading_text.strip():
            builder.add_heading(
                heading_text, int(heading_match.group(1)), span, style=style_name
            )
            for event_index, event in enumerate(events):
                if event.kind != "text":
                    self._emit_structure_event(
                        document, builder, docx, event, span, event_index
                    )
            return
        if not structures_only and style_name.lower() == "title" and heading_text.strip():
            document.title = heading_text.strip()
            builder.add_heading(heading_text, 1, span, style=style_name)
            for event_index, event in enumerate(events):
                if event.kind != "text":
                    self._emit_structure_event(
                        document, builder, docx, event, span, event_index
                    )
            return

        text_buffer: list[str] = []

        def flush() -> None:
            text = "".join(text_buffer)
            text_buffer.clear()
            if not structures_only and text.strip():
                builder.add_content(text, span, metadata={"style": style_name})

        for event_index, event in enumerate(events):
            if event.kind == "text":
                text_buffer.append(event.text)
                continue
            flush()
            self._emit_structure_event(document, builder, docx, event, span, event_index)
        flush()

    @classmethod
    def _paragraph_events(cls, paragraph: Paragraph):
        def walk(node):
            if node.tag in {qn("m:oMath"), qn("m:oMathPara")}:
                omml = etree.tostring(node, encoding="unicode")
                text = "".join(item.text or "" for item in node.iter(qn("m:t"))).strip()
                yield _DocxEvent("formula", text=text, element=node, metadata={"omml": omml})
                return
            if node.tag in {qn("w:drawing"), qn("w:pict"), qn("w:object")}:
                doc_properties = next(iter(node.iter(qn("wp:docPr"))), None)
                metadata = {
                    key: doc_properties.get(key)
                    for key in ("descr", "title", "name")
                    if doc_properties is not None and doc_properties.get(key)
                }
                for blip in node.iter(qn("a:blip")):
                    yield _DocxEvent(
                        "image",
                        element=blip,
                        relationship_id=blip.get(qn("r:embed")),
                        metadata=metadata,
                    )
                return
            if node.tag in {qn("w:t"), qn("w:instrText")}:
                yield _DocxEvent("text", text=node.text or "")
                return
            if node.tag == qn("w:tab"):
                yield _DocxEvent("text", text="\t")
                return
            if node.tag in {qn("w:br"), qn("w:cr")}:
                yield _DocxEvent("text", text="\n")
                return
            for child in node.iterchildren():
                yield from walk(child)

        for child in paragraph._p.iterchildren():
            yield from walk(child)

    def _emit_structure_event(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        docx,
        event: _DocxEvent,
        span: SourceSpan,
        event_index: int,
    ) -> None:
        if event.kind == "image":
            self._add_image(document, builder, docx, event, span, event_index)
        elif event.kind == "formula":
            self._add_formula(document, builder, event, span, event_index)

    def _add_image(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        docx,
        event: _DocxEvent,
        span: SourceSpan,
        event_index: int,
    ) -> None:
        relationship_id = event.relationship_id
        if not relationship_id or relationship_id not in docx.part.rels:
            document.warnings.append("DOCX image has no readable relationship target.")
            return
        relationship = docx.part.rels[relationship_id]
        target_part = relationship.target_part
        blob = target_part.blob
        asset_sha = hashlib.sha256(blob).hexdigest()
        target_name = Path(str(target_part.partname)).name
        safe_name = self._safe_asset_name(target_name, asset_sha)
        asset_path = f"assets/{safe_name}"
        source_path = self._materialize_asset(document, safe_name, blob, asset_sha)
        media_type = getattr(target_part, "content_type", None) or (
            mimetypes.guess_type(target_name)[0] or "application/octet-stream"
        )
        source_target = str(relationship.target_ref).replace("\\", "/")
        image_span = span.model_copy(
            update={
                "image_relationship_id": relationship_id,
                "heading_path": builder.heading_path,
                "metadata": {"relationship_target": source_target},
            }
        )
        asset_id = _stable_id("asset", document.document_id, asset_sha)
        if not any(asset.asset_id == asset_id for asset in document.assets):
            document.assets.append(
                CanonicalAsset(
                    asset_id=asset_id,
                    path=asset_path,
                    media_type=media_type,
                    sha256=asset_sha,
                    source_path=str(source_path),
                    source_spans=[image_span],
                    metadata={
                        "relationship_id": relationship_id,
                        "relationship_target": source_target,
                        "embedded": True,
                        "size_bytes": len(blob),
                    },
                )
            )
        metadata = event.metadata or {}
        caption = metadata.get("descr") or metadata.get("title") or metadata.get("name")
        figure_id = _stable_id(
            "figure",
            document.document_id,
            span.paragraph_id,
            span.table_id,
            span.row_index,
            span.column_index,
            relationship_id,
            event_index,
        )
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
                    **metadata,
                },
            )
        )
        builder.add_content(
            caption or target_name,
            image_span,
            block_type="figure",
            figure_id=figure_id,
            metadata={"relationship_id": relationship_id, **metadata},
        )

    def _materialize_asset(
        self,
        document: CanonicalDocument,
        safe_name: str,
        blob: bytes,
        expected_sha: str,
    ) -> Path:
        root = self.asset_cache_root
        if root is None:
            root = get_settings().cache_dir / "canonical_adapter_assets"
        source_sha = str(document.source_metadata["sha256"])
        directory = Path(root).expanduser().resolve() / source_sha
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / safe_name
        if destination.is_file() and self._file_sha256(destination) == expected_sha:
            return destination
        temporary = directory / f".{safe_name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as output:
                output.write(blob)
                output.flush()
                os.fsync(output.fileno())
            if self._file_sha256(temporary) != expected_sha:
                raise ValueError("materialized DOCX asset hash mismatch")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if self._file_sha256(destination) != expected_sha:
            raise ValueError("persisted DOCX asset hash mismatch")
        return destination

    @staticmethod
    def _safe_asset_name(target_name: str, asset_sha: str) -> str:
        suffix = Path(target_name).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
            suffix = ""
        return f"image-{asset_sha[:24]}{suffix}"

    @staticmethod
    def _file_sha256(path: Path) -> str:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    @staticmethod
    def _add_formula(
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        event: _DocxEvent,
        span: SourceSpan,
        event_index: int,
    ) -> None:
        omml = (event.metadata or {}).get("omml", "")
        source_text = event.text.strip()
        formula_id = _stable_id(
            "formula",
            document.document_id,
            span.paragraph_id,
            span.table_id,
            span.row_index,
            span.column_index,
            event_index,
            omml,
        )
        formula_span = span.model_copy(
            update={
                "heading_path": builder.heading_path,
                "metadata": {"formula_event_index": event_index},
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

    def _add_table(
        self,
        document: CanonicalDocument,
        builder: _DocumentBuilder,
        docx,
        table: Table,
        table_index: int,
    ) -> None:
        table_id = _stable_id("table", document.document_id, "docx", table_index)
        cells: list[CanonicalCell] = []
        grid: list[list[str]] = []
        active_merges: dict[int, CanonicalCell] = {}
        cell_paragraphs: list[tuple[Paragraph, int, int, int]] = []
        raw_rows = list(table._tbl.findall(qn("w:tr")))

        for row_index, raw_row in enumerate(raw_rows):
            row_values: list[str] = []
            column_index = self._grid_before(raw_row)
            if column_index:
                row_values.extend([""] * column_index)
            continued: set[int] = set()
            restarted: set[int] = set()
            for raw_cell in raw_row.findall(qn("w:tc")):
                cell = _Cell(raw_cell, table)
                colspan = self._docx_grid_span(raw_cell)
                vmerge = raw_cell.tcPr.vMerge
                merge_value = str(vmerge.val).lower() if vmerge is not None else ""
                is_restart = vmerge is not None and merge_value == "restart"
                is_continue = vmerge is not None and not is_restart
                required = column_index + colspan
                if len(row_values) < required:
                    row_values.extend([""] * (required - len(row_values)))

                if is_continue and column_index in active_merges:
                    origin = active_merges[column_index]
                    origin.rowspan += 1
                    for logical_column in range(column_index, column_index + colspan):
                        continued.add(logical_column)
                    column_index += colspan
                    continue

                text = cell.text
                row_values[column_index] = text
                cell_span = SourceSpan(
                    table_id=table_id,
                    row_index=row_index,
                    column_index=column_index,
                    heading_path=builder.heading_path,
                )
                canonical_cell = CanonicalCell(
                    text=text,
                    row_index=row_index,
                    column_index=column_index,
                    colspan=colspan,
                    is_header=row_index == 0,
                    source_spans=[cell_span],
                )
                cells.append(canonical_cell)
                for logical_column in range(column_index, column_index + colspan):
                    active_merges.pop(logical_column, None)
                    if is_restart:
                        active_merges[logical_column] = canonical_cell
                        restarted.add(logical_column)
                for paragraph_index, paragraph in enumerate(cell.paragraphs):
                    cell_paragraphs.append(
                        (paragraph, row_index, column_index, paragraph_index)
                    )
                column_index += colspan
            active_merges = {
                column: origin
                for column, origin in active_merges.items()
                if column in continued or column in restarted
            }
            grid.append(row_values)

        width = max((len(row) for row in grid), default=0)
        grid = [(row + [""] * width)[:width] for row in grid]
        headers = grid[0] if grid else []
        rows = grid[1:] if grid else []
        span = SourceSpan(table_id=table_id)
        normalized = _table_markdown(headers, rows)
        document.tables.append(
            CanonicalTable(
                table_id=table_id,
                headers=headers,
                rows=rows,
                cells=cells,
                normalized_markdown=normalized,
                source_spans=[span.model_copy(update={"heading_path": builder.heading_path})],
                metadata={"body_table_index": table_index},
            )
        )
        builder.add_content(normalized, span, block_type="table", table_id=table_id)

        for paragraph, row_index, column_index, paragraph_index in cell_paragraphs:
            paragraph_id = paragraph._p.get(qn("w14:paraId")) or (
                f"table-{table_index}-r{row_index}-c{column_index}-p{paragraph_index}"
            )
            self._emit_paragraph(
                document,
                builder,
                docx,
                paragraph,
                SourceSpan(
                    paragraph_id=paragraph_id,
                    table_id=table_id,
                    row_index=row_index,
                    column_index=column_index,
                ),
                structures_only=True,
            )

    @staticmethod
    def _docx_grid_span(raw_cell) -> int:
        grid_span = raw_cell.tcPr.gridSpan
        if grid_span is None:
            return 1
        try:
            return max(1, int(grid_span.val))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _grid_before(raw_row) -> int:
        grid_before = raw_row.find(f"./{qn('w:trPr')}/{qn('w:gridBefore')}")
        if grid_before is None:
            return 0
        try:
            return max(0, int(grid_before.get(qn("w:val"), "0")))
        except ValueError:
            return 0


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
