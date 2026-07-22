from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

from bs4 import BeautifulSoup

from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityReport,
    CanonicalTable,
    SectionNode,
)
from app.services.canonical_provenance import block_is_generated, source_only_document


_REQUIRED_FILES = {
    "canonical.md",
    "manifest.json",
    "blocks.jsonl",
    "tables.json",
    "figures.json",
    "formulas.json",
}
_REQUIRED_MANIFEST_FIELDS = {
    "canonical_markdown_sha256",
    "input_fingerprint",
    "document_id",
    "version",
    "parser",
    "source",
    "quality",
    "document",
    "warnings",
    "assets",
    "status",
}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARACTERS = set('/\\<>:"|?*')
_REMOTE_URI_SCHEMES = {"http", "https", "s3", "gs", "minio"}
_MAX_TABLE_ROWS = 10_000
_MAX_TABLE_COLUMNS = 1_000
_MAX_TABLE_GRID_CELLS = 1_000_000


class CanonicalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _prepare_document_root(
        self,
        document_id: str,
        *,
        create: bool,
        allow_missing: bool = False,
    ) -> Path:
        if self._is_link_or_reparse_point(self.root):
            raise ValueError(f"store root cannot be a symbolic link: {self.root}")
        if self.root.exists():
            if not self.root.is_dir():
                raise ValueError(f"store root is not a directory: {self.root}")
        elif create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not allow_missing:
            raise FileNotFoundError(f"store root does not exist: {self.root}")

        document_root = self.root / document_id
        if self._is_link_or_reparse_point(document_root):
            raise ValueError(
                f"document root cannot be a symbolic link: {document_root}"
            )
        if document_root.exists():
            if not document_root.is_dir():
                raise ValueError(f"document root is not a directory: {document_root}")
        elif create:
            document_root.mkdir()
        elif not allow_missing:
            raise FileNotFoundError(f"document root does not exist: {document_root}")

        root_resolved = self.root.resolve()
        document_resolved = document_root.resolve()
        if document_resolved.parent != root_resolved:
            raise ValueError(f"document root escapes store root: {document_root}")
        return document_root

    def write_staging(
        self,
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> Path:
        self._validate_component(document_id)
        self._validate_component(version)
        if document.document_id and document.document_id != document_id:
            raise ValueError(
                f"document_id {document.document_id!r} does not match {document_id!r}"
            )
        if document.parse_version and document.parse_version != version:
            raise ValueError(
                f"parse_version {document.parse_version!r} does not match {version!r}"
            )
        document = source_only_document(document)
        document.ensure_json_compatible()
        failed_tables = [
            table.table_id
            for table in document.tables
            if table.status == "validation_failed"
        ]
        if failed_tables or document.metadata.get("table_activation_allowed") is False:
            raise ValueError(
                "canonical table validation failed; staging is not activation-safe: "
                f"{failed_tables}"
            )
        self._validate_record_contract(
            document.blocks,
            document.tables,
            document.figures,
            document.formulas,
            document.assets,
            document.outline,
            document.quality,
        )
        document_root = self._prepare_document_root(document_id, create=True)
        final = document_root / version
        if final.exists():
            raise FileExistsError(f"canonical bundle already exists: {final}")

        staging_candidates = list(document_root.glob(f"{version}.staging-*"))
        if len(staging_candidates) > 1:
            raise ValueError(
                "write_staging found multiple staging directories; "
                f"found {len(staging_candidates)}"
            )
        if staging_candidates:
            existing_staging = staging_candidates[0]
            try:
                if self._is_link_or_reparse_point(existing_staging):
                    raise ValueError("staging path is a symbolic link")
                self._validate_bundle(existing_staging, document_id, version)
            except Exception as exc:
                raise ValueError(
                    f"existing staging bundle is invalid: {existing_staging}"
                ) from exc
            incoming_asset_hashes = self._resolve_asset_hashes(document.assets)
            incoming_payload = self._build_persisted_input_payload(
                document_id,
                version,
                document,
                incoming_asset_hashes,
            )
            manifest = self._read_json(existing_staging / "manifest.json")
            if manifest["input_fingerprint"] != self._input_fingerprint(
                incoming_payload
            ):
                raise ValueError("existing staging input fingerprint mismatch")
            return existing_staging

        staging = document_root / f"{version}.staging-{uuid4().hex}"
        staging.mkdir()

        try:
            (staging / "assets").mkdir()
            asset_hashes = self._copy_assets(staging, document.assets)

            canonical_markdown = self._render_markdown(document_id, version, document)
            canonical_markdown_path = staging / "canonical.md"
            self._write_text(canonical_markdown_path, canonical_markdown)
            self._write_json(
                staging / "manifest.json",
                self._build_manifest(
                    document_id,
                    version,
                    document,
                    canonical_markdown_sha256=self._sha256(canonical_markdown_path),
                    asset_hashes=asset_hashes,
                ),
            )
            self._write_blocks(staging / "blocks.jsonl", document.blocks)
            self._write_json(
                staging / "tables.json",
                [self._bundle_model_dump(table) for table in document.tables],
            )
            self._write_json(
                staging / "figures.json",
                [self._bundle_model_dump(figure) for figure in document.figures],
            )
            self._write_json(
                staging / "formulas.json",
                [self._bundle_model_dump(formula) for formula in document.formulas],
            )
            self._validate_bundle(staging, document_id, version)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return staging

    def promote(self, document_id: str, version: str) -> Path:
        self._validate_component(document_id)
        self._validate_component(version)

        document_root = self._prepare_document_root(
            document_id,
            create=False,
            allow_missing=True,
        )
        final = document_root / version
        if final.exists():
            raise FileExistsError(f"canonical bundle already exists: {final}")

        staging_candidates = list(document_root.glob(f"{version}.staging-*"))
        if len(staging_candidates) != 1:
            raise ValueError(
                "promotion requires exactly one staging directory; "
                f"found {len(staging_candidates)}"
            )

        staging = staging_candidates[0]
        if not staging.is_dir():
            raise ValueError(f"staging path is not a directory: {staging}")
        self._validate_bundle(staging, document_id, version)

        if final.exists():
            raise FileExistsError(f"canonical bundle already exists: {final}")
        os.rename(staging, final)
        return final

    def load(self, document_id: str, version: str) -> CanonicalDocument:
        self._validate_component(document_id)
        self._validate_component(version)
        document_root = self._prepare_document_root(document_id, create=False)
        bundle = document_root / version
        if not bundle.is_dir():
            raise FileNotFoundError(f"canonical bundle does not exist: {bundle}")
        self._validate_bundle(bundle, document_id, version)

        manifest = self._read_json(bundle / "manifest.json")
        document_fields = manifest["document"]
        parser = manifest["parser"]
        source = manifest["source"]
        blocks: list[CanonicalBlock] = []
        with (bundle / "blocks.jsonl").open("r", encoding="utf-8") as block_file:
            for line in block_file:
                blocks.append(CanonicalBlock.model_validate_json(line))

        return CanonicalDocument(
            document_id=document_id,
            parse_version=version,
            source_path=source["path"],
            source_media_type=source["media_type"],
            source_metadata=source["metadata"],
            parser_source=parser["source"],
            parser_metadata=parser["metadata"],
            title=document_fields["title"],
            abstract=document_fields["abstract"],
            keywords=document_fields["keywords"],
            outline=document_fields["outline"],
            metadata=document_fields["metadata"],
            blocks=blocks,
            tables=self._read_model_list(bundle / "tables.json", CanonicalTable),
            figures=self._read_model_list(bundle / "figures.json", CanonicalFigure),
            formulas=self._read_model_list(bundle / "formulas.json", CanonicalFormula),
            assets=[CanonicalAsset.model_validate(item) for item in manifest["assets"]],
            quality=manifest["quality"],
            warnings=manifest["warnings"],
            status=manifest["status"],
        )

    @staticmethod
    def _validate_component(value: str) -> None:
        if (
            not _SAFE_COMPONENT.fullmatch(value)
            or value in {".", ".."}
            or value.endswith((".", " "))
            or CanonicalArtifactStore._is_windows_device_name(value)
        ):
            raise ValueError(f"invalid path component: {value!r}")

    def _copy_assets(
        self,
        staging: Path,
        assets: list[CanonicalAsset],
    ) -> dict[str, str]:
        asset_hashes: dict[str, str] = {}
        for asset in assets:
            if asset.sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", asset.sha256
            ):
                raise ValueError(
                    f"asset sha256 must be lowercase 64-hex for {asset.asset_id!r}"
                )
            destination = self._asset_destination(staging, asset.path)
            if asset.source_path is None:
                raise FileNotFoundError(
                    f"asset source is required for declared asset {asset.asset_id!r}"
                )

            source = Path(asset.source_path)
            if not source.is_file():
                raise FileNotFoundError(f"asset source does not exist: {source}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_file, destination.open("xb") as output_file:
                shutil.copyfileobj(source_file, output_file)
                output_file.flush()
                os.fsync(output_file.fileno())

            actual_sha256 = self._sha256(destination)
            if asset.sha256 is not None and actual_sha256 != asset.sha256:
                raise ValueError(
                    f"asset sha256 mismatch for {asset.asset_id!r}: "
                    f"expected {asset.sha256}, got {actual_sha256}"
                )
            asset_hashes[asset.asset_id] = actual_sha256
        return asset_hashes

    def _resolve_asset_hashes(
        self,
        assets: list[CanonicalAsset],
    ) -> dict[str, str]:
        asset_hashes: dict[str, str] = {}
        for asset in assets:
            if asset.sha256 is not None:
                if not re.fullmatch(r"[0-9a-f]{64}", asset.sha256):
                    raise ValueError(
                        f"asset sha256 must be lowercase 64-hex for {asset.asset_id!r}"
                    )
                asset_hashes[asset.asset_id] = asset.sha256
                continue

            if asset.source_path is None:
                raise FileNotFoundError(
                    "asset source or persisted sha256 is required for "
                    f"{asset.asset_id!r}"
                )

            source = Path(asset.source_path)
            if not source.is_file():
                raise FileNotFoundError(f"asset source does not exist: {source}")
            actual_sha256 = self._sha256(source)
            asset_hashes[asset.asset_id] = actual_sha256
        return asset_hashes

    @staticmethod
    def _asset_destination(staging: Path, asset_path: str) -> Path:
        posix_path = CanonicalArtifactStore._validate_asset_relative_path(asset_path)

        unresolved_assets_root = staging / "assets"
        unresolved_destination = staging / Path(*posix_path.parts)
        asset_candidates: list[Path] = []
        candidate = staging
        for part in posix_path.parts:
            candidate /= part
            asset_candidates.append(candidate)
        for candidate in asset_candidates:
            if CanonicalArtifactStore._is_link_or_reparse_point(candidate):
                raise ValueError(
                    f"canonical bundle cannot contain a symbolic link: {candidate}"
                )

        assets_root = unresolved_assets_root.resolve()
        destination = unresolved_destination.resolve()
        try:
            destination.relative_to(assets_root)
        except ValueError as exc:
            raise ValueError(f"asset path escapes assets directory: {asset_path!r}") from exc
        return destination

    @classmethod
    def _build_manifest(
        cls,
        document_id: str,
        version: str,
        document: CanonicalDocument,
        *,
        canonical_markdown_sha256: str,
        asset_hashes: dict[str, str],
    ) -> dict[str, Any]:
        payload = cls._build_persisted_input_payload(
            document_id,
            version,
            document,
            asset_hashes,
        )
        return {
            "canonical_markdown_sha256": canonical_markdown_sha256,
            "input_fingerprint": cls._input_fingerprint(payload),
            "document_id": payload["document_id"],
            "version": payload["version"],
            "document": payload["document"],
            "parser": payload["parser"],
            "source": payload["source"],
            "quality": payload["quality"],
            "warnings": payload["warnings"],
            "assets": payload["assets"],
            "status": payload["status"],
        }

    @classmethod
    def _build_persisted_input_payload(
        cls,
        document_id: str,
        version: str,
        document: CanonicalDocument,
        asset_hashes: dict[str, str],
    ) -> dict[str, Any]:
        assets = []
        for asset in document.assets:
            item = cls._bundle_model_dump(
                asset,
                exclude={"source_path"},
            )
            item["sha256"] = asset_hashes[asset.asset_id]
            assets.append(item)
        payload = {
            "document_id": document_id,
            "version": version,
            "document": {
                "title": document.title,
                "abstract": document.abstract,
                "keywords": document.keywords,
                "outline": [
                    cls._bundle_model_dump(section)
                    for section in document.outline
                ],
                "metadata": document.metadata,
            },
            "parser": {
                "source": document.parser_source,
                "metadata": document.parser_metadata,
            },
            "source": {
                "path": cls._safe_source_path(document.source_path),
                "media_type": document.source_media_type,
                "metadata": document.source_metadata,
            },
            "quality": cls._bundle_model_dump(document.quality),
            "warnings": document.warnings,
            "assets": assets,
            "status": document.status,
            "blocks": [
                cls._bundle_model_dump(block)
                for block in document.blocks
                if not cls._is_generated_block(block)
            ],
            "tables": [cls._bundle_model_dump(table) for table in document.tables],
            "figures": [cls._bundle_model_dump(figure) for figure in document.figures],
            "formulas": [cls._bundle_model_dump(formula) for formula in document.formulas],
        }
        return cls._sanitize_metadata_paths(payload)

    @staticmethod
    def _build_raw_persisted_input_payload(
        manifest: dict[str, Any],
        raw_blocks: list[Any],
        raw_tables: list[Any],
        raw_figures: list[Any],
        raw_formulas: list[Any],
    ) -> dict[str, Any]:
        return {
            "document_id": manifest["document_id"],
            "version": manifest["version"],
            "document": manifest["document"],
            "parser": manifest["parser"],
            "source": manifest["source"],
            "quality": manifest["quality"],
            "warnings": manifest["warnings"],
            "assets": manifest["assets"],
            "status": manifest["status"],
            "blocks": raw_blocks,
            "tables": raw_tables,
            "figures": raw_figures,
            "formulas": raw_formulas,
        }

    @staticmethod
    def _serialize_input_payload(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _input_fingerprint(cls, payload: dict[str, Any]) -> str:
        return hashlib.sha256(cls._serialize_input_payload(payload)).hexdigest()

    @staticmethod
    def _render_markdown(
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> str:
        lines = [
            "---",
            f"document_id: {json.dumps(document_id, ensure_ascii=False, allow_nan=False)}",
            f"version: {json.dumps(version, ensure_ascii=False, allow_nan=False)}",
            f"title: {json.dumps(document.title, ensure_ascii=False, allow_nan=False)}",
            "---",
            "",
        ]

        if document.title:
            lines.extend([f"# {document.title}", ""])
        if document.abstract:
            lines.extend([document.abstract, ""])

        tables = {table.table_id: table for table in document.tables}
        figures = {figure.figure_id: figure for figure in document.figures}
        formulas = {formula.formula_id: formula for formula in document.formulas}
        emitted_tables: set[str] = set()
        emitted_figures: set[str] = set()
        emitted_formulas: set[str] = set()

        for block in sorted(
            document.blocks,
            key=lambda item: (item.reading_order, item.block_id),
        ):
            if CanonicalArtifactStore._is_generated_block(block):
                continue
            anchor = html.escape(f"block-{block.block_id}", quote=True)
            lines.append(f'<a id="{anchor}"></a>')
            if block.block_type == "heading":
                heading_level = min(max(len(block.section_path), 1), 6)
                text = block.text.lstrip("# ").strip()
                lines.extend([f"{'#' * heading_level} {text}", ""])
            else:
                lines.extend([block.text, ""])

            if block.table_id and block.table_id not in emitted_tables:
                lines.extend(CanonicalArtifactStore._render_table(tables[block.table_id]))
                emitted_tables.add(block.table_id)
            if block.figure_id and block.figure_id not in emitted_figures:
                lines.extend(CanonicalArtifactStore._render_figure(figures[block.figure_id]))
                emitted_figures.add(block.figure_id)
            if block.formula_id and block.formula_id not in emitted_formulas:
                lines.extend(CanonicalArtifactStore._render_formula(formulas[block.formula_id]))
                emitted_formulas.add(block.formula_id)

        for table in document.tables:
            if table.table_id not in emitted_tables:
                lines.extend(CanonicalArtifactStore._render_table(table))
        for figure in document.figures:
            if figure.figure_id not in emitted_figures:
                lines.extend(CanonicalArtifactStore._render_figure(figure))
        for formula in document.formulas:
            if formula.formula_id not in emitted_formulas:
                lines.extend(CanonicalArtifactStore._render_formula(formula))

        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def _render_table(cls, table: CanonicalTable) -> list[str]:
        table_lines: list[str] = []
        markdown = table.normalized_markdown or table.source_markdown
        if markdown and markdown.strip():
            table_lines.extend(markdown.rstrip().splitlines())
        else:
            headers = table.headers
            rows = table.rows
            if not headers and not rows and table.cells:
                headers, rows = cls._table_grid_from_cells(table.cells)
            if not headers and not rows and table.source_html:
                html_cells = cls._table_cells_from_html(table.source_html)
                headers, rows = cls._table_grid_from_cells(html_cells)
            if any(value.strip() for row in [headers, *rows] for value in row):
                table_lines.extend(
                    cls._render_table_grid(table.table_id, headers, rows)
                )
        if not any(line.strip() for line in table_lines):
            raise ValueError(
                f"canonical table {table.table_id!r} has no renderable table evidence"
            )

        lines: list[str] = []
        if table.caption:
            lines.extend([f"### {table.caption}", ""])
        lines.extend(table_lines)
        if lines and lines[-1] != "":
            lines.append("")
        for footnote in table.footnotes:
            lines.extend([f"*{footnote}*", ""])
        return lines

    @classmethod
    def _render_table_grid(
        cls,
        table_id: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> list[str]:
        if headers:
            wrong_widths = [
                index for index, row in enumerate(rows) if len(row) != len(headers)
            ]
            if wrong_widths:
                raise ValueError(
                    f"table row width does not match headers for {table_id!r}: "
                    f"rows={wrong_widths}"
                )
        rendered: list[str] = []
        if headers:
            rendered.append(
                "| "
                + " | ".join(cls._escape_table_value(value) for value in headers)
                + " |"
            )
            rendered.append("| " + " | ".join("---" for _ in headers) + " |")
        rendered.extend(
            "| " + " | ".join(cls._escape_table_value(value) for value in row) + " |"
            for row in rows
        )
        return rendered

    @classmethod
    def _table_grid_from_cells(
        cls,
        cells: list[CanonicalCell],
    ) -> tuple[list[str], list[list[str]]]:
        if not cells:
            return [], []
        row_count = 0
        column_count = 0
        for cell in cells:
            end_row = cell.row_index + cell.rowspan
            end_column = cell.column_index + cell.colspan
            cls._validate_table_dimensions(end_row, end_column)
            row_count = max(row_count, end_row)
            column_count = max(column_count, end_column)
            cls._validate_table_dimensions(row_count, column_count)
        matrix = [["" for _ in range(column_count)] for _ in range(row_count)]
        occupied: set[tuple[int, int]] = set()
        header_rows: set[int] = set()
        for cell in sorted(cells, key=lambda item: (item.row_index, item.column_index)):
            span_coordinates = {
                (row_index, column_index)
                for row_index in range(cell.row_index, cell.row_index + cell.rowspan)
                for column_index in range(
                    cell.column_index,
                    cell.column_index + cell.colspan,
                )
            }
            if occupied.intersection(span_coordinates):
                raise ValueError("canonical table cells contain overlapping spans")
            matrix[cell.row_index][cell.column_index] = cell.text
            if cell.is_header:
                header_rows.add(cell.row_index)
            occupied.update(span_coordinates)
        if not header_rows:
            return [], matrix
        header_index = min(header_rows)
        return matrix[header_index], [
            row for index, row in enumerate(matrix) if index != header_index
        ]

    @classmethod
    def _table_cells_from_html(cls, source_html: str) -> list[CanonicalCell]:
        soup = BeautifulSoup(source_html, "html.parser")
        for unsafe in soup.find_all(["script", "style"]):
            unsafe.decompose()
        table = soup.find("table")
        if table is None:
            return []
        cells: list[CanonicalCell] = []
        occupied: set[tuple[int, int]] = set()
        max_row_count = 0
        max_column_count = 0
        rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
        for row_index, row in enumerate(rows):
            column_index = 0
            for element in row.find_all(["th", "td"], recursive=False):
                try:
                    rowspan = max(1, int(element.get("rowspan", 1)))
                    colspan = max(1, int(element.get("colspan", 1)))
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid HTML table span") from exc
                cls._validate_table_dimensions(
                    row_index + rowspan,
                    column_index + colspan,
                )
                while any(
                    (row_index, candidate_column) in occupied
                    for candidate_column in range(
                        column_index,
                        column_index + colspan,
                    )
                ):
                    column_index += 1
                end_row = row_index + rowspan
                end_column = column_index + colspan
                cls._validate_table_dimensions(end_row, end_column)
                max_row_count = max(max_row_count, end_row)
                max_column_count = max(max_column_count, end_column)
                cls._validate_table_dimensions(
                    max_row_count,
                    max_column_count,
                )
                cells.append(
                    CanonicalCell(
                        text=element.get_text(" ", strip=True),
                        row_index=row_index,
                        column_index=column_index,
                        rowspan=rowspan,
                        colspan=colspan,
                        is_header=(
                            element.name == "th"
                            or element.find_parent("thead") is not None
                        ),
                    )
                )
                for span_row in range(row_index, row_index + rowspan):
                    for span_column in range(
                        column_index,
                        column_index + colspan,
                    ):
                        occupied.add((span_row, span_column))
                column_index += colspan
        return cells

    @staticmethod
    def _validate_table_dimensions(row_count: int, column_count: int) -> None:
        if (
            row_count > _MAX_TABLE_ROWS
            or column_count > _MAX_TABLE_COLUMNS
            or row_count * column_count > _MAX_TABLE_GRID_CELLS
        ):
            raise ValueError(
                "table matrix exceeds canonical limits: "
                f"rows={row_count}, columns={column_count}"
            )

    @staticmethod
    def _render_figure(figure: CanonicalFigure) -> list[str]:
        lines: list[str] = []
        if figure.caption:
            lines.extend([f"### {figure.caption}", ""])
        if figure.asset_path:
            alt = CanonicalArtifactStore._escape_figure_alt(
                figure.caption or figure.figure_id
            )
            destination = quote(figure.asset_path, safe="/")
            lines.extend([f"![{alt}]({destination})", ""])
        if figure.description:
            lines.extend([figure.description, ""])
        return lines

    @staticmethod
    def _render_formula(formula: CanonicalFormula) -> list[str]:
        lines: list[str] = []
        if formula.caption:
            lines.extend([f"### {formula.caption}", ""])
        lines.extend(["$$", formula.latex, "$$", ""])
        if formula.description:
            lines.extend([formula.description, ""])
        return lines

    @classmethod
    def _write_json(cls, path: Path, value: Any) -> None:
        value = cls._sanitize_metadata_paths(value)
        cls._write_text(
            path,
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )

    @classmethod
    def _write_blocks(cls, path: Path, blocks: list[CanonicalBlock]) -> None:
        lines = [
            json.dumps(
                cls._bundle_model_dump(block),
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for block in blocks
            if not cls._is_generated_block(block)
        ]
        cls._write_text(path, "\n".join(lines) + ("\n" if lines else ""))

    @staticmethod
    def _is_generated_block(block: CanonicalBlock) -> bool:
        return block_is_generated(block)

    @classmethod
    def _bundle_model_dump(cls, model: Any, **kwargs: Any) -> Any:
        model.ensure_json_compatible()
        return cls._sanitize_metadata_paths(
            model.model_dump(mode="json", **kwargs)
        )

    @classmethod
    def _sanitize_metadata_paths(
        cls,
        value: Any,
        *,
        in_metadata: bool = False,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._sanitize_metadata_paths(
                    item,
                    in_metadata=in_metadata or key == "metadata",
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._sanitize_metadata_paths(item, in_metadata=in_metadata)
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                cls._sanitize_metadata_paths(item, in_metadata=in_metadata)
                for item in value
            ]
        if in_metadata and isinstance(value, str):
            return cls._safe_metadata_string(value)
        return value

    @classmethod
    def _safe_metadata_string(cls, value: str) -> str:
        return cls._safe_path_string(value)

    @staticmethod
    def _safe_path_string(value: str) -> str:
        parsed = urlparse(value)
        scheme = parsed.scheme.lower()
        if scheme in _REMOTE_URI_SCHEMES:
            return value
        if scheme == "file":
            candidate = unquote(parsed.path)
            if parsed.netloc:
                candidate = f"//{parsed.netloc}{candidate}"
        elif re.match(r"^[A-Za-z]:", value):
            candidate = value
        elif scheme and parsed.path:
            candidate = unquote(parsed.path)
        else:
            candidate = value

        is_posix_rooted = candidate.startswith("/")
        windows_path = PureWindowsPath(candidate)
        is_windows_rooted_or_driven = bool(windows_path.root or windows_path.drive)
        if not is_posix_rooted and not is_windows_rooted_or_driven:
            return value
        path = PurePosixPath(candidate) if is_posix_rooted else windows_path
        return path.name or value

    @staticmethod
    def _escape_table_value(value: str) -> str:
        normalized = re.sub(r"\r\n|\r|\n", "<br>", value)
        return normalized.replace("\\", "\\\\").replace("|", "\\|")

    @staticmethod
    def _escape_figure_alt(value: str) -> str:
        normalized = re.sub(r"\r\n|\r|\n", " ", value)
        return (
            normalized.replace("\\", "\\\\")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        with path.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())

    def _validate_bundle(
        self,
        bundle: Path,
        document_id: str,
        version: str,
    ) -> None:
        if self._is_link_or_reparse_point(bundle):
            raise ValueError(f"canonical bundle cannot be a symbolic link: {bundle}")

        expected_entries = _REQUIRED_FILES | {"assets"}
        actual_entries: set[str] = set()
        linked_entries: list[str] = []
        entry_stats: dict[str, os.stat_result] = {}
        for entry in bundle.iterdir():
            entry_stat = entry.lstat()
            actual_entries.add(entry.name)
            entry_stats[entry.name] = entry_stat
            if self._is_link_or_reparse_point(entry):
                linked_entries.append(entry.name)
        if actual_entries != expected_entries:
            missing_entries = sorted(expected_entries.difference(actual_entries))
            extra_entries = sorted(actual_entries.difference(expected_entries))
            if missing_entries and not extra_entries:
                raise ValueError(
                    "incomplete canonical bundle; missing required artifacts: "
                    + ", ".join(missing_entries)
                )
            raise ValueError(
                "canonical top-level bundle inventory mismatch; "
                f"missing={missing_entries}, extra={extra_entries}"
            )
        if linked_entries:
            raise ValueError(
                "canonical bundle cannot contain a symbolic link: "
                + ", ".join(sorted(linked_entries))
            )

        missing = [
            name
            for name in sorted(_REQUIRED_FILES)
            if not stat.S_ISREG(entry_stats[name].st_mode)
        ]
        if not stat.S_ISDIR(entry_stats["assets"].st_mode):
            missing.append("assets/")
        if missing:
            raise ValueError(
                "incomplete canonical bundle; missing required artifacts: "
                + ", ".join(missing)
            )
        with (bundle / "canonical.md").open("r", encoding="utf-8") as markdown_file:
            if markdown_file.read(4) != "---\n":
                raise ValueError("canonical.md must start with YAML front matter")

        manifest = self._read_json(bundle / "manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("canonical manifest must be a JSON object")
        missing_manifest_fields = _REQUIRED_MANIFEST_FIELDS.difference(manifest)
        if missing_manifest_fields:
            raise ValueError(
                "canonical manifest missing required fields: "
                + ", ".join(sorted(missing_manifest_fields))
            )
        unexpected_manifest_fields = set(manifest).difference(_REQUIRED_MANIFEST_FIELDS)
        if unexpected_manifest_fields:
            raise ValueError(
                "canonical manifest contains unexpected fields: "
                + ", ".join(sorted(unexpected_manifest_fields))
            )
        if manifest.get("document_id") != document_id or manifest.get("version") != version:
            raise ValueError("canonical manifest identity does not match promotion target")
        self._validate_manifest_fields(manifest)
        input_fingerprint = manifest["input_fingerprint"]
        if not isinstance(input_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", input_fingerprint
        ):
            raise ValueError(
                "canonical manifest input_fingerprint must be lowercase 64-hex"
            )
        markdown_sha256 = manifest["canonical_markdown_sha256"]
        if not isinstance(markdown_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", markdown_sha256
        ):
            raise ValueError(
                "canonical manifest canonical_markdown_sha256 must be lowercase 64-hex"
            )
        if self._sha256(bundle / "canonical.md") != markdown_sha256:
            raise ValueError("canonical.md sha256 mismatch")

        raw_blocks, blocks = self._read_jsonl_model_list(
            bundle / "blocks.jsonl",
            CanonicalBlock,
        )
        raw_tables, tables = self._read_model_list_with_raw(
            bundle / "tables.json",
            CanonicalTable,
        )
        raw_figures, figures = self._read_model_list_with_raw(
            bundle / "figures.json",
            CanonicalFigure,
        )
        raw_formulas, formulas = self._read_model_list_with_raw(
            bundle / "formulas.json",
            CanonicalFormula,
        )

        asset_inventory = manifest.get("assets")
        if not isinstance(asset_inventory, list):
            raise ValueError("canonical manifest assets must be a JSON array")
        assets: list[CanonicalAsset] = []
        for item in asset_inventory:
            if not isinstance(item, dict):
                raise ValueError("canonical manifest asset entries must be JSON objects")
            if "source_path" in item:
                raise ValueError("canonical manifest asset entries cannot contain source_path")
            persisted_sha256 = item.get("sha256")
            if not isinstance(persisted_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", persisted_sha256
            ):
                raise ValueError(
                    "canonical manifest asset sha256 must be lowercase 64-hex"
                )
            asset = CanonicalAsset.model_validate(item)
            assets.append(asset)
            destination = self._asset_destination(bundle, asset.path)
            if not destination.is_file():
                raise ValueError(f"declared canonical asset is missing: {asset.path}")
            actual_sha256 = self._sha256(destination)
            if actual_sha256 != asset.sha256:
                raise ValueError(f"canonical asset sha256 mismatch: {asset.path}")
        outline = [
            SectionNode.model_validate(section)
            for section in manifest["document"]["outline"]
        ]
        quality = CanonicalQualityReport.model_validate(manifest["quality"])
        self._validate_record_contract(
            blocks,
            tables,
            figures,
            formulas,
            assets,
            outline,
            quality,
        )
        self._validate_asset_tree(bundle, assets)
        persisted_document = CanonicalDocument(
            document_id=document_id,
            parse_version=version,
            source_path=manifest["source"]["path"],
            source_media_type=manifest["source"]["media_type"],
            source_metadata=manifest["source"]["metadata"],
            parser_source=manifest["parser"]["source"],
            parser_metadata=manifest["parser"]["metadata"],
            title=manifest["document"]["title"],
            abstract=manifest["document"]["abstract"],
            keywords=manifest["document"]["keywords"],
            outline=outline,
            metadata=manifest["document"]["metadata"],
            blocks=blocks,
            tables=tables,
            figures=figures,
            formulas=formulas,
            assets=assets,
            quality=quality,
            warnings=manifest["warnings"],
            status=manifest["status"],
        )
        persisted_asset_hashes = {
            asset.asset_id: asset.sha256 for asset in assets if asset.sha256 is not None
        }
        raw_persisted_payload = self._build_raw_persisted_input_payload(
            manifest,
            raw_blocks,
            raw_tables,
            raw_figures,
            raw_formulas,
        )
        canonical_persisted_payload = self._build_persisted_input_payload(
            document_id,
            version,
            persisted_document,
            persisted_asset_hashes,
        )
        if self._serialize_input_payload(
            raw_persisted_payload
        ) != self._serialize_input_payload(canonical_persisted_payload):
            raise ValueError("persisted canonical input is not canonical")
        if self._input_fingerprint(raw_persisted_payload) != input_fingerprint:
            raise ValueError("canonical input fingerprint mismatch")

    @classmethod
    def _validate_record_contract(
        cls,
        blocks: list[CanonicalBlock],
        tables: list[CanonicalTable],
        figures: list[CanonicalFigure],
        formulas: list[CanonicalFormula],
        assets: list[CanonicalAsset],
        outline: list[SectionNode],
        quality: CanonicalQualityReport,
    ) -> None:
        for asset in assets:
            cls._validate_asset_relative_path(asset.path)
        cls._ensure_unique((block.block_id for block in blocks), "block IDs")
        cls._ensure_unique((table.table_id for table in tables), "table IDs")
        cls._ensure_unique((figure.figure_id for figure in figures), "figure IDs")
        cls._ensure_unique((formula.formula_id for formula in formulas), "formula IDs")
        cls._ensure_unique((asset.asset_id for asset in assets), "asset IDs")
        cls._ensure_unique((asset.path for asset in assets), "asset paths")

        table_ids = {table.table_id for table in tables}
        figure_ids = {figure.figure_id for figure in figures}
        formula_ids = {formula.formula_id for formula in formulas}
        asset_paths = {asset.path for asset in assets}
        for block in blocks:
            references = (
                ("table_id", block.table_id, table_ids),
                ("figure_id", block.figure_id, figure_ids),
                ("formula_id", block.formula_id, formula_ids),
            )
            for field_name, reference, valid_ids in references:
                if reference is not None and reference not in valid_ids:
                    raise ValueError(
                        f"broken block reference {field_name}={reference!r} "
                        f"on {block.block_id!r}"
                    )
        for figure in figures:
            if figure.asset_path and figure.asset_path not in asset_paths:
                raise ValueError(
                    f"figure asset {figure.asset_path!r} is not declared in inventory"
                )

        canonical_block_references: list[tuple[str, str]] = []
        pending_sections = list(outline)
        while pending_sections:
            section = pending_sections.pop()
            pending_sections.extend(section.children)
            if section.block_id:
                canonical_block_references.append(("outline", section.block_id))
        for figure in figures:
            canonical_block_references.extend(
                (f"figure {figure.figure_id!r}", block_id)
                for block_id in figure.nearby_block_ids
            )
        for formula in formulas:
            canonical_block_references.extend(
                (f"formula {formula.formula_id!r}", block_id)
                for block_id in formula.nearby_block_ids
            )
        for issue in quality.issues:
            canonical_block_references.extend(
                (f"quality issue {issue.code!r}", block_id)
                for block_id in issue.block_ids
            )
        block_ids = {block.block_id for block in blocks}
        for owner, block_id in canonical_block_references:
            if block_id not in block_ids:
                raise ValueError(
                    f"broken canonical block reference {block_id!r} in {owner}"
                )

    @staticmethod
    def _ensure_unique(values: Any, label: str) -> None:
        seen: set[str] = set()
        for value in values:
            if value in seen:
                raise ValueError(f"duplicate canonical {label}: {value!r}")
            seen.add(value)

    def _validate_asset_tree(
        self,
        bundle: Path,
        assets: list[CanonicalAsset],
    ) -> None:
        declared_files = {asset.path for asset in assets}
        expected_directories = {"assets"}
        for asset_path in declared_files:
            for parent in PurePosixPath(asset_path).parents:
                if parent.as_posix() == ".":
                    break
                expected_directories.add(parent.as_posix())

        actual_files: set[str] = set()
        actual_directories = {"assets"}
        pending = [bundle / "assets"]
        while pending:
            directory = pending.pop()
            for entry in directory.iterdir():
                relative_path = entry.relative_to(bundle).as_posix()
                if self._is_link_or_reparse_point(entry):
                    raise ValueError(
                        f"canonical bundle cannot contain a symbolic link: {entry}"
                    )
                entry_stat = entry.lstat()
                if stat.S_ISDIR(entry_stat.st_mode):
                    actual_directories.add(relative_path)
                    pending.append(entry)
                elif stat.S_ISREG(entry_stat.st_mode):
                    actual_files.add(relative_path)
                else:
                    raise ValueError(
                        f"canonical asset inventory contains a non-regular file: {entry}"
                    )

        if actual_files != declared_files:
            missing = sorted(declared_files.difference(actual_files))
            extra = sorted(actual_files.difference(declared_files))
            raise ValueError(
                f"canonical asset inventory mismatch; missing={missing}, extra={extra}"
            )
        if actual_directories != expected_directories:
            unexpected = sorted(actual_directories.difference(expected_directories))
            raise ValueError(
                f"canonical asset inventory contains unexpected directories: {unexpected}"
            )

    @staticmethod
    def _validate_manifest_fields(manifest: dict[str, Any]) -> None:
        document = manifest["document"]
        if not isinstance(document, dict):
            raise ValueError("canonical manifest document must be a JSON object")
        required_document_fields = {
            "title",
            "abstract",
            "keywords",
            "outline",
            "metadata",
        }
        missing_document_fields = required_document_fields.difference(document)
        if missing_document_fields:
            raise ValueError(
                "canonical manifest document missing required fields: "
                + ", ".join(sorted(missing_document_fields))
            )
        unexpected_document_fields = set(document).difference(required_document_fields)
        if unexpected_document_fields:
            raise ValueError(
                "canonical manifest document contains unexpected fields: "
                + ", ".join(sorted(unexpected_document_fields))
            )
        if not isinstance(document["title"], str):
            raise ValueError("canonical manifest document title must be a string")
        if document["abstract"] is not None and not isinstance(
            document["abstract"],
            str,
        ):
            raise ValueError("canonical manifest document abstract must be a string or null")
        if not isinstance(document["keywords"], list) or not all(
            isinstance(keyword, str) for keyword in document["keywords"]
        ):
            raise ValueError("canonical manifest document keywords must be a string array")
        if not isinstance(document["outline"], list):
            raise ValueError("canonical manifest document outline must be an array")
        for section in document["outline"]:
            SectionNode.model_validate(section)
        if not isinstance(document["metadata"], dict):
            raise ValueError("canonical manifest document metadata must be a JSON object")

        parser = manifest["parser"]
        if (
            not isinstance(parser, dict)
            or set(parser) != {"source", "metadata"}
            or not isinstance(parser.get("source"), str)
            or not isinstance(parser.get("metadata"), dict)
        ):
            raise ValueError("canonical manifest parser must contain source and metadata")

        source = manifest["source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "media_type", "metadata"}
            or not isinstance(source.get("metadata"), dict)
        ):
            raise ValueError("canonical manifest source must contain metadata")
        if source.get("path") is not None and not isinstance(source["path"], str):
            raise ValueError("canonical manifest source path must be a string or null")
        if source.get("media_type") is not None and not isinstance(
            source["media_type"],
            str,
        ):
            raise ValueError("canonical manifest source media_type must be a string or null")

        try:
            CanonicalQualityReport.model_validate(manifest["quality"])
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical manifest quality must be a quality report") from exc

        warnings = manifest["warnings"]
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            raise ValueError("canonical manifest warnings must be a string array")
        if not isinstance(manifest["assets"], list):
            raise ValueError("canonical manifest assets must be a JSON array")
        if not isinstance(manifest["status"], str):
            raise ValueError("canonical manifest status must be a string")

    @classmethod
    def _read_jsonl_model_list(
        cls,
        path: Path,
        model: type[Any],
    ) -> tuple[list[Any], list[Any]]:
        raw_items: list[Any] = []
        models: list[Any] = []
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise ValueError(f"empty {path.name} line at {line_number}")
                raw_item = json.loads(line)
                raw_items.append(raw_item)
                models.append(model.model_validate(raw_item))
        return raw_items, models

    @classmethod
    def _read_model_list_with_raw(
        cls,
        path: Path,
        model: type[Any],
    ) -> tuple[list[Any], list[Any]]:
        value = cls._read_json(path)
        if not isinstance(value, list):
            raise ValueError(f"canonical artifact must contain a JSON array: {path.name}")
        return value, [model.model_validate(item) for item in value]

    @classmethod
    def _read_model_list(cls, path: Path, model: type[Any]) -> list[Any]:
        _, models = cls._read_model_list_with_raw(path, model)
        return models

    @classmethod
    def _validate_model_list(cls, path: Path, model: type[Any]) -> None:
        cls._read_model_list(path, model)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _safe_source_path(cls, source_path: str | None) -> str | None:
        if source_path is None:
            return None
        return cls._safe_path_string(source_path)

    @staticmethod
    def _is_link_or_reparse_point(path: Path) -> bool:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return False
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        file_attributes = getattr(path_stat, "st_file_attributes", 0)
        return stat.S_ISLNK(path_stat.st_mode) or bool(
            file_attributes & reparse_attribute
        )

    @staticmethod
    def _validate_asset_relative_path(asset_path: str) -> PurePosixPath:
        if "\\" in asset_path:
            raise ValueError(f"invalid asset path: {asset_path!r}")
        posix_path = PurePosixPath(asset_path)
        windows_path = PureWindowsPath(asset_path)
        raw_parts = asset_path.split("/")
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or posix_path.as_posix() != asset_path
            or len(raw_parts) < 2
            or raw_parts[0] != "assets"
            or any(
                not CanonicalArtifactStore._is_portable_asset_component(part)
                for part in raw_parts
            )
        ):
            raise ValueError(f"invalid asset path: {asset_path!r}")
        return posix_path

    @staticmethod
    def _is_portable_asset_component(component: str) -> bool:
        return not (
            component in {"", ".", ".."}
            or component.endswith((".", " "))
            or any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in component)
            or any(ord(character) < 32 for character in component)
            or CanonicalArtifactStore._is_windows_device_name(component)
        )

    @staticmethod
    def _is_windows_device_name(component: str) -> bool:
        stem = component.split(".", maxsplit=1)[0]
        return stem.upper() in _WINDOWS_DEVICE_NAMES
