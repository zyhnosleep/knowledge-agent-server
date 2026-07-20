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
from urllib.parse import quote
from uuid import uuid4

from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityReport,
    CanonicalTable,
    SectionNode,
)


_REQUIRED_FILES = {
    "canonical.md",
    "manifest.json",
    "blocks.jsonl",
    "tables.json",
    "figures.json",
    "formulas.json",
}
_REQUIRED_MANIFEST_FIELDS = {
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


class CanonicalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

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
        self._validate_record_contract(
            document.blocks,
            document.tables,
            document.figures,
            document.formulas,
            document.assets,
            document.outline,
            document.quality,
        )

        document_root = self.root / document_id
        document_root.mkdir(parents=True, exist_ok=True)
        staging = document_root / f"{version}.staging-{uuid4().hex}"
        staging.mkdir()

        try:
            (staging / "assets").mkdir()
            self._copy_assets(staging, document.assets)

            self._write_text(
                staging / "canonical.md",
                self._render_markdown(document_id, version, document),
            )
            self._write_json(
                staging / "manifest.json",
                self._build_manifest(document_id, version, document),
            )
            self._write_blocks(staging / "blocks.jsonl", document.blocks)
            self._write_json(
                staging / "tables.json",
                [table.model_dump(mode="json") for table in document.tables],
            )
            self._write_json(
                staging / "figures.json",
                [figure.model_dump(mode="json") for figure in document.figures],
            )
            self._write_json(
                staging / "formulas.json",
                [formula.model_dump(mode="json") for formula in document.formulas],
            )
            self._validate_bundle(staging, document_id, version)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return staging

    def promote(self, document_id: str, version: str) -> Path:
        self._validate_component(document_id)
        self._validate_component(version)

        document_root = self.root / document_id
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
        bundle = self.root / document_id / version
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

    def _copy_assets(self, staging: Path, assets: list[CanonicalAsset]) -> None:
        for asset in assets:
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
            if actual_sha256 != asset.sha256.lower():
                raise ValueError(
                    f"asset sha256 mismatch for {asset.asset_id!r}: "
                    f"expected {asset.sha256}, got {actual_sha256}"
                )

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

    @staticmethod
    def _build_manifest(
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> dict[str, Any]:
        assets = [
            asset.model_dump(mode="json", exclude={"source_path"})
            for asset in document.assets
        ]
        return {
            "document_id": document_id,
            "version": version,
            "document": {
                "title": document.title,
                "abstract": document.abstract,
                "keywords": document.keywords,
                "outline": [
                    section.model_dump(mode="json") for section in document.outline
                ],
                "metadata": document.metadata,
            },
            "parser": {
                "source": document.parser_source,
                "metadata": document.parser_metadata,
            },
            "source": {
                "path": CanonicalArtifactStore._safe_source_path(document.source_path),
                "media_type": document.source_media_type,
                "metadata": document.source_metadata,
            },
            "quality": document.quality.model_dump(mode="json"),
            "warnings": document.warnings,
            "assets": assets,
            "status": document.status,
        }

    @staticmethod
    def _render_markdown(
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> str:
        lines = [
            "---",
            f"document_id: {json.dumps(document_id, ensure_ascii=False)}",
            f"version: {json.dumps(version, ensure_ascii=False)}",
            f"title: {json.dumps(document.title, ensure_ascii=False)}",
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
            anchor = html.escape(f"block-{block.block_id}", quote=True)
            lines.append(f'<a id="{anchor}"></a>')
            if block.block_type == "heading":
                heading_level = min(max(len(block.section_path), 1), 6)
                text = block.text.lstrip("# ").strip()
                lines.extend([f"{'#' * heading_level} {text}", ""])
            elif block.table_id and block.table_id in tables:
                if block.table_id not in emitted_tables:
                    lines.extend(CanonicalArtifactStore._render_table(tables[block.table_id]))
                    emitted_tables.add(block.table_id)
            elif block.figure_id and block.figure_id in figures:
                if block.figure_id not in emitted_figures:
                    lines.extend(CanonicalArtifactStore._render_figure(figures[block.figure_id]))
                    emitted_figures.add(block.figure_id)
            elif block.formula_id and block.formula_id in formulas:
                if block.formula_id not in emitted_formulas:
                    lines.extend(CanonicalArtifactStore._render_formula(formulas[block.formula_id]))
                    emitted_formulas.add(block.formula_id)
            else:
                lines.extend([block.text, ""])

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

    @staticmethod
    def _render_table(table: CanonicalTable) -> list[str]:
        lines: list[str] = []
        if table.caption:
            lines.extend([f"### {table.caption}", ""])
        markdown = table.normalized_markdown or table.source_markdown
        if markdown is None:
            headers = table.headers
            rows = table.rows
            if headers:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join("---" for _ in headers) + " |")
                lines.extend("| " + " | ".join(row) + " |" for row in rows)
            elif rows:
                lines.extend("| " + " | ".join(row) + " |" for row in rows)
        elif markdown:
            lines.extend(markdown.rstrip().splitlines())
        if lines and lines[-1] != "":
            lines.append("")
        for footnote in table.footnotes:
            lines.extend([f"*{footnote}*", ""])
        return lines

    @staticmethod
    def _render_figure(figure: CanonicalFigure) -> list[str]:
        lines: list[str] = []
        if figure.caption:
            lines.extend([f"### {figure.caption}", ""])
        if figure.asset_path:
            alt = figure.caption or figure.figure_id
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
        cls._write_text(
            path,
            json.dumps(
                value,
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
                block.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for block in blocks
        ]
        cls._write_text(path, "\n".join(lines) + ("\n" if lines else ""))

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

        missing = [name for name in sorted(_REQUIRED_FILES) if not (bundle / name).is_file()]
        if not (bundle / "assets").is_dir():
            missing.append("assets/")
        if missing:
            raise ValueError(
                "incomplete canonical bundle; missing required artifacts: "
                + ", ".join(missing)
            )
        linked_artifacts = [
            name
            for name in sorted(_REQUIRED_FILES)
            if self._is_link_or_reparse_point(bundle / name)
        ]
        if self._is_link_or_reparse_point(bundle / "assets"):
            linked_artifacts.append("assets/")
        if linked_artifacts:
            raise ValueError(
                "canonical bundle cannot contain a symbolic link: "
                + ", ".join(linked_artifacts)
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

        blocks: list[CanonicalBlock] = []
        with (bundle / "blocks.jsonl").open("r", encoding="utf-8") as block_file:
            for line_number, line in enumerate(block_file, start=1):
                if not line.strip():
                    raise ValueError(f"empty blocks.jsonl line at {line_number}")
                blocks.append(CanonicalBlock.model_validate_json(line))

        tables = self._read_model_list(bundle / "tables.json", CanonicalTable)
        figures = self._read_model_list(bundle / "figures.json", CanonicalFigure)
        formulas = self._read_model_list(bundle / "formulas.json", CanonicalFormula)

        asset_inventory = manifest.get("assets")
        if not isinstance(asset_inventory, list):
            raise ValueError("canonical manifest assets must be a JSON array")
        assets: list[CanonicalAsset] = []
        for item in asset_inventory:
            if not isinstance(item, dict):
                raise ValueError("canonical manifest asset entries must be JSON objects")
            if "source_path" in item:
                raise ValueError("canonical manifest asset entries cannot contain source_path")
            asset = CanonicalAsset.model_validate(item)
            assets.append(asset)
            destination = self._asset_destination(bundle, asset.path)
            if not destination.is_file():
                raise ValueError(f"declared canonical asset is missing: {asset.path}")
            actual_sha256 = self._sha256(destination)
            if actual_sha256 != asset.sha256.lower():
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
    def _read_model_list(cls, path: Path, model: type[Any]) -> list[Any]:
        value = cls._read_json(path)
        if not isinstance(value, list):
            raise ValueError(f"canonical artifact must contain a JSON array: {path.name}")
        return [model.model_validate(item) for item in value]

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

    @staticmethod
    def _safe_source_path(source_path: str | None) -> str | None:
        if source_path is None:
            return None
        windows_path = PureWindowsPath(source_path)
        posix_path = PurePosixPath(source_path)
        if windows_path.is_absolute() or windows_path.drive or windows_path.root:
            return windows_path.name
        if posix_path.is_absolute():
            return posix_path.name
        return source_path

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
