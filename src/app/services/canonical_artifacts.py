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
from uuid import uuid4

from app.services.canonical_models import (
    CanonicalAsset,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalFormula,
    CanonicalQualityReport,
    CanonicalTable,
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
    "warnings",
    "assets",
    "status",
}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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

    @staticmethod
    def _validate_component(value: str) -> None:
        if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
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
        if "\\" in asset_path:
            raise ValueError(f"invalid asset path: {asset_path!r}")

        posix_path = PurePosixPath(asset_path)
        windows_path = PureWindowsPath(asset_path)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or posix_path.as_posix() != asset_path
            or len(posix_path.parts) < 2
            or posix_path.parts[0] != "assets"
            or any(part in {"", ".", ".."} for part in posix_path.parts)
        ):
            raise ValueError(f"invalid asset path: {asset_path!r}")

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
            "parser": {
                "source": document.parser_source,
                "metadata": document.parser_metadata,
            },
            "source": {
                "path": document.source_path,
                "media_type": document.source_media_type,
                "metadata": document.source_metadata,
            },
            "quality": document.quality.model_dump(mode="json"),
            "warnings": document.warnings,
            "assets": assets,
            "metadata": document.metadata,
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

        for block in sorted(
            document.blocks,
            key=lambda item: (item.reading_order, item.block_id),
        ):
            anchor = html.escape(f"block-{block.block_id}", quote=True)
            lines.extend([f'<a id="{anchor}"></a>', block.text, ""])

        return "\n".join(lines).rstrip() + "\n"

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
        if manifest.get("document_id") != document_id or manifest.get("version") != version:
            raise ValueError("canonical manifest identity does not match promotion target")
        self._validate_manifest_fields(manifest)

        with (bundle / "blocks.jsonl").open("r", encoding="utf-8") as block_file:
            for line_number, line in enumerate(block_file, start=1):
                if not line.strip():
                    raise ValueError(f"empty blocks.jsonl line at {line_number}")
                CanonicalBlock.model_validate_json(line)

        self._validate_model_list(bundle / "tables.json", CanonicalTable)
        self._validate_model_list(bundle / "figures.json", CanonicalFigure)
        self._validate_model_list(bundle / "formulas.json", CanonicalFormula)

        asset_inventory = manifest.get("assets")
        if not isinstance(asset_inventory, list):
            raise ValueError("canonical manifest assets must be a JSON array")
        for item in asset_inventory:
            if not isinstance(item, dict):
                raise ValueError("canonical manifest asset entries must be JSON objects")
            if "source_path" in item:
                raise ValueError("canonical manifest asset entries cannot contain source_path")
            asset = CanonicalAsset.model_validate(item)
            destination = self._asset_destination(bundle, asset.path)
            if not destination.is_file():
                raise ValueError(f"declared canonical asset is missing: {asset.path}")
            actual_sha256 = self._sha256(destination)
            if actual_sha256 != asset.sha256.lower():
                raise ValueError(f"canonical asset sha256 mismatch: {asset.path}")

    @staticmethod
    def _validate_manifest_fields(manifest: dict[str, Any]) -> None:
        parser = manifest["parser"]
        if (
            not isinstance(parser, dict)
            or not isinstance(parser.get("source"), str)
            or not isinstance(parser.get("metadata"), dict)
        ):
            raise ValueError("canonical manifest parser must contain source and metadata")

        source = manifest["source"]
        if not isinstance(source, dict) or not isinstance(source.get("metadata"), dict):
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
        if "metadata" in manifest and not isinstance(manifest["metadata"], dict):
            raise ValueError("canonical manifest metadata must be a JSON object")

    @classmethod
    def _validate_model_list(cls, path: Path, model: type[Any]) -> None:
        value = cls._read_json(path)
        if not isinstance(value, list):
            raise ValueError(f"canonical artifact must contain a JSON array: {path.name}")
        for item in value:
            model.model_validate(item)

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
