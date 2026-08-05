"""Canonical Artifact Store：把 CanonicalDocument 持久化为本地文件包（bundle）。

一个 canonical bundle 是磁盘上的一个目录，包含：
- canonical.md：渲染后的 Markdown 全文，带 YAML front matter。
- manifest.json：文档元数据、来源信息、质量报告、asset 清单、输入指纹。
- blocks.jsonl：每个 block 一行 JSON。
- tables.json / figures.json / formulas.json：结构化对象数组。
- assets/：图片等附件目录。

本模块负责：
1. write_staging：把 CanonicalDocument 写入临时 staging 目录，并进行激活校验。
2. promote：把 staging 目录原子性地重命名为最终版本目录。
3. load：从 bundle 重新加载为 CanonicalDocument。
4. 渲染 Markdown、拷贝资源、校验 bundle 完整性与安全性。

安全设计：
- 拒绝符号链接 / reparse point，防止目录穿越。
- asset 路径必须是相对路径且以 assets/ 开头。
- 文件写入使用临时文件 + fsync + 原子 rename。
"""

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

from bs4 import BeautifulSoup, NavigableString, Tag

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
from app.services.table_normalization import normalize_fragmented_numeric_spacing


# bundle 顶层必须包含的文件与目录。
_REQUIRED_FILES = {
    "canonical.md",
    "manifest.json",
    "blocks.jsonl",
    "tables.json",
    "figures.json",
    "formulas.json",
}
# manifest.json 必须包含的顶层字段。typed_inventory 现在写入
# manifest["document"]["typed_inventory"]（spec 位置），不再是必填顶层字段。
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
# 仅用于只读/迁移兼容的 legacy 顶层字段；新 bundle 必须使用
# manifest["document"]["typed_inventory"]。
_LEGACY_MANIFEST_FIELDS = {"typed_inventory"}
# 安全的文件名/目录名组件：字母数字开头，可包含 . _ -
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Windows 保留设备名，避免在文件名中使用。
_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARACTERS = set('/\\<>:"|?*')
# 允许的远程 URI scheme，metadata 中保留这些 URI 不被脱敏。
_REMOTE_URI_SCHEMES = {"http", "https", "s3", "gs", "minio"}
# 表格大小上限，防止内存/性能爆炸。
_MAX_TABLE_ROWS = 10_000
_MAX_TABLE_COLUMNS = 1_000
_MAX_TABLE_GRID_CELLS = 1_000_000
# 单个 asset / 整份文档 asset 总大小上限。
_MAX_SINGLE_ASSET_BYTES = 64 * 1024 * 1024
_MAX_DOCUMENT_ASSET_BYTES = 256 * 1024 * 1024
_ASSET_COPY_CHUNK_BYTES = 1024 * 1024


class CanonicalArtifactStore:
    """管理 canonical bundle 的写入、晋升（promote）和读取。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _prepare_document_root(
        self,
        document_id: str,
        *,
        create: bool,
        allow_missing: bool = False,
    ) -> Path:
        """校验并返回文档根目录，防止目录穿越和符号链接攻击。"""
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

    # ------------------------------------------------------------------
    # 公开 API：写入 staging、晋升、加载
    # ------------------------------------------------------------------
    def write_staging(
        self,
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> Path:
        """把文档写入临时 staging 目录，并执行激活前的全部校验。

        激活条件：
        - 记录契约校验通过（ID 唯一、引用合法等）。
        - 没有 status == validation_failed 的表格。
        - quality 为 accepted 或 accepted_with_warnings。
        - metadata 中 table_activation_allowed == True。
        """
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
        from app.services.canonical_adapters import _finalize_structured_evidence

        _finalize_structured_evidence(document)
        document.ensure_json_compatible()
        record_error: ValueError | None = None
        try:
            self._validate_record_contract(
                document.blocks,
                document.tables,
                document.figures,
                document.formulas,
                document.assets,
                document.outline,
                document.quality,
            )
        except ValueError as exc:
            record_error = exc
        failed_table_ids = [
            table.table_id
            for table in document.tables
            if table.status == "validation_failed"
        ]
        quality_allowed = (
            document.quality.accepted is True
            and document.quality.status in {"accepted", "accepted_with_warnings"}
        )
        tables_allowed = all(
            table.status
            in {"accepted_mineru", "repaired_by_vision", "cross_page_merged"}
            for table in document.tables
        )
        activation_allowed = document.metadata.get("table_activation_allowed") is True
        if (
            record_error is not None
            or failed_table_ids
            or not quality_allowed
            or not tables_allowed
            or not activation_allowed
        ):
            issue_codes = sorted({issue.code for issue in document.quality.issues})
            detail = f"; record={record_error}" if record_error is not None else ""
            raise ValueError(
                "canonical activation validation failed: "
                f"quality={document.quality.status}, issues={issue_codes}, "
                f"tables={failed_table_ids}{detail}"
            )
        document_root = self._prepare_document_root(document_id, create=True)
        final = document_root / version
        if final.exists():
            raise FileExistsError(f"canonical bundle already exists: {final}")

        # 如果已经存在相同输入指纹的 staging，则复用，保证幂等。
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
        """把唯一的 staging 目录原子性地重命名为最终版本目录。"""
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
        """从 bundle 加载并重建 CanonicalDocument。"""
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

    # ------------------------------------------------------------------
    # 私有辅助：文件名 / 路径安全
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_component(value: str) -> None:
        """校验 document_id / version 等路径组件是否安全。"""
        if (
            not _SAFE_COMPONENT.fullmatch(value)
            or value in {".", ".."}
            or value.endswith((".", " "))
            or CanonicalArtifactStore._is_windows_device_name(value)
        ):
            raise ValueError(f"invalid path component: {value!r}")

    # ------------------------------------------------------------------
    # 私有辅助：asset 拷贝与哈希校验
    # ------------------------------------------------------------------
    def _copy_assets(
        self,
        staging: Path,
        assets: list[CanonicalAsset],
    ) -> dict[str, str]:
        """把 asset 从 source_path 安全地拷贝到 staging/assets/，并返回 asset_id -> sha256。"""
        asset_hashes: dict[str, str] = {}
        source_stats: dict[str, os.stat_result] = {}
        total_size = 0
        for asset in assets:
            if asset.source_path is None:
                raise FileNotFoundError(
                    f"asset source is required for declared asset {asset.asset_id!r}"
                )
            source = Path(asset.source_path)
            if self._is_link_or_reparse_point(source):
                raise ValueError(f"asset source cannot be a symbolic link: {source}")
            try:
                source_stat = source.stat()
            except OSError as exc:
                raise FileNotFoundError(f"asset source does not exist: {source}") from exc
            if not stat.S_ISREG(source_stat.st_mode):
                raise FileNotFoundError(f"asset source is not a regular file: {source}")
            if source_stat.st_size > _MAX_SINGLE_ASSET_BYTES:
                raise ValueError(
                    f"asset size limit exceeded for {asset.asset_id!r}: {source_stat.st_size}"
                )
            total_size += source_stat.st_size
            if total_size > _MAX_DOCUMENT_ASSET_BYTES:
                raise ValueError("document asset size limit exceeded")
            source_stats[asset.asset_id] = source_stat
        for asset in assets:
            if asset.sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", asset.sha256
            ):
                raise ValueError(
                    f"asset sha256 must be lowercase 64-hex for {asset.asset_id!r}"
                )
            destination = self._asset_destination(staging, asset.path)
            source = Path(asset.source_path)
            initial_stat = source_stats[asset.asset_id]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{uuid4().hex}.tmp"
            )
            digest = hashlib.sha256()
            copied_size = 0
            try:
                with source.open("rb") as source_file, temporary.open("xb") as output_file:
                    opened_stat = os.fstat(source_file.fileno())
                    if (
                        opened_stat.st_size != initial_stat.st_size
                        or opened_stat.st_mtime_ns != initial_stat.st_mtime_ns
                    ):
                        raise ValueError("asset source changed before streaming copy")
                    while chunk := source_file.read(_ASSET_COPY_CHUNK_BYTES):
                        copied_size += len(chunk)
                        if copied_size > initial_stat.st_size:
                            raise ValueError("asset source size changed during streaming copy")
                        digest.update(chunk)
                        output_file.write(chunk)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                    closed_stat = os.fstat(source_file.fileno())
                final_stat = source.stat()
                if (
                    copied_size != initial_stat.st_size
                    or closed_stat.st_size != initial_stat.st_size
                    or final_stat.st_size != initial_stat.st_size
                    or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                    or self._is_link_or_reparse_point(source)
                ):
                    raise ValueError("asset source changed during streaming copy")
                if self._is_link_or_reparse_point(destination):
                    raise ValueError("asset destination became a symbolic link")
                os.replace(temporary, destination)
                if self._is_link_or_reparse_point(destination):
                    raise ValueError("asset destination became a symbolic link")
            finally:
                if temporary.exists() and not self._is_link_or_reparse_point(temporary):
                    temporary.unlink()

            actual_sha256 = digest.hexdigest()
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
        """解析 asset 的 sha256，未提供时从 source_path 计算。"""
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
        """把 asset 的相对路径解析为 staging 下的安全绝对路径。"""
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

    # ------------------------------------------------------------------
    # 私有辅助：manifest 与输入指纹
    # ------------------------------------------------------------------
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
        """构建 manifest.json 的内容。"""
        payload = cls._build_persisted_input_payload(
            document_id,
            version,
            document,
            asset_hashes,
        )
        document_fields = dict(payload["document"])
        # typed_inventory 是派生库存，不参与 input_fingerprint；按 spec 写入
        # manifest["document"]["typed_inventory"]。
        document_fields["typed_inventory"] = cls._build_typed_inventory(
            document_id,
            version,
            document,
        )
        return {
            "canonical_markdown_sha256": canonical_markdown_sha256,
            "input_fingerprint": cls._input_fingerprint(payload),
            "document_id": payload["document_id"],
            "version": payload["version"],
            "document": document_fields,
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
        """构建用于计算 input_fingerprint 的规范化输入负载。

        落盘时排除 asset.source_path，避免缓存路径变化影响指纹；
        排除生成 block，保证持久化的是 source-only 内容。
        """
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
        """从已存在的 manifest 和原始数组重建输入负载，用于校验指纹一致性。

        typed_inventory 是派生库存，不参与 input_fingerprint，因此重建
        负载时从 ``manifest["document"]`` 中剔除，保证新旧 bundle 的
        指纹语义一致。
        """
        document_fields = dict(manifest["document"])
        document_fields.pop("typed_inventory", None)
        return {
            "document_id": manifest["document_id"],
            "version": manifest["version"],
            "document": document_fields,
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
        """把输入负载序列化为规范 JSON 字节串（排序、紧凑、无 NaN）。"""
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _input_fingerprint(cls, payload: dict[str, Any]) -> str:
        """计算输入负载的 SHA-256 指纹。"""
        return hashlib.sha256(cls._serialize_input_payload(payload)).hexdigest()

    # ------------------------------------------------------------------
    # 私有辅助：canonical.md 渲染
    # ------------------------------------------------------------------
    @staticmethod
    def _render_markdown(
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> str:
        """把 CanonicalDocument 渲染为 canonical.md Markdown 全文。"""
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

        # 按阅读顺序遍历 block，在遇到关联 block 时内联渲染表格/图片/公式。
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

        # 未被 block 引用的结构化对象追加在末尾。
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
        """渲染一个 CanonicalTable 为 Markdown 行列表。"""
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
        """根据 headers 和 rows 渲染标准 Markdown 表格。"""
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
        """从单元格列表重建表格网格，返回 (headers, rows)。"""
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
        """从 HTML 表格字符串解析出 CanonicalCell 列表。"""
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
                        text=cls._html_cell_text(table, element),
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

    @classmethod
    def _canonical_table_from_html(
        cls,
        source_html: str,
    ) -> tuple[list[CanonicalCell], list[str], list[list[str]]]:
        """Parse MinerU HTML using one canonical first-logical-row header rule."""
        cells = cls._table_cells_from_html(source_html)
        for cell in cells:
            if cell.row_index == 0:
                cell.is_header = True
        headers, rows = cls._table_grid_from_cells(cells)
        grid = [headers, *rows] if headers else rows
        if grid:
            width = len(grid[0])
            occupied = {
                (row_index, column_index)
                for cell in cells
                for row_index in range(cell.row_index, cell.row_index + cell.rowspan)
                for column_index in range(
                    cell.column_index, cell.column_index + cell.colspan
                )
            }
            for row_index, row in enumerate(grid):
                for column_index in range(width):
                    if (row_index, column_index) in occupied:
                        continue
                    cells.append(
                        CanonicalCell(
                            text=row[column_index],
                            row_index=row_index,
                            column_index=column_index,
                            is_header=row_index == 0,
                            metadata={
                                "synthetic_empty": True,
                                "reason": "html_ragged_grid_gap",
                            },
                        )
                    )
            cells.sort(key=lambda cell: (cell.row_index, cell.column_index))
        return cells, headers, rows

    @staticmethod
    def _html_cell_text(table: Tag, cell: Tag) -> str:
        """提取 HTML 单元格的纯文本，保留图片 alt、LaTeX 公式等内容。"""
        pieces: list[str] = []

        def visit(node: Tag | NavigableString) -> None:
            if isinstance(node, NavigableString):
                if node.find_parent("table") is table and str(node).strip():
                    pieces.append(str(node).strip())
                return
            if not isinstance(node, Tag) or node.name == "table":
                return
            if node.name == "img":
                value = node.get("alt") or node.get("src")
                if value:
                    pieces.append(str(value).strip())
                return
            classes = {str(value).casefold() for value in node.get("class", [])}
            if (
                node.name == "math"
                or "math" in classes
                or node.get("data-latex") is not None
            ):
                annotation = node.find(
                    "annotation", attrs={"encoding": re.compile("tex", re.I)}
                )
                value = node.get("data-latex") or (
                    annotation.get_text("", strip=True)
                    if annotation is not None
                    else node.get_text(" ", strip=True)
                )
                if value:
                    pieces.append(str(value).strip())
                return
            for child in list(node.children):
                visit(child)

        for child in list(cell.children):
            visit(child)
        return normalize_fragmented_numeric_spacing(" ".join(pieces))

    @staticmethod
    def _validate_table_dimensions(row_count: int, column_count: int) -> None:
        """校验表格尺寸不超过安全上限。"""
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
        """渲染图片为 Markdown 图片语法。"""
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
        """渲染公式为 LaTeX 块。"""
        lines: list[str] = []
        if formula.caption:
            lines.extend([f"### {formula.caption}", ""])
        lines.extend(["$$", formula.latex, "$$", ""])
        if formula.description:
            lines.extend([formula.description, ""])
        return lines

    # ------------------------------------------------------------------
    # 私有辅助：文件写入与模型序列化
    # ------------------------------------------------------------------
    @classmethod
    def _write_json(cls, path: Path, value: Any) -> None:
        """以格式化 JSON 写入文件。"""
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
    def _overwrite_json(cls, path: Path, value: Any) -> None:
        """以格式化 JSON 原子覆写已存在的文件（临时文件 + fsync + replace）。"""
        value = cls._sanitize_metadata_paths(value)
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output_file:
                output_file.write(encoded)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def _write_blocks(cls, path: Path, blocks: list[CanonicalBlock]) -> None:
        """把 blocks 以 JSONL 格式写入文件，排除生成 block。"""
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
        """序列化模型为 JSON 兼容字典，并清理 metadata 中的敏感路径。"""
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
        """递归清理 metadata 中的绝对路径，只保留文件名，防止泄露本地路径。

        远程 URI（http/https/s3/gs/minio）保留原样。
        """
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
        """把本地绝对路径脱敏为文件名；保留远程 URI。"""
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
        """转义 Markdown 表格单元格中的 pipe 和反斜杠。"""
        normalized = re.sub(r"\r\n|\r|\n", "<br>", value)
        return normalized.replace("\\", "\\\\").replace("|", "\\|")

    @staticmethod
    def _escape_figure_alt(value: str) -> str:
        """转义 Markdown 图片 alt 文本中的特殊字符。"""
        normalized = re.sub(r"\r\n|\r|\n", " ", value)
        return (
            normalized.replace("\\", "\\\\")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        """以原子方式写入文本文件：独占打开、flush、fsync。"""
        with path.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())

    # ------------------------------------------------------------------
    # 私有辅助：bundle 校验
    # ------------------------------------------------------------------
    def _validate_bundle(
        self,
        bundle: Path,
        document_id: str,
        version: str,
    ) -> None:
        """校验 bundle 目录结构、manifest、asset、记录契约和输入指纹。"""
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
        unexpected_manifest_fields = set(manifest).difference(
            _REQUIRED_MANIFEST_FIELDS | _LEGACY_MANIFEST_FIELDS
        )
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
        typed_inventory = self._extract_typed_inventory(manifest)
        if typed_inventory is not None:
            self._validate_typed_inventory(
                typed_inventory,
                document_id=document_id,
                version=version,
                document=persisted_document,
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
        """校验记录契约：ID 唯一、引用合法、asset 路径安全。"""
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
        """确保一组值没有重复。"""
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
        """校验 assets/ 目录下的文件与 asset 清单完全一致。"""
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
        """校验 manifest 中各字段类型与必填项。"""
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
        unexpected_document_fields = set(document).difference(
            required_document_fields | {"typed_inventory"}
        )
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

    # ------------------------------------------------------------------
    # typed_inventory：结构化表格/图/公式的版本作用域库存
    # ------------------------------------------------------------------
    @classmethod
    def _extract_typed_inventory(cls, manifest: dict[str, Any]) -> dict[str, Any] | None:
        """读取 manifest 的 typed_inventory，支持新 schema 与 legacy 兼容。

        新 bundle 使用 ``manifest["document"]["typed_inventory"]``（spec 位置）；
        legacy 顶层 ``manifest["typed_inventory"]`` 仅作为只读/迁移兼容形式被
        接受。两者同时出现视为歧义并抛错，避免隐式选择某一位置。
        """
        document = manifest.get("document")
        nested = document.get("typed_inventory") if isinstance(document, dict) else None
        top_level = manifest.get("typed_inventory")
        if nested is not None and top_level is not None:
            raise ValueError(
                "canonical manifest contains both top-level and document-level "
                "typed_inventory; migrate the legacy top-level value into "
                "manifest['document']['typed_inventory']"
            )
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError(
                    "canonical manifest document typed_inventory must be a JSON object"
                )
            return nested
        if top_level is not None:
            if not isinstance(top_level, dict):
                raise ValueError(
                    "canonical manifest top-level typed_inventory must be a JSON object"
                )
            return top_level
        return None

    @classmethod
    def _build_typed_inventory(
        cls,
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> dict[str, Any]:
        """构建 canonical 层的确定性 typed_inventory。

        ``tables`` 记录每个表格的 table_id、row_count、来源 block ID 集合，
        以及由后续持久化阶段（index）回填的 child_ids / child_count /
        parent_ids / row_indices。figure_ids / formula_ids 是文档中声明
        的图/公式 ID 集合；``orphan_structured_chunks`` 记录无 table_id /
        无来源 block 映射的结构化分块（index 阶段回填，canonical 层为空）。
        该结构完全来自 canonical artifact，不来自 LLM 输出，并且不参与
        input_fingerprint 计算。
        """
        source_block_ids_by_table: dict[str, list[str]] = {}
        for block in document.blocks:
            if not block.table_id or cls._is_generated_block(block):
                continue
            source_block_ids_by_table.setdefault(block.table_id, []).append(
                block.block_id
            )
        table_records: list[dict[str, Any]] = []
        for table in sorted(document.tables, key=lambda item: item.table_id):
            table_records.append(
                {
                    "table_id": table.table_id,
                    "row_count": len(table.rows),
                    "source_block_ids": sorted(
                        set(source_block_ids_by_table.get(table.table_id, []))
                    ),
                    "child_ids": [],
                    "child_count": 0,
                    "parent_ids": [],
                    "row_indices": [],
                }
            )
        return {
            "document_id": document_id,
            "version": version,
            "tables": table_records,
            "figure_ids": sorted(figure.figure_id for figure in document.figures),
            "formula_ids": sorted(formula.formula_id for formula in document.formulas),
            # 无 table_id / 无来源 block 映射的结构化分块由 index 阶段回填；
            # canonical 层初始为空。存在任何孤儿都会让激活 gate 失败关闭。
            "orphan_structured_chunks": [],
        }

    @classmethod
    def _validate_typed_inventory_shape(cls, typed: Any) -> None:
        """校验 typed_inventory 的结构（类型、唯一性、计数一致性）。

        只检查形状，不校验与 canonical 文档的一致性（后者需要文档对象）。
        """
        if not isinstance(typed, dict):
            raise ValueError("canonical manifest typed_inventory must be a JSON object")
        tables = typed.get("tables")
        if not isinstance(tables, list):
            raise ValueError("canonical manifest typed_inventory tables must be an array")
        table_ids: list[str] = []
        for record in tables:
            if not isinstance(record, dict):
                raise ValueError(
                    "canonical typed_inventory table records must be JSON objects"
                )
            required = {
                "table_id",
                "row_count",
                "source_block_ids",
                "child_ids",
                "child_count",
                "parent_ids",
                "row_indices",
            }
            missing = sorted(required.difference(record))
            if missing:
                raise ValueError(
                    "canonical typed_inventory table record missing fields: "
                    + ", ".join(missing)
                )
            table_id = record["table_id"]
            if not isinstance(table_id, str) or not table_id:
                raise ValueError(
                    "canonical typed_inventory table_id must be a non-empty string"
                )
            table_ids.append(table_id)
            row_count = record["row_count"]
            if (
                isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count < 0
            ):
                raise ValueError(
                    f"canonical typed_inventory row_count invalid for {table_id!r}"
                )
            for field in ("source_block_ids", "child_ids", "parent_ids", "row_indices"):
                values = record[field]
                if not isinstance(values, list) or not all(
                    isinstance(value, str)
                    if field in {"source_block_ids", "child_ids", "parent_ids"}
                    else isinstance(value, int) and value >= 0
                    for value in values
                ):
                    raise ValueError(
                        f"canonical typed_inventory {field} invalid for {table_id!r}"
                    )
                if len(set(values)) != len(values):
                    raise ValueError(
                        f"canonical typed_inventory duplicate {field} for {table_id!r}"
                    )
            child_count = record["child_count"]
            if (
                isinstance(child_count, bool)
                or not isinstance(child_count, int)
                or child_count != len(record["child_ids"])
            ):
                raise ValueError(
                    f"canonical typed_inventory child_count mismatch for {table_id!r}"
                )
        if len(set(table_ids)) != len(table_ids):
            raise ValueError("canonical typed_inventory duplicate table IDs")
        for key in ("figure_ids", "formula_ids"):
            ids = typed.get(key)
            if not isinstance(ids, list) or not all(
                isinstance(value, str) for value in ids
            ):
                raise ValueError(
                    f"canonical typed_inventory {key} must be a string array"
                )
            if len(set(ids)) != len(ids):
                raise ValueError(f"canonical typed_inventory duplicate {key}")
        orphans = typed.get("orphan_structured_chunks")
        if orphans is not None:
            if not isinstance(orphans, list) or not all(
                isinstance(value, str) and value for value in orphans
            ):
                raise ValueError(
                    "canonical typed_inventory orphan_structured_chunks must be "
                    "a non-empty string array"
                )
            if len(set(orphans)) != len(orphans):
                raise ValueError(
                    "canonical typed_inventory duplicate orphan chunk IDs"
                )

    @classmethod
    def _validate_typed_inventory(
        cls,
        typed: Any,
        *,
        document_id: str,
        version: str,
        document: CanonicalDocument,
    ) -> None:
        """校验 typed_inventory 的形状及与 persisted canonical 文档的一致性。

        只比较 canonical 层字段（table ID 集合、row_count、source_block_ids、
        figure/formula ID 集合）；child_ids / parent_ids / row_indices 由
        index 阶段从持久化 stage 数据回填，不属于 canonical 层，因此不在此比较。
        """
        cls._validate_typed_inventory_shape(typed)
        if typed.get("document_id") != document_id or typed.get("version") != version:
            raise ValueError(
                "canonical typed_inventory identity does not match the bundle"
            )
        expected = cls._build_typed_inventory(document_id, version, document)
        expected_tables = {record["table_id"]: record for record in expected["tables"]}
        stored_tables = {record["table_id"]: record for record in typed["tables"]}
        if set(stored_tables) != set(expected_tables):
            raise ValueError(
                "canonical typed_inventory table IDs do not match the persisted document"
            )
        for table_id in sorted(stored_tables):
            stored_record = stored_tables[table_id]
            expected_record = expected_tables[table_id]
            if stored_record["row_count"] != expected_record["row_count"]:
                raise ValueError(
                    f"canonical typed_inventory row_count mismatch for {table_id!r}"
                )
            if sorted(stored_record["source_block_ids"]) != sorted(
                expected_record["source_block_ids"]
            ):
                raise ValueError(
                    f"canonical typed_inventory source_block_ids mismatch for "
                    f"{table_id!r}"
                )
        if sorted(typed.get("figure_ids", [])) != sorted(expected["figure_ids"]):
            raise ValueError(
                "canonical typed_inventory figure IDs do not match the persisted document"
            )
        if sorted(typed.get("formula_ids", [])) != sorted(expected["formula_ids"]):
            raise ValueError(
                "canonical typed_inventory formula IDs do not match the persisted document"
            )

    def load_typed_inventory(self, document_id: str, version: str) -> dict[str, Any]:
        """加载并校验某个 (document_id, version) 的 manifest typed_inventory。

        新 bundle 从 ``manifest["document"]["typed_inventory"]`` 读取；legacy
        顶层 ``manifest["typed_inventory"]`` 仅作为只读/迁移兼容形式被接受。
        bundle 或 typed_inventory 缺失时失败关闭，并给出重建/迁移诊断。
        """
        self._validate_component(document_id)
        self._validate_component(version)
        document_root = self._prepare_document_root(document_id, create=False)
        bundle = document_root / version
        if not bundle.is_dir():
            raise FileNotFoundError(
                f"canonical bundle does not exist: {bundle}; rebuild is required"
            )
        self._validate_bundle(bundle, document_id, version)
        manifest = self._read_json(bundle / "manifest.json")
        typed = self._extract_typed_inventory(manifest)
        if typed is None:
            raise ValueError(
                "canonical manifest has no typed_inventory; the bundle predates "
                "typed inventory and must be rebuilt/migrated before activation "
                f"({document_id} {version})"
            )
        return typed

    def update_typed_inventory(
        self,
        document_id: str,
        version: str,
        child_inventory: dict[str, Any],
    ) -> Path:
        """把 index 阶段持久化的 Child 库存回填进 manifest 的 typed_inventory。

        ``child_inventory`` 形如 ``{"tables": {table_id: {"child_ids": [...],
        "parent_ids": [...]}}, "row_indices": {table_id: [...]},
        "orphan_structured_chunks": [...]}``。回填后再次校验形状并原子写回。
        若某个被分块的表格覆盖的行集合与 canonical row_count 不一致则抛错
        （fail-closed），阻止把丢行的版本激活。行覆盖只统计子块，父块不能
        掩盖缺失的 child 行。无 typed_inventory 或仍处于 legacy 顶层位置的
        bundle 一律失败关闭，要求重建/迁移。
        """
        self._validate_component(document_id)
        self._validate_component(version)
        if not isinstance(child_inventory, dict):
            raise ValueError("child inventory must be a JSON object")
        child_tables = child_inventory.get("tables")
        payload_rows = child_inventory.get("row_indices") or {}
        if not isinstance(child_tables, dict) or not isinstance(payload_rows, dict):
            raise ValueError("child inventory must contain a tables mapping")

        document_root = self._prepare_document_root(document_id, create=False)
        bundle = document_root / version
        if not bundle.is_dir():
            raise FileNotFoundError(f"canonical bundle does not exist: {bundle}")
        manifest_path = bundle / "manifest.json"
        manifest = self._read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("canonical manifest must be a JSON object")
        if manifest.get("document_id") != document_id or manifest.get("version") != version:
            raise ValueError("canonical manifest identity does not match update target")
        typed = self._extract_typed_inventory(manifest)
        if typed is None:
            raise ValueError(
                "canonical manifest has no typed_inventory; the bundle predates "
                "typed inventory and must be rebuilt/migrated before typed "
                "inventory updates"
            )
        if "typed_inventory" not in (manifest.get("document") or {}):
            raise ValueError(
                "canonical manifest typed_inventory is not in the new schema "
                "location manifest['document']['typed_inventory']; legacy "
                "top-level bundles must be rebuilt/migrated before typed "
                "inventory updates"
            )
        if not isinstance(typed.get("tables"), list):
            raise ValueError("canonical manifest is missing a valid typed_inventory")

        orphans = child_inventory.get("orphan_structured_chunks")
        if orphans is None:
            typed["orphan_structured_chunks"] = []
        elif isinstance(orphans, list) and all(
            isinstance(value, str) and value for value in orphans
        ):
            typed["orphan_structured_chunks"] = sorted(set(orphans))
        else:
            raise ValueError(
                "child inventory orphan_structured_chunks must be a non-empty "
                "string array"
            )

        known_tables = {
            record.get("table_id")
            for record in typed["tables"]
            if isinstance(record, dict) and isinstance(record.get("table_id"), str)
        }
        unknown_tables = sorted(set(child_tables).difference(known_tables))
        if unknown_tables:
            raise ValueError(
                "typed child inventory references unknown tables: "
                + ", ".join(unknown_tables)
            )

        for record in typed["tables"]:
            if not isinstance(record, dict) or not isinstance(record.get("table_id"), str):
                raise ValueError(
                    "canonical typed_inventory contains an invalid table record"
                )
            table_id = record["table_id"]
            row_count = record.get("row_count")
            data = child_tables.get(table_id)
            if data is None or not isinstance(data, dict):
                record["child_ids"] = []
                record["child_count"] = 0
                record["parent_ids"] = []
                record["row_indices"] = []
                if isinstance(row_count, int) and row_count > 0:
                    raise ValueError(
                        "typed inventory row coverage for "
                        f"{table_id!r} does not match row_count {row_count}: "
                        "covered=[]"
                    )
                continue
            child_ids = sorted(
                {
                    str(value)
                    for value in data.get("child_ids") or []
                    if str(value)
                }
            )
            record["child_ids"] = child_ids
            record["child_count"] = len(child_ids)
            record["parent_ids"] = sorted(
                {
                    str(value)
                    for value in data.get("parent_ids") or []
                    if str(value)
                }
            )
            covered = sorted(
                {
                    int(value)
                    for value in payload_rows.get(table_id) or []
                    if isinstance(value, int) and value >= 0
                }
            )
            record["row_indices"] = covered
            if (
                isinstance(row_count, int)
                and row_count > 0
                and covered != list(range(row_count))
            ):
                raise ValueError(
                    "typed inventory row coverage for "
                    f"{table_id!r} does not match row_count {row_count}: "
                    f"covered={covered}"
                )

        self._validate_typed_inventory_shape(typed)
        self._overwrite_json(manifest_path, manifest)
        return bundle

    # ------------------------------------------------------------------
    # 私有辅助：读取与哈希
    # ------------------------------------------------------------------
    @classmethod
    def _read_jsonl_model_list(
        cls,
        path: Path,
        model: type[Any],
    ) -> tuple[list[Any], list[Any]]:
        """读取 JSONL 文件，返回原始 dict 列表和验证后的模型列表。"""
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
        """读取 JSON 数组文件，返回原始列表和验证后的模型列表。"""
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
        return cls._safe_path_string(source_path) if source_path is not None else None

    @staticmethod
    def _is_link_or_reparse_point(path: Path) -> bool:
        """判断路径是否为符号链接或 Windows reparse point。"""
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
        """校验 asset 相对路径安全，返回 PurePosixPath。"""
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
        """判断 asset 路径组件是否可在跨平台环境中安全传输。"""
        return not (
            component in {"", ".", ".."}
            or component.endswith((".", " "))
            or any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in component)
            or any(ord(character) < 32 for character in component)
            or CanonicalArtifactStore._is_windows_device_name(component)
        )

    @staticmethod
    def _is_windows_device_name(component: str) -> bool:
        """判断名称是否为 Windows 保留设备名。"""
        stem = component.split(".", maxsplit=1)[0]
        return stem.upper() in _WINDOWS_DEVICE_NAMES
