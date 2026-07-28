from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.records import Document, DocumentChunk, DocumentParseVersion
from app.services.canonical_artifacts import CanonicalArtifactStore
from app.services.canonical_quality import CanonicalQualityGate
from app.services.ingestion_stages import IngestionStageRunner
from app.services.pipeline import IngestionPipeline
from app.services.vector_store import get_vector_store


STRICT_REBUILD_METRICS = (
    "contextual_prefix_completeness",
    "embedding_completeness",
    "table_validation_rate",
    "source_span_validity",
    "artifact_link_validity",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentRebuildResult:
    document_id: str
    source_path: str
    source_exists: bool
    planned_version: str
    status: str
    active_version: str | None = None
    failure_stage: str | None = None
    error: str | None = None


@dataclass
class RebuildReport:
    started_at: str
    finished_at: str
    dry_run: bool
    resume: bool
    total: int
    succeeded_documents: int
    failed_documents: int
    failed_document_ids: list[str]
    documents: list[DocumentRebuildResult] = field(default_factory=list)
    parse_completeness: float = 0.0
    contextualized_children: int = 0
    embedded_children: int = 0
    indexed_children: int = 0
    pgvector_rows: int | None = None
    contextual_prefix_completeness: float = 0.0
    embedding_completeness: float = 0.0
    pgvector_completeness: float | None = None
    table_validation_rate: float = 0.0
    source_span_validity: float = 0.0
    artifact_link_validity: float = 0.0
    ready_for_acceptance: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanupReport:
    deleted_chunks: int = 0
    deleted_parse_versions: int = 0
    deleted_vector_versions: int = 0
    deleted_artifact_directories: int = 0
    deleted_mineru_directories: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _planned_version(document: Document) -> str:
    return f"{get_settings().canonical_pipeline_version}-{document.sha256[:12]}"


def _selected_documents(db: Session, document_id: str | None) -> list[Document]:
    statement = select(Document).order_by(Document.id)
    if document_id:
        statement = statement.where(Document.id == document_id)
    documents = list(db.scalars(statement))
    if document_id and not documents:
        raise ValueError(f"Document {document_id!r} does not exist.")
    return documents


def _ratio(valid: int, total: int, *, empty_is_complete: bool = False) -> float:
    if total <= 0:
        return 1.0 if empty_is_complete else 0.0
    return round(valid / total, 6)


def collect_integrity_metrics(
    db: Session,
    *,
    artifact_root: Path | None = None,
    document_ids: set[str] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    documents = _selected_documents(db, None)
    if document_ids is not None:
        documents = [document for document in documents if document.id in document_ids]
    store = CanonicalArtifactStore(artifact_root or settings.canonical_artifacts_dir)
    dimensions = settings.ollama_embedding_dimensions
    vector_store = get_vector_store(db)

    active_documents = [
        document for document in documents if bool(document.active_parse_version)
    ]
    active_pairs = {
        (document.id, str(document.active_parse_version)) for document in active_documents
    }
    children = [
        chunk
        for chunk in db.scalars(
            select(DocumentChunk).where(DocumentChunk.chunk_role == "child")
        )
        if (chunk.document_id, chunk.parse_version) in active_pairs
    ]
    contextualized = sum(
        bool(chunk.contextual_prefix)
        and bool(chunk.contextualization_model)
        and bool(chunk.contextualization_version)
        and bool(chunk.contextualization_prompt_version)
        and chunk.contextualized_at is not None
        and chunk.embedding_text == f"{chunk.contextual_prefix}\n\n{chunk.text}"
        for chunk in children
    )
    embedded = sum(
        isinstance(chunk.embedding, list)
        and len(chunk.embedding) == dimensions
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in chunk.embedding
        )
        for chunk in children
    )
    valid_spans = sum(IngestionPipeline._chunk_has_valid_source_spans(chunk) for chunk in children)

    indexed = 0
    pgvector_rows: int | None = None
    is_postgresql = db.get_bind().dialect.name == "postgresql"
    if vector_store.available():
        indexed = sum(
            vector_store.count_document_chunks(document.id, str(document.active_parse_version))
            for document in active_documents
        )
        if is_postgresql:
            pgvector_rows = indexed
    elif is_postgresql:
        indexed = 0
        pgvector_rows = 0
    else:
        indexed = embedded

    valid_artifacts = 0
    valid_tables = 0
    total_tables = 0
    for document in active_documents:
        try:
            canonical = store.load(document.id, str(document.active_parse_version))
        except (FileNotFoundError, ValueError, OSError):
            continue
        valid_artifacts += 1
        report = CanonicalQualityGate().evaluate(canonical)
        invalid_table_ids = {
            str(issue.metadata.get("table_id"))
            for issue in report.issues
            if issue.code == "table_invalid" and issue.metadata.get("table_id")
        }
        total_tables += len(canonical.tables)
        valid_tables += sum(
            table.status != "validation_failed" and table.table_id not in invalid_table_ids
            for table in canonical.tables
        )

    pgvector_completeness: float | None = None
    if is_postgresql:
        pgvector_completeness = _ratio(indexed, len(children))
    return {
        "parse_completeness": _ratio(len(active_documents), len(documents)),
        "total_children": len(children),
        "contextualized_children": contextualized,
        "embedded_children": embedded,
        "indexed_children": indexed,
        "pgvector_rows": pgvector_rows,
        "contextual_prefix_completeness": _ratio(contextualized, len(children)),
        "embedding_completeness": _ratio(embedded, len(children)),
        "pgvector_completeness": pgvector_completeness,
        "table_validation_rate": _ratio(valid_tables, total_tables, empty_is_complete=True),
        "source_span_validity": _ratio(valid_spans, len(children)),
        "artifact_link_validity": _ratio(valid_artifacts, len(active_documents)),
    }


def rebuild(
    db: Session,
    *,
    artifact_root: Path | None = None,
    dry_run: bool = False,
    resume: bool = False,
    document_id: str | None = None,
) -> RebuildReport:
    started_at = _utc_now()
    documents = _selected_documents(db, document_id)
    results: list[DocumentRebuildResult] = []

    if dry_run:
        for document in documents:
            source = Path(document.raw_path).expanduser()
            results.append(
                DocumentRebuildResult(
                    document_id=document.id,
                    source_path=str(source),
                    source_exists=source.is_file(),
                    planned_version=_planned_version(document),
                    status="planned" if source.is_file() else "missing_source",
                    failure_stage=None if source.is_file() else "source_check",
                    error=None if source.is_file() else "Original source file is missing.",
                )
            )
        failed_ids = [row.document_id for row in results if not row.source_exists]
        return RebuildReport(
            started_at=started_at,
            finished_at=_utc_now(),
            dry_run=True,
            resume=resume,
            total=len(documents),
            succeeded_documents=len(documents) - len(failed_ids),
            failed_documents=len(failed_ids),
            failed_document_ids=failed_ids,
            documents=results,
            ready_for_acceptance=False,
        )

    if artifact_root is not None and artifact_root != get_settings().canonical_artifacts_dir:
        raise ValueError("artifact_root overrides are supported only for dry-run inspection.")

    for document in documents:
        source = Path(document.raw_path).expanduser()
        planned_version = _planned_version(document)
        if not source.is_file():
            results.append(
                DocumentRebuildResult(
                    document_id=document.id,
                    source_path=str(source),
                    source_exists=False,
                    planned_version=planned_version,
                    status="failed",
                    failure_stage="source_check",
                    error="Original source file is missing.",
                )
            )
            continue
        try:
            existing = db.scalar(
                select(DocumentParseVersion).where(
                    DocumentParseVersion.document_id == document.id,
                    DocumentParseVersion.version_key == planned_version,
                )
            )
            if existing is not None and existing.stage_state and not resume:
                raise RuntimeError(
                    "Existing canonical stage checkpoints require --resume."
                )
            pipeline = IngestionPipeline(db)
            version = pipeline._get_or_create_parse_version(document)  # noqa: SLF001
            runner = IngestionStageRunner(
                db,
                handlers=pipeline.ingestion_stage_handlers(),
                dispatcher=None,
            )
            runner.run_until_blocked(
                document.id,
                version.version_key,
                include_activation=True,
            )
            db.refresh(document)
            results.append(
                DocumentRebuildResult(
                    document_id=document.id,
                    source_path=str(source),
                    source_exists=True,
                    planned_version=version.version_key,
                    status="succeeded",
                    active_version=document.active_parse_version,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report each document and continue
            db.rollback()
            results.append(
                DocumentRebuildResult(
                    document_id=document.id,
                    source_path=str(source),
                    source_exists=True,
                    planned_version=planned_version,
                    status="failed",
                    failure_stage="rebuild",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    failed_ids = [row.document_id for row in results if row.status != "succeeded"]
    metrics = collect_integrity_metrics(
        db,
        artifact_root=artifact_root,
        document_ids={document.id for document in documents},
    )
    strict_metrics_pass = all(metrics[name] == 1.0 for name in STRICT_REBUILD_METRICS)
    ready = (
        not failed_ids
        and metrics["parse_completeness"] == 1.0
        and strict_metrics_pass
        and (
            metrics["pgvector_completeness"] in {None, 1.0}
        )
    )
    return RebuildReport(
        started_at=started_at,
        finished_at=_utc_now(),
        dry_run=False,
        resume=resume,
        total=len(documents),
        succeeded_documents=len(documents) - len(failed_ids),
        failed_documents=len(failed_ids),
        failed_document_ids=failed_ids,
        documents=results,
        ready_for_acceptance=ready,
        **{key: value for key, value in metrics.items() if key != "total_children"},
    )


def _safe_remove_directory(path: Path, *, allowed_root: Path) -> bool:
    root = allowed_root.expanduser().resolve()
    target = path.expanduser().resolve()
    if target == root or root not in target.parents or not target.exists():
        return False
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return True


def _clear_temp_root(root: Path | None) -> int:
    if root is None:
        return 0
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved == Path(resolved.anchor):
        return 0
    deleted = 0
    for child in list(resolved.iterdir()):
        if _safe_remove_directory(child, allowed_root=resolved):
            deleted += 1
    return deleted


def cleanup_old_data(
    db: Session,
    *,
    artifact_root: Path,
    rebuild_report: dict[str, Any],
    delete_old_after_acceptance: bool,
    confirm_delete_old_data: bool,
    mineru_output_dir: Path | None = None,
) -> CleanupReport:
    if not (delete_old_after_acceptance and confirm_delete_old_data):
        raise ValueError("Cleanup requires both deletion guards.")
    if (
        rebuild_report.get("ready_for_acceptance") is not True
        or int(rebuild_report.get("failed_documents") or 0) != 0
    ):
        raise ValueError("Rebuild report is not ready for acceptance cleanup.")

    result = CleanupReport()
    documents = list(db.scalars(select(Document).order_by(Document.id)))
    vector_store = get_vector_store(db)
    old_versions: list[DocumentParseVersion] = []
    old_keys_by_document: dict[str, set[str]] = {}
    for document in documents:
        active = document.active_parse_version
        versions = list(
            db.scalars(
                select(DocumentParseVersion).where(
                    DocumentParseVersion.document_id == document.id
                )
            )
        )
        inactive = [version for version in versions if version.version_key != active]
        old_versions.extend(inactive)
        old_keys_by_document[document.id] = {
            "legacy",
            *(version.version_key for version in inactive),
        }
        if vector_store.available():
            for version_key in old_keys_by_document[document.id]:
                try:
                    vector_store.delete_document(
                        document.id, parse_version=version_key
                    )
                except TypeError:
                    # SQLiteVecStore exposes version-scoped deletion through
                    # its internal row operation; its public method deletes
                    # every version and would also remove the active index.
                    vector_store._ensure_meta_table()  # noqa: SLF001
                    vector_store._delete_document_rows(  # noqa: SLF001
                        document.id, parse_version=version_key
                    )
                result.deleted_vector_versions += 1

    for document in documents:
        old_keys = old_keys_by_document[document.id]
        chunk_ids = list(
            db.scalars(
                select(DocumentChunk.id).where(
                    DocumentChunk.document_id == document.id,
                    DocumentChunk.parse_version.in_(old_keys),
                )
            )
        )
        if chunk_ids:
            db.execute(delete(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))
            result.deleted_chunks += len(chunk_ids)

    for version in old_versions:
        candidates = {
            Path(version.artifact_dir),
            artifact_root / version.document_id / version.version_key,
            artifact_root / version.document_id / f"{version.version_key}.pipeline",
            artifact_root / version.document_id / f"{version.version_key}.staging",
        }
        for candidate in candidates:
            if _safe_remove_directory(candidate, allowed_root=artifact_root):
                result.deleted_artifact_directories += 1
        db.delete(version)
        result.deleted_parse_versions += 1

    db.commit()
    result.deleted_mineru_directories = _clear_temp_root(mineru_output_dir)
    return result


def run_cli_rebuild(**kwargs) -> RebuildReport:
    with SessionLocal() as db:
        return rebuild(db, **kwargs)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the active canonical RAG index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--document-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--delete-old-after-acceptance", action="store_true")
    parser.add_argument("--confirm-delete-old-data", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    if args.delete_old_after_acceptance or args.confirm_delete_old_data:
        if not (args.delete_old_after_acceptance and args.confirm_delete_old_data):
            _parser().error(
                "--delete-old-after-acceptance and --confirm-delete-old-data must be used together"
            )
        if not args.report.is_file():
            _parser().error("--report must reference a successful existing rebuild report")
        rebuild_report = json.loads(args.report.read_text(encoding="utf-8"))
        with SessionLocal() as db:
            cleanup = cleanup_old_data(
                db,
                artifact_root=settings.canonical_artifacts_dir,
                mineru_output_dir=settings.mineru_output_dir or settings.cache_dir / "mineru",
                rebuild_report=rebuild_report,
                delete_old_after_acceptance=True,
                confirm_delete_old_data=True,
            )
        rebuild_report["cleanup"] = cleanup.to_dict()
        rebuild_report["cleanup_completed_at"] = _utc_now()
        _write_json(args.report, rebuild_report)
        return 0

    report = run_cli_rebuild(
        dry_run=args.dry_run,
        resume=args.resume,
        document_id=args.document_id,
    )
    payload = report.to_dict()
    _write_json(args.report, payload)
    if int(payload.get("failed_documents") or 0) or (
        not args.dry_run and payload.get("ready_for_acceptance") is not True
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
