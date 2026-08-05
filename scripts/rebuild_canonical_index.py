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
from app.services.ai import OllamaClient
from app.services.canonical_artifacts import CanonicalArtifactStore
from app.services.canonical_quality import CanonicalQualityGate
from app.services.contextualization_policy import (
    requires_contextualization,
    valid_contextualized_embedding,
    valid_plain_embedding,
)
from app.services.ingestion_stages import IngestionStageRunner
from app.services.ingestion_identity import (
    TokenizerUnavailableError,
    build_ingestion_config_snapshot,
    build_parse_version_key,
    canonical_ingestion_config_hash,
)
from app.services.pipeline import (
    IngestionPipeline,
    compare_typed_inventory,
    derive_db_typed_inventory,
)
from app.services.structured_evidence import StructuredEvidenceBuilder
from app.services.vector_store import get_vector_store


STRICT_REBUILD_METRICS = (
    "source_fidelity_completeness",
    "structured_limit_completeness",
    "contextual_prefix_completeness",
    "plain_embedding_completeness",
    "embedding_completeness",
    "table_validation_rate",
    "source_span_validity",
    "artifact_link_validity",
    "config_identity_completeness",
    "source_version_identity_completeness",
    "child_token_limit_completeness",
)

_REBUILD_TOKEN_PROVIDER = StructuredEvidenceBuilder(strict_tokenizer=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _release_loaded_models() -> list[str]:
    return OllamaClient().unload_loaded_models()


def _strict_child_token_count(text: str) -> int:
    return _REBUILD_TOKEN_PROVIDER.estimate_tokens(text)


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
    parse_version_map: dict[str, str] = field(default_factory=dict)
    tokenizer_available: bool = False
    tokenizer_error: str | None = None
    ingestion_config: dict[str, Any] | None = None
    ingestion_config_sha256: str | None = None
    parse_completeness: float = 0.0
    contextualization_eligible_children: int = 0
    contextualized_children: int = 0
    plain_embedding_children: int = 0
    embedded_children: int = 0
    indexed_children: int = 0
    pgvector_rows: int | None = None
    source_fidelity_completeness: float = 0.0
    structured_limit_completeness: float = 0.0
    contextual_prefix_completeness: float = 0.0
    plain_embedding_completeness: float = 0.0
    embedding_completeness: float = 0.0
    pgvector_completeness: float | None = None
    table_validation_rate: float = 0.0
    source_span_validity: float = 0.0
    artifact_link_validity: float = 0.0
    config_identity_completeness: float = 0.0
    source_version_identity_completeness: float = 0.0
    child_token_limit_completeness: float = 0.0
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


def _planned_version(
    document: Document,
    *,
    ingestion_config: dict[str, Any] | None = None,
    ingestion_config_sha256: str | None = None,
) -> str:
    snapshot = ingestion_config or build_ingestion_config_snapshot()
    config_hash = ingestion_config_sha256 or canonical_ingestion_config_hash(snapshot)
    return build_parse_version_key(
        document.sha256,
        snapshot=snapshot,
        config_sha256=config_hash,
        settings=get_settings(),
    )


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


def _exact_ratio(valid: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return valid / total


def collect_integrity_metrics(
    db: Session,
    *,
    artifact_root: Path | None = None,
    document_ids: set[str] | None = None,
    expected_ingestion_config: dict[str, Any] | None = None,
    expected_ingestion_config_sha256: str | None = None,
    parse_version_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    documents = _selected_documents(db, None)
    if document_ids is not None:
        documents = [document for document in documents if document.id in document_ids]
    store = CanonicalArtifactStore(artifact_root or settings.canonical_artifacts_dir)
    dimensions = settings.ollama_embedding_dimensions
    vector_store = get_vector_store(db)

    selected_version_by_document = {
        document.id: str(parse_version_map[document.id])
        for document in documents
        if parse_version_map is not None and document.id in parse_version_map
    } if parse_version_map is not None else {
        document.id: str(document.active_parse_version)
        for document in documents
        if bool(document.active_parse_version)
    }
    active_documents = [
        document for document in documents if document.id in selected_version_by_document
    ]
    active_pairs = set(selected_version_by_document.items())
    active_versions = {
        (version.document_id, version.version_key): version
        for version in db.scalars(select(DocumentParseVersion))
        if (version.document_id, version.version_key) in active_pairs
    }
    completed_fidelity_checkpoints = {
        "source_fidelity_completeness": 0,
        "structured_limit_completeness": 0,
    }
    for pair in active_pairs:
        version = active_versions.get(pair)
        stage_state = version.stage_state if version is not None else None
        semantic_split = (
            stage_state.get("semantic_split") if isinstance(stage_state, dict) else None
        )
        if not isinstance(semantic_split, dict) or semantic_split.get("status") != "completed":
            continue
        output = (
            semantic_split.get("output")
            if isinstance(semantic_split, dict)
            else None
        )
        if not isinstance(output, dict):
            continue
        for metric_name in completed_fidelity_checkpoints:
            value = output.get(metric_name)
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) == 1.0
            ):
                completed_fidelity_checkpoints[metric_name] += 1
    matching_config_identities = 0
    matching_source_version_identities = 0
    for pair in active_pairs:
        version = active_versions.get(pair)
        if version is None or expected_ingestion_config is None or expected_ingestion_config_sha256 is None:
            continue
        manifest = version.manifest_json or {}
        stored_snapshot = manifest.get("ingestion_config")
        stored_hash = manifest.get("ingestion_config_sha256")
        if (
            stored_snapshot == expected_ingestion_config
            and stored_hash == expected_ingestion_config_sha256
            and isinstance(stored_snapshot, dict)
            and canonical_ingestion_config_hash(stored_snapshot) == stored_hash
            and version.version_key.endswith(
                f"-{expected_ingestion_config_sha256[:12]}"
            )
        ):
            matching_config_identities += 1
    if (
        expected_ingestion_config is not None
        and expected_ingestion_config_sha256 is not None
    ):
        for document in active_documents:
            expected_version_key = build_parse_version_key(
                document.sha256,
                snapshot=expected_ingestion_config,
                config_sha256=expected_ingestion_config_sha256,
                settings=settings,
            )
            if selected_version_by_document.get(document.id) == expected_version_key:
                matching_source_version_identities += 1
    children = [
        chunk
        for chunk in db.scalars(
            select(DocumentChunk).where(DocumentChunk.chunk_role == "child")
        )
        if (chunk.document_id, chunk.parse_version) in active_pairs
    ]
    eligible = [
        chunk for chunk in children if requires_contextualization(chunk.block_type)
    ]
    plain = [
        chunk for chunk in children if not requires_contextualization(chunk.block_type)
    ]
    contextualized = sum(valid_contextualized_embedding(chunk) for chunk in eligible)
    plain_embedded = sum(valid_plain_embedding(chunk) for chunk in plain)
    child_token_max: int | None = None
    if expected_ingestion_config is not None:
        semantic_splitting = expected_ingestion_config.get("semantic_splitting")
        child_tokens = (
            semantic_splitting.get("child_tokens")
            if isinstance(semantic_splitting, dict)
            else None
        )
        configured_max = (
            child_tokens.get("max") if isinstance(child_tokens, dict) else None
        )
        if (
            not isinstance(configured_max, bool)
            and isinstance(configured_max, int)
            and configured_max > 0
        ):
            child_token_max = configured_max
    valid_child_token_limits = 0
    if child_token_max is not None:
        for chunk in children:
            actual_token_count = _strict_child_token_count(chunk.text)
            if (
                actual_token_count == chunk.token_count
                and actual_token_count <= child_token_max
            ):
                valid_child_token_limits += 1
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
            vector_store.count_document_chunks(
                document.id,
                selected_version_by_document[document.id],
            )
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
            canonical = store.load(
                document.id,
                selected_version_by_document[document.id],
            )
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
        "parse_completeness": _ratio(len(active_versions), len(documents)),
        "total_children": len(children),
        "contextualization_eligible_children": len(eligible),
        "contextualized_children": contextualized,
        "plain_embedding_children": plain_embedded,
        "embedded_children": embedded,
        "indexed_children": indexed,
        "pgvector_rows": pgvector_rows,
        "source_fidelity_completeness": _exact_ratio(
            completed_fidelity_checkpoints["source_fidelity_completeness"],
            len(active_documents),
        ),
        "structured_limit_completeness": _exact_ratio(
            completed_fidelity_checkpoints["structured_limit_completeness"],
            len(active_documents),
        ),
        "contextual_prefix_completeness": _ratio(
            contextualized,
            len(eligible),
            empty_is_complete=True,
        ),
        "plain_embedding_completeness": _ratio(
            plain_embedded,
            len(plain),
            empty_is_complete=True,
        ),
        "embedding_completeness": _ratio(embedded, len(children)),
        "pgvector_completeness": pgvector_completeness,
        "table_validation_rate": _ratio(valid_tables, total_tables, empty_is_complete=True),
        "source_span_validity": _ratio(valid_spans, len(children)),
        "artifact_link_validity": _ratio(valid_artifacts, len(active_documents)),
        "config_identity_completeness": _ratio(
            matching_config_identities,
            len(active_documents),
        ),
        "source_version_identity_completeness": _ratio(
            matching_source_version_identities,
            len(active_documents),
        ),
        "child_token_limit_completeness": _exact_ratio(
            valid_child_token_limits,
            len(children),
        ),
    }


def activate_rebuild_batch(
    db: Session,
    *,
    parse_version_map: dict[str, str],
    acceptance_report: dict[str, Any],
    artifact_root: Path | None = None,
) -> list[DocumentParseVersion]:
    requested = {
        str(document_id): str(version_key)
        for document_id, version_key in parse_version_map.items()
        if str(document_id) and str(version_key)
    }
    if not requested or len(requested) != len(parse_version_map):
        raise ValueError("Activation requires a non-empty staged version map.")
    accepted_map_raw = acceptance_report.get("parse_version_map")
    accepted_map = (
        {
            str(document_id): str(version_key)
            for document_id, version_key in accepted_map_raw.items()
            if str(document_id) and str(version_key)
        }
        if isinstance(accepted_map_raw, dict)
        else {}
    )
    if accepted_map != requested:
        raise ValueError(
            "Acceptance report version map does not match the activation request."
        )
    if acceptance_report.get("strict_pass") is not True:
        raise RuntimeError("Strict staged acceptance must pass before activation.")

    ingestion_config = build_ingestion_config_snapshot()
    ingestion_config_sha256 = canonical_ingestion_config_hash(ingestion_config)
    metrics = collect_integrity_metrics(
        db,
        artifact_root=artifact_root,
        document_ids=set(requested),
        expected_ingestion_config=ingestion_config,
        expected_ingestion_config_sha256=ingestion_config_sha256,
        parse_version_map=requested,
    )
    failed_metrics = [
        name
        for name in ("parse_completeness", *STRICT_REBUILD_METRICS)
        if metrics.get(name) != 1.0
    ]
    if metrics.get("pgvector_completeness") not in {None, 1.0}:
        failed_metrics.append("pgvector_completeness")
    if failed_metrics:
        raise RuntimeError(
            "Staged integrity regressed before activation: "
            + ", ".join(failed_metrics)
        )

    store = CanonicalArtifactStore(
        artifact_root or get_settings().canonical_artifacts_dir
    )
    for document_id in sorted(requested):
        version_key = requested[document_id]
        try:
            manifest_inventory = store.load_typed_inventory(document_id, version_key)
        except (FileNotFoundError, ValueError) as exc:
            # canonical bundle 缺失或 typed_inventory 缺失/损坏/legacy 迁移
            # 需求：批量激活必须失败关闭，禁止静默跳过。
            raise RuntimeError(
                "Staged typed inventory is unavailable for activation "
                f"({document_id} {version_key}): {exc}"
            ) from exc
        chunk_rows = list(
            db.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.parse_version == version_key,
                )
            ).all()
        )
        observed = derive_db_typed_inventory(manifest_inventory, chunk_rows)
        mismatches = compare_typed_inventory(manifest_inventory, observed)
        if mismatches:
            raise RuntimeError(
                "Staged typed inventory mismatch blocks activation: "
                + "; ".join(mismatches)
            )

    pipeline = IngestionPipeline(db)
    runner = IngestionStageRunner(
        db,
        handlers=pipeline.ingestion_stage_handlers(),
        dispatcher=None,
        pre_stage_validator=getattr(
            pipeline,
            "validate_ingestion_identity",
            None,
        ),
    )
    return runner.activate_batch(requested)


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
    tokenizer_available = False
    tokenizer_error: str | None = None
    ingestion_config: dict[str, Any] | None = None
    ingestion_config_sha256: str | None = None
    try:
        ingestion_config = build_ingestion_config_snapshot()
        ingestion_config_sha256 = canonical_ingestion_config_hash(ingestion_config)
        tokenizer_available = True
    except TokenizerUnavailableError as exc:
        tokenizer_error = str(exc)

    if dry_run:
        for document in documents:
            source = Path(document.raw_path).expanduser()
            planned_version = (
                _planned_version(
                    document,
                    ingestion_config=ingestion_config,
                    ingestion_config_sha256=ingestion_config_sha256,
                )
                if tokenizer_available
                and ingestion_config is not None
                and ingestion_config_sha256 is not None
                else f"{get_settings().canonical_pipeline_version}-{document.sha256[:12]}-unavailable"
            )
            results.append(
                DocumentRebuildResult(
                    document_id=document.id,
                    source_path=str(source),
                    source_exists=source.is_file(),
                    planned_version=planned_version,
                    status=(
                        "planned"
                        if source.is_file() and tokenizer_available
                        else "missing_source"
                        if not source.is_file()
                        else "tokenizer_unavailable"
                    ),
                    failure_stage=(
                        None
                        if source.is_file() and tokenizer_available
                        else "source_check"
                        if not source.is_file()
                        else "tokenizer_preflight"
                    ),
                    error=(
                        None
                        if source.is_file() and tokenizer_available
                        else "Original source file is missing."
                        if not source.is_file()
                        else tokenizer_error
                    ),
                )
            )
        failed_ids = [
            row.document_id
            for row in results
            if not row.source_exists or not tokenizer_available
        ]
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
            tokenizer_available=tokenizer_available,
            tokenizer_error=tokenizer_error,
            ingestion_config=ingestion_config,
            ingestion_config_sha256=ingestion_config_sha256,
            ready_for_acceptance=False,
        )

    if artifact_root is not None and artifact_root != get_settings().canonical_artifacts_dir:
        raise ValueError("artifact_root overrides are supported only for dry-run inspection.")

    if not tokenizer_available or ingestion_config is None or ingestion_config_sha256 is None:
        results = [
            DocumentRebuildResult(
                document_id=document.id,
                source_path=str(Path(document.raw_path).expanduser()),
                source_exists=Path(document.raw_path).expanduser().is_file(),
                planned_version=(
                    f"{get_settings().canonical_pipeline_version}-"
                    f"{document.sha256[:12]}-unavailable"
                ),
                status="tokenizer_unavailable",
                failure_stage="tokenizer_preflight",
                error=tokenizer_error or "Required tokenizer identity is unavailable.",
            )
            for document in documents
        ]
        return RebuildReport(
            started_at=started_at,
            finished_at=_utc_now(),
            dry_run=False,
            resume=resume,
            total=len(documents),
            succeeded_documents=0,
            failed_documents=len(documents),
            failed_document_ids=[document.id for document in documents],
            documents=results,
            tokenizer_available=False,
            tokenizer_error=tokenizer_error,
            ingestion_config=None,
            ingestion_config_sha256=None,
            ready_for_acceptance=False,
        )

    for document in documents:
        source = Path(document.raw_path).expanduser()
        planned_version = _planned_version(
            document,
            ingestion_config=ingestion_config,
            ingestion_config_sha256=ingestion_config_sha256,
        )
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
                pre_stage_validator=getattr(
                    pipeline, "validate_ingestion_identity", None
                ),
            )
            runner.run_until_blocked(
                document.id,
                version.version_key,
                include_activation=False,
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
        finally:
            _release_loaded_models()

    failed_ids = [row.document_id for row in results if row.status != "succeeded"]
    metrics = collect_integrity_metrics(
        db,
        artifact_root=artifact_root,
        document_ids={document.id for document in documents},
        expected_ingestion_config=ingestion_config,
        expected_ingestion_config_sha256=ingestion_config_sha256,
        parse_version_map={
            row.document_id: row.planned_version
            for row in results
            if row.status == "succeeded"
        },
    )
    strict_metrics_pass = all(metrics[name] == 1.0 for name in STRICT_REBUILD_METRICS)
    ready = (
        not failed_ids
        and tokenizer_available
        and ingestion_config_sha256 is not None
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
        parse_version_map={
            row.document_id: row.planned_version
            for row in results
            if row.status == "succeeded"
        },
        tokenizer_available=tokenizer_available,
        tokenizer_error=tokenizer_error,
        ingestion_config=ingestion_config,
        ingestion_config_sha256=ingestion_config_sha256,
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
    version_map_raw = rebuild_report.get("parse_version_map")
    activation = rebuild_report.get("activation")
    version_map = (
        {
            str(document_id): str(version_key)
            for document_id, version_key in version_map_raw.items()
            if str(document_id) and str(version_key)
        }
        if isinstance(version_map_raw, dict)
        else {}
    )
    activated_document_ids = (
        {
            str(document_id)
            for document_id in activation.get("activated_document_ids", [])
            if str(document_id)
        }
        if isinstance(activation, dict)
        and activation.get("status") == "completed"
        and isinstance(activation.get("activated_document_ids"), list)
        else set()
    )
    if not version_map or activated_document_ids != set(version_map):
        raise ValueError(
            "Cleanup requires a completed activated batch bound to the version map."
        )
    activated_rows = db.execute(
        select(
            Document.id,
            Document.active_parse_version,
            DocumentParseVersion.status,
        )
        .join(
            DocumentParseVersion,
            (DocumentParseVersion.document_id == Document.id)
            & (
                DocumentParseVersion.version_key
                == Document.active_parse_version
            ),
        )
        .where(Document.id.in_(sorted(version_map)))
    ).all()
    activated_state = {
        str(row.id): (str(row.active_parse_version), str(row.status))
        for row in activated_rows
    }
    if any(
        activated_state.get(document_id) != (version_key, "active")
        for document_id, version_key in version_map.items()
    ):
        raise ValueError(
            "Cleanup requires the activated batch to match current active pointers."
        )

    result = CleanupReport()
    documents = list(
        db.scalars(
            select(Document)
            .where(Document.id.in_(sorted(version_map)))
            .order_by(Document.id)
        )
    )
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
        inactive = [
            version
            for version in versions
            if version.version_key != active and version.status == "superseded"
        ]
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
