from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.records import (
    Document,
    DocumentParseVersion,
    PipelineRun,
    RunStatus,
)
from app.services.parse_versions import (
    ALLOWED_TRANSITIONS,
    FAILED_STAGE_RETRIES,
    ParseVersionService,
)


INGESTION_STAGES = (
    "parse",
    "repair",
    "canonicalize",
    "semantic_split",
    "contextualize",
    "embed",
    "index",
    "activate",
)

STAGE_QUEUES = {stage: f"ingest.{stage}" for stage in INGESTION_STAGES}

_COMPLETED_STATUS = {
    "parse": "quality_checking",
    "repair": "canonicalizing",
    "canonicalize": "chunking",
    "semantic_split": "contextualizing",
    "contextualize": "embedding",
    "embed": "indexing",
    "index": "ready_to_activate",
    "activate": "active",
}
_RUNNING_STATUS = {
    "parse": "parsing",
    "repair": "repairing",
    "canonicalize": "canonicalizing",
    "semantic_split": "chunking",
    "contextualize": "contextualizing",
    "embed": "embedding",
    "index": "indexing",
    "activate": "ready_to_activate",
}
_FAILED_STATUS = {
    "parse": "parse_failed",
    "repair": "table_repair_failed",
    "canonicalize": "parse_failed",
    "semantic_split": "parse_failed",
    "contextualize": "contextualization_failed",
    "embed": "embedding_failed",
    "index": "embedding_failed",
    "activate": "activation_failed",
}


class StageAlreadyClaimed(RuntimeError):
    """Raised when another worker has committed a claim for the same stage."""


class StageHandlerUnavailable(RuntimeError):
    """Raised instead of recording a stage as complete without real work."""


class StageCheckpointTooLarge(ValueError):
    """Raised before committing a checkpoint that exceeds the durable JSON cap."""


@dataclass(frozen=True)
class IngestionStageContext:
    db: Session
    document: Document
    version: DocumentParseVersion
    stage: str
    input: dict[str, Any]
    attempt: int


StageHandler = Callable[[IngestionStageContext], Any]


class IngestionStageRunner:
    """Run version-scoped ingestion stages with durable, resumable checkpoints."""

    def __init__(
        self,
        db: Session,
        *,
        handlers: Mapping[str, StageHandler] | None = None,
        dispatcher: Any | None = None,
        clock: Callable[[], datetime] | None = None,
        claim_owner: str | None = None,
        claim_ttl_seconds: int = 900,
        max_checkpoint_bytes: int = 256 * 1024,
    ) -> None:
        self.db = db
        self.handlers = dict(handlers or {})
        unknown_handlers = set(self.handlers) - set(INGESTION_STAGES)
        if unknown_handlers:
            names = ", ".join(sorted(unknown_handlers))
            raise ValueError(f"Unknown ingestion stage handlers: {names}.")
        self.dispatcher = dispatcher
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.claim_owner = claim_owner or uuid4().hex
        if claim_ttl_seconds <= 0:
            raise ValueError("claim_ttl_seconds must be positive")
        if max_checkpoint_bytes <= 0:
            raise ValueError("max_checkpoint_bytes must be positive")
        self.claim_ttl_seconds = claim_ttl_seconds
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.versions = ParseVersionService(db)

    def run_until_blocked(
        self,
        document_id: str,
        version_key: str,
        *,
        include_activation: bool = False,
    ) -> DocumentParseVersion:
        stages = INGESTION_STAGES if include_activation else INGESTION_STAGES[:-1]
        version: DocumentParseVersion | None = None
        for stage in stages:
            version = self.run_stage(
                document_id,
                version_key,
                stage,
                enqueue_next=False,
            )
        if version is None:  # pragma: no cover - the stage contract is never empty
            raise RuntimeError("No ingestion stages are configured.")
        return version

    def run_stage(
        self,
        document_id: str,
        version_key: str,
        stage: str,
        *,
        enqueue_next: bool = True,
    ) -> DocumentParseVersion:
        self._validate_stage(stage)
        document, version = self._load_locked(document_id, version_key)
        checkpoint = dict((version.stage_state or {}).get(stage) or {})

        if checkpoint.get("status") == "completed":
            self.db.commit()
            if enqueue_next and not checkpoint.get("next_enqueued", False):
                self._enqueue_next(version, stage)
            return version
        if checkpoint.get("status") == "running":
            if not self._claim_expired(checkpoint):
                self.db.rollback()
                raise StageAlreadyClaimed(
                    f"Ingestion stage {stage!r} is already claimed for "
                    f"document {document_id!r}, version {version_key!r}."
                )

        self._validate_order(version, stage)
        self._prepare_status_for_attempt(version, stage)
        document.status = self._document_running_status(stage)
        self._set_pipeline_progress(document.id, stage, completed=False)
        attempt = int(checkpoint.get("attempts") or 0) + 1
        input_checkpoint = self._stage_input(version, stage)
        running = {
            "status": "running",
            "attempts": attempt,
            "progress": 0,
            "input": input_checkpoint,
            "output": checkpoint.get("output"),
            "error": None,
            "started_at": self._timestamp(),
            "completed_at": None,
            "failed_at": None,
            "next_enqueued": False,
            "claim_owner": self.claim_owner,
            "lease_expires_at": self._timestamp(
                self._now() + timedelta(seconds=self.claim_ttl_seconds)
            ),
        }
        self._set_checkpoint(version, stage, running)
        self.db.commit()

        try:
            document, version = self._load_locked(document_id, version_key)
            handler = self.handlers.get(stage)
            if handler is None:
                raise StageHandlerUnavailable(
                    f"No production handler is configured for ingestion stage {stage!r}."
                )
            output = handler(
                IngestionStageContext(
                    db=self.db,
                    document=document,
                    version=version,
                    stage=stage,
                    input=input_checkpoint,
                    attempt=attempt,
                )
            )
            safe_output = _json_safe(output)
            if stage == "activate":
                self.versions.activate(document, version)
                document.status = "ready"
            else:
                self.versions.transition(version, _COMPLETED_STATUS[stage])
                if _COMPLETED_STATUS[stage] != "ready_to_activate":
                    document.status = _COMPLETED_STATUS[stage]
            self._set_pipeline_progress(document.id, stage, completed=True)
            completed = {
                **running,
                "status": "completed",
                "progress": 100,
                "output": safe_output,
                "completed_at": self._timestamp(),
            }
            self._set_checkpoint(version, stage, completed)
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            self._record_failure(document_id, version_key, stage, running, exc)
            raise

        if enqueue_next:
            self._enqueue_next(version, stage)
        return version

    def _load_locked(
        self, document_id: str, version_key: str
    ) -> tuple[Document, DocumentParseVersion]:
        document = self.db.scalar(
            select(Document).where(Document.id == document_id).with_for_update()
        )
        if document is None:
            raise ValueError(f"Document {document_id!r} does not exist.")
        version = self.db.scalar(
            select(DocumentParseVersion)
            .where(
                DocumentParseVersion.document_id == document_id,
                DocumentParseVersion.version_key == version_key,
            )
            .with_for_update()
        )
        if version is None:
            raise ValueError(
                f"Parse version {version_key!r} does not exist for document {document_id!r}."
            )
        return document, version

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in INGESTION_STAGES:
            raise ValueError(f"Unknown ingestion stage {stage!r}.")

    @staticmethod
    def _validate_order(version: DocumentParseVersion, stage: str) -> None:
        state = version.stage_state or {}
        stage_index = INGESTION_STAGES.index(stage)
        missing = [
            previous
            for previous in INGESTION_STAGES[:stage_index]
            if (state.get(previous) or {}).get("status") != "completed"
        ]
        if missing:
            raise ValueError(
                f"Ingestion stage {stage!r} is out of order; incomplete stages: "
                + ", ".join(missing)
                + "."
            )

    def _prepare_status_for_attempt(
        self, version: DocumentParseVersion, stage: str
    ) -> None:
        expected = _RUNNING_STATUS[stage]
        if version.status in FAILED_STAGE_RETRIES:
            self.versions.retry_failed_stage(version)
            for target in self._status_path(version.status, expected):
                self.versions.transition(version, target)
            return
        if version.status == expected:
            return
        self.versions.transition(version, expected)

    @staticmethod
    def _status_path(source: str, target: str) -> list[str]:
        if source == target:
            return []
        pending: deque[tuple[str, list[str]]] = deque([(source, [])])
        visited = {source}
        while pending:
            current, path = pending.popleft()
            for candidate in ALLOWED_TRANSITIONS.get(current, set()):
                if candidate.endswith("_failed") or candidate == "active":
                    continue
                candidate_path = [*path, candidate]
                if candidate == target:
                    return candidate_path
                if candidate not in visited:
                    visited.add(candidate)
                    pending.append((candidate, candidate_path))
        raise ValueError(
            f"Parse-version status {source!r} cannot resume at {target!r}."
        )

    @staticmethod
    def _stage_input(
        version: DocumentParseVersion, stage: str
    ) -> dict[str, Any]:
        stage_index = INGESTION_STAGES.index(stage)
        previous_output = None
        previous_stage = None
        if stage_index:
            previous_stage = INGESTION_STAGES[stage_index - 1]
            previous_output = (version.stage_state or {}).get(previous_stage, {}).get(
                "output"
            )
        return _json_safe(
            {
                "document_id": version.document_id,
                "version_key": version.version_key,
                "stage": stage,
                "previous_stage": previous_stage,
                "previous_output": previous_output,
            }
        )

    def _record_failure(
        self,
        document_id: str,
        version_key: str,
        stage: str,
        running: dict[str, Any],
        exc: Exception,
    ) -> None:
        document, version = self._load_locked(document_id, version_key)
        failed_status = _FAILED_STATUS[stage]
        if version.status != failed_status:
            self.versions.transition(version, failed_status)
        document.status = failed_status
        self._set_pipeline_failure(document.id, stage, exc)
        failed = {
            **running,
            "status": "failed",
            "progress": 0,
            "error": str(exc),
            "failed_at": self._timestamp(),
        }
        self._set_checkpoint(version, stage, failed)
        self.db.commit()

    def _enqueue_next(self, version: DocumentParseVersion, stage: str) -> None:
        stage_index = INGESTION_STAGES.index(stage)
        if stage_index == len(INGESTION_STAGES) - 1:
            self._mark_enqueued(version, stage)
            return
        if self.dispatcher is None:
            return
        next_stage = INGESTION_STAGES[stage_index + 1]
        self.dispatcher.enqueue_stage(
            version.document_id,
            version.version_key,
            next_stage,
        )
        self._mark_enqueued(version, stage)

    def _mark_enqueued(self, version: DocumentParseVersion, stage: str) -> None:
        _document, locked_version = self._load_locked(
            version.document_id, version.version_key
        )
        checkpoint = dict((locked_version.stage_state or {}).get(stage) or {})
        checkpoint["next_enqueued"] = True
        checkpoint["next_enqueued_at"] = self._timestamp()
        self._set_checkpoint(locked_version, stage, checkpoint)
        self.db.commit()

    def _set_checkpoint(
        self,
        version: DocumentParseVersion, stage: str, checkpoint: dict[str, Any]
    ) -> None:
        state = dict(version.stage_state or {})
        state[stage] = _json_safe(checkpoint)
        encoded = json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > self.max_checkpoint_bytes:
            raise StageCheckpointTooLarge(
                f"Ingestion checkpoint is {len(encoded)} bytes; "
                f"maximum is {self.max_checkpoint_bytes} bytes."
            )
        version.stage_state = state

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _timestamp(self, value: datetime | None = None) -> str:
        value = value or self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _claim_expired(self, checkpoint: Mapping[str, Any]) -> bool:
        raw_expiry = checkpoint.get("lease_expires_at")
        if not isinstance(raw_expiry, str):
            return False
        try:
            expiry = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry.astimezone(timezone.utc) <= self._now()

    def _latest_pipeline_run(self, document_id: str) -> PipelineRun | None:
        return self.db.scalar(
            select(PipelineRun)
            .where(PipelineRun.document_id == document_id)
            .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        )

    def _set_pipeline_progress(
        self, document_id: str, stage: str, *, completed: bool
    ) -> None:
        run = self._latest_pipeline_run(document_id)
        if run is None:
            return
        stage_index = INGESTION_STAGES.index(stage)
        percent = round(
            100 * (stage_index + (1 if completed else 0)) / len(INGESTION_STAGES)
        )
        report = dict(run.provider_report or {})
        report["progress"] = {
            "percent": percent,
            "stage": "completed" if stage == "activate" and completed else stage,
            "message": (
                f"Ingestion stage {stage} completed."
                if completed
                else f"Ingestion stage {stage} started."
            ),
        }
        run.provider_report = report
        run.status = (
            RunStatus.completed.value
            if stage == "activate" and completed
            else RunStatus.running.value
        )

    def _set_pipeline_failure(
        self, document_id: str, stage: str, exc: Exception
    ) -> None:
        run = self._latest_pipeline_run(document_id)
        if run is None:
            return
        report = dict(run.provider_report or {})
        report["error"] = str(exc)
        report["progress"] = {
            "percent": round(
                100 * INGESTION_STAGES.index(stage) / len(INGESTION_STAGES)
            ),
            "stage": f"{stage}_failed",
            "message": str(exc),
        }
        run.provider_report = report
        run.status = RunStatus.failed.value
        run.notes = str(exc)

    @staticmethod
    def _document_running_status(stage: str) -> str:
        if stage == "activate":
            return "indexing"
        return _RUNNING_STATUS[stage]


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Stage checkpoints cannot contain non-finite numbers.")
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, Mapping):
        converted = {str(key): _json_safe(item) for key, item in value.items()}
        json.dumps(converted, allow_nan=False)
        return converted
    if isinstance(value, (list, tuple, set, frozenset)):
        converted = [_json_safe(item) for item in value]
        json.dumps(converted, allow_nan=False)
        return converted
    raise TypeError(f"Stage checkpoint value {type(value).__name__} is not JSON serializable.")
