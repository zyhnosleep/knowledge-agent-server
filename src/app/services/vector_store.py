from __future__ import annotations

import logging
import math
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    distance: float
    parse_version: str = "legacy"


@dataclass(frozen=True)
class ChunkVector:
    chunk_id: str
    document_id: str
    embedding: list[float]
    parse_version: str = "legacy"


class SQLiteVecStore:
    """Optional sqlite-vec backed index for document chunk embeddings."""

    _META_TABLE = "document_chunk_vector_index"
    _TABLE_PREFIX = "document_chunk_vec_"

    def __init__(self, db: Session) -> None:
        self.db = db
        self._available: bool | None = None
        self._sqlite_vec: Any | None = None

    def available(self) -> bool:
        if not settings.vector_store_enabled:
            return False
        if settings.vector_store_backend != "sqlite-vec":
            return False
        if self.db.get_bind().dialect.name != "sqlite":
            return False
        if self._available is not None:
            return self._available
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except Exception:
            self._available = False
            return False
        try:
            raw_connection = self._raw_connection()
            self._set_extension_loading(raw_connection, True)
            try:
                sqlite_vec.load(raw_connection)
            finally:
                self._set_extension_loading(raw_connection, False)
            raw_connection.execute("select vec_version()").fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.info("sqlite-vec is not available on this connection: %s", exc)
            self._sqlite_vec = None
            self._available = False
            return False
        self._sqlite_vec = sqlite_vec
        self._available = True
        return True

    def replace_document_chunks(self, document_id: str, vectors: Iterable[ChunkVector]) -> None:
        vectors = [
            ChunkVector(
                chunk_id=vector.chunk_id,
                document_id=vector.document_id,
                embedding=normalized_embedding,
                parse_version=vector.parse_version,
            )
            for vector in vectors
            if self._valid_embedding(vector.embedding)
            for normalized_embedding in [self._normalize_embedding(vector.embedding)]
            if normalized_embedding
        ]
        if not self.available():
            return
        try:
            with self.db.begin_nested():
                self._ensure_meta_table()
                parse_versions = {vector.parse_version for vector in vectors}
                for parse_version in parse_versions:
                    self._delete_document_rows(document_id, parse_version=parse_version)
                for vector in vectors:
                    dimensions = len(vector.embedding)
                    self._ensure_vector_table(dimensions)
                    self._delete_chunk_row(vector.chunk_id)
                    row_id = self._insert_mapping(
                        vector.chunk_id,
                        vector.document_id,
                        dimensions,
                        parse_version=vector.parse_version,
                    )
                    self.db.execute(
                        text(
                            f"INSERT INTO {self._vector_table_name(dimensions)}(rowid, embedding) "
                            "VALUES (:rowid, :embedding)"
                        ),
                        {
                            "rowid": row_id,
                            "embedding": self._serialize(vector.embedding),
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite-vec indexing failed for document %s; JSON embeddings remain available: %s", document_id, exc)

    def delete_document(self, document_id: str) -> None:
        if not self.available():
            return
        try:
            with self.db.begin_nested():
                self._ensure_meta_table()
                self._delete_document_rows(document_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite-vec cleanup failed for document %s; JSON embeddings remain available: %s", document_id, exc)

    def _delete_document_rows(
        self, document_id: str, *, parse_version: str | None = None
    ) -> None:
        version_filter = ""
        parameters = {"document_id": document_id}
        if parse_version is not None:
            version_filter = " AND parse_version = :parse_version"
            parameters["parse_version"] = parse_version
        rows = self.db.execute(
            text(
                f"SELECT id, dimensions FROM {self._META_TABLE} "
                f"WHERE document_id = :document_id{version_filter}"
            ),
            parameters,
        ).all()
        for row_id, dimensions in rows:
            self._delete_vector_row(int(row_id), int(dimensions))
        self.db.execute(
            text(
                f"DELETE FROM {self._META_TABLE} "
                f"WHERE document_id = :document_id{version_filter}"
            ),
            parameters,
        )

    def _delete_chunk_row(self, chunk_id: str) -> None:
        row = self.db.execute(
            text(f"SELECT id, dimensions FROM {self._META_TABLE} WHERE chunk_id = :chunk_id"),
            {"chunk_id": chunk_id},
        ).first()
        if row is None:
            return
        self._delete_vector_row(int(row.id), int(row.dimensions))
        self.db.execute(text(f"DELETE FROM {self._META_TABLE} WHERE id = :rowid"), {"rowid": int(row.id)})

    def _delete_vector_row(self, row_id: int, dimensions: int) -> None:
        table_name = self._vector_table_name(dimensions)
        self._ensure_vector_table(dimensions)
        self.db.execute(text(f"DELETE FROM {table_name} WHERE rowid = :rowid"), {"rowid": row_id})

    def search(self, embedding: list[float], *, limit: int, document_ids: list[str] | None = None) -> list[VectorHit]:
        if not self._valid_embedding(embedding) or limit <= 0 or not self.available():
            return []
        normalized_embedding = self._normalize_embedding(embedding)
        if not normalized_embedding:
            return []
        scoped_document_ids = [str(document_id) for document_id in (document_ids or []) if str(document_id or "").strip()]
        dimensions = len(normalized_embedding)
        try:
            self._ensure_meta_table()
            self._ensure_vector_table(dimensions)
            total_rows = self._indexed_row_count(dimensions)
            if total_rows <= 0:
                return []
            vector_limit = min(total_rows, max(limit, 50))
            while True:
                rows = self._search_vector_rows(dimensions, normalized_embedding, vector_limit)
                if not rows:
                    return []
                hits = self._hits_from_vector_rows(rows, scoped_document_ids)
                if len(hits) >= limit or vector_limit >= total_rows:
                    return hits[:limit]
                next_limit = min(total_rows, max(vector_limit + 1, vector_limit * 2))
                if next_limit == vector_limit:
                    return hits[:limit]
                vector_limit = next_limit
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite-vec search failed; falling back to JSON embeddings: %s", exc)
            return []

    def _search_vector_rows(self, dimensions: int, embedding: list[float], limit: int) -> list[Any]:
        return self.db.execute(
            text(
                f"SELECT rowid, distance FROM {self._vector_table_name(dimensions)} "
                "WHERE embedding MATCH :embedding AND k = :limit "
                "ORDER BY distance"
            ),
            {"embedding": self._serialize(embedding), "limit": limit},
        ).all()

    def _hits_from_vector_rows(self, rows: list[Any], scoped_document_ids: list[str]) -> list[VectorHit]:
        if not rows:
            return []
        row_ids = [int(row.rowid) for row in rows]
        parameters: dict[str, Any] = {f"row_id_{index}": row_id for index, row_id in enumerate(row_ids)}
        row_id_placeholders = ",".join(f":row_id_{index}" for index in range(len(row_ids)))
        document_filter = ""
        if scoped_document_ids:
            parameters.update({f"document_id_{index}": document_id for index, document_id in enumerate(scoped_document_ids)})
            document_placeholders = ",".join(f":document_id_{index}" for index in range(len(scoped_document_ids)))
            document_filter = f"AND idx.document_id IN ({document_placeholders}) "
        mapping_rows = self.db.execute(
            text(
                f"SELECT idx.id, idx.chunk_id, idx.parse_version FROM {self._META_TABLE} AS idx "
                "JOIN document_chunks AS chunk ON chunk.id = idx.chunk_id "
                "JOIN documents AS document ON document.id = idx.document_id "
                f"WHERE idx.id IN ({row_id_placeholders}) "
                "AND (idx.parse_version = document.active_parse_version "
                "OR (document.active_parse_version IS NULL AND idx.parse_version = 'legacy')) "
                "AND chunk.parse_version = idx.parse_version "
                f"{document_filter}"
            ),
            parameters,
        ).all()
        chunks_by_row_id = {
            int(row.id): (str(row.chunk_id), str(row.parse_version))
            for row in mapping_rows
        }
        return [
            VectorHit(
                chunk_id=chunks_by_row_id[int(row.rowid)][0],
                distance=float(row.distance),
                parse_version=chunks_by_row_id[int(row.rowid)][1],
            )
            for row in rows
            if int(row.rowid) in chunks_by_row_id
        ]

    def _indexed_row_count(self, dimensions: int) -> int:
        return int(
            self.db.execute(
                text(f"SELECT COUNT(*) FROM {self._META_TABLE} WHERE dimensions = :dimensions"),
                {"dimensions": dimensions},
            ).scalar_one()
        )

    def count_document_chunks(self, document_id: str, parse_version: str) -> int:
        if not self.available():
            return 0
        self._ensure_meta_table()
        return int(
            self.db.execute(
                text(
                    f"SELECT COUNT(*) FROM {self._META_TABLE} "
                    "WHERE document_id = :document_id AND parse_version = :parse_version"
                ),
                {"document_id": document_id, "parse_version": parse_version},
            ).scalar_one()
        )

    def _ensure_meta_table(self) -> None:
        self.db.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {self._META_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL UNIQUE,
                    document_id TEXT NOT NULL,
                    parse_version TEXT NOT NULL DEFAULT 'legacy',
                    dimensions INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        )
        columns = {
            str(row.name)
            for row in self.db.execute(
                text(f"PRAGMA table_info({self._META_TABLE})")
            ).all()
        }
        if "parse_version" not in columns:
            self.db.execute(
                text(
                    f"ALTER TABLE {self._META_TABLE} ADD COLUMN "
                    "parse_version TEXT NOT NULL DEFAULT 'legacy'"
                )
            )
        self.db.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS ix_{self._META_TABLE}_document_id "
                f"ON {self._META_TABLE} (document_id)"
            )
        )

    def _ensure_vector_table(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("Vector dimensions must be positive")
        self.db.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vector_table_name(dimensions)} "
                f"USING vec0(embedding float[{dimensions}])"
            )
        )

    def _insert_mapping(
        self,
        chunk_id: str,
        document_id: str,
        dimensions: int,
        parse_version: str = "legacy",
    ) -> int:
        now = datetime.utcnow().isoformat(timespec="seconds")
        self.db.execute(
            text(
                f"""
                INSERT INTO {self._META_TABLE}
                    (chunk_id, document_id, parse_version, dimensions, created_at, updated_at)
                VALUES
                    (:chunk_id, :document_id, :parse_version, :dimensions, :created_at, :updated_at)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    parse_version = excluded.parse_version,
                    dimensions = excluded.dimensions,
                    updated_at = excluded.updated_at
                """
            ),
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "parse_version": parse_version,
                "dimensions": dimensions,
                "created_at": now,
                "updated_at": now,
            },
        )
        row_id = self.db.execute(
            text(f"SELECT id FROM {self._META_TABLE} WHERE chunk_id = :chunk_id"),
            {"chunk_id": chunk_id},
        ).scalar_one()
        return int(row_id)

    def _serialize(self, embedding: list[float]) -> bytes:
        if self._sqlite_vec is not None:
            return self._sqlite_vec.serialize_float32(embedding)
        raise RuntimeError("sqlite-vec serializer is unavailable")

    def _raw_connection(self) -> Any:
        proxied = self.db.connection().connection
        return getattr(proxied, "driver_connection", proxied)

    @staticmethod
    def _set_extension_loading(raw_connection: Any, enabled: bool) -> None:
        enable_load_extension = getattr(raw_connection, "enable_load_extension", None)
        if callable(enable_load_extension):
            enable_load_extension(enabled)

    @classmethod
    def _vector_table_name(cls, dimensions: int) -> str:
        if dimensions <= 0:
            raise ValueError("Vector dimensions must be positive")
        return f"{cls._TABLE_PREFIX}{dimensions}"

    @staticmethod
    def _valid_embedding(embedding: list[float] | None) -> bool:
        return bool(embedding) and all(isinstance(value, int | float) and math.isfinite(float(value)) for value in embedding)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        norm = math.sqrt(sum(float(value) * float(value) for value in embedding))
        if not math.isfinite(norm) or norm <= 0:
            return []
        return [float(value) / norm for value in embedding]


class PGVectorStore:
    """PostgreSQL pgvector index for document chunk embeddings."""

    _TABLE_NAME = "document_chunk_pgvector_index"

    def __init__(self, db: Session) -> None:
        self.db = db
        self._available: bool | None = None

    def available(self) -> bool:
        if not settings.vector_store_enabled:
            return False
        if settings.vector_store_backend != "pgvector":
            return False
        if self.db.get_bind().dialect.name != "postgresql":
            return False
        if self._available is not None:
            return self._available
        try:
            installed = self.db.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
            self._available = bool(installed)
        except Exception as exc:  # noqa: BLE001
            logger.info("pgvector is not available on this connection: %s", exc)
            self._available = False
        return self._available

    def replace_document_chunks(
        self, document_id: str, vectors: Iterable[ChunkVector]
    ) -> None:
        normalized = [
            ChunkVector(
                chunk_id=vector.chunk_id,
                document_id=vector.document_id,
                embedding=embedding,
                parse_version=vector.parse_version,
            )
            for vector in vectors
            if self._valid_embedding(vector.embedding)
            for embedding in [self._normalize_embedding(vector.embedding)]
            if len(embedding) == settings.ollama_embedding_dimensions
        ]
        if not self.available():
            return
        try:
            with self.db.begin_nested():
                parse_versions = {vector.parse_version for vector in normalized}
                for parse_version in parse_versions:
                    self.delete_document(document_id, parse_version=parse_version)
                if normalized:
                    self.db.execute(
                        text(
                            f"INSERT INTO {self._TABLE_NAME} "
                            "(chunk_id, document_id, parse_version, embedding) "
                            "VALUES (:chunk_id, :document_id, :parse_version, CAST(:embedding AS vector)) "
                            "ON CONFLICT (chunk_id) DO UPDATE SET "
                            "document_id = EXCLUDED.document_id, "
                            "parse_version = EXCLUDED.parse_version, "
                            "embedding = EXCLUDED.embedding"
                        ),
                        [
                            {
                                "chunk_id": vector.chunk_id,
                                "document_id": vector.document_id,
                                "parse_version": vector.parse_version,
                                "embedding": self._serialize(vector.embedding),
                            }
                            for vector in normalized
                        ],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pgvector indexing failed for document %s; JSON embeddings remain available: %s",
                document_id,
                exc,
            )

    def delete_document(
        self, document_id: str, *, parse_version: str | None = None
    ) -> None:
        if not self.available():
            return
        sql = f"DELETE FROM {self._TABLE_NAME} WHERE document_id = :document_id"
        parameters = {"document_id": document_id}
        if parse_version is not None:
            sql += " AND parse_version = :parse_version"
            parameters["parse_version"] = parse_version
        self.db.execute(text(sql), parameters)

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        if not self._valid_embedding(embedding) or limit <= 0 or not self.available():
            return []
        normalized = self._normalize_embedding(embedding)
        if len(normalized) != settings.ollama_embedding_dimensions:
            return []
        scoped_document_ids = [
            str(document_id)
            for document_id in (document_ids or [])
            if str(document_id or "").strip()
        ]
        try:
            rows = self._search_rows(normalized, limit, scoped_document_ids)
            return [
                VectorHit(
                    chunk_id=str(row.chunk_id),
                    distance=float(row.distance),
                    parse_version=str(getattr(row, "parse_version", "legacy")),
                )
                for row in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector search failed; falling back to JSON embeddings: %s", exc)
            return []

    def count_document_chunks(self, document_id: str, parse_version: str) -> int:
        if not self.available():
            return 0
        return int(
            self.db.execute(
                text(
                    f"SELECT COUNT(*) FROM {self._TABLE_NAME} "
                    "WHERE document_id = :document_id AND parse_version = :parse_version"
                ),
                {"document_id": document_id, "parse_version": parse_version},
            ).scalar_one()
        )

    def _search_rows(
        self, embedding: list[float], limit: int, document_ids: list[str]
    ) -> list[Any]:
        parameters: dict[str, Any] = {
            "embedding": self._serialize(embedding),
            "limit": limit,
        }
        scope_sql = ""
        if document_ids:
            placeholders = []
            for index, document_id in enumerate(document_ids):
                key = f"document_id_{index}"
                parameters[key] = document_id
                placeholders.append(f":{key}")
            scope_sql = f"AND idx.document_id IN ({','.join(placeholders)}) "
        return self.db.execute(
            text(
                f"SELECT idx.chunk_id, idx.parse_version, "
                "idx.embedding <=> CAST(:embedding AS vector) AS distance "
                f"FROM {self._TABLE_NAME} AS idx "
                "JOIN documents AS document ON document.id = idx.document_id "
                "WHERE (idx.parse_version = document.active_parse_version "
                "OR (document.active_parse_version IS NULL AND idx.parse_version = 'legacy')) "
                f"{scope_sql}"
                "ORDER BY distance LIMIT :limit"
            ),
            parameters,
        ).all()

    @staticmethod
    def _serialize(embedding: list[float]) -> str:
        return json.dumps(embedding, separators=(",", ":"))

    @staticmethod
    def _valid_embedding(embedding: list[float] | None) -> bool:
        return SQLiteVecStore._valid_embedding(embedding)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        return SQLiteVecStore._normalize_embedding(embedding)


def get_vector_store(db: Session) -> SQLiteVecStore | PGVectorStore:
    if (
        settings.vector_store_enabled
        and settings.vector_store_backend == "pgvector"
        and db.get_bind().dialect.name == "postgresql"
    ):
        return PGVectorStore(db)
    return SQLiteVecStore(db)
