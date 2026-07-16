from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, Table, create_engine, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models import records  # noqa: F401


class MigrationError(RuntimeError):
    pass


def _rows(connection: Connection, table: Table) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(select(table)).mappings()]


def _primary_key(table: Table, row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column.name] for column in table.primary_key.columns)


def _rows_by_primary_key(
    rows: Iterable[dict[str, Any]], table: Table
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {_primary_key(table, row): row for row in rows}


def _insert_rows(
    connection: Connection,
    table: Table,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> None:
    if not rows:
        return
    dialect = connection.dialect.name
    primary_key_columns = [column.name for column in table.primary_key.columns]
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        if dialect == "postgresql":
            statement = postgresql_insert(table).values(batch)
            statement = statement.on_conflict_do_nothing(
                index_elements=primary_key_columns
            )
        elif dialect == "sqlite":
            statement = sqlite_insert(table).values(batch).prefix_with("OR IGNORE")
        else:
            statement = table.insert().values(batch)
        connection.execute(statement)


def _assert_existing_rows_match(
    table: Table,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> None:
    if not target_rows:
        return
    source_by_key = _rows_by_primary_key(source_rows, table)
    target_by_key = _rows_by_primary_key(target_rows, table)
    unexpected = sorted(set(target_by_key) - set(source_by_key), key=str)
    if unexpected:
        raise MigrationError(
            f"Target table {table.name} contains unexpected rows: {unexpected[:5]}"
        )
    mismatched = [
        key
        for key, target_row in target_by_key.items()
        if source_by_key.get(key) != target_row
    ]
    if mismatched:
        raise MigrationError(
            f"Target table {table.name} contains mismatched rows: {mismatched[:5]}"
        )


def _validate_target_schema(target_engine: Engine) -> None:
    existing = set()
    with target_engine.connect() as connection:
        existing = set(
            connection.dialect.get_table_names(connection)
        )
    missing = set(Base.metadata.tables) - existing
    if missing:
        raise MigrationError(
            "Target schema is missing Alembic-managed tables: "
            + ", ".join(sorted(missing))
        )


def _reindex_postgres_vectors(target_engine: Engine, dimensions: int = 4096) -> int:
    if target_engine.dialect.name != "postgresql":
        return 0
    chunks = Base.metadata.tables["document_chunks"]
    indexed = 0
    with target_engine.begin() as connection:
        vector_table_exists = connection.execute(
            text("SELECT to_regclass('document_chunk_pgvector_index')")
        ).scalar_one_or_none()
        if not vector_table_exists:
            raise MigrationError("PostgreSQL pgvector index table is missing")
        rows = connection.execute(
            select(chunks.c.id, chunks.c.document_id, chunks.c.embedding).where(
                chunks.c.embedding.is_not(None)
            )
        ).mappings()
        for row in rows:
            embedding = list(row["embedding"] or [])
            if len(embedding) != dimensions:
                continue
            connection.execute(
                text(
                    "INSERT INTO document_chunk_pgvector_index "
                    "(chunk_id, document_id, embedding) "
                    "VALUES (:chunk_id, :document_id, CAST(:embedding AS vector)) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET "
                    "document_id = EXCLUDED.document_id, embedding = EXCLUDED.embedding"
                ),
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "embedding": json.dumps(embedding, separators=(",", ":")),
                },
            )
            indexed += 1
    return indexed


def migrate_database(
    source_url: str,
    target_url: str,
    *,
    dry_run: bool = False,
    batch_size: int = 250,
    reindex_vectors: bool = True,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source_engine = create_engine(source_url, future=True)
    target_engine = create_engine(target_url, future=True)
    if source_engine.dialect.name != "sqlite":
        raise MigrationError("Source database must be SQLite")
    _validate_target_schema(target_engine)

    counts: dict[str, int] = {}
    with source_engine.connect() as source_connection:
        source_tables = set(inspect(source_connection).get_table_names())
        if dry_run:
            for table in Base.metadata.sorted_tables:
                counts[table.name] = (
                    int(
                        source_connection.execute(
                            select(func.count()).select_from(table)
                        ).scalar_one()
                    )
                    if table.name in source_tables
                    else 0
                )
            return counts

        with target_engine.begin() as target_connection:
            for table in Base.metadata.sorted_tables:
                source_rows = (
                    _rows(source_connection, table)
                    if table.name in source_tables
                    else []
                )
                target_rows = _rows(target_connection, table)
                _assert_existing_rows_match(table, source_rows, target_rows)
                _insert_rows(target_connection, table, source_rows, batch_size)
                migrated_count = int(
                    target_connection.execute(
                        select(func.count()).select_from(table)
                    ).scalar_one()
                )
                if migrated_count != len(source_rows):
                    raise MigrationError(
                        f"Row count mismatch for {table.name}: "
                        f"source={len(source_rows)} target={migrated_count}"
                    )
                counts[table.name] = migrated_count

    if reindex_vectors:
        counts["document_chunk_pgvector_index"] = _reindex_postgres_vectors(
            target_engine
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate an LLM Wiki SQLite database into an Alembic-managed PostgreSQL database."
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--skip-vector-reindex", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    counts = migrate_database(
        args.source_url,
        args.target_url,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        reindex_vectors=not args.skip_vector_reindex,
    )
    payload = {
        "status": "dry-run" if args.dry_run else "completed",
        "tables": counts,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
