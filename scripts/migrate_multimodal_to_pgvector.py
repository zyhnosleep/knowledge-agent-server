"""Copy a frozen pilot SQLite snapshot into an EMPTY, migrated PostgreSQL DB.

Run with DATABASE_URL pointing to the target and VECTOR_STORE_BACKEND=pgvector.
Stop the pilot API first. This does not change .env or delete the source DB.
"""
import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import Session

from app.db.session import Base, engine
from app.models.records import Document, DocumentChunk
from app.services.vector_store import ChunkVector, PGVectorStore, get_vector_store


def digest(rows):
    def normalize(value):
        if isinstance(value, datetime):
            if value.tzinfo:
                value = value.astimezone(timezone.utc).replace(tzinfo=None)
            return value.isoformat()
        raise TypeError(type(value).__name__)
    encoded = [json.dumps(dict(row), sort_keys=True, ensure_ascii=False,
                          default=normalize, separators=(",", ":")) for row in rows]
    return hashlib.sha256("\n".join(sorted(encoded)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Target must be PostgreSQL")
    if not args.source.is_file():
        raise RuntimeError("Source SQLite database missing")
    args.backup_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.backup_dir / "multimodal.sqlite"
    with sqlite3.connect(args.source.resolve().as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(snapshot) as backup:
            source.backup(backup)
            if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite snapshot failed integrity check")
    source_engine = create_engine("sqlite:///" + str(snapshot.resolve()))
    report = {"snapshot": str(snapshot.resolve()), "tables": {},
              "backend": "pgvector", "distance": "cosine (<=>)",
              "reembedded": False}
    source_tables = set(inspect(source_engine).get_table_names())
    tables = [table for table in Base.metadata.sorted_tables if table.name in source_tables]
    with source_engine.connect() as src, engine.begin() as dst:
        for table in tables:
            if dst.scalar(select(func.count()).select_from(table)):
                raise RuntimeError(f"Refusing to overwrite nonempty target table: {table.name}")
        pending = []
        for table in tables:
            rows = [dict(row) for row in src.execute(select(table)).mappings()]
            report["tables"][table.name] = {"count": len(rows), "source_sha256": digest(rows)}
            self_columns = {fk.parent.name for fk in table.foreign_keys
                            if fk.column.table.name == table.name}
            inserts = []
            for row in rows:
                item = dict(row)
                links = {key: item[key] for key in self_columns if item[key] is not None}
                if links:
                    if any(not table.c[key].nullable for key in links):
                        raise RuntimeError("Cannot stage non-nullable self references")
                    restored = dict(links)
                    # SQLAlchemy's onupdate would otherwise replace source timestamps.
                    if "updated_at" in table.c:
                        restored["updated_at"] = row["updated_at"]
                    pending.append((table, {c.name: row[c.name] for c in table.primary_key}, restored))
                    item.update({key: None for key in links})
                inserts.append(item)
            if inserts:
                dst.execute(table.insert(), inserts)
        for table, keys, links in pending:
            update = table.update()
            for key, value in keys.items():
                update = update.where(table.c[key] == value)
            dst.execute(update.values(**links))
        for table in tables:
            rows = list(dst.execute(select(table)).mappings())
            actual = digest(rows)
            expected = report["tables"][table.name]
            if len(rows) != expected["count"] or actual != expected["source_sha256"]:
                key_names = [column.name for column in table.primary_key]
                source_by_key = {tuple(row[key] for key in key_names): dict(row)
                                 for row in src.execute(select(table)).mappings()}
                differences = []
                for row in rows:
                    wanted = source_by_key[tuple(row[key] for key in key_names)]
                    if digest([row]) != digest([wanted]):
                        differences = [key for key in wanted
                                       if digest([{key: wanted[key]}]) != digest([{key: row[key]}])]
                        break
                raise RuntimeError(f"Migration verification failed: {table.name}; fields={differences}")
            expected["target_sha256"] = actual
        # Join the copy transaction: index failure rolls back the data copy too.
        with Session(bind=dst) as db:
            store = get_vector_store(db)
            if not isinstance(store, PGVectorStore) or not store.available():
                raise RuntimeError("Actual PGVectorStore is unavailable")
            children = db.scalars(select(DocumentChunk).where(DocumentChunk.chunk_role == "child")).all()
            for document in db.scalars(select(Document)).all():
                selected = [c for c in children if c.document_id == document.id]
                if any(len(c.embedding or []) != 2560 for c in selected):
                    raise RuntimeError("Invalid existing embedding dimensions")
                store.replace_document_chunks(document.id, [ChunkVector(
                    chunk_id=c.id, document_id=c.document_id, embedding=c.embedding,
                    parse_version=c.parse_version) for c in selected])
            count = db.scalar(text("SELECT count(*) FROM document_chunk_pgvector_index"))
            if count != len(children):
                raise RuntimeError(f"Index incomplete: {count}/{len(children)}")
            report["indexed_children"] = count
            report["pgvector_version"] = db.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
            db.commit()
    (args.backup_dir / "migration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
