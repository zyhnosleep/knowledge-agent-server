"""数据库引擎与会话管理。

本模块负责：
1. 创建 SQLAlchemy engine 与 sessionmaker（支持 SQLite / PostgreSQL）。
2. SQLite 专用：启用 WAL 模式和外键约束。
3. 初始化数据库表（create_all + 轻量迁移）。
4. SQLite 的兼容性迁移：为旧表补充缺失列、重建 document_chunks schema、
   回填 parse_version、建立索引与唯一约束。

主要入口：
- get_db(): FastAPI 依赖，提供每请求一个 session。
- init_db(): 应用启动时初始化数据库结构。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from datetime import datetime

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    """SQLite 连接建立时启用 WAL 与外键约束。"""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


settings = get_settings()

# SQLite 需要允许跨线程使用并加长锁等待，PostgreSQL 不需要特殊参数。
connect_args = (
    {"check_same_thread": False, "timeout": 30}
    if settings.database_url.startswith("sqlite")
    else {}
)
engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
if settings.database_url.startswith("sqlite"):
    event.listen(engine, "connect", _enable_sqlite_foreign_keys)
# 每请求会话工厂：关闭 autoflush/autocommit，显式提交。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


# document_chunks 表中必须为 NOT NULL 的列（用于 schema 重建校验）。
_REQUIRED_CHUNK_COLUMNS = {
    "parse_version",
    "chunk_role",
    "block_type",
    "section_path",
    "source_block_ids",
    "source_spans",
    "embedding_text",
    "token_count",
}
# chunk 自引用外键的约束名与 ondelete 行为。
_CHUNK_SELF_FOREIGN_KEYS = {
    "parent_chunk_id": ("fk_document_chunks_parent_chunk_id", "CASCADE"),
    "previous_chunk_id": ("fk_document_chunks_previous_chunk_id", "SET NULL"),
    "next_chunk_id": ("fk_document_chunks_next_chunk_id", "SET NULL"),
}
# 用于检查自引用外键是否有悬空引用（孤儿行）的 SQL。
_CHUNK_SELF_REFERENCE_CHECKS = (
    (
        "parent_chunk_id",
        "SELECT 1 FROM document_chunks AS source "
        "LEFT JOIN document_chunks AS target "
        "ON target.id = source.parent_chunk_id "
        "WHERE source.parent_chunk_id IS NOT NULL AND target.id IS NULL LIMIT 1",
    ),
    (
        "previous_chunk_id",
        "SELECT 1 FROM document_chunks AS source "
        "LEFT JOIN document_chunks AS target "
        "ON target.id = source.previous_chunk_id "
        "WHERE source.previous_chunk_id IS NOT NULL AND target.id IS NULL LIMIT 1",
    ),
    (
        "next_chunk_id",
        "SELECT 1 FROM document_chunks AS source "
        "LEFT JOIN document_chunks AS target "
        "ON target.id = source.next_chunk_id "
        "WHERE source.next_chunk_id IS NOT NULL AND target.id IS NULL LIMIT 1",
    ),
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求创建一个数据库 session，请求结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """初始化数据库：建表，并对 SQLite 执行兼容性迁移。"""
    from app.models import records  # noqa: F401

    if settings.database_url.startswith("sqlite"):
        _configure_sqlite_engine(engine)
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()
    _ensure_sqlite_unique_indexes()


def _configure_sqlite_engine(target_engine: Engine) -> None:
    """确保 SQLite engine 开启 WAL 与外键。"""
    if not event.contains(target_engine, "connect", _enable_sqlite_foreign_keys):
        event.listen(target_engine, "connect", _enable_sqlite_foreign_keys)
    with target_engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()


def _ensure_sqlite_columns() -> None:
    """为 SQLite 旧表补充缺失列（轻量迁移）。"""
    if not settings.database_url.startswith("sqlite"):
        return
    columns = (
        ("conversation_sessions", "document_id", "TEXT"),
        ("conversation_turns", "citations", "TEXT"),
        ("documents", "active_parse_version", "VARCHAR(128)"),
        ("document_chunks", "parse_version", "VARCHAR(128)"),
        ("document_chunks", "parent_chunk_id", "VARCHAR(36)"),
        ("document_chunks", "chunk_role", "VARCHAR(20)"),
        ("document_chunks", "block_type", "VARCHAR(40)"),
        ("document_chunks", "section_path", "JSON"),
        ("document_chunks", "source_block_ids", "JSON"),
        ("document_chunks", "source_spans", "JSON"),
        ("document_chunks", "contextual_prefix", "TEXT"),
        ("document_chunks", "embedding_text", "TEXT"),
        ("document_chunks", "contextualization_model", "VARCHAR(120)"),
        ("document_chunks", "contextualization_version", "VARCHAR(120)"),
        (
            "document_chunks",
            "contextualization_prompt_version",
            "VARCHAR(120)",
        ),
        ("document_chunks", "contextualized_at", "DATETIME"),
        ("document_chunks", "parser_name", "VARCHAR(120)"),
        ("document_chunks", "parser_version", "VARCHAR(120)"),
        ("document_chunks", "splitter_name", "VARCHAR(120)"),
        ("document_chunks", "splitter_version", "VARCHAR(120)"),
        ("document_chunks", "splitting_model", "VARCHAR(120)"),
        ("document_chunks", "semantic_boundary_score", "FLOAT"),
        ("document_chunks", "token_count", "INTEGER"),
        ("document_chunks", "previous_chunk_id", "VARCHAR(36)"),
        ("document_chunks", "next_chunk_id", "VARCHAR(36)"),
    )
    with engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        for table_name, column_name, column_type in columns:
            if table_name not in table_names:
                continue
            try:
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                )
            except OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
    _backfill_sqlite_parse_versions()
    _ensure_sqlite_chunk_schema()
    _ensure_sqlite_indexes()


def _ensure_sqlite_chunk_schema() -> None:
    """校验 document_chunks 的必填列与外键，必要时重建表。"""
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.connect() as connection:
        schema = inspect(connection)
        if "document_chunks" not in schema.get_table_names():
            return
        columns = {column["name"]: column for column in schema.get_columns("document_chunks")}
        foreign_keys = {
            foreign_key["constrained_columns"][0]: foreign_key
            for foreign_key in schema.get_foreign_keys("document_chunks")
            if foreign_key["referred_table"] == "document_chunks"
        }
        _check_sqlite_chunk_self_references(connection)
        _check_sqlite_chunk_foreign_keys(connection)
        nullable_mismatch = any(
            columns[column_name]["nullable"]
            for column_name in _REQUIRED_CHUNK_COLUMNS
        )
        missing_foreign_keys = {
            column_name
            for column_name, (_constraint_name, ondelete) in _CHUNK_SELF_FOREIGN_KEYS.items()
            if column_name not in foreign_keys
            or foreign_keys[column_name].get("options", {}).get("ondelete", "").upper()
            != ondelete
        }
        if not nullable_mismatch and not missing_foreign_keys:
            return
        conflicting_foreign_keys = missing_foreign_keys & set(foreign_keys)
        if conflicting_foreign_keys:
            names = ", ".join(sorted(conflicting_foreign_keys))
            raise RuntimeError(
                f"SQLite document_chunks has incompatible foreign keys: {names}."
            )

    _rebuild_sqlite_chunk_schema(columns, missing_foreign_keys)


def _rebuild_sqlite_chunk_schema(
    columns: dict[str, dict], missing_foreign_keys: set[str]
) -> None:
    """重建 document_chunks 表：把必填列改为 NOT NULL、补上缺失外键。"""
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.commit()
        try:
            with connection.begin():
                context = MigrationContext.configure(connection)
                operations = Operations(context)
                with operations.batch_alter_table(
                    "document_chunks", recreate="always"
                ) as batch_op:
                    for column_name in sorted(_REQUIRED_CHUNK_COLUMNS):
                        if columns[column_name]["nullable"]:
                            batch_op.alter_column(
                                column_name,
                                existing_type=columns[column_name]["type"],
                                nullable=False,
                            )
                    for column_name in sorted(missing_foreign_keys):
                        constraint_name, ondelete = _CHUNK_SELF_FOREIGN_KEYS[column_name]
                        batch_op.create_foreign_key(
                            constraint_name,
                            "document_chunks",
                            [column_name],
                            ["id"],
                            ondelete=ondelete,
                        )
                _check_sqlite_chunk_foreign_keys(connection)
        finally:
            if connection.in_transaction():
                connection.rollback()
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.commit()


def _check_sqlite_chunk_self_references(connection) -> None:
    """检查自引用外键是否有孤儿行。"""
    orphaned_columns = [
        column_name
        for column_name, statement in _CHUNK_SELF_REFERENCE_CHECKS
        if connection.exec_driver_sql(statement).first() is not None
    ]
    if orphaned_columns:
        names = ", ".join(orphaned_columns)
        raise RuntimeError(
            "SQLite document_chunks has foreign-key violations in self-reference "
            f"columns: {names}."
        )


def _check_sqlite_chunk_foreign_keys(connection) -> None:
    """使用 PRAGMA 检查外键完整性。"""
    violations = connection.exec_driver_sql(
        "PRAGMA foreign_key_check(document_chunks)"
    ).all()
    if violations:
        raise RuntimeError("SQLite document_chunks has foreign-key violations.")


def _backfill_sqlite_parse_versions() -> None:
    """把旧 chunk 数据回填为 legacy 解析版本。"""
    if not settings.database_url.startswith("sqlite"):
        return
    empty_json = json.dumps([])
    with engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        if not {"document_chunks", "document_parse_versions"} <= table_names:
            return
        connection.execute(
            text(
                "UPDATE document_chunks SET "
                "parse_version = COALESCE(parse_version, 'legacy'), "
                "chunk_role = COALESCE(chunk_role, 'child'), "
                "block_type = COALESCE(block_type, 'narrative'), "
                "section_path = COALESCE(section_path, :empty_json), "
                "source_block_ids = COALESCE(source_block_ids, :empty_json), "
                "source_spans = COALESCE(source_spans, :empty_json), "
                "embedding_text = COALESCE(embedding_text, text), "
                "token_count = COALESCE(token_count, token_estimate, 0)"
            ),
            {"empty_json": empty_json},
        )
        document_ids = connection.execute(
            text(
                "SELECT DISTINCT document_id FROM document_chunks "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM document_parse_versions "
                "WHERE document_parse_versions.document_id = document_chunks.document_id "
                "AND document_parse_versions.version_key = 'legacy')"
            )
        ).scalars()
        now = datetime.utcnow()
        for document_id in document_ids:
            connection.execute(
                text(
                    "INSERT INTO document_parse_versions "
                    "(id, document_id, version_key, status, artifact_dir, parser_name, "
                    "parser_version, manifest_json, quality_json, stage_state, activated_at, "
                    "created_at, updated_at) VALUES "
                    "(:id, :document_id, 'legacy', 'quarantined', 'legacy', NULL, NULL, "
                    ":empty_object, :empty_object, :stage_state, NULL, :now, :now)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "empty_object": json.dumps({}),
                    "stage_state": json.dumps({"migration": "legacy"}),
                    "now": now,
                },
            )


def _ensure_sqlite_indexes() -> None:
    """为常用查询建立索引。"""
    if not settings.database_url.startswith("sqlite"):
        return
    statements = (
        (
            "conversation_sessions",
            "CREATE INDEX IF NOT EXISTS ix_conversation_sessions_document_id "
            "ON conversation_sessions (document_id)",
        ),
        (
            "document_chunks",
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_document_id "
            "ON document_chunks (document_id)",
        ),
        (
            "document_chunks",
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_document_parse_version_role "
            "ON document_chunks (document_id, parse_version, chunk_role)",
        ),
    )
    with engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        for table_name, statement in statements:
            if table_name not in table_names:
                continue
            connection.execute(text(statement))


def _ensure_sqlite_unique_indexes() -> None:
    """建立唯一索引（entities 的 project+name 唯一）。"""
    if not settings.database_url.startswith("sqlite"):
        return
    statements = (
        (
            "uq_entities_project_name",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_project_name ON entities (project_id, name)",
        ),
    )
    with engine.begin() as connection:
        for index_name, statement in statements:
            try:
                connection.execute(text(statement))
            except IntegrityError as exc:
                raise RuntimeError(f"Cannot create unique index {index_name}; duplicate rows already exist.") from exc
            except OperationalError as exc:
                raise RuntimeError(f"Cannot create unique index {index_name}: {exc}") from exc
