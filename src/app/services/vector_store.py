"""向量存储模块(sqlite-vec / pgvector 两种后端)。

封装两类可选的向量库后端,为文档 chunk 的 embedding 提供近似最近邻
(KNN)检索,服务于 RAG 检索层:

- :class:`SQLiteVecStore`:基于 SQLite 扩展 sqlite-vec 的本地索引。
  它按 embedding 维度创建不同的 vec0 虚拟表(document_chunk_vec_<N>),
  并用一张元数据映射表(document_chunk_vector_index)把 chunk_id、
  document_id、parse_version、dimensions 与向量行关联起来;
- :class:`PGVectorStore`:基于 PostgreSQL pgvector 扩展的索引,单张表
  存储,利用 vector 类型与余弦距离(<=>)排序。

两个后端共同的约定:
- 写入前对 embedding 做 L2 归一化(用归一化后的向量做余弦相似度检索);
- 用 ``available()`` 惰性探测后端是否可用;不可用时调用方应回退到
  JSON embedding 的普通(非向量)检索路径;
- 支持 parse_version(解析版本)隔离:检索时会根据文档的 active 版本
  或调用方传入的 parse_version_map 决定命中哪个版本的 chunk;
- 支持按 document_ids 范围过滤检索。

:func:`get_vector_store` 根据配置项与当前数据库方言,选择合适的后端
实例返回。
"""

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
    """一次向量检索命中的结果。

    - ``chunk_id``:命中的 chunk 标识;
    - ``distance``:查询向量与命中向量的距离(越小越相似);
    - ``parse_version``:该 chunk 所属的解析版本,默认 "legacy"。
    """

    chunk_id: str
    distance: float
    parse_version: str = "legacy"


@dataclass(frozen=True)
class ChunkVector:
    """一个待写入向量索引的 chunk 向量。

    - ``chunk_id`` / ``document_id``:chunk 与其所属文档;
    - ``embedding``:归一化后的向量(调用方负责归一化);
    - ``parse_version``:该 chunk 的解析版本,默认 "legacy"。
    """

    chunk_id: str
    document_id: str
    embedding: list[float]
    parse_version: str = "legacy"


class SQLiteVecStore:
    """基于 sqlite-vec 扩展的可选 embedding 索引(sqlite 后端)。

    通过一张元数据表 + 若干按维度划分的 vec0 虚拟表实现:
    - 元数据表 ``_META_TABLE`` 记录 chunk_id、document_id、parse_version、
      dimensions 与行 id 的映射;
    - 每个维度一个虚拟表 ``_TABLE_PREFIX + dimensions``,实际存放向量。
    这种设计让同一索引同时容纳不同维度的 embedding。
    """

    _META_TABLE = "document_chunk_vector_index"
    _TABLE_PREFIX = "document_chunk_vec_"

    def __init__(self, db: Session) -> None:
        """保存 SQLAlchemy Session,并初始化惰性探测用的缓存字段。

        - ``_available``:None 表示尚未探测,True/False 为探测结果;
        - ``_sqlite_vec``:缓存的 sqlite_vec 模块对象(探测成功后才有)。
        """
        self.db = db
        self._available: bool | None = None
        self._sqlite_vec: Any | None = None

    def available(self) -> bool:
        """惰性探测 sqlite-vec 扩展在当前连接上是否可用。

        依次检查:功能开关(settings.vector_store_enabled)、后端类型、
        数据库方言是否 sqlite;若之前已探测过则直接返回缓存结果。然后
        尝试导入 sqlite_vec 模块并在底层连接上加载扩展,并执行一次
        ``select vec_version()`` 验证。
        """
        # 前置开关:未启用向量存储、后端不是 sqlite-vec、或方言不是
        # sqlite 时都视为不可用。
        if not settings.vector_store_enabled:
            return False
        if settings.vector_store_backend != "sqlite-vec":
            return False
        if self.db.get_bind().dialect.name != "sqlite":
            return False
        # 已探测过则直接返回缓存结果。
        if self._available is not None:
            return self._available
        # 尝试导入扩展模块;导入失败即不可用。
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except Exception:
            self._available = False
            return False
        try:
            # 在底层原始连接上临时开启扩展加载,加载 sqlite_vec 后关闭。
            raw_connection = self._raw_connection()
            self._set_extension_loading(raw_connection, True)
            try:
                sqlite_vec.load(raw_connection)
            finally:
                self._set_extension_loading(raw_connection, False)
            # 执行版本查询,验证扩展确实可用。
            raw_connection.execute("select vec_version()").fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.info("sqlite-vec is not available on this connection: %s", exc)
            self._sqlite_vec = None
            self._available = False
            return False
        # 探测成功,缓存模块对象与结果。
        self._sqlite_vec = sqlite_vec
        self._available = True
        return True

    def replace_document_chunks(self, document_id: str, vectors: Iterable[ChunkVector]) -> None:
        """以"整体替换"的方式写入某文档的 chunk 向量。

        先过滤掉无效 embedding 并对有效向量做 L2 归一化;然后在单个嵌套
        事务中:确保元数据表存在、删除该文档在各 parse_version 下的旧行、
        再逐 chunk 插入(先写元数据映射,再写向量虚拟表)。任一环节失败
        都会回滚,并留下"JSON embedding 仍然可用"的告警日志。
        """
        # 过滤非法 embedding,并替换为归一化后的向量(规范化失败则丢弃)。
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
        # 后端不可用时直接跳过(调用方应使用 JSON embedding 兜底)。
        if not self.available():
            return
        try:
            # 嵌套事务:整体替换具有原子性,失败可整体回滚。
            with self.db.begin_nested():
                self._ensure_meta_table()
                # 按 parse_version 分组清理旧行,避免版本间互相覆盖。
                parse_versions = {vector.parse_version for vector in vectors}
                for parse_version in parse_versions:
                    self._delete_document_rows(document_id, parse_version=parse_version)
                for vector in vectors:
                    dimensions = len(vector.embedding)
                    # 确保对应维度的虚拟表存在(首个该维度向量时创建)。
                    self._ensure_vector_table(dimensions)
                    # 若该 chunk 已存在,先删除旧行再做插入(幂等替换)。
                    self._delete_chunk_row(vector.chunk_id)
                    # 1) 写入元数据映射表,拿到自增行 id。
                    row_id = self._insert_mapping(
                        vector.chunk_id,
                        vector.document_id,
                        dimensions,
                        parse_version=vector.parse_version,
                    )
                    # 2) 用该行 id 作为虚拟表的 rowid,写入序列化向量。
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
        """删除某文档全部(所有 parse_version)的向量索引条目。"""
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
        """删除某文档(可按 parse_version 过滤)在索引中的全部行。

        先从元数据表查出涉及的行 id 与维度,删除对应虚拟表中的向量,
        最后删除元数据行。
        """
        version_filter = ""
        parameters = {"document_id": document_id}
        # 若指定了 parse_version,则只删除该版本的行。
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
        # 先清理每个条目对应的向量虚拟表行。
        for row_id, dimensions in rows:
            self._delete_vector_row(int(row_id), int(dimensions))
        # 再清理元数据行。
        self.db.execute(
            text(
                f"DELETE FROM {self._META_TABLE} "
                f"WHERE document_id = :document_id{version_filter}"
            ),
            parameters,
        )

    def _delete_chunk_row(self, chunk_id: str) -> None:
        """删除指定 chunk 的向量条目(元数据 + 向量虚拟表行)。

        仅当该 chunk 在元数据表中存在时才执行;用于写入前保证幂等。
        """
        row = self.db.execute(
            text(f"SELECT id, dimensions FROM {self._META_TABLE} WHERE chunk_id = :chunk_id"),
            {"chunk_id": chunk_id},
        ).first()
        if row is None:
            return
        self._delete_vector_row(int(row.id), int(row.dimensions))
        self.db.execute(text(f"DELETE FROM {self._META_TABLE} WHERE id = :rowid"), {"rowid": int(row.id)})

    def _delete_vector_row(self, row_id: int, dimensions: int) -> None:
        """按行 id 与维度从对应的 vec0 虚拟表中删除向量行。"""
        table_name = self._vector_table_name(dimensions)
        self._ensure_vector_table(dimensions)
        self.db.execute(text(f"DELETE FROM {table_name} WHERE rowid = :rowid"), {"rowid": row_id})

    def search(
        self,
        embedding: list[float],
        *,
        limit: int,
        document_ids: list[str] | None = None,
        parse_version_map: dict[str, str] | None = None,
    ) -> list[VectorHit]:
        """执行 KNN 向量检索,返回按距离升序的命中列表。

        过程:
        1. 校验 embedding、limit 与后端可用性;归一化查询向量;
        2. 在对应维度虚拟表中做近似检索(先取较大的初始候选数);
        3. 把命中行映射回 chunk(join 文档与 chunk 表,并做 parse_version
           过滤/文档范围过滤);
        4. 若有效命中数不足 limit 且候选数未达上限,则按指数增长候选数
           重试;最终返回至多 limit 条命中。
        任一步失败则告警并返回空列表(调用方回退到 JSON embedding)。
        """
        # 前置校验:查询向量无效、limit 非正、后端不可用则直接返回空。
        if not self._valid_embedding(embedding) or limit <= 0 or not self.available():
            return []
        normalized_embedding = self._normalize_embedding(embedding)
        if not normalized_embedding:
            return []
        # 归一化文档范围过滤条件,丢弃空白 id。
        scoped_document_ids = [str(document_id) for document_id in (document_ids or []) if str(document_id or "").strip()]
        dimensions = len(normalized_embedding)
        try:
            self._ensure_meta_table()
            self._ensure_vector_table(dimensions)
            # 该维度下已索引的总行数。
            total_rows = self._indexed_row_count(dimensions)
            if total_rows <= 0:
                return []
            # 初始候选数:至少取 max(limit, 50),但不超过总行数。
            vector_limit = min(total_rows, max(limit, 50))
            while True:
                # 1) 向量检索,取前 vector_limit 近邻。
                rows = self._search_vector_rows(dimensions, normalized_embedding, vector_limit)
                if not rows:
                    return []
                # 2) 把命中行映射回 chunk,并应用文档/版本过滤。
                hits = self._hits_from_vector_rows(
                    rows,
                    scoped_document_ids,
                    parse_version_map=parse_version_map,
                )
                # 3) 有效命中足够,或候选已到上限,则返回。
                if len(hits) >= limit or vector_limit >= total_rows:
                    return hits[:limit]
                # 4) 否则扩大候选数(至少 +1,成倍增长)再试。
                next_limit = min(total_rows, max(vector_limit + 1, vector_limit * 2))
                if next_limit == vector_limit:
                    return hits[:limit]
                vector_limit = next_limit
        except Exception as exc:  # noqa: BLE001
            logger.warning("sqlite-vec search failed; falling back to JSON embeddings: %s", exc)
            return []

    def _search_vector_rows(self, dimensions: int, embedding: list[float], limit: int) -> list[Any]:
        """在指定维度虚拟表中执行 sqlite-vec KNN 检索。

        返回按距离升序的 (rowid, distance) 行;``k = :limit`` 控制近邻数。
        """
        return self.db.execute(
            text(
                f"SELECT rowid, distance FROM {self._vector_table_name(dimensions)} "
                "WHERE embedding MATCH :embedding AND k = :limit "
                "ORDER BY distance"
            ),
            {"embedding": self._serialize(embedding), "limit": limit},
        ).all()

    def _hits_from_vector_rows(
        self,
        rows: list[Any],
        scoped_document_ids: list[str],
        *,
        parse_version_map: dict[str, str] | None = None,
    ) -> list[VectorHit]:
        """把向量检索得到的 (rowid, distance) 行映射回 chunk 命中。

        通过一次 join 查询同时拿到 chunk_id、文档的 active_parse_version
        等信息,并按版本规则过滤:
        - 若 parse_version_map 提供了某文档的版本覆盖(影子版本),用该版本;
        - 否则使用文档当前 active 版本(缺省为 "legacy")。
        只在版本匹配的行上构造 VectorHit。
        """
        if not rows:
            return []
        row_ids = [int(row.rowid) for row in rows]
        # 为行 id 构造 :row_id_N 具名参数与占位符列表。
        parameters: dict[str, Any] = {f"row_id_{index}": row_id for index, row_id in enumerate(row_ids)}
        row_id_placeholders = ",".join(f":row_id_{index}" for index in range(len(row_ids)))
        # 若指定了文档范围,附加 IN 过滤条件。
        document_filter = ""
        if scoped_document_ids:
            parameters.update({f"document_id_{index}": document_id for index, document_id in enumerate(scoped_document_ids)})
            document_placeholders = ",".join(f":document_id_{index}" for index in range(len(scoped_document_ids)))
            document_filter = f"AND idx.document_id IN ({document_placeholders}) "
        # 联表查询:元数据表 + chunk 表 + documents 表,拿到 chunk_id、
        # 文档 active 版本等信息;同时保证 chunk 行版本与索引版本一致。
        mapping_rows = self.db.execute(
            text(
                f"SELECT idx.id, idx.chunk_id, idx.parse_version, "
                "document.id AS document_id, "
                "document.active_parse_version AS active_parse_version "
                f"FROM {self._META_TABLE} AS idx "
                "JOIN document_chunks AS chunk ON chunk.id = idx.chunk_id "
                "JOIN documents AS document ON document.id = idx.document_id "
                f"WHERE idx.id IN ({row_id_placeholders}) "
                "AND chunk.parse_version = idx.parse_version "
                f"{document_filter}"
            ),
            parameters,
        ).all()
        # 影子版本:调用方显式指定的文档->版本映射,优先于 active 版本。
        shadow_versions = {
            str(document_id): str(version_key)
            for document_id, version_key in (parse_version_map or {}).items()
        }
        # 构造 {rowid: (chunk_id, parse_version)};只有索引版本与"影子版本
        # 或 active 版本"一致的行才被保留。
        chunks_by_row_id = {
            int(row.id): (str(row.chunk_id), str(row.parse_version))
            for row in mapping_rows
            if str(row.parse_version)
            == (
                shadow_versions.get(str(row.document_id))
                or str(row.active_parse_version or "legacy")
            )
        }
        # 按原距离顺序返回,跳过未通过版本过滤的行。
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
        """统计指定维度下已索引的条目数。"""
        return int(
            self.db.execute(
                text(f"SELECT COUNT(*) FROM {self._META_TABLE} WHERE dimensions = :dimensions"),
                {"dimensions": dimensions},
            ).scalar_one()
        )

    def count_document_chunks(self, document_id: str, parse_version: str) -> int:
        """统计某文档某解析版本已索引的 chunk 数。"""
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
        """确保元数据映射表存在,并做必要的旧库迁移。

        创建表(含 id/chunk_id/document_id/parse_version/dimensions/
        created_at/updated_at 字段);若旧库缺 parse_version 列则补列;
        最后确保有按 document_id 的索引。
        """
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
        # 旧库迁移:早期版本可能没有 parse_version 列。
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
        """确保指定维度的 vec0 虚拟表存在;维度必须为正整数。"""
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
        """在元数据表中插入(或更新)一个 chunk 的映射,返回自增行 id。

        使用 ``ON CONFLICT(chunk_id) DO UPDATE`` 实现 upsert:同一 chunk
        重复写入时更新其文档/版本/维度与时间戳。
        """
        # 记录创建/更新时间(UTC,精确到秒)。
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
        # 回查自增主键,作为向量虚拟表里的 rowid。
        row_id = self.db.execute(
            text(f"SELECT id FROM {self._META_TABLE} WHERE chunk_id = :chunk_id"),
            {"chunk_id": chunk_id},
        ).scalar_one()
        return int(row_id)

    def _serialize(self, embedding: list[float]) -> bytes:
        """把 embedding 序列化为 sqlite-vec 需要的 float32 字节串。"""
        if self._sqlite_vec is not None:
            return self._sqlite_vec.serialize_float32(embedding)
        raise RuntimeError("sqlite-vec serializer is unavailable")

    def _raw_connection(self) -> Any:
        """穿透 SQLAlchemy 代理,拿到底层 DBAPI 原始连接。

        部分驱动(如 sqlite3)在 SQLAlchemy 连接下还包了一层
        driver_connection,这里一并处理。
        """
        proxied = self.db.connection().connection
        return getattr(proxied, "driver_connection", proxied)

    @staticmethod
    def _set_extension_loading(raw_connection: Any, enabled: bool) -> None:
        """开启/关闭底层连接上的扩展加载开关(部分 sqlite 驱动不支持)。"""
        enable_load_extension = getattr(raw_connection, "enable_load_extension", None)
        if callable(enable_load_extension):
            enable_load_extension(enabled)

    @classmethod
    def _vector_table_name(cls, dimensions: int) -> str:
        """按维度生成对应的 vec0 虚拟表名。"""
        if dimensions <= 0:
            raise ValueError("Vector dimensions must be positive")
        return f"{cls._TABLE_PREFIX}{dimensions}"

    @staticmethod
    def _valid_embedding(embedding: list[float] | None) -> bool:
        """校验 embedding 是否合法:非空且每个元素都是有限数值。"""
        return bool(embedding) and all(isinstance(value, int | float) and math.isfinite(float(value)) for value in embedding)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        """对 embedding 做 L2 归一化,使其模长为 1。

        归一化后向量间的点积等价于余弦相似度;若范数为 0 或非有限值
        则返回空列表(表示无法归一化,调用方应丢弃该向量)。
        """
        norm = math.sqrt(sum(float(value) * float(value) for value in embedding))
        if not math.isfinite(norm) or norm <= 0:
            return []
        return [float(value) / norm for value in embedding]


class PGVectorStore:
    """基于 PostgreSQL pgvector 扩展的 chunk embedding 索引。

    用单张表(_TABLE_NAME)存储 chunk_id、document_id、parse_version 与
    vector 类型的 embedding;检索时用余弦距离运算符 ``<=>`` 排序。
    """

    _TABLE_NAME = "document_chunk_pgvector_index"

    def __init__(self, db: Session) -> None:
        """保存 SQLAlchemy Session 并初始化可用性缓存。"""
        self.db = db
        self._available: bool | None = None

    def available(self) -> bool:
        """惰性探测 pgvector 扩展是否可用。

        依次检查开关、后端类型、方言是否为 postgresql,然后查询
        ``pg_extension`` 中是否安装了名为 'vector' 的扩展。
        """
        # 前置开关:未启用、后端不对、方言不对则不可用。
        if not settings.vector_store_enabled:
            return False
        if settings.vector_store_backend != "pgvector":
            return False
        if self.db.get_bind().dialect.name != "postgresql":
            return False
        # 已探测过则直接返回缓存结果。
        if self._available is not None:
            return self._available
        try:
            # 查询是否安装了 vector 扩展。
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
        """以"整体替换"方式写入某文档的 chunk 向量(pgvector 后端)。

        先过滤/归一化向量,并只保留维度与配置的 ollama embedding 维度
        一致的向量;然后在嵌套事务中:删除该文档各 parse_version 的旧行,
        再批量 upsert 新行。
        """
        # 过滤非法向量、归一化,并校验维度必须等于配置的嵌入维度。
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
                # 先按版本清空该文档的旧向量,保证整体替换语义。
                parse_versions = {vector.parse_version for vector in normalized}
                for parse_version in parse_versions:
                    self.delete_document(document_id, parse_version=parse_version)
                if normalized:
                    # 批量 upsert:同一 chunk 重复时更新其文档/版本/向量。
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
        """删除某文档的向量行(可选按 parse_version 过滤)。"""
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
        parse_version_map: dict[str, str] | None = None,
    ) -> list[VectorHit]:
        """执行 pgvector 余弦距离检索,返回按距离升序的命中列表。

        查询向量需先归一化且维度等于配置维度;支持按文档范围过滤与
        版本覆盖(parse_version_map)。失败时告警并返回空列表(调用方
        回退到 JSON embedding)。
        """
        # 前置校验:查询向量无效、limit 非正、后端不可用则返回空。
        if not self._valid_embedding(embedding) or limit <= 0 or not self.available():
            return []
        normalized = self._normalize_embedding(embedding)
        # pgvector 表结构固定,维度必须与配置一致。
        if len(normalized) != settings.ollama_embedding_dimensions:
            return []
        # 归一化文档范围过滤条件。
        scoped_document_ids = [
            str(document_id)
            for document_id in (document_ids or [])
            if str(document_id or "").strip()
        ]
        try:
            # 有版本覆盖时走带 shadow 版本的查询,否则走常规查询。
            rows = (
                self._search_rows(
                    normalized,
                    limit,
                    scoped_document_ids,
                    parse_version_map=parse_version_map,
                )
                if parse_version_map
                else self._search_rows(normalized, limit, scoped_document_ids)
            )
            # 直接把行转换为 VectorHit(查询已含版本过滤)。
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
        """统计某文档某解析版本已索引的 chunk 数(pgvector 后端)。"""
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
        self,
        embedding: list[float],
        limit: int,
        document_ids: list[str],
        *,
        parse_version_map: dict[str, str] | None = None,
    ) -> list[Any]:
        """执行 pgvector KNN 查询并返回原始行。

        版本匹配规则:
        - 常规:索引版本须等于文档 active 版本;active 为空时须为 legacy;
        - 带影子版本(parse_version_map):对这些文档优先用映射中的版本,
          其余文档仍走常规规则。
        通过 ``embedding <=> CAST(... AS vector)`` 计算余弦距离并按距离
        升序取前 limit 行。
        """
        parameters: dict[str, Any] = {
            "embedding": self._serialize(embedding),
            "limit": limit,
        }
        # 文档范围过滤(可选)。
        scope_sql = ""
        if document_ids:
            placeholders = []
            for index, document_id in enumerate(document_ids):
                key = f"document_id_{index}"
                parameters[key] = document_id
                placeholders.append(f":{key}")
            scope_sql = f"AND idx.document_id IN ({','.join(placeholders)}) "
        # 默认版本规则:索引版本 == 文档 active 版本(缺省用 legacy)。
        version_sql = (
            "(idx.parse_version = document.active_parse_version "
            "OR (document.active_parse_version IS NULL AND idx.parse_version = 'legacy'))"
        )
        # 影子版本:调用方显式指定某些文档应使用的版本。
        shadow_versions = {
            str(document_id): str(version_key)
            for document_id, version_key in (parse_version_map or {}).items()
        }
        if shadow_versions:
            shadow_conditions: list[str] = []
            shadow_document_keys: list[str] = []
            # 为每个影子版本文档生成 (document_id = X AND parse_version = Y)。
            for index, document_id in enumerate(sorted(shadow_versions)):
                document_key = f"shadow_document_id_{index}"
                version_key = f"shadow_parse_version_{index}"
                parameters[document_key] = document_id
                parameters[version_key] = shadow_versions[document_id]
                shadow_document_keys.append(f":{document_key}")
                shadow_conditions.append(
                    f"(idx.document_id = :{document_key} "
                    f"AND idx.parse_version = :{version_key})"
                )
            # 版本条件 = 任一影子文档命中,或"非影子文档按常规规则"。
            version_sql = (
                "(" + " OR ".join(shadow_conditions) + " OR "
                f"(idx.document_id NOT IN ({','.join(shadow_document_keys)}) AND "
                f"{version_sql}))"
            )
        return self.db.execute(
            text(
                f"SELECT idx.chunk_id, idx.parse_version, "
                "idx.embedding <=> CAST(:embedding AS vector) AS distance "
                f"FROM {self._TABLE_NAME} AS idx "
                "JOIN documents AS document ON document.id = idx.document_id "
                f"WHERE {version_sql} "
                f"{scope_sql}"
                "ORDER BY distance LIMIT :limit"
            ),
            parameters,
        ).all()

    @staticmethod
    def _serialize(embedding: list[float]) -> str:
        """把 embedding 序列化为 pgvector 的 JSON 字符串(CAST 时使用)。"""
        return json.dumps(embedding, separators=(",", ":"))

    @staticmethod
    def _valid_embedding(embedding: list[float] | None) -> bool:
        """复用 SQLiteVecStore 的校验逻辑(非空且元素均为有限数值)。"""
        return SQLiteVecStore._valid_embedding(embedding)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        """复用 SQLiteVecStore 的 L2 归一化逻辑。"""
        return SQLiteVecStore._normalize_embedding(embedding)


def get_vector_store(db: Session) -> SQLiteVecStore | PGVectorStore:
    """根据配置与数据库方言,返回合适的向量存储后端实例。

    当配置启用了向量存储、后端为 pgvector 且当前数据库是 PostgreSQL 时
    返回 PGVectorStore;否则返回 SQLiteVecStore。
    """
    if (
        settings.vector_store_enabled
        and settings.vector_store_backend == "pgvector"
        and db.get_bind().dialect.name == "postgresql"
    ):
        return PGVectorStore(db)
    return SQLiteVecStore(db)
