"""Strict deployment must fail closed, never relabel historical vectors."""
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import Base
from app.models.records import Document, DocumentChunk, DocumentParseVersion, Project
from app.services.model_readiness import ModelReadiness
from app.services.search import QueryService
from app.services.vector_store import PGVectorStore, get_vector_store


def strict_settings(**overrides):
    return Settings(_env_file=None, **{
        "VECTOR_STORE_STRICT": True, "VECTOR_STORE_BACKEND": "pgvector",
        "DATABASE_URL": "postgresql+psycopg://localhost/contract",
        "OLLAMA_EMBEDDING_MODEL": "qwen3-vl-embedding:2b",
        "OLLAMA_EMBEDDING_DIMENSIONS": 2048,
        "EMBEDDING_REVISION": "weights-r1", "EMBEDDING_PROCESSOR_HASH": "processor-p1",
        **overrides,
    })


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id="p", slug="p", name="Contract"))
        session.add(Document(id="d", project_id="p", title="D", file_name="d.txt",
            sha256="a" * 64, raw_path="/unused", status="ready", active_parse_version="v5"))
        session.flush()
        yield session
    engine.dispose()


def add_version(db, identity):
    db.add(DocumentParseVersion(document_id="d", version_key="v5", status="ready",
        artifact_dir="/unused", manifest_json={"ingestion_config": {"embedding": identity}},
        stage_state={"embed": {"status": "completed"}, "index": {"status": "completed"}}))
    db.flush()


def identity_dict(**overrides):
    return {"provider": "ollama", "model": "qwen3-vl-embedding:2b", "dimensions": 2048,
        "revision": "weights-r1", "processor_hash": "processor-p1", **overrides}


def test_strict_profile_rejects_sqlite_without_json_fallback(db, monkeypatch):
    import app.services.search as search
    import app.services.vector_store as vectors
    configured = strict_settings(OLLAMA_EMBEDDING_DIMENSIONS=2)
    monkeypatch.setattr(search, "settings", configured)
    monkeypatch.setattr(vectors, "settings", configured)
    document = db.get(Document, "d")
    document.active_parse_version = None
    db.add(DocumentChunk(id="c", document_id="d", parse_version="legacy",
        chunk_role="child", ordinal=0, text="probe", embedding=[1.0, 0.0]))
    db.flush()
    json_calls = []
    monkeypatch.setattr(search, "cosine_similarity", lambda *args: json_calls.append(args) or 1.0)
    with pytest.raises(RuntimeError, match="pgvector_config_invalid"):
        QueryService(db)._search_source_chunks("probe", "p", ["d"], question_vector=[1.0, 0.0])
    assert json_calls == []


def test_missing_identity_is_unverified_not_backfilled(db):
    from app.services.runtime_contract import EmbeddingIdentity, RuntimeContractError, check_embedding_contract
    historical = identity_dict()
    del historical["revision"]
    del historical["processor_hash"]
    add_version(db, historical)
    before = deepcopy(db.scalar(select(DocumentParseVersion)).manifest_json)
    with pytest.raises(RuntimeContractError, match="embedding_identity_unverified"):
        check_embedding_contract(db, EmbeddingIdentity(**identity_dict()), {"d": "v5"})
    assert db.scalar(select(DocumentParseVersion)).manifest_json == before


@pytest.mark.parametrize("field,value", [("revision", "weights-r2"),
    ("processor_hash", "processor-p2"), ("model", "other-model"), ("dimensions", 2560)])
def test_historical_identity_mismatch_rejects_even_same_dimensions(db, field, value):
    from app.services.runtime_contract import EmbeddingIdentity, RuntimeContractError, check_embedding_contract
    add_version(db, identity_dict(**{field: value}))
    with pytest.raises(RuntimeContractError, match="embedding_identity_mismatch"):
        check_embedding_contract(db, EmbeddingIdentity(**identity_dict()), {"d": "v5"})


def test_verified_frozen_version_is_not_replaced_by_active_pointer(db):
    from app.services.runtime_contract import EmbeddingIdentity, check_embedding_contract
    add_version(db, identity_dict())
    db.get(Document, "d").active_parse_version = "v6"
    check_embedding_contract(db, EmbeddingIdentity(**identity_dict()), {"d": "v5"})


def test_missing_frozen_version_is_not_accepted_as_empty(db):
    from app.services.runtime_contract import EmbeddingIdentity, RuntimeContractError, check_embedding_contract
    with pytest.raises(RuntimeContractError, match="parse_version_missing"):
        check_embedding_contract(db, EmbeddingIdentity(**identity_dict()), {"d": "v5"})


def test_same_dimensions_different_model_revision_blocks_ready(monkeypatch):
    import app.services.model_readiness as readiness_module
    def handler(request):
        if request.url.path == "/api/embedding_identity":
            return httpx.Response(200, json=identity_dict(revision="weights-r2"))
        return httpx.Response(200, json={"models": [
            {"model": "qwen3.5:9b"}, {"model": "qwen3-vl-embedding:2b"}]})
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(readiness_module.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    result = ModelReadiness(strict_settings(), cache_seconds=0).check()
    assert result["status"] == "degraded"
    assert result["models"]["embedding"]["status"] != "ready"
    assert result["models"]["embedding"]["error"] == "embedding_identity_mismatch"


def test_pgvector_search_backend_failure_is_not_empty_success(monkeypatch):
    import app.services.vector_store as vectors
    monkeypatch.setattr(vectors, "settings", strict_settings(OLLAMA_EMBEDDING_DIMENSIONS=2))
    store = PGVectorStore(SimpleNamespace())
    monkeypatch.setattr(store, "available", lambda: True)
    def failed(*args, **kwargs):
        raise OSError("private credential must not leak")
    monkeypatch.setattr(store, "_search_rows", failed)
    with pytest.raises(RuntimeError, match="pgvector_search_failed") as exc:
        store.search([1.0, 0.0], limit=1)
    assert "credential" not in str(exc.value)


def test_valid_empty_pgvector_search_does_not_read_json_embeddings(monkeypatch, db):
    import app.services.search as search
    import app.services.vector_store as vectors
    configured = strict_settings(OLLAMA_EMBEDDING_DIMENSIONS=2)
    monkeypatch.setattr(search, "settings", configured)
    monkeypatch.setattr(vectors, "settings", configured)
    db.get(Document, "d").active_parse_version = None
    db.add(DocumentChunk(id="c", document_id="d", parse_version="legacy", chunk_role="child", ordinal=0,
        text="probe", embedding=[1.0, 0.0]))
    db.flush()
    monkeypatch.setattr(search, "get_vector_store", lambda db: SimpleNamespace(search=lambda *a, **k: []))
    def forbidden(*args):
        pytest.fail("strict pgvector searched JSON embeddings after a valid empty result")
    monkeypatch.setattr(search, "cosine_similarity", forbidden)
    service = QueryService(db)
    contexts = service._search_source_chunks("probe", "p", ["d"], question_vector=[1.0, 0.0])
    assert contexts  # Normal lexical evidence may still supplement a valid empty vector result.
    assert service.retrieval_backend == "pgvector"


def test_pgvector_contract_rejects_sqlite_without_printing_connection_url(db):
    from app.services.runtime_contract import RuntimeContractError, check_pgvector_contract
    with pytest.raises(RuntimeContractError, match="pgvector_config_invalid") as exc:
        check_pgvector_contract(db, strict_settings(DATABASE_URL="postgresql+psycopg://user:secret@localhost/db"))
    assert "secret" not in str(exc.value)


def test_new_ingestion_identity_includes_verified_embedding_fingerprint():
    from app.services.ingestion_identity import build_ingestion_config_snapshot
    snapshot = build_ingestion_config_snapshot(strict_settings(), tokenizer_identity={
        "name": "tokenizer", "revision": "token-r1", "content_sha256": "t1"})
    assert snapshot["embedding"] == identity_dict()


@pytest.mark.parametrize("vector", [[0.0, 0.0], [True, 0.0], [float("nan"), 0.0], [1.0]])
def test_strict_invalid_query_vector_is_not_reported_as_zero_hits(monkeypatch, vector):
    import app.services.vector_store as vectors
    monkeypatch.setattr(vectors, "settings", strict_settings(OLLAMA_EMBEDDING_DIMENSIONS=2))
    store = PGVectorStore(SimpleNamespace())
    monkeypatch.setattr(store, "available", lambda: True)
    with pytest.raises(RuntimeError, match="pgvector_invalid_embedding"):
        store.search(vector, limit=1)


class ContractDatabase:
    """Use real ORM records; stub only the PostgreSQL-specific catalog boundary."""
    def __init__(self, db, *, extension="0.8.6", column="vector(2048)", indexed=2):
        self.db, self.extension, self.column, self.indexed = db, extension, column, indexed

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name="postgresql", driver="psycopg"))

    def execute(self, statement, parameters=None):
        sql = str(statement)
        if "pg_extension" in sql:
            return SimpleNamespace(scalar_one_or_none=lambda: self.extension)
        if "pg_attribute" in sql:
            return SimpleNamespace(scalar_one_or_none=lambda: self.column)
        if "count(idx.chunk_id)" in sql:
            return SimpleNamespace(one=lambda: SimpleNamespace(expected=2, indexed=self.indexed))
        return self.db.execute(statement, parameters or {})

    def scalar(self, statement):
        return self.db.scalar(statement)

    def scalars(self, statement):
        return self.db.scalars(statement)


@pytest.mark.parametrize("overrides,reason", [({"extension": None}, "pgvector_extension_missing"),
    ({"column": "vector(2560)"}, "pgvector_dimension_mismatch"),
    ({"indexed": 1}, "active_index_incomplete")])
def test_incomplete_physical_backend_blocks_ready(db, overrides, reason):
    from app.services.runtime_contract import RuntimeContractError, check_pgvector_contract
    add_version(db, identity_dict())
    db.scalar(select(DocumentParseVersion)).status = "active"
    db.flush()
    with pytest.raises(RuntimeContractError, match=reason):
        check_pgvector_contract(ContractDatabase(db, **overrides), strict_settings())


def test_fully_verified_contract_reports_actual_backend(db):
    from app.services.runtime_contract import check_pgvector_contract
    add_version(db, identity_dict())
    db.scalar(select(DocumentParseVersion)).status = "active"
    db.flush()
    result = check_pgvector_contract(ContractDatabase(db), strict_settings())
    assert result == {"status": "ready", "backend": "pgvector", "extension": "0.8.6",
        "column_type": "vector(2048)", "embedding_identity": identity_dict(), "active_documents": 1}
