# Canonical Contextual Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current character-truncated ingestion path with versioned canonical parsing, semantic Parent/Child chunks, mandatory contextual prefixes, contextualized embeddings, precise source locations, and an all-or-nothing historical rebuild.

**Architecture:** Format-specific adapters produce one shared `CanonicalDocument` model and immutable artifact bundle. A staged Redis pipeline validates and repairs the canonical content, creates semantic Parent/Child chunks, contextualizes every retrievable Child, builds a version-scoped pgvector index, and atomically activates the new parse version. Retrieval searches only the active version and expands Child hits to their Parent while citations remain bound to original source spans.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL/pgvector, Redis/RQ, Ollama (`qwen3.5:9b`, `qwen3-embedding:4b`), MinerU, PyMuPDF, pypdf, python-docx, BeautifulSoup, trafilatura, pytest, dependency-free HTML/CSS/JavaScript frontend.

**Design:** `docs/superpowers/specs/2026-07-20-canonical-contextual-ingestion-design.md`

---

## File Map

### New production modules

- `src/app/services/canonical_models.py`: canonical document, block, table, figure, formula, source span, and quality types.
- `src/app/services/canonical_artifacts.py`: version-key generation, staging directories, canonical bundle rendering, validation, promotion, and cleanup.
- `src/app/services/canonical_adapters.py`: adapter protocol and PDF/DOCX/HTML/TXT/Markdown adapters.
- `src/app/services/canonical_quality.py`: deterministic quality gates and targeted repair decisions.
- `src/app/services/structured_evidence.py`: table validation/row grouping plus Figure/Formula derived data.
- `src/app/services/semantic_chunking.py`: section-first semantic Parent/Child construction.
- `src/app/services/contextualization.py`: batched contextual-prefix generation, validation, retries, and checkpoints.
- `src/app/services/parse_versions.py`: parse-version persistence, status transitions, activation, and obsolete-data deletion.
- `src/app/services/ingestion_stages.py`: resumable stage orchestration.
- `scripts/rebuild_canonical_index.py`: maintenance-mode historical rebuild and final cleanup.
- `scripts/evaluate_canonical_retrieval.py`: parsing/RAG/citation acceptance runner.

### Existing modules to modify

- `src/app/core/config.py`: parser, tokenizer, semantic splitting, contextualization, artifact, and stage queue settings.
- `src/app/models/records.py`: parse-version model, active version, Parent/Child and source fields.
- `src/app/db/session.py`: SQLite compatibility columns for local tests.
- `src/app/services/parser.py`: temporary compatibility facade over canonical adapters.
- `src/app/services/pipeline.py`: replace monolithic parse/chunk/embed execution with staged orchestration.
- `src/app/services/queue.py`: named stage queues.
- `src/app/workers/jobs.py`: stage job entry point.
- `src/app/services/ai.py`: configurable contextualization and semantic-embedding calls.
- `src/app/services/vector_store.py`: version-scoped writes and active-version queries.
- `src/app/services/search.py`: Child retrieval, Parent expansion, active-version filters, source-span citations.
- `src/app/schemas/common.py`: parse-version, source-location, and citation response models.
- `src/app/api/routes.py`: canonical parse/status/download/location endpoints.
- `src/app/static/index.html`: parse progress, Markdown view/download, and source highlighting.
- `.env.development.example`, `.env.test.example`: explicit new settings.
- `deploy/systemd/knowledge-agent-dev-worker.service`, `deploy/systemd/knowledge-agent-test-worker.service`: staged queue worker configuration.
- `pyproject.toml`: tokenizer dependency.

### New tests and fixtures

- `tests/test_canonical_models.py`
- `tests/test_canonical_artifacts.py`
- `tests/test_canonical_adapters.py`
- `tests/test_canonical_quality.py`
- `tests/test_structured_evidence.py`
- `tests/test_semantic_chunking.py`
- `tests/test_contextualization.py`
- `tests/test_ingestion_stages.py`
- `tests/test_parse_versions.py`
- `tests/test_canonical_api.py`
- `tests/test_canonical_retrieval.py`
- `tests/fixtures/canonical/`

---

### Task 1: Add Configuration and Dependency Contracts

**Files:**
- Modify: `src/app/core/config.py`
- Modify: `.env.development.example`
- Modify: `.env.test.example`
- Modify: `pyproject.toml`
- Create: `tests/test_contextual_ingestion_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
from app.core.config import Settings


def test_contextual_ingestion_defaults_are_strict() -> None:
    settings = Settings(_env_file=None)
    assert settings.canonical_artifacts_dir.name == "parsed"
    assert settings.semantic_splitting_model == "qwen3-embedding:4b"
    assert settings.contextualization_model == "qwen3.5:9b"
    assert settings.contextualization_batch_size == 12
    assert settings.contextualization_max_retries == 2
    assert settings.contextualization_max_sentences == 2
    assert settings.parent_token_limits == (500, 1200, 1800)
    assert settings.child_token_limits == (180, 400, 600)
    assert settings.child_overlap_tokens == 50
    assert settings.mineru_enabled is True
    assert settings.maintenance_mode_enabled is False
```

- [ ] **Step 2: Run the test and verify the missing settings fail**

Run: `pytest tests/test_contextual_ingestion_config.py -v`

Expected: FAIL with an `AttributeError` for `canonical_artifacts_dir`.

- [ ] **Step 3: Add typed settings and tuple accessors**

Add fields to `Settings`:

```python
canonical_artifacts_dir: Path = Field(default=Path("./data/parsed"), alias="CANONICAL_ARTIFACTS_DIR")
canonical_pipeline_version: str = Field(default="canonical-v1", alias="CANONICAL_PIPELINE_VERSION")
semantic_splitting_enabled: bool = Field(default=True, alias="SEMANTIC_SPLITTING_ENABLED")
semantic_splitting_model: str = Field(default="qwen3-embedding:4b", alias="SEMANTIC_SPLITTING_MODEL")
semantic_break_percentile: int = Field(default=20, ge=1, le=99, alias="SEMANTIC_BREAK_PERCENTILE")
semantic_parent_min_tokens: int = Field(default=500, gt=0, alias="SEMANTIC_PARENT_MIN_TOKENS")
semantic_parent_target_tokens: int = Field(default=1200, gt=0, alias="SEMANTIC_PARENT_TARGET_TOKENS")
semantic_parent_max_tokens: int = Field(default=1800, gt=0, alias="SEMANTIC_PARENT_MAX_TOKENS")
semantic_child_min_tokens: int = Field(default=180, gt=0, alias="SEMANTIC_CHILD_MIN_TOKENS")
semantic_child_target_tokens: int = Field(default=400, gt=0, alias="SEMANTIC_CHILD_TARGET_TOKENS")
semantic_child_max_tokens: int = Field(default=600, gt=0, alias="SEMANTIC_CHILD_MAX_TOKENS")
semantic_child_overlap_tokens: int = Field(default=50, ge=0, alias="SEMANTIC_CHILD_OVERLAP_TOKENS")
semantic_tokenizer_name: str = Field(default="Qwen/Qwen3-Embedding-4B", alias="SEMANTIC_TOKENIZER_NAME")
contextualization_enabled: bool = Field(default=True, alias="CONTEXTUALIZATION_ENABLED")
contextualization_base_url: str = Field(default="http://localhost:11435", alias="CONTEXTUALIZATION_BASE_URL")
contextualization_model: str = Field(default="qwen3.5:9b", alias="CONTEXTUALIZATION_MODEL")
contextualization_batch_size: int = Field(default=12, gt=0, alias="CONTEXTUALIZATION_BATCH_SIZE")
contextualization_max_retries: int = Field(default=2, ge=0, alias="CONTEXTUALIZATION_MAX_RETRIES")
contextualization_timeout: int = Field(default=180, gt=0, alias="CONTEXTUALIZATION_TIMEOUT")
contextualization_max_sentences: int = Field(default=2, ge=1, le=2, alias="CONTEXTUALIZATION_MAX_SENTENCES")
contextualization_prompt_version: str = Field(default="context-v1", alias="CONTEXTUALIZATION_PROMPT_VERSION")
figure_analysis_model: str = Field(default="qwen3.5:9b", alias="FIGURE_ANALYSIS_MODEL")
formula_analysis_model: str = Field(default="qwen3.5:9b", alias="FORMULA_ANALYSIS_MODEL")
maintenance_mode_enabled: bool = Field(default=False, alias="MAINTENANCE_MODE_ENABLED")

@property
def parent_token_limits(self) -> tuple[int, int, int]:
    return (self.semantic_parent_min_tokens, self.semantic_parent_target_tokens, self.semantic_parent_max_tokens)

@property
def child_token_limits(self) -> tuple[int, int, int]:
    return (self.semantic_child_min_tokens, self.semantic_child_target_tokens, self.semantic_child_max_tokens)
```

Change the default `mineru_enabled` to `True`. Add `transformers>=4.51.0,<5.0` to dependencies for the configured Qwen tokenizer.

- [ ] **Step 4: Update both environment examples with explicit values**

Add every field above and set `MINERU_ENABLED=true`, `SAC_KG_ENABLED=false`, `AGENT_SYNTHESIS_PROVIDER=local`, and `MAINTENANCE_MODE_ENABLED=false` in both files. Use `CONTEXTUALIZATION_BASE_URL=http://127.0.0.1:11435` for development and `http://127.0.0.1:11436` for test. Set `CANONICAL_ARTIFACTS_DIR=./runtime/data/parsed` so artifacts follow each environment's `DATA_DIR`.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_contextual_ingestion_config.py tests/test_deployment_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/core/config.py .env.development.example .env.test.example pyproject.toml tests/test_contextual_ingestion_config.py
git commit -m "Add contextual ingestion configuration"
```

### Task 2: Define Canonical Models and Immutable Artifacts

**Files:**
- Create: `src/app/services/canonical_models.py`
- Create: `src/app/services/canonical_artifacts.py`
- Create: `tests/test_canonical_models.py`
- Create: `tests/test_canonical_artifacts.py`

- [ ] **Step 1: Write model serialization tests**

```python
from app.services.canonical_models import CanonicalBlock, CanonicalDocument, SourceSpan


def test_canonical_block_preserves_source_spans() -> None:
    span = SourceSpan(page_index=7, page_label="8", bbox=[10.0, 20.0, 200.0, 120.0], source_block_id="b-8")
    block = CanonicalBlock(
        block_id="block-8",
        block_type="table",
        text="| Model | F1 |\n|---|---|\n| SAC-KG | 74.7 |",
        section_path=["Experiments", "Main Results"],
        reading_order=12,
        source_spans=[span],
        parser_source="mineru",
    )
    restored = CanonicalBlock.model_validate_json(block.model_dump_json())
    assert restored.source_spans[0].page_index == 7
    assert restored.block_type == "table"


def test_reference_blocks_are_not_retrievable_by_default() -> None:
    block = CanonicalBlock(block_id="r1", block_type="reference", text="[1] Paper", reading_order=1, parser_source="mineru")
    assert block.retrievable is False
```

- [ ] **Step 2: Run the tests to verify imports fail**

Run: `pytest tests/test_canonical_models.py -v`

Expected: FAIL with `ModuleNotFoundError: app.services.canonical_models`.

- [ ] **Step 3: Implement canonical Pydantic models**

Define `SourceSpan`, `CanonicalBlock`, `CanonicalCell`, `CanonicalTable`, `CanonicalFigure`, `CanonicalFormula`, `CanonicalAsset`, `CanonicalQualityIssue`, `CanonicalQualityReport`, and `CanonicalDocument`. Validate `block_type` with a `Literal` and make `retrievable` false only for `heading` and `reference` by default.

- [ ] **Step 4: Write artifact round-trip tests**

```python
def test_artifact_store_writes_self_contained_bundle(tmp_path: Path, canonical_document) -> None:
    store = CanonicalArtifactStore(tmp_path)
    staging = store.write_staging("doc-1", "canonical-v1-abcd", canonical_document)
    assert (staging / "canonical.md").read_text("utf-8").startswith("---\n")
    assert (staging / "manifest.json").exists()
    assert (staging / "blocks.jsonl").exists()
    final = store.promote("doc-1", "canonical-v1-abcd")
    assert final.exists()
    assert not staging.exists()
```

- [ ] **Step 5: Implement artifact writing and atomic promotion**

`CanonicalArtifactStore` must write to `<version>.staging-<uuid>`, fsync JSON/Markdown files, validate that every asset path resolves beneath the staging `assets/`, then atomically rename the directory to `<version>`. `canonical.md` renders source blocks only; generated Figure/Formula analysis and contextual prefixes stay in JSON.

- [ ] **Step 6: Run tests and diff checks**

Run: `pytest tests/test_canonical_models.py tests/test_canonical_artifacts.py -v`

Expected: PASS.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add src/app/services/canonical_models.py src/app/services/canonical_artifacts.py tests/test_canonical_models.py tests/test_canonical_artifacts.py
git commit -m "Add canonical document artifacts"
```

### Task 3: Add Parse-Version and Parent/Child Database Schema

**Files:**
- Create: `src/app/db/alembic/versions/a4e2c7f90120_add_canonical_parse_versions.py`
- Modify: `src/app/models/records.py`
- Modify: `src/app/db/session.py`
- Create: `src/app/services/parse_versions.py`
- Create: `tests/test_parse_versions.py`
- Modify: `tests/test_migrations.py`

- [ ] **Step 1: Write failing model and activation tests**

```python
def test_parse_version_activation_is_atomic(db, ready_document) -> None:
    service = ParseVersionService(db)
    version = service.create(ready_document.id, "canonical-v1-abcd", "/tmp/doc/version")
    service.transition(version, "ready_to_activate")
    service.activate(ready_document, version)
    db.flush()
    assert ready_document.active_parse_version == "canonical-v1-abcd"
    assert version.status == "active"


def test_parent_child_chunk_keeps_original_and_embedding_text(db, ready_document) -> None:
    parent = DocumentChunk(document_id=ready_document.id, parse_version="v1", ordinal=0, chunk_role="parent", block_type="narrative", text="Parent")
    db.add(parent)
    db.flush()
    child = DocumentChunk(
        document_id=ready_document.id,
        parse_version="v1",
        parent_chunk_id=parent.id,
        ordinal=1,
        chunk_role="child",
        block_type="narrative",
        text="Original evidence",
        contextual_prefix="该分块来自方法部分。",
        embedding_text="该分块来自方法部分。\n\nOriginal evidence",
    )
    db.add(child)
    db.flush()
    assert child.parent_chunk_id == parent.id
    assert child.text == "Original evidence"
```

- [ ] **Step 2: Run tests and verify schema fields are missing**

Run: `pytest tests/test_parse_versions.py -v`

Expected: FAIL for missing `DocumentParseVersion` or model keyword arguments.

- [ ] **Step 3: Implement SQLAlchemy models**

Add `DocumentParseVersion` with `document_id`, `version_key`, `status`, `artifact_dir`, parser metadata, `manifest_json`, `quality_json`, `stage_state`, and `activated_at`, with a unique constraint on `(document_id, version_key)`. Add `Document.active_parse_version`. Extend `DocumentStatus` with `parsing`, `quality_checking`, `repairing`, `canonicalizing`, `chunking`, `contextualizing`, `embedding`, `indexing`, `parse_failed`, `table_repair_failed`, `contextualization_failed`, `embedding_failed`, and `activation_failed` so API status serialization remains explicit. During the migration backfill, mark `legacy` chunks as quarantined and exclude them from active retrieval until the rebuild activates a canonical version.

Extend `DocumentChunk` with the fields in the design. Use JSON for `section_path`, `source_block_ids`, and `source_spans`. Add a self-referencing nullable `parent_chunk_id` with `ON DELETE CASCADE` and indexes on `(document_id, parse_version, chunk_role)`.

- [ ] **Step 4: Implement the Alembic migration**

Set `down_revision = "81f6b74cc203"`. Create `document_parse_versions`, add nullable columns, backfill existing chunks with `parse_version='legacy'`, `chunk_role='child'`, `block_type='narrative'`, `embedding_text=text`, then make required columns non-null. Add `parse_version` to `document_chunk_pgvector_index` and backfill `legacy` on PostgreSQL.

- [ ] **Step 5: Implement ParseVersionService transitions**

Permit only:

```python
ALLOWED_TRANSITIONS = {
    "queued": {"parsing", "parse_failed"},
    "parsing": {"quality_checking", "parse_failed"},
    "quality_checking": {"repairing", "canonicalizing", "parse_failed"},
    "repairing": {"canonicalizing", "table_repair_failed", "parse_failed"},
    "canonicalizing": {"chunking", "parse_failed"},
    "chunking": {"contextualizing", "parse_failed"},
    "contextualizing": {"embedding", "contextualization_failed"},
    "embedding": {"indexing", "embedding_failed"},
    "indexing": {"ready_to_activate", "embedding_failed"},
    "ready_to_activate": {"active", "activation_failed"},
}
```

Activation updates the document pointer and version status in one transaction.

- [ ] **Step 6: Run migration and model tests**

Run: `pytest tests/test_parse_versions.py tests/test_migrations.py -v`

Expected: PASS on SQLite test databases; PostgreSQL-specific DDL assertions pass without executing pgvector SQL on SQLite.

- [ ] **Step 7: Commit**

```bash
git add src/app/models/records.py src/app/db/session.py src/app/db/alembic/versions/a4e2c7f90120_add_canonical_parse_versions.py src/app/services/parse_versions.py tests/test_parse_versions.py tests/test_migrations.py
git commit -m "Add versioned parent child chunk schema"
```

### Task 4: Implement Canonical Adapters for Existing Formats

**Files:**
- Create: `src/app/services/canonical_adapters.py`
- Modify: `src/app/services/parser.py`
- Create: `tests/test_canonical_adapters.py`
- Create: `tests/fixtures/canonical/sample.md`
- Create: `tests/fixtures/canonical/sample.html`
- Create: `tests/fixtures/canonical/sample.docx`

- [ ] **Step 1: Write adapter contract tests**

```python
@pytest.mark.parametrize("name", ["sample.md", "sample.html", "sample.docx"])
def test_existing_formats_produce_canonical_blocks(fixture_dir: Path, name: str) -> None:
    document = parse_canonical_document(fixture_dir / name)
    assert document.blocks
    assert [block.reading_order for block in document.blocks] == list(range(len(document.blocks)))
    assert all(block.block_id and block.parser_source for block in document.blocks)


def test_markdown_references_are_preserved_but_not_retrievable(fixture_dir: Path) -> None:
    document = parse_canonical_document(fixture_dir / "sample.md")
    references = [block for block in document.blocks if block.block_type == "reference"]
    assert references
    assert all(not block.retrievable for block in references)
```

- [ ] **Step 2: Run the tests and verify the canonical entry point is absent**

Run: `pytest tests/test_canonical_adapters.py -v`

Expected: FAIL with missing `parse_canonical_document`.

- [ ] **Step 3: Implement the adapter protocol and dispatcher**

```python
class CanonicalAdapter(Protocol):
    def parse(self, path: Path) -> CanonicalDocument:
        raise NotImplementedError


def parse_canonical_document(path: Path) -> CanonicalDocument:
    suffix = path.suffix.lower()
    adapter = {
        ".pdf": PDFCanonicalAdapter(),
        ".docx": DocxCanonicalAdapter(),
        ".html": HtmlCanonicalAdapter(),
        ".htm": HtmlCanonicalAdapter(),
        ".md": MarkdownCanonicalAdapter(),
        ".markdown": MarkdownCanonicalAdapter(),
        ".txt": TextCanonicalAdapter(),
    }.get(suffix, TextCanonicalAdapter())
    return adapter.parse(path)
```

- [ ] **Step 4: Implement DOCX, HTML, Markdown, and TXT adapters**

DOCX must map headings, paragraphs, tables, inline image relationship IDs, formulas where present, and paragraph/table locators. HTML must preserve DOM selectors for headings, paragraphs, tables, figures, captions, code, and math while removing scripts/styles/navigation. Markdown must preserve headings, fenced blocks, tables, formulas, images, references, and line ranges. TXT creates paragraph blocks with line/character ranges.

- [ ] **Step 5: Keep `parse_document()` as a compatibility facade**

Convert `CanonicalDocument` to the current `ParsedDocument` only for tests and callers not yet migrated. Mark the conversion function internal and do not truncate canonical blocks.

- [ ] **Step 6: Run adapter and legacy parser tests**

Run: `pytest tests/test_canonical_adapters.py tests/test_parser_document_intelligence.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app/services/canonical_adapters.py src/app/services/parser.py tests/test_canonical_adapters.py tests/fixtures/canonical
git commit -m "Add canonical adapters for supported formats"
```

### Task 5: Make MinerU the Primary PDF Parser with Quality-Gated Repair

**Files:**
- Modify: `src/app/services/canonical_adapters.py`
- Create: `src/app/services/canonical_quality.py`
- Modify: `src/app/services/parser.py`
- Create: `tests/test_canonical_quality.py`
- Modify: `tests/test_parser_document_intelligence.py`

- [ ] **Step 1: Write parser-order and single-page-tolerance tests**

```python
def test_pdf_uses_mineru_when_pypdf_text_extraction_fails(monkeypatch, pdf_path: Path) -> None:
    monkeypatch.setattr(parser, "_extract_pdf_text_layer", lambda _path: (_ for _ in ()).throw(RuntimeError("page text failed")))
    monkeypatch.setattr(canonical_adapters, "run_mineru", lambda _path: canonical_pdf_fixture())
    result = PDFCanonicalAdapter().parse(pdf_path)
    assert result.metadata["primary_parser"] == "mineru"


def test_quality_gate_requests_targeted_abstract_repair() -> None:
    report = CanonicalQualityGate().evaluate(document_with_unclassified_abstract())
    assert report.accepted is False
    assert any(issue.code == "abstract_missing" and issue.repair_scope == "pages:1-2" for issue in report.issues)
```

- [ ] **Step 2: Run tests and verify current pypdf-first behavior fails**

Run: `pytest tests/test_canonical_quality.py tests/test_parser_document_intelligence.py -v`

Expected: FAIL because `_parse_pdf()` raises before MinerU.

- [ ] **Step 3: Refactor PDF orchestration**

Split basic PDF validation/page count from best-effort text extraction. Run MinerU first after validation. Evaluate the canonical result. Invoke Document Intelligence only for issue scopes returned by the gate; invoke full Document Intelligence only on fatal MinerU failure. Use pypdf text blocks only when both structured paths fail.

- [ ] **Step 4: Implement deterministic quality issues**

Codes must include `page_missing`, `content_empty`, `reading_order_invalid`, `abstract_missing`, `asset_invalid`, `table_invalid`, `figure_caption_missing`, and `formula_analysis_missing`. Assign fatal, repairable, or warning severity exactly as defined by the design.

- [ ] **Step 5: Remove every parser-level `[:4000]` truncation**

Replace truncation with complete canonical blocks. Add a regression assertion:

```python
def test_mineru_long_table_is_not_truncated() -> None:
    markdown = make_table_with_rows(600)
    document = mineru_document_with_table(markdown)
    table = next(block for block in document.blocks if block.block_type == "table")
    assert table.text.endswith("| row-599 |")
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_canonical_quality.py tests/test_parser_document_intelligence.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app/services/canonical_adapters.py src/app/services/canonical_quality.py src/app/services/parser.py tests/test_canonical_quality.py tests/test_parser_document_intelligence.py
git commit -m "Make MinerU the quality gated PDF parser"
```

### Task 6: Add Structured Table, Figure, and Formula Processing

**Files:**
- Create: `src/app/services/structured_evidence.py`
- Modify: `src/app/services/canonical_adapters.py`
- Modify: `src/app/services/canonical_artifacts.py`
- Create: `tests/test_structured_evidence.py`

- [ ] **Step 1: Write strict table tests**

```python
def test_long_table_children_repeat_caption_and_header() -> None:
    table = canonical_table(rows=120, caption="Table 5. Main results", headers=["Dataset", "Model", "F1", "AUC"])
    parent, children = StructuredEvidenceBuilder().table_chunks(table, max_tokens=180)
    assert len(children) > 1
    assert all("Table 5. Main results" in child.text for child in children)
    assert all("| Dataset | Model | F1 | AUC |" in child.text for child in children)


def test_invalid_table_requires_repair() -> None:
    result = TableValidator().validate(canonical_table(rows=[["Model", "F1"], ["SAC-KG"]]))
    assert result.accepted is False
    assert result.status == "validation_failed"
```

- [ ] **Step 2: Write Figure/Formula provenance tests**

```python
def test_generated_figure_analysis_is_not_rendered_as_source_markdown(tmp_path: Path) -> None:
    document = canonical_document_with_figure(generated_summary="AI-only trend")
    path = CanonicalArtifactStore(tmp_path).write_staging("doc", "v1", document) / "canonical.md"
    markdown = path.read_text("utf-8")
    assert "![Figure" in markdown
    assert "Original caption" in markdown
    assert "AI-only trend" not in markdown
```

- [ ] **Step 3: Run tests and verify the module is absent**

Run: `pytest tests/test_structured_evidence.py -v`

Expected: FAIL with missing module/classes.

- [ ] **Step 4: Implement table validation and row-group chunking**

Preserve cells, rowspan/colspan, bbox, footnotes, source HTML/Markdown, and normalized Markdown. Join cross-page continuation before chunking. Return `accepted_mineru`, `repaired_by_vision`, `cross_page_merged`, or `validation_failed`. Invalid tables produce a repair request and block activation if repair remains invalid.

- [ ] **Step 5: Implement Figure and Formula records**

Copy assets into the canonical staging bundle before MinerU temp cleanup. Save generated analysis only in `figures.json`/`formulas.json`. Build source evidence from original caption/LaTeX and nearby blocks. Generated analysis warnings do not reject the canonical version.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_structured_evidence.py tests/test_canonical_artifacts.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app/services/structured_evidence.py src/app/services/canonical_adapters.py src/app/services/canonical_artifacts.py tests/test_structured_evidence.py tests/test_canonical_artifacts.py
git commit -m "Add structured canonical evidence"
```

### Task 7: Build Section-Aware Semantic Parent/Child Chunks

**Files:**
- Create: `src/app/services/semantic_chunking.py`
- Create: `tests/test_semantic_chunking.py`

- [ ] **Step 1: Write boundary, token, and structure tests**

```python
def test_semantic_chunker_keeps_children_within_limits(fake_embedder, token_counter) -> None:
    chunks = SemanticChunker(fake_embedder, token_counter).build(canonical_narrative_document())
    children = [chunk for chunk in chunks if chunk.chunk_role == "child"]
    assert children
    assert all(180 <= chunk.token_count <= 600 for chunk in children)
    assert all(chunk.parent_local_id for chunk in children)


def test_table_blocks_never_merge_with_narrative(fake_embedder, token_counter) -> None:
    chunks = SemanticChunker(fake_embedder, token_counter).build(document_with_paragraph_table_paragraph())
    table_children = [chunk for chunk in chunks if chunk.block_type == "table" and chunk.chunk_role == "child"]
    assert table_children
    assert all("ordinary paragraph" not in chunk.text for chunk in table_children)


def test_overlap_copies_whole_sentences() -> None:
    children = split_children(long_parent(), overlap_tokens=50)
    assert children[0].text.split("。")[-2] in children[1].text
```

- [ ] **Step 2: Run tests and verify the chunker is absent**

Run: `pytest tests/test_semantic_chunking.py -v`

Expected: FAIL with missing `SemanticChunker`.

- [ ] **Step 3: Implement token counting and semantic boundaries**

Load the configured Qwen tokenizer once per process. Split canonical narrative into paragraph/sentence units, batch raw-unit embeddings, calculate adjacent cosine similarity, and prefer boundaries in the lowest configured percentile after target size. Force a sentence boundary at maximum. Keep section boundaries hard.

- [ ] **Step 4: Implement Parent/Child and neighbor metadata**

Define `ChunkDraft.parent_local_id` for in-memory relationships, then assign database `parent_chunk_id` during persistence. Derive stable local IDs from parse version, source block IDs, role, and ordinal. Preserve all source spans. Populate previous/next Child IDs only within the same logical structure. Route tables, figures, and formulas through their dedicated chunk builders.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_semantic_chunking.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/semantic_chunking.py tests/test_semantic_chunking.py
git commit -m "Add semantic parent child chunking"
```

### Task 8: Generate Mandatory Batched Contextual Prefixes

**Files:**
- Create: `src/app/services/contextualization.py`
- Modify: `src/app/services/ai.py`
- Create: `tests/test_contextualization.py`

- [ ] **Step 1: Write batch and strict-failure tests**

```python
def test_contextualizer_uses_document_parent_and_child_context(fake_llm) -> None:
    result = Contextualizer(fake_llm).contextualize(document_context(), child_chunks(2))
    prompt = fake_llm.prompts[0]
    assert "论文标题" in prompt
    assert "Experiments > Main Results" in prompt
    assert "当前 Parent" in prompt
    assert result[0].embedding_text == result[0].contextual_prefix + "\n\n" + result[0].text


def test_contextualizer_rejects_missing_child_after_retries(fake_llm_missing_id) -> None:
    with pytest.raises(ContextualizationFailed) as exc:
        Contextualizer(fake_llm_missing_id, max_retries=2).contextualize(document_context(), child_chunks(3))
    assert exc.value.failed_child_ids


def test_contextual_prefix_never_becomes_citation_text(fake_llm) -> None:
    child = Contextualizer(fake_llm).contextualize(document_context(), child_chunks(1))[0]
    assert child.text not in child.contextual_prefix
```

- [ ] **Step 2: Run tests and verify the contextualizer is absent**

Run: `pytest tests/test_contextualization.py -v`

Expected: FAIL with missing module/classes.

- [ ] **Step 3: Add a dedicated structured Ollama call**

Define response models:

```python
class ContextualPrefixItem(BaseModel):
    child_id: str
    prefix: str


class ContextualPrefixBatch(BaseModel):
    items: list[ContextualPrefixItem]
```

Use the independent base URL, model, timeout, and prompt version. The prompt requests one or two Chinese sentences, preserves English proper nouns, and forbids specific numeric values not needed to identify a table/metric.

- [ ] **Step 4: Implement validation, checkpoints, and retry splitting**

Validate exact Child ID sets, uniqueness, non-empty prefixes, sentence count, length, and source-entity consistency. Attempt batch 12, retry after 2 seconds with correction feedback, then split the failed batch after 8 seconds. Persist successful batch outputs through a callback. After all attempts, raise `ContextualizationFailed`; never set `embedding_text=text` as a fallback.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_contextualization.py tests/test_ai_client.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/contextualization.py src/app/services/ai.py tests/test_contextualization.py tests/test_ai_client.py
git commit -m "Add strict batched contextualization"
```

### Task 9: Implement Resumable Ingestion Stages and Named Queues

**Files:**
- Create: `src/app/services/ingestion_stages.py`
- Modify: `src/app/services/pipeline.py`
- Modify: `src/app/services/queue.py`
- Modify: `src/app/workers/jobs.py`
- Modify: `deploy/systemd/knowledge-agent-dev-worker.service`
- Modify: `deploy/systemd/knowledge-agent-test-worker.service`
- Create: `tests/test_ingestion_stages.py`
- Modify: `tests/test_deployment_config.py`

- [ ] **Step 1: Write stage checkpoint and resume tests**

```python
def test_failed_contextualization_resumes_without_reparsing(stage_runner, fakes) -> None:
    fakes.contextualizer.fail_once = True
    with pytest.raises(ContextualizationFailed):
        stage_runner.run_until_blocked("doc-1", "v1")
    assert fakes.parser.calls == 1
    stage_runner.run_until_blocked("doc-1", "v1")
    assert fakes.parser.calls == 1
    assert stage_runner.version("doc-1", "v1").status == "ready_to_activate"


def test_table_repair_failure_never_enqueues_embedding(stage_runner, fakes) -> None:
    fakes.table_repair.always_fail = True
    stage_runner.run_stage("doc-1", "v1", "repair")
    assert stage_runner.version("doc-1", "v1").status == "table_repair_failed"
    assert not fakes.queue.has_stage("embedding")
```

- [ ] **Step 2: Run tests and verify stage orchestration is absent**

Run: `pytest tests/test_ingestion_stages.py -v`

Expected: FAIL with missing `IngestionStageRunner`.

- [ ] **Step 3: Implement idempotent stage handlers**

Each handler reads its input artifact/checkpoint and writes a complete output before transitioning status. Stages are `parse`, `repair`, `canonicalize`, `semantic_split`, `contextualize`, `embed`, `index`, and `activate`. Re-running a completed stage returns its existing output without duplicate rows or model calls.

- [ ] **Step 4: Add named RQ queues and one stage job entry point**

```python
STAGE_QUEUES = {
    "parse": "ingest.parse",
    "repair": "ingest.repair",
    "canonicalize": "ingest.canonicalize",
    "semantic_split": "ingest.semantic_split",
    "contextualize": "ingest.contextualize",
    "embed": "ingest.embed",
    "index": "ingest.index",
    "activate": "ingest.activate",
}


def run_ingestion_stage(document_id: str, version_key: str, stage: str) -> str:
    db = SessionLocal()
    try:
        return IngestionStageRunner(db).run(document_id, version_key, stage)
    finally:
        db.close()
```

The job enqueues only the next stage after committing the current checkpoint. Configure one worker process to consume all stage queues in order, preserving GPU concurrency 1.

- [ ] **Step 5: Convert `IngestionPipeline.process_document()` into a compatibility entry point**

It creates the version and starts/resumes staged ingestion. Preserve current run progress API fields while adding stage-specific status and errors.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_ingestion_stages.py tests/test_pipeline_sac_kg.py tests/test_deployment_config.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app/services/ingestion_stages.py src/app/services/pipeline.py src/app/services/queue.py src/app/workers/jobs.py deploy/systemd/knowledge-agent-dev-worker.service deploy/systemd/knowledge-agent-test-worker.service tests/test_ingestion_stages.py tests/test_pipeline_sac_kg.py tests/test_deployment_config.py
git commit -m "Add resumable canonical ingestion stages"
```

### Task 10: Build Version-Scoped Contextual Embeddings and Atomic Activation

**Files:**
- Modify: `src/app/services/vector_store.py`
- Modify: `src/app/services/ingestion_stages.py`
- Modify: `src/app/services/parse_versions.py`
- Modify: `tests/test_vector_retrieval.py`
- Create: `tests/test_canonical_indexing.py`

- [ ] **Step 1: Write active-version isolation tests**

```python
def test_pgvector_search_returns_only_active_parse_version(pgvector_store, db, document) -> None:
    seed_versioned_vectors(db, document, old="old-v", new="new-v")
    document.active_parse_version = "new-v"
    db.commit()
    hits = pgvector_store.search([1.0, 0.0], limit=10, document_ids=[document.id])
    assert hits
    assert {hit.parse_version for hit in hits} == {"new-v"}


def test_activation_rejects_incomplete_contextual_embeddings(stage_runner) -> None:
    version = version_with_child_counts(total=10, contextualized=10, embedded=9, indexed=9)
    with pytest.raises(ActivationError, match="embedding completeness"):
        stage_runner.activate(version)
```

- [ ] **Step 2: Run tests and verify the current vector index leaks versions**

Run: `pytest tests/test_canonical_indexing.py tests/test_vector_retrieval.py -v`

Expected: FAIL because vector hits have no parse version filter.

- [ ] **Step 3: Version vector writes and active queries**

Add `parse_version` to `ChunkVector` and `VectorHit`. Write only Child chunks with complete contextual prefixes and correct 2560 dimensions. Query pgvector by joining documents and requiring `index.parse_version = documents.active_parse_version`. Apply equivalent active filters to SQLite-vec and JSON fallbacks.

- [ ] **Step 4: Add activation invariants**

Before activation, compare retrievable Child count, contextualized count, embedded JSON count, pgvector count, valid source span count, and artifact validation. Require equality and non-zero retrievable content. Commit `active_parse_version` only after all checks pass.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_canonical_indexing.py tests/test_vector_retrieval.py tests/test_parse_versions.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/services/vector_store.py src/app/services/ingestion_stages.py src/app/services/parse_versions.py tests/test_vector_retrieval.py tests/test_canonical_indexing.py
git commit -m "Index contextual embeddings by parse version"
```

### Task 11: Retrieve Child Hits and Expand to Parent Evidence

**Files:**
- Modify: `src/app/services/search.py`
- Modify: `src/app/services/rag_adapter.py`
- Modify: `src/app/schemas/common.py`
- Create: `tests/test_canonical_retrieval.py`

- [ ] **Step 1: Write Parent expansion and citation-boundary tests**

```python
def test_child_hit_expands_parent_but_cites_original_child(db, canonical_document_rows) -> None:
    response = QueryService(db).retrieve_evidence("project", "SAC-KG 在 OIE2016 的主结果", limit=5)
    item = response.items[0]
    assert item.parent_chunk_id
    assert "Experiments" in item.context_text
    assert item.excerpt == canonical_document_rows.child.text
    assert canonical_document_rows.child.contextual_prefix not in item.excerpt


def test_reference_chunks_do_not_enter_normal_retrieval(db, reference_rows) -> None:
    contexts = QueryService(db)._search_source_chunks("related method", reference_rows.project_id, [], limit=10)
    assert all(context.citation.chunk_id != reference_rows.reference_child.id for context in contexts)


def test_table_hit_includes_caption_header_and_row_group(db, table_rows) -> None:
    contexts = QueryService(db)._search_source_chunks("OIE2016 F1", table_rows.project_id, [], limit=5)
    text = contexts[0].prompt_text
    assert "Table 5" in text
    assert "Dataset | Model | F1 | AUC" in text
    assert "OIE2016" in text
```

- [ ] **Step 2: Run tests and verify retrieval has no Parent expansion**

Run: `pytest tests/test_canonical_retrieval.py -v`

Expected: FAIL for missing `parent_chunk_id` and `context_text`.

- [ ] **Step 3: Add active Child candidate retrieval**

Filter by active parse version, `chunk_role='child'`, `block_type != 'reference'`, and optional document scope. Use `embedding_text` only through its vector; never insert it into citation excerpts.

- [ ] **Step 4: Add type-aware Parent and neighbor expansion**

Build an `ExpandedEvidence` object with Child citation text, Parent context, selected neighbor text, and source spans. Ordinary facts get Parent; overview/comparison may add neighbors; table/Figure/Formula use their dedicated expansion rules. Enforce a token budget before `_draft_answer()`.

- [ ] **Step 5: Extend schemas without breaking existing clients**

Add optional `parse_version`, `parent_chunk_id`, `block_type`, `source_spans`, `asset_id`, `table_id`, `figure_id`, and `formula_id` to `Citation` and `EvidenceItem`. Keep existing fields and response shapes valid.

- [ ] **Step 6: Run retrieval regression**

Run: `pytest tests/test_canonical_retrieval.py tests/test_query_service.py tests/test_vector_retrieval.py tests/test_rag_adapter.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app/services/search.py src/app/services/rag_adapter.py src/app/schemas/common.py src/app/schemas/agent.py tests/test_canonical_retrieval.py tests/test_query_service.py tests/test_vector_retrieval.py tests/test_rag_adapter.py
git commit -m "Expand contextual child hits to source parents"
```

### Task 12: Add Canonical Parse and Source-Location APIs

**Files:**
- Modify: `src/app/schemas/common.py`
- Modify: `src/app/api/routes.py`
- Create: `tests/test_canonical_api.py`

- [ ] **Step 1: Write API contract and path-safety tests**

```python
def test_current_parse_markdown_and_download(client, active_parse) -> None:
    status = client.get(f"/api/documents/{active_parse.document_id}/parse").json()
    assert status["version"] == active_parse.version_key
    markdown = client.get(f"/api/documents/{active_parse.document_id}/parse/markdown")
    assert markdown.status_code == 200
    assert "# Paper" in markdown.json()["markdown"]
    download = client.get(f"/api/documents/{active_parse.document_id}/parse/download")
    assert download.headers["content-disposition"].startswith("attachment;")


def test_location_route_never_serves_other_version_or_path(client, active_parse, old_chunk) -> None:
    response = client.get(f"/api/documents/{active_parse.document_id}/citations/{old_chunk.id}/location")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests and verify routes are absent**

Run: `pytest tests/test_canonical_api.py -v`

Expected: FAIL with HTTP 404.

- [ ] **Step 3: Implement read-only current-version endpoints**

Return only the active parse version. Resolve artifact paths with the same containment checks used by document file serving. The status response includes parser, progress stage, quality summary, repair pages, warning counts, and download availability. Do not expose contextual prefixes or historical versions.

- [ ] **Step 4: Implement citation location responses**

Return source spans plus a source URL appropriate to PDF, DOCX, HTML, or text. Reject a chunk not belonging to the document's active parse version.

- [ ] **Step 5: Run API tests**

Run: `pytest tests/test_canonical_api.py tests/test_api_routes.py tests/test_auth.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app/schemas/common.py src/app/api/routes.py tests/test_canonical_api.py tests/test_api_routes.py
git commit -m "Expose canonical parse artifacts and locations"
```

### Task 13: Add Parse Progress, Markdown View, Download, and Highlight UI

**Files:**
- Modify: `src/app/static/index.html`
- Modify: `tests/test_static_frontend.py`

- [ ] **Step 1: Write static frontend contracts**

```python
def test_frontend_exposes_current_parse_actions_only() -> None:
    html = _read_frontend()
    assert "查看解析 Markdown" in html
    assert "下载 canonical.md" in html
    assert "/parse/markdown" in html
    assert "/parse/download" in html
    assert "历史解析版本" not in html
    assert "contextual_prefix" not in html


def test_source_view_uses_location_spans_for_highlighting() -> None:
    html = _read_frontend()
    assert "/citations/" in html
    assert "source_spans" in html
    assert "highlightSourceSpans" in html
```

- [ ] **Step 2: Run tests and verify controls are missing**

Run: `pytest tests/test_static_frontend.py -k 'parse or highlight' -v`

Expected: FAIL.

- [ ] **Step 3: Implement parse state and actions**

Add icon buttons with tooltips for view/download. The parse panel displays current version, parser, stage, percent, quality status, repair pages, and warnings. Render canonical Markdown as text/structured DOM without `innerHTML` injection.

- [ ] **Step 4: Implement source highlighting**

On citation click, fetch the location endpoint, open the current source drawer, navigate to the page/element/line range, and render one or more bbox overlays. Keep PDF and Markdown viewports independently scrollable. Mark AI-only Figure observations as image analysis, not source text.

- [ ] **Step 5: Run static and JavaScript syntax tests**

Run: `pytest tests/test_static_frontend.py -v`

Expected: PASS, including `node --check` extraction.

- [ ] **Step 6: Perform browser verification**

Run the local API against canonical fixtures, then verify with Playwright at 1440x900 and 390x844:

- Markdown opens and scrolls.
- Download returns the current file.
- PDF citation navigates and highlights bbox.
- Long source text and overlays do not overlap controls.
- No historical-version or contextual-prefix UI is visible.

- [ ] **Step 7: Commit**

```bash
git add src/app/static/index.html tests/test_static_frontend.py
git commit -m "Add canonical parse and citation UI"
```

### Task 14: Add Rebuild and Acceptance Tooling

**Files:**
- Create: `scripts/rebuild_canonical_index.py`
- Create: `scripts/evaluate_canonical_retrieval.py`
- Create: `tests/test_rebuild_canonical_index.py`
- Create: `tests/test_canonical_acceptance.py`
- Create: `docs/query_acceptance/canonical_ingestion_v1.json`

- [ ] **Step 1: Write dry-run and all-or-nothing tests**

```python
def test_rebuild_dry_run_does_not_mutate_documents(db, documents, tmp_path: Path) -> None:
    report = rebuild(db, artifact_root=tmp_path, dry_run=True)
    assert report.total == len(documents)
    assert all(document.active_parse_version is None for document in documents)


def test_rebuild_blocks_testing_when_one_document_fails(fake_runner, documents) -> None:
    fake_runner.fail_document_id = documents[-1].id
    report = rebuild_all(fake_runner, documents)
    assert report.ready_for_acceptance is False
    assert report.failed_document_ids == [documents[-1].id]
```

- [ ] **Step 2: Run tests and verify scripts are absent**

Run: `pytest tests/test_rebuild_canonical_index.py tests/test_canonical_acceptance.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement the maintenance rebuild CLI**

Arguments must include `--dry-run`, `--resume`, `--document-id`, `--report`, `--delete-old-after-acceptance`, and an explicit `--confirm-delete-old-data` guard. Dry-run reports source availability and planned versions without writing. Resume skips completed stages using checkpoints.

- [ ] **Step 4: Implement the acceptance runner**

Read the existing full30 artifacts and the new JSON case set. Record parse completeness, contextualization/embedding counts, pgvector counts, table accuracy, source-location validity, recall@5/10, answer/citation result, P50/P95 retrieval latency, and failure attribution. Exit non-zero if any strict gate is below 100% or any required query case fails.

- [ ] **Step 5: Add concrete canonical cases**

The JSON file must include current documents and questions for narrative retrieval, Chinese-to-English retrieval, long and cross-page tables, Figure, Formula, appendix, reference exclusion, contextual-prefix citation exclusion, and bbox location. Each case contains expected document ID or stable source identity, block type, page label, required terms, and location requirement.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_rebuild_canonical_index.py tests/test_canonical_acceptance.py tests/test_retrieval_acceptance.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/rebuild_canonical_index.py scripts/evaluate_canonical_retrieval.py tests/test_rebuild_canonical_index.py tests/test_canonical_acceptance.py docs/query_acceptance/canonical_ingestion_v1.json
git commit -m "Add canonical rebuild acceptance tooling"
```

### Task 15: Full Regression, Test-Environment Rebuild, Acceptance, and Old-Data Deletion

**Files:**
- Modify: `docs/work.md`
- Modify: `deploy/internal-pilot.md`
- Generated report: `runtime/canonical-rebuild-report.json`
- Generated report: `runtime/canonical-acceptance-report.json`

- [ ] **Step 1: Run the complete local regression before touching the server**

Run: `pytest -q`

Expected: all tests pass with zero failures.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 2: Deploy code without starting the rebuild**

Update the test environment code and dependencies, apply Alembic through `a4e2c7f90120`, and verify the API/worker starts with existing data still untouched.

Run on the test server:

```bash
alembic current
python -m pytest tests/test_contextual_ingestion_config.py tests/test_migrations.py tests/test_deployment_config.py -q
```

Expected: migration head is `a4e2c7f90120`; focused tests pass.

- [ ] **Step 3: Enter maintenance mode and run a dry run**

Disable upload/query routes through a maintenance setting, keep health/progress endpoints available, then run:

```bash
python scripts/rebuild_canonical_index.py --dry-run --report runtime/canonical-rebuild-dry-run.json
```

Expected: every current source file is present and every document has a planned version; no database or artifact mutation.

- [ ] **Step 4: Run the full historical rebuild**

```bash
python scripts/rebuild_canonical_index.py --resume --report runtime/canonical-rebuild-report.json
```

Expected strict report fields:

```text
failed_documents = 0
contextual_prefix_completeness = 1.0
embedding_completeness = 1.0
table_validation_rate = 1.0
source_span_validity = 1.0
artifact_link_validity = 1.0
ready_for_acceptance = true
```

Do not continue if any field differs.

- [ ] **Step 5: Run full30 and canonical acceptance**

Run the existing full30 command used by the project, then:

```bash
python scripts/evaluate_canonical_retrieval.py --cases docs/query_acceptance/canonical_ingestion_v1.json --report runtime/canonical-acceptance-report.json
```

Expected: all strict parse/location cases pass; full30 does not regress below its established gate; every failure, if the existing suite permits any, has attribution and no source/citation regression.

- [ ] **Step 6: Delete old RAG data only after acceptance succeeds**

```bash
python scripts/rebuild_canonical_index.py --delete-old-after-acceptance --confirm-delete-old-data --report runtime/canonical-rebuild-report.json
```

Expected:

- Old DocumentChunk rows and pgvector rows are removed.
- Old parse directories and MinerU temp directories are removed.
- Original source files remain.
- Active canonical artifacts, chunks, contextual prefixes, embeddings, and vectors remain.
- No offline old-data backup is retained, per the approved design.

- [ ] **Step 7: Re-run integrity and query acceptance after deletion**

```bash
python scripts/evaluate_canonical_retrieval.py --cases docs/query_acceptance/canonical_ingestion_v1.json --report runtime/canonical-acceptance-post-cleanup.json
```

Expected: same strict pass result as before cleanup; database queries confirm no `legacy` parse-version chunks or vectors.

- [ ] **Step 8: Exit maintenance mode and verify the user workflow**

Verify authenticated page load, upload queueing, parse progress, current Markdown view/download, Agent query, citation navigation/highlight, and health/queue status. Confirm no pending/started/failed stage jobs remain.

- [ ] **Step 9: Record the release**

Update `docs/work.md` and `deploy/internal-pilot.md` with the deployed commit, document counts, chunk counts, repair counts, acceptance report paths, cleanup result, model versions, P50/P95, and current service URLs.

- [ ] **Step 10: Commit release documentation**

```bash
git add docs/work.md deploy/internal-pilot.md
git commit -m "Record canonical ingestion rollout"
```

---

## Final Verification Checklist

- [ ] No parser path contains `[:4000]` or another silent source truncation.
- [ ] MinerU runs before pypdf text fallback and pypdf single-page failures are non-fatal.
- [ ] PDF, DOCX, HTML/HTM, TXT, and Markdown emit the same canonical model.
- [ ] `canonical.md`, `manifest.json`, `blocks.jsonl`, structured evidence JSON, and assets are self-contained.
- [ ] Tables are complete, structured, row-grouped, and strict-gated.
- [ ] Figure/Formula generated analysis is separate from canonical source text.
- [ ] Parent/Child limits and semantic boundaries are tested.
- [ ] Every retrievable Child has a contextual prefix and contextualized embedding.
- [ ] Contextual prefixes never appear in citation excerpts.
- [ ] Retrieval searches only active-version Child vectors and expands to Parent context.
- [ ] Source locations highlight PDF bbox or the equivalent locator for other formats.
- [ ] Redis stage retries resume from checkpoints and GPU work is serialized.
- [ ] Current-version Markdown view/download works; history and rollback UI do not exist.
- [ ] Historical rebuild is 100% complete before testing starts.
- [ ] full30 and canonical acceptance pass before maintenance mode ends.
- [ ] Old RAG data and temp outputs are deleted only after post-build acceptance.
- [ ] Original uploaded files remain intact.
