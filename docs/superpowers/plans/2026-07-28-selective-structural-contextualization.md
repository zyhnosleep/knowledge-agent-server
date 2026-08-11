# Selective Structural Contextualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed every retrievable Child directly without an ingestion-time LLM call, with strict raw-embedding integrity gates and a clean `canonical-v4` artifact version.

**Architecture:** The policy module classifies every retrievable Child as direct embedding. The pipeline carries semantic drafts through the contextualize stage without constructing the LLM service, embeds exact `child.text`, and activates only when raw embedding completeness is 100%. Compatibility reports retain contextual metrics with an empty eligible set.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy, PostgreSQL/pgvector, Ollama, pytest, systemd user services.

---

## Hard Boundaries

- Modify and run only development: `/home/zhangyh/knowledge-agent-dev`, API `8002`, Ollama `11435`, database `knowledge_agent_dev`.
- Test remains the unchanged old-version baseline at API `8001` and database `knowledge_agent_test`.
- Development uses GPU0 only. Do not stop, inspect through intrusive tooling, or schedule work on GPU1.
- Preserve every `canonical-v1` parse version, chunk, vector, artifact, MinerU output, and source file.
- Do not use either old-data deletion flag.
- Do not run `opls5`, full-17, or retrieval comparison until the new `ff14sb` canary passes.

### Task 1: Define The Child Embedding Policy

**Files:**
- Create: `src/app/services/contextualization_policy.py`
- Create: `tests/test_contextualization_policy.py`

- [ ] **Step 1: Write failing classification and validation tests**

Create tests that assert:

```python
def test_only_structured_evidence_requires_llm_context() -> None:
    assert CONTEXTUALIZED_BLOCK_TYPES == frozenset()
    for block_type in ("narrative", "caption", "appendix", "table", "figure", "formula"):
        assert requires_contextualization(block_type) is False


def test_plain_child_requires_exact_raw_embedding_text_and_no_context_fields() -> None:
    child = _child(block_type="narrative", text="Raw child", embedding_text="Raw child")
    assert valid_plain_embedding(child) is True
    child["embedding_text"] = "Section context\n\nRaw child"
    assert valid_plain_embedding(child) is False


def test_no_child_type_accepts_contextualized_embedding() -> None:
    child = _contextualized_child(block_type="figure")
    assert valid_contextualized_embedding(child) is False
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_contextualization_policy.py -q
```

Expected: collection fails because `app.services.contextualization_policy` does not exist.

- [ ] **Step 3: Implement the minimal policy module**

Implement these public contracts:

```python
CONTEXTUALIZED_BLOCK_TYPES = frozenset()
PLAIN_EMBEDDING_BLOCK_TYPES = frozenset({"narrative", "caption", "appendix", "table", "figure", "formula"})
RETRIEVABLE_BLOCK_TYPES = CONTEXTUALIZED_BLOCK_TYPES | PLAIN_EMBEDDING_BLOCK_TYPES


def requires_contextualization(block_type: str) -> bool:
    if block_type not in RETRIEVABLE_BLOCK_TYPES:
        raise ValueError(f"unsupported retrievable block type: {block_type}")
    return block_type in CONTEXTUALIZED_BLOCK_TYPES


def valid_contextualized_embedding(chunk: object) -> bool:
    prefix = _field(chunk, "contextual_prefix")
    text = _field(chunk, "text")
    return bool(
        requires_contextualization(str(_field(chunk, "block_type")))
        and prefix
        and _field(chunk, "contextualization_model")
        and _field(chunk, "contextualization_version")
        and _field(chunk, "contextualization_prompt_version")
        and _field(chunk, "contextualized_at") is not None
        and _field(chunk, "embedding_text") == f"{prefix}\n\n{text}"
    )


def valid_plain_embedding(chunk: object) -> bool:
    return bool(
        not requires_contextualization(str(_field(chunk, "block_type")))
        and all(_field(chunk, name) is None for name in CONTEXT_FIELDS)
        and _field(chunk, "embedding_text") == _field(chunk, "text")
    )
```

`_field` must support both mappings from stage JSON and SQLAlchemy/Pydantic objects. Unknown block types must fail closed.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
python -m pytest tests/test_contextualization_policy.py -q
```

Expected: all policy tests pass.

### Task 2: Bypass The LLM For Every Child

**Files:**
- Modify: `src/app/services/pipeline.py`
- Modify: `tests/test_ingestion_stages.py`

- [ ] **Step 1: Write a failing mixed contextualization-stage test**

Build one Parent and six Children, one per retrievable block type. Replace `ContextualizationService` with a fake whose constructor raises. Assert:

```python
assert output["child_count"] == 6
assert output["contextualized_child_count"] == 0
assert output["plain_child_count"] == 6

artifact = json.loads(Path(output["artifact_path"]).read_text("utf-8"))
by_type = {item["block_type"]: item for item in artifact if item["chunk_role"] == "child"}
for block_type in ("narrative", "caption", "appendix", "table", "figure", "formula"):
    assert by_type[block_type]["embedding_text"] == by_type[block_type]["text"]
    assert "contextual_prefix" not in by_type[block_type]
```

Add a second test with only narrative Children and assert the fake service is never constructed or called.

- [ ] **Step 2: Verify RED**

Run the two new node IDs with `pytest -q`. Expected: the fake currently receives all six Children, and the no-structured test constructs/calls the service.

- [ ] **Step 3: Implement partition, contextualize, and stable merge**

In `_run_contextualize_stage`:

```python
structured = [child for child in children if requires_contextualization(child.block_type)]
plain = [child for child in children if not requires_contextualization(child.block_type)]
contextualized = (
    ContextualizationService().contextualize(
        document=document_context,
        children=structured,
        parents=parents,
    )
    if structured
    else []
)
contextualized_by_id = {child.local_id: child for child in contextualized}
combined = [
    (
        contextualized_by_id[draft.local_id].model_dump(mode="json")
        if draft.chunk_role == "child" and requires_contextualization(draft.block_type)
        else draft.model_dump(mode="json")
    )
    for draft in drafts
]
```

Reject duplicate/missing structured results and preserve the exact original draft order. Report total, contextualized, and plain Child counts.

- [ ] **Step 4: Verify GREEN and focused regressions**

Run:

```powershell
python -m pytest tests/test_ingestion_stages.py -k "contextualize or production_staged_parse" -q
python -m pytest tests/test_contextualization.py -q
```

Expected: all selected tests pass; direct `ContextualizationService` validation behavior remains unchanged.

### Task 3: Embed And Persist The Mixed Child Inventory

**Files:**
- Modify: `src/app/services/pipeline.py`
- Modify: `tests/test_ingestion_stages.py`
- Modify: `tests/test_canonical_indexing.py`

- [ ] **Step 1: Write failing embed and index tests**

Change the embed-stage fixture to include plain narrative, table, and figure Children. Assert Ollama receives, in Child order:

```python
assert calls == [["Raw narrative", "| A | B |", "Figure 1"]]
```

Add failures for any Child with modified `embedding_text` or contextualization fields. Add an index test proving all Child types persist with vectors and none has contextualization columns.

- [ ] **Step 2: Verify RED**

Run the new node IDs. Expected: `_run_embed_stage` rejects the plain Child with `Embedding requires contextualized child chunks`, and index persistence rejects it as `ContextualizedChunk`.

- [ ] **Step 3: Implement policy validation in embed and persistence**

Replace the all-context requirement with:

```python
for item in children:
    if requires_contextualization(str(item.get("block_type"))):
        valid = valid_contextualized_embedding(item)
    else:
        valid = valid_plain_embedding(item)
    if not valid:
        raise RuntimeError("Embedding input violates the Child contextualization policy.")
```

In `_persist_versioned_chunks`, validate every Child with `valid_plain_embedding` and `ChunkDraft`. Use the common draft interface for parent linkage, vector creation, and previous/next linkage. Do not synthesize context fields.

- [ ] **Step 4: Verify GREEN**

Run:

```powershell
python -m pytest tests/test_ingestion_stages.py -k "embed_stage or production_staged_parse" -q
python -m pytest tests/test_canonical_indexing.py -q
```

Expected: mixed inventories embed, index, and preserve old active versions until activation.

### Task 4: Make Activation And Reports Policy-Aware

**Files:**
- Modify: `src/app/services/pipeline.py`
- Modify: `scripts/rebuild_canonical_index.py`
- Modify: `scripts/evaluate_canonical_retrieval.py`
- Modify: `tests/test_canonical_indexing.py`
- Modify: `tests/test_rebuild_canonical_index.py`
- Modify: `tests/test_canonical_acceptance.py`

- [ ] **Step 1: Write failing activation and metric tests**

Add a successful activation fixture with plain narrative, table, and figure Children. Add failures for altered embedding text or contextualization fields. In rebuild metrics assert:

```python
assert metrics["contextualization_eligible_children"] == 0
assert metrics["contextualized_children"] == 0
assert metrics["plain_embedding_children"] == 3
assert metrics["contextual_prefix_completeness"] == 1.0
assert metrics["plain_embedding_completeness"] == 1.0
```

Add a no-structured document case expecting `contextual_prefix_completeness == 1.0`. Extend the canonical acceptance strict-gate test so `plain_embedding_completeness=0.0` is reported as a failed gate.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest tests/test_canonical_indexing.py tests/test_rebuild_canonical_index.py tests/test_canonical_acceptance.py -q
```

Expected: mixed activation fails the old all-context equality check and new report fields are missing.

- [ ] **Step 3: Implement explicit activation counts**

Compute separate sets and counts:

```python
eligible = [chunk for chunk in children if requires_contextualization(chunk.block_type)]
plain = [chunk for chunk in children if not requires_contextualization(chunk.block_type)]
contextualized_count = sum(valid_contextualized_embedding(chunk) for chunk in eligible)
plain_embedding_count = sum(valid_plain_embedding(chunk) for chunk in plain)
```

Activation must require `contextualized_count == len(eligible)`, `plain_embedding_count == len(plain)`, and the existing embedded/indexed/span/artifact counts equal all retrievable Children. Return explicit eligible and plain counts in the activation output.

- [ ] **Step 4: Implement report fields and strict gates**

Add these `RebuildReport` fields:

```python
contextualization_eligible_children: int = 0
plain_embedding_children: int = 0
plain_embedding_completeness: float = 0.0
```

Use `_ratio(contextualized, eligible_count, empty_is_complete=True)` and `_ratio(plain_valid, plain_count, empty_is_complete=True)`. Add `plain_embedding_completeness` to both `STRICT_REBUILD_METRICS` and `STRICT_INTEGRITY_GATES`.

- [ ] **Step 5: Verify GREEN**

Run the three files again. Expected: all tests pass and incomplete plain or structured policy data blocks acceptance.

### Task 5: Start A Clean Canonical V4 And Update Task 15 Documentation

**Files:**
- Modify: `src/app/core/config.py`
- Modify: `tests/test_rebuild_canonical_index.py`
- Modify: `docs/superpowers/plans/2026-07-28-task15-development-rebuild-comparison.md`
- Modify: `docs/work.md`

- [ ] **Step 1: Write a failing default-version test**

Update the dry-run expectation to require every planned version to start with `canonical-v4-`. Verify it fails while the default is still `canonical-v3`.

- [ ] **Step 2: Bump only the new development code default**

Change:

```python
canonical_pipeline_version: str = Field(
    default="canonical-v4", alias="CANONICAL_PIPELINE_VERSION"
)
```

Do not edit the test environment or its runtime environment. Verify the development `runtime/app.env` does not override `CANONICAL_PIPELINE_VERSION`; if it does, stop and remove only that development override after recording its prior value.

- [ ] **Step 3: Amend Task 15 gates and progress notes**

Record that V1/V2/V3 artifacts are preserved and inactive, `canonical-v4` is the all-direct policy, every Child embedding is raw, and the contextualization eligible count is zero. Add `plain_embedding_completeness=1.0` to every canary/full rebuild gate.

- [ ] **Step 4: Run the full focused local suite**

Run:

```powershell
python -m pytest tests/test_contextualization_policy.py tests/test_contextualization.py tests/test_ingestion_stages.py tests/test_canonical_indexing.py tests/test_rebuild_canonical_index.py tests/test_canonical_acceptance.py -q
git diff --check
```

Expected: all tests pass and no whitespace errors exist.

### Task 6: Deploy Development Only And Re-run `ff14sb`

**Files:**
- Deploy only changed source/script files to `/home/zhangyh/knowledge-agent-dev`
- Generate: `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-ff14sb-v4.json`

- [ ] **Step 1: Capture and compare deployment hashes**

Record local SHA-256 values for every changed source and script. Copy only those files to development, then require remote hashes to match. Do not copy to `/home/zhangyh/knowledge-agent-test`.

- [ ] **Step 2: Run focused tests on development**

With `PYTHONPATH=src`, run the same focused test set in development. Require all tests to pass before starting a rebuild.

- [ ] **Step 3: Reconfirm execution boundaries**

Require no rebuild process, development worker `CUDA_VISIBLE_DEVICES=0`, test counts unchanged, source SHA unchanged, and no `CANONICAL_PIPELINE_VERSION` override in the development environment.

- [ ] **Step 4: Run the clean V4 canary on GPU0**

```bash
cd /home/zhangyh/knowledge-agent-dev
set -a
. runtime/app.env
set +a
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=src .venv/bin/python scripts/rebuild_canonical_index.py --resume \
  --document-id 098a4ce8-d772-46a5-ae67-e0594a355460 \
  --report runtime/task15/canary-ff14sb-v4.json
```

Require the selected version to start with `canonical-v4-`; no V1/V2/V3 stage may be resumed or overwritten.

- [ ] **Step 5: Strictly inspect the report and artifacts**

Require:

```text
failed_documents = 0
contextual_prefix_completeness = 1.0
plain_embedding_completeness = 1.0
embedding_completeness = 1.0
pgvector_completeness = 1.0
table_validation_rate = 1.0
source_span_validity = 1.0
artifact_link_validity = 1.0
ready_for_acceptance = true
```

Load persisted Children and prove that no row has contextual fields, all Children have `embedding_text == text`, and every Child has a vector. Reconfirm MinerU is primary, page coverage is 50/50, and exactly pages 7, 8, 9, 15, 16, 45, 46, 48, and 50 use `pypdf_text_layer` with `mineru_page_missing`.

- [ ] **Step 6: Continue Task 15 only after pass**

If the canary passes, update the main Task 15 plan and proceed to the `opls5` table canary. If any strict check fails, keep maintenance enabled, stop before `opls5`, and diagnose the exact stage with a failing regression test.
