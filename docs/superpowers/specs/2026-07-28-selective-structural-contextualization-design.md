# Selective Structural Contextualization Design

**Date:** 2026-07-28

**Status:** Approved final approach

## 1. Problem

The current canonical ingestion pipeline sends every retrievable Child chunk to
`qwen3.5:9b` for a generated contextual prefix. On the real `ff14sb` paper this
produced 465 chunks and more than 120 generation requests before the canary was
stopped. Long prompts were truncated at the Ollama context limit, which caused
validation retries and made contextualization dominate rebuild time.

The final user-approved policy removes generated contextual prefixes from the
ingestion path. Every retrievable Child is embedded directly as its original
text. Figure and formula Children already contain the parser-produced
description or representation that belongs in the searchable document.

## 2. Approved Policy

| Child block type | LLM contextualization | Embedding input |
| --- | --- | --- |
| `table` | forbidden | `child.text` |
| `figure` | forbidden | `child.text` |
| `formula` | forbidden | `child.text` |
| `narrative` | forbidden | `child.text` |
| `caption` | forbidden | `child.text` |
| `appendix` | forbidden | `child.text` |

Parent chunks remain unembedded. They continue to provide hierarchy, expansion,
and display context for retrievable Child chunks.

The policy is explicit and fail-closed. No Child may carry contextualization
fields or a modified embedding input.

## 3. Data Flow

```text
semantic Child chunks
  -> validate embedding_text == child.text
  -> embed every Child
  -> persist every Child and its vector
  -> activation integrity gate
```

`ContextualizationService` is no longer constructed by the ingestion pipeline.
It remains an isolated utility for compatibility until a separate cleanup task
decides whether other callers still require it.

## 4. Artifact And Persistence Contract

The contextualization-stage artifact contains a mixed Child inventory:

- All Children validate as `ChunkDraft`; their contextualization fields must
  be absent and `embedding_text` must equal `text` exactly.
- Parent chunks remain `ChunkDraft` values and receive no vector.
- Chunk IDs, source spans, source block IDs, ordering links, metadata, and
  original `text` remain unchanged for both paths.

The embedding and index stages validate the policy from `block_type` rather than
accepting arbitrary missing fields. This prevents an LLM failure from being
mistaken for an intentional raw Child embedding.

## 5. Integrity Metrics

`contextual_prefix_completeness` remains in reports for compatibility. Its
eligible denominator is always empty, so the value must be `1.0`.

The rebuild report adds:

- `contextualization_eligible_children`
- `plain_embedding_children`
- `plain_embedding_completeness`

Strict acceptance requires:

- `contextual_prefix_completeness == 1.0`
- `plain_embedding_completeness == 1.0`
- `embedding_completeness == 1.0`
- `pgvector_completeness == 1.0` when pgvector is active
- the existing table, source-span, and artifact checks remain `1.0`

The activation gate separately proves that the eligible set is empty, every
Child has unmodified raw-text embedding input, and every Child has a valid
vector and source span.

## 6. Failure And Resume Behavior

- Children never call the contextualization model and therefore cannot
  fail due to contextualization output formatting.
- The all-direct policy uses a new `canonical-v4-<source-sha-prefix>` parse
  version. It never resumes or activates a `canonical-v1`, `canonical-v2`, or
  `canonical-v3` checkpoint produced by an earlier contextualization policy.
- The interrupted `ff14sb` V1, V2, and V3 checkpoints remain preserved as
  inactive development artifacts; the canary starts a clean `canonical-v4`
  pipeline instead of deleting or rewriting them.
- Existing active versions, old chunks, vectors, artifacts, MinerU outputs, and
  source files remain untouched.
- The test environment remains a read-only old-version baseline.
- All development execution remains bound to GPU0; GPU1 remains out of scope.

## 7. Test Design

Tests are written before implementation and must prove:

1. The pipeline never constructs or calls the contextualization model.
2. Narrative, caption, appendix, table, figure, and formula Children retain
   `embedding_text == text` and have no contextual fields.
3. All Child inventories embed, persist, link, and activate without loss.
4. Context fields or modified embedding text on any Child fail closed.
5. Rebuild metrics report zero eligible Children and complete plain embeddings.
6. Every document reports contextual prefix completeness as `1.0` without an
   LLM call.

## 8. Task 15 Execution Change

The stopped V3 `ff14sb` run is not accepted. After local and development tests
pass, Task 15 restarts at the `ff14sb` canary with the all-direct V4 policy. `opls5`, the
17-document rebuild, and retrieval comparison remain blocked until the canary
meets every strict integrity requirement.
