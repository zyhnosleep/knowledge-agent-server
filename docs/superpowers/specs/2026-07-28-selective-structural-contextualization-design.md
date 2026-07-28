# Selective Structural Contextualization Design

**Date:** 2026-07-28

**Status:** Approved approach, pending written-spec review

## 1. Problem

The current canonical ingestion pipeline sends every retrievable Child chunk to
`qwen3.5:9b` for a generated contextual prefix. On the real `ff14sb` paper this
produced 465 chunks and more than 120 generation requests before the canary was
stopped. Long prompts were truncated at the Ollama context limit, which caused
validation retries and made contextualization dominate rebuild time.

The user-approved policy is to generate LLM context only where a chunk is
structurally difficult to understand in isolation. Ordinary text must be
embedded directly as the original Child text.

## 2. Approved Policy

| Child block type | LLM contextualization | Embedding input |
| --- | --- | --- |
| `table` | required | `contextual_prefix + "\n\n" + child.text` |
| `figure` | required | `contextual_prefix + "\n\n" + child.text` |
| `formula` | required | `contextual_prefix + "\n\n" + child.text` |
| `narrative` | forbidden | `child.text` |
| `caption` | forbidden | `child.text` |
| `appendix` | forbidden | `child.text` |

Parent chunks remain unembedded. They continue to provide hierarchy, expansion,
and display context for retrievable Child chunks.

The policy is explicit and fail-closed. A structured Child cannot silently fall
back to raw-text embedding when contextualization fails. A plain Child cannot
carry contextualization fields or a modified embedding input.

## 3. Data Flow

```text
semantic Child chunks
  -> partition by block type
     -> table/figure/formula -> LLM prefix -> prefix + Child embedding text
     -> narrative/caption/appendix -------> raw Child embedding text
  -> merge in original chunk order
  -> embed every Child
  -> persist every Child and its vector
  -> activation integrity gate
```

`ContextualizationService` keeps its single responsibility: it receives only
eligible structured Children and returns `ContextualizedChunk` objects. The
pipeline owns policy selection and merges contextualized and plain chunks.

## 4. Artifact And Persistence Contract

The contextualization-stage artifact contains a mixed Child inventory:

- Structured Children validate as `ContextualizedChunk` and must contain all
  contextualization provenance fields.
- Plain Children validate as `ChunkDraft`; their contextualization fields must
  be absent and `embedding_text` must equal `text` exactly.
- Parent chunks remain `ChunkDraft` values and receive no vector.
- Chunk IDs, source spans, source block IDs, ordering links, metadata, and
  original `text` remain unchanged for both paths.

The embedding and index stages validate the policy from `block_type` rather than
accepting arbitrary missing fields. This prevents an LLM failure from being
mistaken for an intentional raw Child embedding.

## 5. Integrity Metrics

`contextual_prefix_completeness` changes meaning from "all Children have a
prefix" to "all context-eligible structured Children have a valid prefix".
An empty eligible set is complete.

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

The activation gate separately proves that every eligible structured Child is
contextualized, every plain Child is unmodified raw-text embedding input, and
every Child has a valid vector and source span.

## 6. Failure And Resume Behavior

- A failed structured contextualization blocks activation.
- Plain Children never call the contextualization model and therefore cannot
  fail due to contextualization output formatting.
- The interrupted `ff14sb` contextualization checkpoint is development-only and
  will be released through the existing stage recovery mechanism before the
  canary is resumed with the new code.
- Existing active versions, old chunks, vectors, artifacts, MinerU outputs, and
  source files remain untouched.
- The test environment remains a read-only old-version baseline.
- All development execution remains bound to GPU0; GPU1 remains out of scope.

## 7. Test Design

Tests are written before implementation and must prove:

1. The pipeline sends only `table`, `figure`, and `formula` Children to the LLM.
2. Plain Children retain `embedding_text == text` and have no contextual fields.
3. Structured Children retain `embedding_text == prefix + "\n\n" + text`.
4. Mixed Child inventories embed, persist, link, and activate without loss.
5. Missing context on an eligible structured Child fails closed.
6. Context fields or modified embedding text on a plain Child fail closed.
7. Rebuild metrics use the eligible denominator and report plain completeness.
8. A document with no structured Children reports contextual prefix completeness
   as `1.0` without making any LLM call.

## 8. Task 15 Execution Change

The stopped `ff14sb` run is not accepted. After local and development tests pass,
Task 15 restarts at the `ff14sb` canary with the selective policy. `opls5`, the
17-document rebuild, and retrieval comparison remain blocked until the canary
meets every strict integrity requirement.
