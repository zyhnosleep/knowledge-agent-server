# Canonical Table Full-Evidence Design

## 1. Problem

The server test version uses a table-first path: for an explicit table or
metric question, it reads the matching table from `document_intelligence.tables`
instead of relying only on ordinary vector chunks. The development version
already has canonical table Children, but table evidence can still be lost or
misrepresented between table retrieval, answer construction, and Agent
synthesis.

The goal is to preserve the useful test-version idea while making table
completeness independent of a fixed character window or an LLM's ability to
remember every row.

## 2. Goals

- Keep ordinary narrative and mechanism questions on the existing Child/Parent
  retrieval path.
- For explicit table/metric questions, locate the target canonical table and
  assemble all of its rows in source order.
- Extract requested row, column, and value facts deterministically from the
  complete table before invoking an LLM.
- Ensure values after the old 4000-character boundary remain available.
- Preserve table-specific citations, table IDs, parse-version isolation, and
  source spans.
- Make Agent synthesis consume the same table facts without allowing the
  generic evidence excerpt cap to remove required values.
- Keep the test environment and its active data unchanged.

## 3. Non-goals

- Do not replace MinerU or reparse PDFs at query time.
- Do not merge tables into narrative chunks.
- Do not make every query load every table in a document.
- Do not remove the existing row-level Children; they remain useful for table
  discovery, ranking, and citation.
- Do not use an unbounded full-table prompt as the only protection against
  missing values.

## 4. Proposed architecture

### 4.1 Table discovery and assembly

For a `table_or_metric` query, the query service will first identify the
document and target table using the existing table selectors and canonical
table Child retrieval. It will then assemble a `TableContext` keyed by:

```text
document_id + parse_version + table_id
```

The context will contain the caption/label, complete normalized headers, all
canonical data rows in ordinal order, page/source metadata, and the Child IDs
that contributed each row. No `[:4000]` character slice is used for this
assembly.

### 4.2 Deterministic table facts

The server will scan the complete `TableContext` and produce a bounded,
structured `TableFacts` result containing the requested selectors, column
names, exact source values, row indices, and table identity. The extractor
must support both familiar metrics (`F1`, `AUC`, `Recall`, etc.) and arbitrary
paper-specific columns such as `Asp`, `C6`, `exptl`, dipole moment, or surface
tension.

The facts are the completeness boundary: if a requested value exists in the
assembled table, it must be present in `TableFacts` before answer generation.
If the canonical table itself is incomplete, the service must report an
ingestion/parse-quality problem rather than infer that the value is absent.

### 4.3 RAG answer

For numeric/table questions, the deterministic answer builder will use
`TableFacts` to produce the factual portion of the answer and table-specific
citations. The LLM may add concise explanation or comparison wording, but it
must not be the component responsible for discovering or preserving numeric
values. A completeness check will verify that all requested facts survive the
answer stage; otherwise the deterministic answer is retained or used for
repair.

For a request to show an entire table, the service may return the canonical
Markdown directly or in explicitly labelled table segments. It will not
silently truncate the table at a fixed character count.

### 4.4 Agent synthesis

Agent synthesis will receive the extracted `TableFacts` and table identity as
a dedicated table evidence section. The generic evidence-pack excerpt limit
will not be allowed to truncate required table facts. If synthesis omits a
required fact, the existing RAG answer/fact-preserving fallback remains the
source of truth.

## 5. Long-table policy

- The backend always scans the complete canonical table.
- Exact metric questions send the model the bounded extracted facts plus the
  minimum supporting table rows needed for citation.
- Summary questions may use row-group summaries, but each group is generated
  from complete rows and retains its table identity.
- Full-table requests return complete canonical Markdown, potentially in
  labelled segments.
- Fixed character truncation such as `block[:4000]` is never used as a
  completeness rule.

This separates data completeness from prompt size: a model context limit may
reduce explanatory text, but it cannot make an existing source value disappear
from the server-side facts.

## 6. Error handling and compatibility

- Canonical table IDs and parse versions remain isolated from legacy data and
  staged versions.
- The legacy `document_intelligence.tables` path remains available for legacy
  documents, but its fixed window is not reused for canonical table facts.
- Missing headers, malformed rows, or incomplete canonical table artifacts are
  recorded as table-quality failures and cannot be converted into a confident
  “value absent” answer.
- Ordinary narrative, figure, formula, and mechanism queries retain their
  existing retrieval behavior.

## 7. TDD and acceptance tests

Tests will be written before implementation and will cover:

1. A canonical table whose requested value occurs after 4000 characters.
2. Assembly of multiple table Children into one ordered TableContext.
3. Preservation of multi-row/multi-level headers and arbitrary column names.
4. Exact extraction of requested rows and values for the current failing table
   cases.
5. Full-table output without silent truncation.
6. Agent synthesis receiving table facts despite the generic excerpt cap.
7. Required-term completeness gating for retrieval and answer evaluation.
8. Regression coverage proving ordinary narrative queries are unchanged.

The staged development evaluation will be rerun only after the focused table
tests pass. The test environment remains read-only and unchanged.
