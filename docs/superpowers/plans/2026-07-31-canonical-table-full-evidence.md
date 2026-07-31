# Canonical Table Full-Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with TDD checkpoints.

**Goal:** Make explicit table/metric questions read complete canonical table data on the server, extract requested facts deterministically, and preserve those facts through RAG and Agent synthesis without changing the test environment.

**Architecture:** Keep row-level canonical table Children for discovery, ranking, and citations. Add a pure table-evidence layer that assembles all Children for \`document_id + parse_version + table_id\` and extracts exact facts from the complete table. Pass bounded facts—not a fixed-character table slice—to answer and Agent synthesis; use canonical Markdown directly only for explicit full-table requests.

**Tech Stack:** Python 3.12, SQLAlchemy, Pydantic, pytest, existing \`normalize_table_text()\`, canonical \`DocumentChunk\` metadata, Ollama synthesis.

---

## Task 1: Add failing pure table-evidence tests

**Files:**
- Create: \`tests/test_table_evidence.py\`
- Reference: \`src/app/services/table_normalization.py\`

- [ ] **Step 1: Write the failing tests**

Add tests for the public pure functions that will be introduced:

~~~python
from app.services.table_evidence import (
    CanonicalTableChunk,
    assemble_table_context,
    extract_table_facts,
)


def test_assembly_keeps_rows_after_old_4000_character_boundary() -> None:
    rows = [f"| Model-{i} | {i}.123 |" for i in range(500)]
    chunks = [
        CanonicalTableChunk(
            chunk_id=f"row-{i}",
            document_id="doc-1",
            parse_version="canonical-v4",
            table_id="table-9",
            ordinal=i,
            text=("Table 9\n| Model | Value |\n| --- | --- |\n" + row),
        )
        for i, row in enumerate(rows)
    ]

    table = assemble_table_context(chunks)

    assert "Model-499" in table.markdown
    assert table.row_count == 500


def test_extract_table_facts_supports_arbitrary_columns_and_exact_values() -> None:
    table = assemble_table_context([
        CanonicalTableChunk(
            chunk_id="row-1",
            document_id="doc-1",
            parse_version="canonical-v4",
            table_id="table-7",
            ordinal=1,
            text=(
                "Table 7\n| Model | Asp | C6 | exptl |\n"
                "| --- | --- | --- | --- |\n"
                "| OPLS5 | 21.0 | 8.95 | 95.3 |"
            ),
        ),
    ])

    facts = extract_table_facts(
        "What are the OPLS5 Asp, C6, and exptl values in Table 7?",
        table,
    )

    assert {fact.column for fact in facts} == {"Asp", "C6", "exptl"}
    assert {fact.value for fact in facts} == {"21.0", "8.95", "95.3"}
    assert all(fact.table_id == "table-7" for fact in facts)


def test_assembly_deduplicates_repeated_headers_but_preserves_source_rows() -> None:
    chunks = [
        CanonicalTableChunk(
            chunk_id="header-a", document_id="doc-1", parse_version="v1",
            table_id="table-1", ordinal=1,
            text="Table 1\n| Model | Score |\n| --- | --- |\n| A | 1.0 |",
        ),
        CanonicalTableChunk(
            chunk_id="header-b", document_id="doc-1", parse_version="v1",
            table_id="table-1", ordinal=2,
            text="| Model | Score |\n| --- | --- |\n| B | 2.0 |",
        ),
    ]

    table = assemble_table_context(chunks)

    assert table.markdown.count("| Model | Score |") == 1
    assert "| A | 1.0 |" in table.markdown
    assert "| B | 2.0 |" in table.markdown
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run from the development worktree:

~~~powershell
cd D:\LLM_wiki\.worktrees\internal-pilot
$env:PYTHONPATH = "src"
python -m pytest tests/test_table_evidence.py -q
~~~

Expected: FAIL with \`ModuleNotFoundError: No module named 'app.services.table_evidence'\`.

- [ ] **Step 3: Commit the failing tests**

~~~powershell
git add tests/test_table_evidence.py
git commit -m "test: define complete canonical table evidence behavior"
~~~

## Task 2: Implement complete canonical table assembly and fact extraction

**Files:**
- Create: \`src/app/services/table_evidence.py\`
- Modify: \`src/app/services/table_extraction.py\` only when a shared parser helper is needed
- Test: \`tests/test_table_evidence.py\`

- [ ] **Step 1: Implement the minimal data model and assembly**

Create a pure module with these stable interfaces:

~~~python
@dataclass(frozen=True)
class CanonicalTableChunk:
    chunk_id: str
    document_id: str
    parse_version: str
    table_id: str
    ordinal: int
    text: str
    page_label: str | None = None
    source_spans: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TableFact:
    table_id: str
    document_id: str
    parse_version: str
    row_label: str
    column: str
    value: str
    row_index: int
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class TableContext:
    document_id: str
    parse_version: str
    table_id: str
    label: str | None
    markdown: str
    headers: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    source_chunk_ids: tuple[str, ...]
    row_count: int
    quality_flags: tuple[str, ...] = ()


def assemble_table_context(chunks: Sequence[CanonicalTableChunk]) -> TableContext: ...
def extract_table_facts(question: str, table: TableContext) -> list[TableFact]: ...
~~~

\`assemble_table_context()\` must sort by \`(ordinal, chunk_id)\`, normalize cells with \`normalize_table_text()\`, retain the first complete header, remove only repeated header/separator rows, preserve every data row, and flag an empty or malformed table. It must never use a fixed character slice.

\`extract_table_facts()\` must use the existing selector normalization rules and support arbitrary column names. It should return exact source cell strings and never convert numeric formatting such as \`8.95\` or \`95.3\` to a new format.

- [ ] **Step 2: Run the focused tests**

~~~powershell
python -m pytest tests/test_table_evidence.py -q
~~~

Expected: PASS.

- [ ] **Step 3: Commit the table-evidence module**

~~~powershell
git add src/app/services/table_evidence.py src/app/services/table_extraction.py tests/test_table_evidence.py
git commit -m "feat: assemble complete canonical table evidence"
~~~

## Task 3: Integrate canonical table assembly into QueryService

**Files:**
- Modify: \`src/app/services/search.py\`
- Test: \`tests/test_query_service.py\`
- Test: \`tests/test_canonical_retrieval.py\`

- [ ] **Step 1: Add failing QueryService tests**

Add an integration fixture with two canonical table Children for one \`table_id\`, where the requested value is only in the final row. Assert that the retrieved table context exposes a complete table facts payload while every citation still points to its original Child.

~~~python
def test_canonical_table_assembly_exposes_complete_facts_to_answer_path() -> None:
    service = service_for_table_children(
        rows=[
            "| OPLS5 | 1.0 |",
            "| OPLS5 | 8.95 |",
        ],
        table_id="table-7",
    )

    contexts = service._search_source_chunks(
        "What is the OPLS5 C6 value in Table 7?", "p1", ["d1"], limit=1
    )
    contexts = service._fit_contexts_to_token_budget(
        contexts, question="What is the OPLS5 C6 value in Table 7?"
    )

    assert any("8.95" in str(context.table_facts) for context in contexts)
    assert {context.citation.chunk_id for context in contexts} >= {"row-1", "row-2"}
~~~

- [ ] **Step 2: Run the test and confirm failure**

~~~powershell
python -m pytest tests/test_query_service.py -k complete_facts_to_answer_path -q
~~~

Expected: FAIL because \`RetrievedContext\` has no table-facts payload and the answer path only sees individual row prompt text.

- [ ] **Step 3: Add table payload fields and assembly hook**

Extend \`RetrievedContext\` with optional table-only fields:

~~~python
table_context: TableContext | None = None
table_facts: tuple[TableFact, ...] = ()
~~~

In the canonical table expansion path, group selected Children by \`document_id + parse_version + table_id\`, construct \`CanonicalTableChunk\` records, call \`assemble_table_context()\`, and attach the same assembled table and extracted facts to the table contexts. Keep each Child citation, \`source_spans\`, \`page_label\`, and \`chunk_id\` unchanged.

Update \`_prompt_context_text()\` and \`_context_table_evidence_text()\` so the answer path can use complete table facts while citations continue to expose the original row excerpt. The token budget must count the bounded facts block, not silently cut the server-side \`TableContext\`.

- [ ] **Step 4: Run canonical retrieval and query tests**

~~~powershell
python -m pytest tests/test_query_service.py -k "table or metric" -q
python -m pytest tests/test_canonical_retrieval.py -q
~~~

Expected: PASS, with no regression in active/staged parse-version isolation.

- [ ] **Step 5: Commit QueryService integration**

~~~powershell
git add src/app/services/search.py tests/test_query_service.py tests/test_canonical_retrieval.py
git commit -m "feat: attach complete canonical table evidence to retrieval"
~~~

## Task 4: Make deterministic table answers use complete facts

**Files:**
- Modify: \`src/app/services/search.py\`
- Modify: \`src/app/services/table_evidence.py\`
- Test: \`tests/test_query_service.py\`

- [ ] **Step 1: Add failing answer tests for current omissions**

Add regression cases for \`Asp\`, \`C6\`, \`exptl\`, and a value after the 4000-character boundary. Assert that \`_deterministic_table_answer_if_supported()\` contains every extracted value and never returns a “missing” answer when facts exist.

~~~python
def test_deterministic_table_answer_reports_all_extracted_arbitrary_columns() -> None:
    contexts = contexts_with_complete_table(
        "Table 7", [
            ("OPLS5", {"Asp": "21.0", "C6": "8.95", "exptl": "95.3"}),
        ]
    )

    answer = service._deterministic_table_answer_if_supported(
        "What are the OPLS5 Asp, C6, and exptl values in Table 7?",
        contexts,
        "low",
    )

    assert answer is not None
    assert all(
        value in answer.answer_markdown
        for value in ("21.0", "8.95", "95.3")
    )
~~~

- [ ] **Step 2: Run the tests and confirm failure**

~~~powershell
python -m pytest tests/test_query_service.py -k all_extracted_arbitrary_columns -q
~~~

Expected: FAIL because the current metric extractor only understands a subset of fixed metric headers and does not consume the assembled table facts.

- [ ] **Step 3: Implement fact-first answer construction**

Update \`_extract_requested_metric_values()\` and the generic table answer path to prefer \`context.table_facts\`. Preserve exact values and table/row labels, then fall back to the existing Markdown parser for legacy contexts. Add a completeness helper:

~~~python
def _table_facts_cover_question(
    question: str,
    facts: Sequence[TableFact],
) -> bool:
    """Return True only when every requested selector has a source fact."""
~~~

Use this helper before allowing an LLM repair or “value absent” answer. When facts exist but the generated answer omits one, return the deterministic fact answer or repair using the exact facts.

- [ ] **Step 4: Run focused and full query-service tests**

~~~powershell
python -m pytest tests/test_query_service.py -k "table or metric" -q
python -m pytest tests/test_query_service.py -q
~~~

Expected: PASS.

- [ ] **Step 5: Commit deterministic table answering**

~~~powershell
git add src/app/services/search.py src/app/services/table_evidence.py tests/test_query_service.py
git commit -m "feat: answer table metrics from deterministic complete facts"
~~~

## Task 5: Carry table facts through EvidencePack and Agent synthesis

**Files:**
- Modify: \`src/app/schemas/agent.py\`
- Modify: \`src/app/services/search.py\`
- Modify: \`src/app/services/agent_executor.py\` only if pack merging needs table fields preserved
- Modify: \`src/app/services/agent_synthesizer.py\`
- Test: \`tests/test_agent_synthesizer.py\`
- Test: \`tests/test_agent_executor.py\`

- [ ] **Step 1: Add failing Agent tests**

Add a test that constructs an evidence pack containing complete table facts whose source excerpt is longer than \`MAX_EXCERPT_CHARS\`, then asserts the synthesis prompt contains all exact values in the dedicated table section.

~~~python
def test_table_facts_bypass_generic_excerpt_cap(monkeypatch) -> None:
    evidence_pack = {
        "status": "ok",
        "items": [{"index": 0, "evidence_kind": "table", "excerpt": "short"}],
        "table_facts": [{
            "table_id": "table-7",
            "row_label": "OPLS5",
            "column": "C6",
            "value": "8.95",
        }],
    }

    prompt = AgentSynthesizer._format_table_facts_section(evidence_pack)

    assert "table-7" in prompt
    assert "OPLS5" in prompt
    assert "C6" in prompt
    assert "8.95" in prompt
~~~

- [ ] **Step 2: Run and confirm failure**

~~~powershell
python -m pytest tests/test_agent_synthesizer.py -k table_facts_bypass -q
~~~

Expected: FAIL because no dedicated table-facts formatter exists.

- [ ] **Step 3: Add typed table facts and dedicated prompt section**

Add a small Pydantic \`TableFactEvidence\` model and an optional \`table_facts\` field to \`EvidencePack\`. Populate it in \`QueryService.retrieve_evidence()\` from the assembled table contexts. Preserve it in Agent evidence-pack merges.

Implement:

~~~python
@staticmethod
def _format_table_facts_section(
    evidence_pack: dict[str, Any] | None,
) -> str:
    """Render exact table facts without the generic 300-character excerpt cap."""
~~~

Include this section in Ollama and external synthesis prompts. Keep the generic excerpt cap for ordinary evidence. The prompt must instruct the model to retain every provided exact value and cite the associated table index.

- [ ] **Step 4: Verify executor propagation and synthesis fallback**

~~~powershell
python -m pytest tests/test_agent_synthesizer.py -q
python -m pytest tests/test_agent_executor.py -k "evidence_pack or synthesize" -q
~~~

Expected: PASS; if synthesis omits a required fact, the fact-preserving RAG answer remains available as the fallback.

- [ ] **Step 5: Commit Agent table-fact propagation**

~~~powershell
git add src/app/schemas/agent.py src/app/services/search.py src/app/services/agent_executor.py src/app/services/agent_synthesizer.py tests/test_agent_synthesizer.py tests/test_agent_executor.py
git commit -m "feat: preserve canonical table facts through agent synthesis"
~~~

## Task 6: Enforce required-term completeness in evaluation

**Files:**
- Modify: \`scripts/query_eval.py\`
- Test: \`tests/test_query_eval.py\`

- [ ] **Step 1: Add the failing evaluator test**

Create a case where retrieval citations exist but one \`required_terms\` value is absent. Assert the case status is \`fail\` and the failure reason includes \`required_term_missing\`.

- [ ] **Step 2: Run the test and confirm failure**

~~~powershell
python -m pytest tests/test_query_eval.py -k required_term_missing -q
~~~

- [ ] **Step 3: Add the completeness gate**

Apply the gate after citation extraction and before setting the case status to pass. Keep \`required_terms_passed\` in the JSON report and add the explicit failure reason; do not alter unrelated case scoring.

- [ ] **Step 4: Run evaluator tests and commit**

~~~powershell
python -m pytest tests/test_query_eval.py -q
git add scripts/query_eval.py tests/test_query_eval.py
git commit -m "test: fail retrieval cases missing required table terms"
~~~

## Task 7: Full local verification and controlled development-server test

**Files:**
- Modify: \`docs/work.md\`
- Do not modify: server test source/data or test services

- [ ] **Step 1: Run the complete focused regression suite**

~~~powershell
cd D:\LLM_wiki\.worktrees\internal-pilot
$env:PYTHONPATH = "src"
python -m pytest tests/test_table_evidence.py tests/test_query_service.py tests/test_canonical_retrieval.py tests/test_agent_synthesizer.py tests/test_agent_executor.py tests/test_query_eval.py -q
~~~

Expected: PASS with no new failures.

- [ ] **Step 2: Run existing canonical retrieval and parser regression suites**

~~~powershell
python -m pytest tests/test_structured_evidence.py tests/test_semantic_chunking.py tests/test_parser_document_intelligence.py tests/test_vector_retrieval.py -q
~~~

- [ ] **Step 3: Commit all implementation changes before server sync**

~~~powershell
git status --short
git add src tests scripts
git commit -m "feat: add complete table evidence path"
~~~

- [ ] **Step 4: Sync only verified development files to \`/home/zhangyh/knowledge-agent-dev\`**

Use the existing development deployment procedure with \`CUDA_VISIBLE_DEVICES=0\`. Do not write to \`/home/zhangyh/knowledge-agent-test\`, do not change its active pointer, and do not delete old data.

- [ ] **Step 5: Run staged retrieval-only first**

Require Recall@5 and Recall@10 to remain at the existing baseline and require all table required terms to pass. Stop before full-answer evaluation on any retrieval regression.

- [ ] **Step 6: Run table-focused full-answer cases, then the full 30-case RAG evaluation**

Record table-answer value coverage, citation correctness, latency, and remaining failures in \`docs/work.md\`. Run Agent full evaluation only after direct RAG meets the agreed gate.

