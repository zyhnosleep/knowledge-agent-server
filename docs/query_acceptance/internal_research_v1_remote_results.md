# Internal Research Query Benchmark Results

Run date: 2026-06-24

Environment:

- Remote host: `<private-lan-host>`
- Repository path: `~/llm_wiki_server`
- API: `http://127.0.0.1:8000`
- Benchmark: `benchmarks/query/internal_research_v1.json`
- Batch size: 5 cases
- Per-query timeout: 180 seconds

## Summary

- Total cases: 30
- Passed: 1
- Failed: 29

The benchmark now confirms that the system has moved beyond ingest-only validation: it can produce actionable query failure diagnostics. The current query layer is not yet stable enough for real-paper QA across the 10-paper set.

## Failure Counts

| Failure reason | Count |
| --- | ---: |
| `no_citation` | 16 |
| `missing_expected_answer_text` | 14 |
| `http_error` | 13 |
| `wrong_source_hint` | 7 |
| `cross_paper_contamination` | 7 |
| `forbidden_answer_text` | 4 |
| `non_chinese_answer` | 3 |
| `missing_citation_text` | 2 |
| `citation_not_table` | 1 |
| `unsupported_claim` | 1 |

## Case Results

| Case | Status | Failure reasons |
| --- | --- | --- |
| `charmm36_overview` | pass | - |
| `charmm36_table_metrics` | fail | `missing_citation_text`, `citation_not_table` |
| `charmm36_mechanism` | fail | `missing_expected_answer_text` |
| `charmm36idpsff_overview` | fail | `missing_expected_answer_text` |
| `charmm36idpsff_table_metrics` | fail | `http_error`, `no_citation` |
| `charmm36idpsff_mechanism` | fail | `missing_expected_answer_text` |
| `charmm36m_overview` | fail | `http_error`, `no_citation` |
| `charmm36m_table_metrics` | fail | `http_error`, `no_citation` |
| `charmm36m_mechanism` | fail | `missing_expected_answer_text`, `forbidden_answer_text`, `wrong_source_hint`, `cross_paper_contamination` |
| `ff14sb_overview` | fail | `http_error`, `no_citation` |
| `ff14sb_table_metrics` | fail | `http_error`, `no_citation` |
| `ff14sb_mechanism` | fail | `no_citation`, `missing_expected_answer_text`, `wrong_source_hint`, `cross_paper_contamination`, `non_chinese_answer` |
| `ff19sb_overview` | fail | `no_citation`, `missing_expected_answer_text`, `wrong_source_hint`, `cross_paper_contamination`, `non_chinese_answer` |
| `ff19sb_parameter_metrics` | fail | `http_error`, `no_citation` |
| `ff19sb_mechanism` | fail | `no_citation`, `missing_expected_answer_text`, `wrong_source_hint`, `cross_paper_contamination`, `non_chinese_answer` |
| `ff99sb_disp_overview` | fail | `http_error`, `no_citation` |
| `ff99sb_disp_table_metrics` | fail | `http_error`, `no_citation` |
| `ff99sb_disp_mechanism` | fail | `http_error`, `no_citation` |
| `ff99sb_ildn_overview` | fail | `missing_expected_answer_text`, `forbidden_answer_text`, `wrong_source_hint`, `cross_paper_contamination` |
| `ff99sb_ildn_table_parameters` | fail | `http_error`, `no_citation` |
| `ff99sb_ildn_mechanism` | fail | `http_error`, `no_citation` |
| `oplsaa_overview` | fail | `missing_expected_answer_text` |
| `oplsaa_table_metrics` | fail | `missing_citation_text`, `unsupported_claim` |
| `oplsaa_mechanism` | fail | `missing_expected_answer_text` |
| `opls4_overview` | fail | `missing_expected_answer_text`, `forbidden_answer_text`, `wrong_source_hint`, `cross_paper_contamination` |
| `opls4_table_metrics` | fail | `http_error`, `no_citation` |
| `opls4_mechanism` | fail | `missing_expected_answer_text`, `forbidden_answer_text`, `wrong_source_hint`, `cross_paper_contamination` |
| `opls5_overview` | fail | `missing_expected_answer_text` |
| `opls5_table_metrics` | fail | `http_error`, `no_citation` |
| `opls5_mechanism` | fail | `missing_expected_answer_text` |

## Repair Priorities

1. Stabilize query latency and timeout behavior. Thirteen cases timed out at 180 seconds, so result quality cannot improve reliably until query runtime is bounded.
2. Strengthen source targeting. Seven cases cited the wrong force-field paper, especially around CHARMM36/CHARMM36m/CHARMM36IDPSFF and OPLS4/OPLS5.
3. Fix table evidence retrieval and citation selection. Table answers can include correct values while citing nearby or wrong table excerpts.
4. Improve false-negative repair. Several answers say evidence is absent even when the relevant paper has been ingested.
5. Revisit strict expected atoms after query routing is fixed. Some `missing_expected_answer_text` failures are likely benchmark wording strictness, but source/citation failures should be addressed first.

## Follow-up Query Fix Sample

After removing index-overview contamination, adding prompt context windowing, fixing `ff19SB` metric-query misclassification, and narrowing placeholder-page filtering, a five-case representative rerun still failed 5/5 but showed source-routing improvement:

| Case | Result after fix | Observation |
| --- | --- | --- |
| `ff19sb_overview` | fail: `missing_expected_answer_text` | Now cites `sources/ff19sb-amino-acid-specific-protein-backbone-parameters`; previous no-evidence/source-routing failure is resolved. |
| `opls4_mechanism` | fail: `missing_expected_answer_text` | Now cites `sources/opls4-force-field-development-and-validation`; previous OPLS5 wrong-source failure is resolved. |
| `opls5_overview` | fail: `missing_expected_answer_text` | Cites OPLS5 and OPLS4; likely benchmark atom strictness or answer wording issue. |
| `charmm36_table_metrics` | fail: `missing_citation_text`, `citation_not_table` | Answer contains requested values, but citations point to the wrong table excerpts. |
| `oplsaa_table_metrics` | fail: `http_error`, `no_citation` | Still times out at 180 seconds. |

This confirms the next repair round should prioritize table block targeting/citation selection and query latency, then tune overly exact expected atoms.
