# Query Failure Taxonomy

Use these categories when reviewing `scripts/query_eval.py` output and deciding whether to fix ingest, wiki rendering, retrieval, answer generation, or citation handling.

| Code | Meaning | Typical next step |
| --- | --- | --- |
| `http_error` | `/api/query` returned a non-200 response or could not be reached. | Check API process, logs, and request payload. |
| `no_citation` | The answer has claims but no returned citation. | Inspect final citation selection and answer citation normalization. |
| `wrong_source_hint` | The answer cites the wrong source page or another paper. | Inspect wiki ranking, source disambiguation, and cross-paper retrieval. |
| `missing_expected_answer_text` | The answer omits required concepts, entities, or numeric anchors. | Inspect contexts first; if contexts are correct, repair answer generation. |
| `forbidden_answer_text` | The answer contains known false-negative or contamination phrases. | Check table/metric repair and answer constraints. |
| `missing_citation_text` | Citation text does not include required table/page/evidence anchors. | Fix table/figure excerpt selection and citation dedup. |
| `citation_index_mismatch` | Answer contains citation markers outside the returned citation list. | Fix final citation renumbering/drop logic. |
| `non_chinese_answer` | A Chinese question received a mostly non-Chinese answer. | Tighten language constraints or deterministic fallback. |
| `table_false_negative` | Answer says values are absent while source has table evidence. | Inspect table-first context promotion and table evidence extraction. |
| `citation_not_table` | A table/metric answer cites prose instead of the table block. | Prefer table citations over summary/raw citation for table intent. |
| `cross_paper_contamination` | Answer mixes facts from a different force-field paper. | Strengthen source targeting and source-page citation constraints. |
| `source_alias_confusion` | A duplicate source alias is treated as a separate primary source. | Normalize source aliases or prefer canonical source pages. |
| `unsupported_claim` | The answer adds claims not supported by citation excerpts. | Add stricter answer constraints and evidence checks. |
