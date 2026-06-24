# Internal Research Query Acceptance v1

This benchmark checks whether the `internal-research` project can answer real questions over the 10 newly ingested force-field papers, not just complete ingest.

## Sources

The benchmark covers these source wiki pages:

- `sources/charmm36-force-field-refinement-for-proteins`
- `sources/charmm36idpsff`
- `sources/charmm36m-force-field`
- `sources/ff14sb`
- `sources/ff19sb-amino-acid-specific-protein-backbone-parameters`
- `sources/ff99sb-disp`
- `sources/ff99sb-ildn`
- `sources/opls-aa-force-field-development-and-validation`
- `sources/opls4`
- `sources/opls5-force-field-development-and-validation`

`sources/knowledge-graph-sac-kg-framework-overview` is intentionally excluded. For OPLS4, accept either `sources/opls4` or the current re-ingested slug `sources/opls4-force-field-development-and-validation`; citations to OPLS5 should still fail source-hint checks.

## Case Design

Each paper has three acceptance questions:

- overview: main contribution, target problem, validation strategy
- table or parameter: concrete values from tables, formulas, or parameterization anchors
- mechanism or comparison: why the method improves over prior force fields

Each case requires:

- a citation
- a Chinese answer for Chinese questions
- expected answer atoms
- forbidden false-negative phrases
- citation source hints matching the expected source page
- for table/metric cases, citation excerpt atoms such as table labels and key numeric values
- for multi-table cases, grouped citation excerpt atoms where every group must appear within a single citation excerpt

The executable benchmark lives at `benchmarks/query/internal_research_v1.json`.

## Run

```bash
python scripts/query_eval.py benchmarks/query/internal_research_v1.json \
  --base-url http://127.0.0.1:8000 \
  --output tmp/query_benchmark/internal_research_v1.json \
  --markdown tmp/query_benchmark/internal_research_v1.md
```

Use `--fail-on-error` in CI-like checks when any failed case should return a nonzero exit code.

For long remote runs, split the benchmark into batches and keep incremental reports on disk:

```bash
python scripts/query_eval.py benchmarks/query/internal_research_v1.json \
  --base-url http://127.0.0.1:8000 \
  --offset 0 \
  --limit 5 \
  --output tmp/query_benchmark/internal_research_v1_00_04.json \
  --markdown tmp/query_benchmark/internal_research_v1_00_04.md
```
