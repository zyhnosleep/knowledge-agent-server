"""Read-only quality report scanner for loop run manifests.

Scans QUALITY_REPORTS_DIR recursively for manifest.json files, sorts by
filesystem modified time, and returns compact summaries. Never executes
loop scripts or trusts embedded paths from manifest contents.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class QualityReportsService:
    """Read-only scanner for MinerU/RAG/Agent loop run manifests."""

    def __init__(self, reports_dir: Path) -> None:
        self._reports_dir = reports_dir.resolve()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def collect_runs(self, *, limit: int) -> dict[str, Any]:
        """Return structured dashboard data: runs, total_manifests,
        malformed_count, valid_total.

        Scans ALL manifests under *reports_dir* so that *malformed_count*
        and *valid_total* are accurate across the entire directory, not
        just the first *limit* entries.
        """
        manifests = self._discover_manifests()
        runs: list[dict[str, Any]] = []
        malformed_count = 0
        for manifest_path, _mtime in manifests:
            summary = self._build_run_summary(manifest_path)
            if summary is None:
                malformed_count += 1
            else:
                runs.append(summary)
        return {
            "runs": runs[:limit],
            "total_manifests": len(manifests),
            "malformed_count": malformed_count,
            "valid_total": len(runs),
        }

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    def _discover_manifests(self) -> list[tuple[Path, float]]:
        """Find every ``manifest.json`` under *reports_dir*, returning
        ``(path, mtime)`` sorted by mtime descending."""
        if not self._reports_dir.is_dir():
            return []
        candidates: list[tuple[Path, float]] = []
        for path in sorted(self._reports_dir.rglob("manifest.json")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            candidates.append((path, mtime))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------

    def _read_json_object(self, path: Path) -> dict[str, Any] | None:
        """Read *path* and return a dict if the content is a JSON object,
        otherwise return ``None``."""
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    # ------------------------------------------------------------------
    # summary builders
    # ------------------------------------------------------------------

    def _build_run_summary(self, manifest_path: Path) -> dict[str, Any] | None:
        manifest = self._read_json_object(manifest_path)
        if manifest is None:
            return None

        run_dir = manifest_path.parent
        relative_path = self._relative(run_dir)
        modified_time = self._iso_mtime(manifest_path)

        return {
            "run_id": run_dir.name,
            "relative_path": relative_path,
            "modified_time": modified_time,
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "profile": manifest.get("profile"),
            "overall_status": manifest.get("overall_status"),
            "base_url": manifest.get("base_url"),
            "benchmark": manifest.get("benchmark"),
            "command_statuses": self._build_command_statuses(manifest),
            "query_summary": self._build_query_summary(manifest, run_dir),
            "query_gate": self._build_query_gate(manifest),
            "mineru_smoke_summary": self._build_mineru_smoke_summary(manifest),
            "service_ingest_summary": self._build_service_ingest_summary(manifest),
            "failed_cases": self._build_failed_cases(run_dir),
            "agent_metrics": self._build_agent_metrics(manifest, run_dir),
        }

    def _build_command_statuses(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        commands = manifest.get("commands")
        if not isinstance(commands, list):
            return []
        return [
            {
                "name": c.get("name"),
                "exit_code": c.get("exit_code"),
                "duration_seconds": c.get("duration_seconds"),
            }
            for c in commands
            if isinstance(c, dict)
        ]

    def _build_query_summary(
        self, manifest: dict[str, Any], run_dir: Path
    ) -> dict[str, Any] | None:
        """Extract query summary from manifest, falling back to
        same-directory ``query_eval.json``."""
        qe = manifest.get("query_eval")
        if isinstance(qe, dict):
            summary = qe.get("summary")
            if isinstance(summary, dict):
                return {
                    "total": summary.get("total"),
                    "passed": summary.get("passed"),
                    "failed": summary.get("failed"),
                }
        # Fallback: read run_dir / "query_eval.json"
        return self._read_query_eval_json_fallback(run_dir)

    def _read_query_eval_json_fallback(self, run_dir: Path) -> dict[str, Any] | None:
        """Read ``run_dir / "query_eval.json"`` and extract its ``summary``.

        Never uses manifest-embedded paths — only *run_dir* (the actual
        manifest parent under QUALITY_REPORTS_DIR)."""
        qe_path = run_dir / "query_eval.json"
        data = self._read_json_object(qe_path)
        if data is None:
            return None
        summary = data.get("summary")
        if not isinstance(summary, dict):
            return None
        return {
            "total": summary.get("total"),
            "passed": summary.get("passed"),
            "failed": summary.get("failed"),
        }

    def _build_query_gate(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        qe = manifest.get("query_eval")
        if not isinstance(qe, dict):
            return None
        gate = qe.get("gate")
        if not isinstance(gate, dict):
            return None
        return {
            "enabled": gate.get("enabled", False),
            "passed": gate.get("passed"),
            "checks": gate.get("checks", {}),
        }

    def _build_mineru_smoke_summary(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        ms = manifest.get("mineru_smoke")
        if not isinstance(ms, dict):
            return None
        summary = ms.get("summary")
        if not isinstance(summary, dict):
            return None
        return {
            "status": summary.get("status"),
            "parser_mode": summary.get("parser_mode"),
            "chunks": summary.get("chunks"),
            "error": summary.get("error"),
        }

    def _build_service_ingest_summary(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        si = manifest.get("service_ingest")
        if not isinstance(si, dict):
            return None
        summary = si.get("summary")
        if not isinstance(summary, dict):
            return None
        return {
            "status": summary.get("status"),
            "error": summary.get("error"),
        }

    def _build_failed_cases(self, run_dir: Path) -> list[dict[str, Any]]:
        """Read same-directory ``query_attribution.json`` and return at most
        10 failed case summaries."""
        attr_path = run_dir / "query_attribution.json"
        data = self._read_json_object(attr_path)
        if data is None:
            return []
        cases = data.get("cases")
        if not isinstance(cases, list):
            return []
        failed = [c for c in cases if isinstance(c, dict) and c.get("status") != "pass"]
        summaries: list[dict[str, Any]] = []
        for case in failed[:10]:
            answer_terms = case.get("answer_expected_terms") or {}
            citation_terms = case.get("citation_expected_terms") or {}
            source_hint = case.get("citation_source_hint") or {}
            summaries.append({
                "id": case.get("id"),
                "status": case.get("status"),
                "likely_stage": case.get("likely_stage") or "unknown",
                "failure_reasons": case.get("failure_reasons") or [],
                "missing_answer_terms": answer_terms.get("missing") if isinstance(answer_terms, dict) else [],
                "missing_citation_terms": citation_terms.get("missing") if isinstance(citation_terms, dict) else [],
                "source_hint_matched": source_hint.get("matched") if isinstance(source_hint, dict) else None,
            })
        return summaries

    # ------------------------------------------------------------------
    # agent / evidence metrics
    # ------------------------------------------------------------------

    def _build_agent_metrics(
        self, manifest: dict[str, Any], run_dir: Path
    ) -> dict[str, Any] | None:
        """Build ``agent_metrics`` from same-directory artifacts:
        manifest ``query_eval.attribution_summary``, ``query_eval.json``,
        ``query_attribution.json``, and optional agent trace sidecars.

        Never trusts manifest-embedded paths — only reads from *run_dir*.
        Returns ``None`` when no query/agent artifacts are available.
        """
        # ---------- query totals ----------
        query_total: int | None = None
        query_selected: int | None = None
        query_completed: int | None = None
        query_passed: int | None = None
        query_failed: int | None = None

        qe_data = self._read_json_object(run_dir / "query_eval.json")
        if isinstance(qe_data, dict):
            summary = qe_data.get("summary")
            if isinstance(summary, dict):
                query_total = _safe_int(summary.get("total"))
                query_selected = _safe_int(summary.get("selected"))
                query_completed = _safe_int(summary.get("completed"))
                query_passed = _safe_int(summary.get("passed"))
                query_failed = _safe_int(summary.get("failed"))

        # Also try manifest-embedded query_eval summary as a fallback
        qe_block = manifest.get("query_eval")
        if isinstance(qe_block, dict):
            m_summary = qe_block.get("summary")
            if isinstance(m_summary, dict):
                if query_total is None:
                    query_total = _safe_int(m_summary.get("total"))
                if query_selected is None:
                    query_selected = _safe_int(m_summary.get("selected"))
                if query_completed is None:
                    query_completed = _safe_int(m_summary.get("completed"))
                if query_passed is None:
                    query_passed = _safe_int(m_summary.get("passed"))
                if query_failed is None:
                    query_failed = _safe_int(m_summary.get("failed"))

        # ---------- stage / failure counts ----------
        failure_reason_counts: dict[str, int] = {}
        likely_stage_counts: dict[str, int] = {}
        failed_likely_stage_counts: dict[str, int] = {}
        unattributed_case_ids: list[str] = []

        # First try manifest attribution_summary (already parsed by loop runner)
        if isinstance(qe_block, dict):
            attr_summary = qe_block.get("attribution_summary")
            if isinstance(attr_summary, dict):
                failure_reason_counts = _coerce_string_keys(
                    attr_summary.get("failure_reason_counts")
                )
                likely_stage_counts = _coerce_string_keys(
                    attr_summary.get("likely_stage_counts")
                )
                failed_likely_stage_counts = _coerce_string_keys(
                    attr_summary.get("failed_likely_stage_counts")
                )

        # Gate unattributed case ids (from manifest gate details)
        if isinstance(qe_block, dict):
            gate = qe_block.get("gate")
            if isinstance(gate, dict) and isinstance(gate.get("unattributed_case_ids"), list):
                unattributed_case_ids = [
                    str(cid) for cid in gate["unattributed_case_ids"]
                ]

        # ---------- attribution file (query_attribution.json) ----------
        attr_data = self._read_json_object(run_dir / "query_attribution.json")

        # If manifest attribution_summary was absent, derive from the file
        if not failure_reason_counts or not likely_stage_counts:
            if isinstance(attr_data, dict):
                cases = attr_data.get("cases")
                if isinstance(cases, list):

                    if not failure_reason_counts:
                        frc: Counter[str] = Counter()
                        for case in cases:
                            if isinstance(case, dict):
                                for reason in case.get("failure_reasons") or []:
                                    frc[str(reason)] += 1
                        failure_reason_counts = dict(frc)

                    if not likely_stage_counts:
                        lsc: Counter[str] = Counter()
                        flsc: Counter[str] = Counter()
                        for case in cases:
                            if isinstance(case, dict):
                                stage = str(case.get("likely_stage") or "unknown")
                                lsc[stage] += 1
                                if case.get("status") != "pass":
                                    flsc[stage] += 1
                        likely_stage_counts = dict(lsc)
                        failed_likely_stage_counts = dict(flsc)

        # Gate unattributed case ids from attribution file fallback
        if not unattributed_case_ids and isinstance(attr_data, dict):
            cases = attr_data.get("cases")
            if isinstance(cases, list):
                unattributed_case_ids = [
                    str(case.get("id"))
                    for case in cases
                    if isinstance(case, dict)
                    and case.get("status") != "pass"
                    and case.get("likely_stage") in (None, "", "unknown")
                ]

        # ---------- retrieval coverage (from attribution cases) ----------
        retrieval_coverage: dict[str, Any] | None = None
        if isinstance(attr_data, dict):
            cases = attr_data.get("cases")
            if isinstance(cases, list):
                total = len(cases)
                with_hints = 0
                hint_matched = 0
                hint_unmatched = 0
                with_citations = 0
                for case in cases:
                    if not isinstance(case, dict):
                        continue
                    src = case.get("citation_source_hint")
                    if isinstance(src, dict):
                        with_hints += 1
                        if src.get("matched") is True:
                            hint_matched += 1
                        elif src.get("matched") is False:
                            hint_unmatched += 1
                    sources = case.get("citation_sources")
                    if isinstance(sources, list) and len(sources) > 0:
                        with_citations += 1
                retrieval_coverage = {
                    "total_cases": total,
                    "cases_with_source_hints": with_hints,
                    "source_hint_matched": hint_matched,
                    "source_hint_unmatched": hint_unmatched,
                    "cases_with_citations": with_citations,
                    "cases_without_citations": total - with_citations,
                }

        # ---------- table evidence cases (from attribution cases) ----------
        table_evidence_cases: list[dict[str, Any]] = []
        if isinstance(attr_data, dict):
            cases = attr_data.get("cases")
            if isinstance(cases, list):
                for case in cases:
                    if not isinstance(case, dict):
                        continue
                    cid = case.get("id")
                    ceg = case.get("citation_excerpt_groups")
                    if not isinstance(ceg, list) or not ceg:
                        continue
                    groups_total = len(ceg)
                    groups_matched = sum(
                        1 for g in ceg if isinstance(g, dict) and g.get("matched")
                    )
                    has_table_failure = any(
                        "table" in str(r).lower()
                        for r in (case.get("failure_reasons") or [])
                    )
                    table_evidence_cases.append({
                        "id": str(cid),
                        "table_groups_total": groups_total,
                        "table_groups_matched": groups_matched,
                        "has_table_failure": has_table_failure,
                    })

        # ---------- agent trace sidecar (optional) ----------
        agent_tool_counts: dict[str, int] = {}
        agent_provider_counts: dict[str, int] = {}

        trace_data = self._read_agent_trace_sidecar(run_dir)
        if isinstance(trace_data, dict):
            agent_tool_counts = _coerce_string_keys(
                trace_data.get("tool_counts")
            )
            agent_provider_counts = _coerce_string_keys(
                trace_data.get("provider_counts")
            )
            # If no tool counts but has individual traces/runs, try to aggregate
            if not agent_tool_counts or not agent_provider_counts:
                runs = trace_data.get("runs") or trace_data.get("traces")
                if isinstance(runs, list):
                    tool_counter: dict[str, int] = {}
                    provider_counter: dict[str, int] = {}
                    for run_entry in runs:
                        if not isinstance(run_entry, dict):
                            continue
                        provider = str(
                            run_entry.get("provider")
                            or run_entry.get("model")
                            or ""
                        ).strip()
                        if provider:
                            provider_counter[provider] = (
                                provider_counter.get(provider, 0) + 1
                            )
                        tools = run_entry.get("tool_names") or run_entry.get("tools") or []
                        if isinstance(tools, list):
                            for tool in tools:
                                name = str(tool).strip()
                                if name:
                                    tool_counter[name] = tool_counter.get(name, 0) + 1
                    if not agent_tool_counts:
                        agent_tool_counts = tool_counter
                    if not agent_provider_counts:
                        agent_provider_counts = provider_counter

        # Only return metrics when at least one query or trace value is present
        if (
            query_total is None
            and query_passed is None
            and query_failed is None
            and not failure_reason_counts
            and not likely_stage_counts
            and not agent_tool_counts
            and not agent_provider_counts
        ):
            return None

        return {
            "query_total": query_total,
            "query_selected": query_selected,
            "query_completed": query_completed,
            "query_passed": query_passed,
            "query_failed": query_failed,
            "failure_reason_counts": failure_reason_counts,
            "likely_stage_counts": likely_stage_counts,
            "failed_likely_stage_counts": failed_likely_stage_counts,
            "unattributed_case_ids": unattributed_case_ids,
            "retrieval_coverage": retrieval_coverage,
            "table_evidence_cases": table_evidence_cases,
            "agent_tool_counts": agent_tool_counts,
            "agent_provider_counts": agent_provider_counts,
        }

    def _read_agent_trace_sidecar(self, run_dir: Path) -> dict[str, Any] | None:
        """Read optional agent trace sidecar from *run_dir*.

        Checks ``agent_trace_summary.json`` first, then ``agent_traces.json``.
        """
        for name in ("agent_trace_summary.json", "agent_traces.json"):
            data = self._read_json_object(run_dir / name)
            if data is not None:
                return data
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _relative(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self._reports_dir)
        except ValueError:
            return path.name
        return rel.as_posix()

    @staticmethod
    def _iso_mtime(path: Path) -> str:
        try:
            ts = path.stat().st_mtime
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except OSError:
            return ""


def _safe_int(value: Any) -> int | None:
    """Coerce *value* to int, returning ``None`` when it can't be parsed."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_string_keys(raw: Any) -> dict[str, int]:
    """Convert a raw dict-like value to ``{str: int}``, skipping non-int values."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for k, v in raw.items():
        try:
            result[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return result
