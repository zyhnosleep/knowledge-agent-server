"""Tests for QualityReportsService — read-only scanner of loop run manifests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.services.quality_reports import QualityReportsService


def _write_manifest(dir_path: Path, manifest: dict, *, mtime_offset: float = 0) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    if mtime_offset != 0:
        path.touch()
        # Adjust mtime by sleeping, since we can't easily set it on Windows
    return path


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


class TestScanAndSort:
    """Service must scan QUALITY_REPORTS_DIR recursively for manifest.json,
    sort by filesystem modified time descending, and parse only JSON objects."""

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert result["runs"] == []

    def test_single_manifest_returns_one_entry(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "run1", {
            "started_at": "2026-06-27T00:00:00Z",
            "finished_at": "2026-06-27T00:05:00Z",
            "profile": "quick",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "run1"),
            "dry_run": False,
            "commands": [],
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert len(result["runs"]) == 1
        assert result["runs"][0]["run_id"] == "run1"

    def test_nested_manifests_discovered(self, tmp_path: Path) -> None:
        for name in ("run_a", "run_b"):
            _write_manifest(tmp_path / name, {
                "started_at": "2026-06-27T00:00:00Z",
                "finished_at": "2026-06-27T00:05:00Z",
                "profile": "quick",
                "overall_status": "passed",
                "base_url": "http://127.0.0.1:8000",
                "benchmark": "benchmarks/query/internal_research_v1.json",
                "out_dir": str(tmp_path / name),
                "dry_run": False,
                "commands": [],
            })
        # Add a nested manifest
        nested = tmp_path / "deep" / "nested_run"
        _write_manifest(nested, {
            "started_at": "2026-06-27T00:00:00Z",
            "finished_at": "2026-06-27T00:05:00Z",
            "profile": "full",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(nested),
            "dry_run": False,
            "commands": [],
        })

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=10)
        assert len(result["runs"]) == 3

    def test_sorted_by_mtime_descending(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "old", {
            "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "old"), "dry_run": False, "commands": [],
        })
        time.sleep(0.15)
        _write_manifest(tmp_path / "new", {
            "started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "new"), "dry_run": False, "commands": [],
        })

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=10)
        assert result["runs"][0]["run_id"] == "new"
        assert result["runs"][1]["run_id"] == "old"

    def test_respects_limit(self, tmp_path: Path) -> None:
        for i in range(10):
            _write_manifest(tmp_path / f"run{i}", {
                "started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:05:00Z",
                "profile": "quick", "overall_status": "passed",
                "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
                "out_dir": str(tmp_path / f"run{i}"), "dry_run": False, "commands": [],
            })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=3)
        assert len(result["runs"]) == 3

    def test_skips_malformed_manifest(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "good", {
            "started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "good"), "dry_run": False, "commands": [],
        })
        (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
        (tmp_path / "bad" / "manifest.json").write_text("not valid json {{{", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=10)
        assert len(result["runs"]) == 1
        assert result["runs"][0]["run_id"] == "good"

    def test_skips_non_object_manifest(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "good", {
            "started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "good"), "dry_run": False, "commands": [],
        })
        (tmp_path / "list_dir").mkdir(parents=True, exist_ok=True)
        (tmp_path / "list_dir" / "manifest.json").write_text('["list","not","dict"]', encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=10)
        assert len(result["runs"]) == 1
        assert result["runs"][0]["run_id"] == "good"


class TestRunSummary:
    """Run summaries must expose compact fields and never trust embedded paths."""

    def test_essential_fields_present(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "r1", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "r1"),
            "dry_run": False,
            "commands": [{
                "name": "pytest_full_1", "argv": ["python", "-m", "pytest"],
                "exit_code": 0, "duration_seconds": 12.5,
                "stdout_path": "tmp/out.txt", "stderr_path": "tmp/err.txt",
            }],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        entry = result["runs"][0]
        assert entry["run_id"] == "r1"
        assert entry["started_at"] == "2026-06-27T10:00:00Z"
        assert entry["finished_at"] == "2026-06-27T10:05:00Z"
        assert entry["profile"] == "full"
        assert entry["overall_status"] == "passed"
        assert entry["base_url"] == "http://127.0.0.1:8000"
        assert entry["benchmark"] == "benchmarks/query/internal_research_v1.json"
        assert entry["relative_path"] is not None

    def test_command_statuses_extracted(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "r2", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "query",
            "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "r2"),
            "dry_run": False,
            "commands": [
                {"name": "pytest_quick_1", "argv": ["python", "-m", "pytest", "-q"],
                 "exit_code": 0, "duration_seconds": 5.0,
                 "stdout_path": "", "stderr_path": ""},
                {"name": "query_eval", "argv": ["python", "scripts/query_eval.py"],
                 "exit_code": 1, "duration_seconds": 30.0,
                 "stdout_path": "", "stderr_path": ""},
            ],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        commands = result["runs"][0]["command_statuses"]
        assert isinstance(commands, list)
        assert len(commands) == 2
        assert commands[0] == {"name": "pytest_quick_1", "exit_code": 0, "duration_seconds": 5.0}
        assert commands[1] == {"name": "query_eval", "exit_code": 1, "duration_seconds": 30.0}

    def test_query_summary_present(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "r3", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "query",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "r3"),
            "dry_run": False,
            "commands": [],
            "query_eval": {
                "report_path": "tmp/query_eval.json",
                "summary": {"total": 10, "passed": 8, "failed": 2},
                "gate": {"enabled": True, "passed": True, "checks": {}},
            },
            "mineru_smoke": None,
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        qs = result["runs"][0].get("query_summary")
        assert qs is not None
        assert qs["passed"] == 8
        assert qs["failed"] == 2

    def test_query_gate_present(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "r4", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "r4"),
            "dry_run": False,
            "commands": [],
            "query_eval": {
                "report_path": "tmp/query_eval.json",
                "summary": {"total": 10, "passed": 4, "failed": 6},
                "gate": {"enabled": True, "passed": False, "checks": {"min_query_passed": False}},
            },
            "mineru_smoke": None,
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        gate = result["runs"][0].get("query_gate")
        assert gate is not None
        assert gate["enabled"] is True
        assert gate["passed"] is False

    def test_mineru_smoke_summary(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "r5", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "r5"),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": {
                "summary_path": "tmp/mineru_parser_smoke.json",
                "summary": {"status": "passed", "chunks": 42, "tables": 3},
            },
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        ms = result["runs"][0].get("mineru_smoke_summary")
        assert ms is not None
        assert ms["status"] == "passed"

    def test_service_ingest_summary(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "r6", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(tmp_path / "r6"),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": {
                "summary_path": "tmp/service_ingest_smoke.json",
                "summary": {"status": "passed", "checks": {"document_ready": True}},
            },
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        si = result["runs"][0].get("service_ingest_summary")
        assert si is not None
        assert si["status"] == "passed"

    def test_relative_path_is_relative_to_reports_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "nested" / "deep_run"
        _write_manifest(run_dir, {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(run_dir),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        relative = result["runs"][0]["relative_path"]
        assert not Path(relative).is_absolute()
        assert "nested" in relative
        assert "deep_run" in relative

    def test_does_not_trust_out_dir_from_manifest(self, tmp_path: Path) -> None:
        """The service must NOT expose the manifest's out_dir field."""
        _write_manifest(tmp_path / "safe", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": "/etc/passwd",
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        # out_dir must not appear as-is
        entry_str = json.dumps(result["runs"])
        assert "/etc/passwd" not in entry_str


class TestAttributionSideload:
    """Service must read optional same-directory query_attribution.json and query_eval.json."""

    def test_attribution_file_sideloaded(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_attr"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(run_dir),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "case1", "status": "fail", "failure_reasons": ["wrong_source_hint"],
                 "likely_stage": "source", "stage_reasons": {}},
                {"id": "case2", "status": "pass", "failure_reasons": [],
                 "likely_stage": None, "stage_reasons": {}},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        failed_cases = result["runs"][0].get("failed_cases")
        assert isinstance(failed_cases, list)
        assert len(failed_cases) == 1
        assert failed_cases[0]["id"] == "case1"
        assert failed_cases[0]["status"] == "fail"
        assert failed_cases[0]["likely_stage"] == "source"

    def test_attribution_capped_at_10(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_many"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(run_dir),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": f"case{i}", "status": "fail", "failure_reasons": ["http_error"],
                 "likely_stage": "http", "stage_reasons": {"http_error": "http"}}
                for i in range(25)
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        failed_cases = result["runs"][0].get("failed_cases")
        assert len(failed_cases) == 10

    def test_attribution_malformed_json_skipped(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_bad_attr"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(run_dir),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text("not json {{{{{", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        # Should not fail; failed_cases should be empty
        assert result["runs"][0].get("failed_cases") == []

    def test_failed_case_fields_complete(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_complete"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full",
            "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000",
            "benchmark": "benchmarks/query/internal_research_v1.json",
            "out_dir": str(run_dir),
            "dry_run": False,
            "commands": [],
            "query_eval": None,
            "mineru_smoke": None,
            "service_ingest": None,
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [{
                "id": "Q001",
                "status": "fail",
                "failure_reasons": ["wrong_source_hint", "no_citation"],
                "likely_stage": "source",
                "stage_reasons": {"wrong_source_hint": "source", "no_citation": "citation"},
                "answer_expected_terms": {"missing": ["term1", "term2"]},
                "citation_expected_terms": {"missing": ["cite1"]},
                "citation_source_hint": {"matched": False, "matching_values": []},
            }],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        fc = result["runs"][0]["failed_cases"][0]
        assert fc["id"] == "Q001"
        assert fc["status"] == "fail"
        assert fc["likely_stage"] == "source"
        assert "wrong_source_hint" in fc["failure_reasons"]
        assert "term1" in fc["missing_answer_terms"]
        assert "cite1" in fc["missing_citation_terms"]
        assert fc["source_hint_matched"] is False


class TestSkippedSemantics:
    """skipped_reports counts only malformed manifests, not limit truncation."""

    def test_limit_truncation_does_not_increment_skipped(self, tmp_path: Path) -> None:
        """With 3 valid manifests and limit=1, skipped_reports must be 0."""
        for i in range(3):
            _write_manifest(tmp_path / f"run{i}", {
                "started_at": "2026-06-27T10:00:00Z",
                "finished_at": "2026-06-27T10:05:00Z",
                "profile": "quick", "overall_status": "passed",
                "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
                "out_dir": str(tmp_path / f"run{i}"),
                "dry_run": False, "commands": [],
            })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=1)
        assert len(result["runs"]) == 1
        assert result["malformed_count"] == 0
        assert result["valid_total"] == 3

    def test_malformed_counted_separately(self, tmp_path: Path) -> None:
        """1 valid + 1 malformed within limit=5 → runs=1, malformed=1."""
        _write_manifest(tmp_path / "good", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "good"),
            "dry_run": False, "commands": [],
        })
        (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
        (tmp_path / "bad" / "manifest.json").write_text("not json {{{", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert len(result["runs"]) == 1
        assert result["malformed_count"] == 1
        assert result["valid_total"] == 1

    def test_skipped_zero_with_malformed_beyond_limit(self, tmp_path: Path) -> None:
        """limit=1, first manifest valid, second malformed → runs=1, malformed=1 (counted across all), not truncated."""
        _write_manifest(tmp_path / "good", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "good"),
            "dry_run": False, "commands": [],
        })
        (tmp_path / "bad").mkdir(parents=True, exist_ok=True)
        (tmp_path / "bad" / "manifest.json").write_text("invalid", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=1)
        assert len(result["runs"]) == 1
        # malformed_count counts across ALL manifests, not just the limit window
        assert result["malformed_count"] == 1
        assert result["valid_total"] == 1

    def test_collect_runs_returns_structured_result(self, tmp_path: Path) -> None:
        """collect_runs must return a dict with runs, total_manifests, malformed_count, valid_total."""
        _write_manifest(tmp_path / "r1", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "r1"),
            "dry_run": False, "commands": [],
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert isinstance(result, dict)
        assert isinstance(result["runs"], list)
        assert isinstance(result["total_manifests"], int)
        assert isinstance(result["malformed_count"], int)
        assert isinstance(result["valid_total"], int)
        assert result["total_manifests"] == 1
        assert result["malformed_count"] == 0
        assert result["valid_total"] == 1


class TestQueryEvalJsonFallback:
    """When manifest query_eval.summary is absent, read same-directory query_eval.json."""

    def test_fallback_reads_query_eval_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_fb"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir),
            "dry_run": False, "commands": [],
            "query_eval": {
                "report_path": "tmp/qe.json",
                "summary": None,
                "gate": None,
            },
        }), encoding="utf-8")
        (run_dir / "query_eval.json").write_text(json.dumps({
            "summary": {"total": 30, "passed": 25, "failed": 5},
            "cases": [],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        qs = result["runs"][0].get("query_summary")
        assert qs is not None
        assert qs["total"] == 30
        assert qs["passed"] == 25
        assert qs["failed"] == 5

    def test_fallback_when_query_eval_key_missing(self, tmp_path: Path) -> None:
        """When manifest has no query_eval key at all, fallback to file."""
        run_dir = tmp_path / "run_no_qe"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir),
            "dry_run": False, "commands": [],
        }), encoding="utf-8")
        (run_dir / "query_eval.json").write_text(json.dumps({
            "summary": {"total": 10, "passed": 9, "failed": 1},
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        qs = result["runs"][0].get("query_summary")
        assert qs is not None
        assert qs["total"] == 10

    def test_fallback_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """When manifest has no query_eval and no file exists, query_summary is None."""
        run_dir = tmp_path / "run_no_fb"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir),
            "dry_run": False, "commands": [],
        }), encoding="utf-8")
        # no query_eval.json file

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert result["runs"][0].get("query_summary") is None

    def test_fallback_skips_malformed_file(self, tmp_path: Path) -> None:
        """Malformed query_eval.json is silently treated as missing."""
        run_dir = tmp_path / "run_bad_qe"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir),
            "dry_run": False, "commands": [],
        }), encoding="utf-8")
        (run_dir / "query_eval.json").write_text("not valid {{{{{", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert result["runs"][0].get("query_summary") is None

    def test_fallback_never_uses_embedded_path(self, tmp_path: Path) -> None:
        """Even if manifest has a report_path pointing elsewhere, fallback only uses run_dir."""
        run_dir = tmp_path / "run_safe_fb"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir),
            "dry_run": False, "commands": [],
            "query_eval": {
                "report_path": "/etc/hacked/qe.json",
            },
        }), encoding="utf-8")
        # The real query_eval.json is in run_dir — the embedded path is ignored
        (run_dir / "query_eval.json").write_text(json.dumps({
            "summary": {"total": 5, "passed": 3, "failed": 2},
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        qs = result["runs"][0].get("query_summary")
        assert qs is not None
        assert qs["total"] == 5


class TestAgentMetrics:
    """agent_metrics must be built from same-directory artifacts and survive
    missing or malformed optional sidecars."""

    def test_null_when_no_artifacts(self, tmp_path: Path) -> None:
        """agent_metrics is None when run has no query or agent artifacts."""
        _write_manifest(tmp_path / "r1", {
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(tmp_path / "r1"), "dry_run": False, "commands": [],
        })
        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert result["runs"][0].get("agent_metrics") is None

    def test_from_manifest_attribution_summary(self, tmp_path: Path) -> None:
        """manifest query_eval.attribution_summary is surfaced into agent_metrics."""
        run_dir = tmp_path / "r2"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 30, "selected": 30, "completed": 30, "passed": 25, "failed": 5},
                "attribution_summary": {
                    "failure_reason_counts": {"missing_expected_answer_text": 5},
                    "likely_stage_counts": {"answer": 5, "citation": 10, "source": 15},
                    "failed_likely_stage_counts": {"answer": 5},
                },
                "gate": {
                    "enabled": True, "passed": False, "checks": {},
                    "unattributed_case_ids": [],
                },
            },
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["query_total"] == 30
        assert am["query_passed"] == 25
        assert am["query_failed"] == 5
        assert am["failure_reason_counts"] == {"missing_expected_answer_text": 5}
        assert am["likely_stage_counts"] == {"answer": 5, "citation": 10, "source": 15}
        assert am["failed_likely_stage_counts"] == {"answer": 5}

    def test_from_query_attribution_fallback(self, tmp_path: Path) -> None:
        """Same-directory query_attribution.json can populate metrics even when
        manifest lacks attribution_summary."""
        run_dir = tmp_path / "r3"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 10, "passed": 7, "failed": 3},
                # No attribution_summary here!
                "gate": {"enabled": True, "passed": False, "checks": {}},
            },
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "C1", "status": "pass", "failure_reasons": [], "likely_stage": "source",
                 "citation_sources": [{"page_slug": "test"}],
                 "citation_source_hint": {"matched": True, "expected": ["test"]}},
                {"id": "C2", "status": "fail", "failure_reasons": ["missing_expected_answer_text"],
                 "likely_stage": "answer", "citation_sources": [],
                 "citation_source_hint": {"matched": False}},
                {"id": "C3", "status": "fail", "failure_reasons": ["no_citation"],
                 "likely_stage": None, "citation_sources": [],
                 "citation_source_hint": None},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["query_total"] == 10
        assert am["query_passed"] == 7
        assert am["query_failed"] == 3
        assert am["failure_reason_counts"] == {"missing_expected_answer_text": 1, "no_citation": 1}
        assert am["likely_stage_counts"] == {"source": 1, "answer": 1, "unknown": 1}
        assert am["failed_likely_stage_counts"] == {"answer": 1, "unknown": 1}
        assert "C3" in am["unattributed_case_ids"]

    def test_retrieval_coverage(self, tmp_path: Path) -> None:
        """retrieval_coverage is derived from citation/source-hint fields in
        query_attribution.json."""
        run_dir = tmp_path / "r4"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 5, "passed": 5, "failed": 0},
                "gate": {"enabled": False, "passed": None, "checks": {}},
            },
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "C1", "status": "pass",
                 "citation_source_hint": {"matched": True, "expected": ["term1"]},
                 "citation_sources": [{"page_slug": "doc1"}]},
                {"id": "C2", "status": "pass",
                 "citation_source_hint": {"matched": False, "expected": ["term2"]},
                 "citation_sources": [{"page_slug": "doc2"}]},
                {"id": "C3", "status": "pass",
                 "citation_source_hint": None,
                 "citation_sources": []},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        rc = am["retrieval_coverage"]
        assert rc is not None
        assert rc["total_cases"] == 3
        assert rc["cases_with_source_hints"] == 2
        assert rc["source_hint_matched"] == 1
        assert rc["source_hint_unmatched"] == 1
        assert rc["cases_with_citations"] == 2
        assert rc["cases_without_citations"] == 1

    def test_table_evidence_cases(self, tmp_path: Path) -> None:
        """table_evidence_cases derived from citation excerpt groups and
        table-related failures."""
        run_dir = tmp_path / "r5"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 3, "passed": 1, "failed": 2},
                "gate": {"enabled": False, "passed": None, "checks": {}},
            },
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "C1", "status": "pass",
                 "failure_reasons": [],
                 "citation_excerpt_groups": [
                     {"terms": ["t1", "t2"], "matched": True},
                     {"terms": ["t3"], "matched": True},
                 ]},
                {"id": "C2", "status": "fail",
                 "failure_reasons": ["table_false_negative"],
                 "citation_excerpt_groups": [
                     {"terms": ["t4"], "matched": False},
                 ]},
                {"id": "C3", "status": "fail",
                 "failure_reasons": ["missing_expected_answer_text"],
                 "citation_excerpt_groups": []},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        tec = am["table_evidence_cases"]
        assert len(tec) == 2  # C1 and C2 have citation_excerpt_groups with items
        c1 = next(c for c in tec if c["id"] == "C1")
        assert c1["table_groups_total"] == 2
        assert c1["table_groups_matched"] == 2
        assert c1["has_table_failure"] is False
        c2 = next(c for c in tec if c["id"] == "C2")
        assert c2["table_groups_total"] == 1
        assert c2["table_groups_matched"] == 0
        assert c2["has_table_failure"] is True

    def test_agent_trace_sidecar(self, tmp_path: Path) -> None:
        """agent_tool_counts and agent_provider_counts from optional
        agent_trace_summary.json or agent_traces.json."""
        run_dir = tmp_path / "r6"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "gate": {"enabled": False, "passed": None, "checks": {}},
            },
        }), encoding="utf-8")
        (run_dir / "agent_trace_summary.json").write_text(json.dumps({
            "tool_counts": {"rag.retrieve_evidence": 30, "rag.answer": 30, "answer.synthesize": 30},
            "provider_counts": {"deepseek-v4-pro": 30},
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["agent_tool_counts"] == {
            "rag.retrieve_evidence": 30, "rag.answer": 30, "answer.synthesize": 30,
        }
        assert am["agent_provider_counts"] == {"deepseek-v4-pro": 30}

    def test_agent_trace_sidecar_fallback_to_traces(self, tmp_path: Path) -> None:
        """agent_traces.json used when agent_trace_summary.json is absent."""
        run_dir = tmp_path / "r7"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "gate": {"enabled": False, "passed": None, "checks": {}},
            },
        }), encoding="utf-8")
        (run_dir / "agent_traces.json").write_text(json.dumps({
            "trace_count": 3,
            "traces": [
                {"provider": "deepseek-v4-pro", "tool_names": ["rag.retrieve_evidence", "rag.answer"]},
                {"provider": "deepseek-v4-pro", "tool_names": ["rag.retrieve_evidence", "answer.synthesize"]},
                {"provider": "local-fallback", "tools": ["rag.answer"]},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["agent_tool_counts"] == {
            "rag.retrieve_evidence": 2, "rag.answer": 2, "answer.synthesize": 1,
        }
        assert am["agent_provider_counts"] == {
            "deepseek-v4-pro": 2, "local-fallback": 1,
        }

    def test_malformed_sidecar_treated_as_missing(self, tmp_path: Path) -> None:
        """Malformed optional sidecars (query_attribution.json,
        agent_trace_summary.json) are treated as missing, not as malformed manifests."""
        run_dir = tmp_path / "r8"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 5, "passed": 3, "failed": 2},
                "gate": {"enabled": False, "passed": None, "checks": {}},
            },
        }), encoding="utf-8")
        # Malformed sidecars
        (run_dir / "query_attribution.json").write_text("not valid {{{{{", encoding="utf-8")
        (run_dir / "agent_trace_summary.json").write_text("also not valid {{{", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        # Must not fail — the run is still valid
        assert len(result["runs"]) == 1
        am = result["runs"][0].get("agent_metrics")
        assert am is not None  # Still has query totals from manifest
        assert am["query_total"] == 5
        assert am["query_passed"] == 3
        assert am["query_failed"] == 2
        # Sidecars are missing → derived fields empty
        assert am["agent_tool_counts"] == {}
        assert am["agent_provider_counts"] == {}
        assert am["retrieval_coverage"] is None

    def test_manifest_embedded_paths_ignored(self, tmp_path: Path) -> None:
        """Manifest-embedded paths (report_path, attribution_path) are
        ignored for sidecar reads. Only same-directory artifacts are used."""
        run_dir = tmp_path / "r9"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create a decoy directory with fake data
        decoy_dir = tmp_path / "decoy"
        decoy_dir.mkdir(parents=True, exist_ok=True)
        (decoy_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "DECOY", "status": "fail", "failure_reasons": ["decoy_error"],
                 "likely_stage": "decoy"},
            ],
        }), encoding="utf-8")

        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 1, "passed": 1, "failed": 0},
                "report_path": str(decoy_dir / "query_eval.json"),
                "attribution_path": str(decoy_dir / "query_attribution.json"),
                "gate": {"enabled": False, "passed": None, "checks": {}},
            },
        }), encoding="utf-8")
        # Real attribution in run_dir (not decoy)
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "REAL", "status": "fail", "failure_reasons": ["real_error"],
                 "likely_stage": "answer"},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        # Should have the REAL case, not the DECOY
        assert am["failure_reason_counts"] == {"real_error": 1}
        assert am["likely_stage_counts"] == {"answer": 1}

    def test_unattributed_case_ids_from_manifest_gate(self, tmp_path: Path) -> None:
        """unattributed_case_ids are read from manifest gate details."""
        run_dir = tmp_path / "r10"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 5, "passed": 2, "failed": 3},
                "gate": {
                    "enabled": True, "passed": False, "checks": {},
                    "unattributed_case_ids": ["case_a", "case_b"],
                },
            },
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["unattributed_case_ids"] == ["case_a", "case_b"]

    def test_query_totals_from_query_eval_json(self, tmp_path: Path) -> None:
        """Query totals are read from same-directory query_eval.json."""
        run_dir = tmp_path / "r11"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            # No query_eval in manifest at all
        }), encoding="utf-8")
        (run_dir / "query_eval.json").write_text(json.dumps({
            "summary": {"total": 30, "selected": 30, "completed": 30, "passed": 25, "failed": 5},
            "cases": [],
        }), encoding="utf-8")
        (run_dir / "query_attribution.json").write_text(json.dumps({
            "cases": [
                {"id": "C1", "status": "fail", "failure_reasons": ["no_citation"],
                 "likely_stage": "citation"},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["query_total"] == 30
        assert am["query_selected"] == 30
        assert am["query_completed"] == 30
        assert am["query_passed"] == 25
        assert am["query_failed"] == 5
        assert am["failure_reason_counts"] == {"no_citation": 1}

    def test_existing_fields_still_work(self, tmp_path: Path) -> None:
        """Existing dashboard fields (query_summary, query_gate, failed_cases,
        mineru_smoke_summary, service_ingest_summary) remain backward-compatible."""
        run_dir = tmp_path / "r12"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 10, "passed": 5, "failed": 5},
                "attribution_summary": {
                    "failure_reason_counts": {"no_citation": 5},
                    "likely_stage_counts": {"citation": 5},
                    "failed_likely_stage_counts": {"citation": 5},
                },
                "gate": {"enabled": True, "passed": False, "checks": {"min_query_passed": False}},
            },
            "mineru_smoke": {
                "summary": {"status": "passed", "chunks": 42},
            },
            "service_ingest": {
                "summary": {"status": "passed"},
            },
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        entry = result["runs"][0]
        # Existing fields
        assert entry.get("query_summary") is not None
        assert entry["query_summary"]["total"] == 10
        assert entry.get("query_gate") is not None
        assert entry["query_gate"]["enabled"] is True
        assert entry["query_gate"]["passed"] is False
        assert entry.get("mineru_smoke_summary") is not None
        assert entry["mineru_smoke_summary"]["status"] == "passed"
        assert entry.get("service_ingest_summary") is not None
        assert entry["service_ingest_summary"]["status"] == "passed"
        assert entry.get("failed_cases") == []
        # New field
        assert entry.get("agent_metrics") is not None
        assert entry["agent_metrics"]["failure_reason_counts"]["no_citation"] == 5


class TestTraceOnlyMetrics:
    """Trace-only runs with agent_trace_summary.json or agent_traces.json
    must produce non-null agent_metrics even without query artifacts."""

    def test_trace_summary_only_no_query_artifacts(self, tmp_path: Path) -> None:
        """agent_metrics is non-null when agent_trace_summary.json exists,
        even without query_eval.json, query_attribution.json, or manifest query_eval."""
        run_dir = tmp_path / "trace_only_1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            # No query_eval in manifest
        }), encoding="utf-8")
        # No query_eval.json, no query_attribution.json
        (run_dir / "agent_trace_summary.json").write_text(json.dumps({
            "tool_counts": {"rag.retrieve_evidence": 15, "rag.answer": 15},
            "provider_counts": {"deepseek-v4-pro": 15},
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        # Query fields are None — no query artifacts
        assert am["query_total"] is None
        assert am["query_passed"] is None
        assert am["query_failed"] is None
        # Trace fields populated
        assert am["agent_tool_counts"] == {"rag.retrieve_evidence": 15, "rag.answer": 15}
        assert am["agent_provider_counts"] == {"deepseek-v4-pro": 15}
        # Non-trace fields empty/default
        assert am["failure_reason_counts"] == {}
        assert am["likely_stage_counts"] == {}
        assert am["unattributed_case_ids"] == []
        assert am["retrieval_coverage"] is None

    def test_traces_json_only_no_query_artifacts(self, tmp_path: Path) -> None:
        """agent_metrics is non-null when agent_traces.json exists,
        even without query artifacts."""
        run_dir = tmp_path / "trace_only_2"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
        }), encoding="utf-8")
        (run_dir / "agent_traces.json").write_text(json.dumps({
            "traces": [
                {"provider": "deepseek-v4-pro", "tool_names": ["rag.retrieve_evidence"]},
                {"provider": "local-fallback", "tools": ["rag.answer"]},
            ],
        }), encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is not None
        assert am["query_total"] is None
        assert am["agent_tool_counts"] == {"rag.retrieve_evidence": 1, "rag.answer": 1}
        assert am["agent_provider_counts"] == {"deepseek-v4-pro": 1, "local-fallback": 1}

    def test_malformed_trace_no_query_still_null(self, tmp_path: Path) -> None:
        """Malformed trace sidecar with no query artifacts → agent_metrics still None."""
        run_dir = tmp_path / "trace_only_3"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
        }), encoding="utf-8")
        # Malformed trace sidecar
        (run_dir / "agent_trace_summary.json").write_text("not valid {{{{{", encoding="utf-8")

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        am = result["runs"][0].get("agent_metrics")
        assert am is None

    def test_null_when_no_artifacts_at_all(self, tmp_path: Path) -> None:
        """agent_metrics is still None when no query and no trace artifacts exist."""
        run_dir = tmp_path / "trace_only_4"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
        }), encoding="utf-8")
        # No query_eval.json, no query_attribution.json, no trace sidecars

        service = QualityReportsService(reports_dir=tmp_path)
        result = service.collect_runs(limit=5)
        assert result["runs"][0].get("agent_metrics") is None
