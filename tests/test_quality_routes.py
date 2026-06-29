"""Tests for quality dashboard API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.quality_routes import quality_router


class _FakeSettings:
    quality_reports_dir: Path = Path("./tmp")


def make_client(reports_dir: Path) -> TestClient:
    """Build a minimal FastAPI test app with the quality router mounted."""
    import app.api.quality_routes as qr

    app = FastAPI()
    app.include_router(quality_router, prefix="/api")

    fake = _FakeSettings()
    fake.quality_reports_dir = reports_dir
    qr.settings = fake

    return TestClient(app, raise_server_exceptions=True)


class TestDashboardRoute:
    """GET /api/quality/dashboard returns a read-only quality dashboard response."""

    def test_returns_200_when_no_reports(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["runs"] == []
        assert body["skipped_reports"] == 0
        assert "reports_dir" in body
        assert "message" in body

    def test_limit_validated_minimum(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=0")
        assert resp.status_code == 422

    def test_limit_validated_maximum(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=21")
        assert resp.status_code == 422

    def test_limit_default_applied(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard")
        assert resp.status_code == 200
        body = resp.json()
        assert "runs" in body

    def test_read_only_no_side_effects(self, tmp_path: Path) -> None:
        """The route must never start loop scripts or long-running work."""
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=5")
        assert resp.status_code == 200
        files_before = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}
        resp2 = client.get("/api/quality/dashboard?limit=5")
        assert resp2.status_code == 200
        files_after = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()}
        assert files_before == files_after

    def test_response_schema_has_required_top_level_fields(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert "reports_dir" in body
        assert "runs" in body
        assert isinstance(body["runs"], list)
        assert "skipped_reports" in body
        assert isinstance(body["skipped_reports"], int)
        assert "message" in body


class TestSkippedSemanticsViaRoute:
    """skipped_reports must not include limit truncation."""

    def test_limit_truncation_not_in_skipped(self, tmp_path: Path) -> None:
        """3 valid manifests, limit=1 → skipped_reports=0."""
        import json as _json
        for i in range(3):
            d = tmp_path / f"run{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "manifest.json").write_text(_json.dumps({
                "started_at": "2026-06-27T10:00:00Z",
                "finished_at": "2026-06-27T10:05:00Z",
                "profile": "quick", "overall_status": "passed",
                "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
                "out_dir": str(d), "dry_run": False, "commands": [],
            }), encoding="utf-8")
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=1")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["runs"]) == 1
        assert body["skipped_reports"] == 0

    def test_malformed_increments_skipped(self, tmp_path: Path) -> None:
        """1 valid + 1 malformed → skipped_reports=1."""
        import json as _json
        d = tmp_path / "good"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(_json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(d), "dry_run": False, "commands": [],
        }), encoding="utf-8")
        bad = tmp_path / "bad"
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "manifest.json").write_text("not json {{{", encoding="utf-8")
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["runs"]) == 1
        assert body["skipped_reports"] == 1
        assert "malformed" in body["message"].lower()


class TestAgentMetricsViaRoute:
    """Route response must include agent_metrics in each run."""

    def test_route_response_includes_agent_metrics(self, tmp_path: Path) -> None:
        """GET /api/quality/dashboard returns agent_metrics when artifacts exist."""
        import json as _json
        run_dir = tmp_path / "run_am"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(_json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "full", "overall_status": "failed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(run_dir), "dry_run": False, "commands": [],
            "query_eval": {
                "summary": {"total": 30, "passed": 25, "failed": 5},
                "attribution_summary": {
                    "failure_reason_counts": {"missing_expected_answer_text": 5},
                    "likely_stage_counts": {"answer": 5},
                    "failed_likely_stage_counts": {"answer": 5},
                },
                "gate": {"enabled": True, "passed": False, "checks": {}},
            },
        }), encoding="utf-8")
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["runs"]) == 1
        run = body["runs"][0]
        assert "agent_metrics" in run
        am = run["agent_metrics"]
        assert am is not None
        assert am["query_total"] == 30
        assert am["failure_reason_counts"] == {"missing_expected_answer_text": 5}
        assert am["failed_likely_stage_counts"] == {"answer": 5}

    def test_route_agent_metrics_null_when_no_artifacts(self, tmp_path: Path) -> None:
        """agent_metrics is null when run has no query/agent artifacts."""
        import json as _json
        d = tmp_path / "plain_run"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(_json.dumps({
            "started_at": "2026-06-27T10:00:00Z",
            "finished_at": "2026-06-27T10:05:00Z",
            "profile": "quick", "overall_status": "passed",
            "base_url": "http://127.0.0.1:8000", "benchmark": "b.json",
            "out_dir": str(d), "dry_run": False, "commands": [],
        }), encoding="utf-8")
        client = make_client(tmp_path)
        resp = client.get("/api/quality/dashboard?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        run = body["runs"][0]
        assert "agent_metrics" in run
        assert run["agent_metrics"] is None
