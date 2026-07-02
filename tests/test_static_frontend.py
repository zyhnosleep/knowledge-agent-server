from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup


def test_static_frontend_contains_pipeline_dashboard_shell() -> None:
    html = Path("src/app/static/index.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    assert soup.select_one('[data-view="runs"]') is not None
    assert soup.select_one("#pipelineTopicGrid") is not None
    assert soup.select_one("#pipelineRunsBody") is not None
    assert soup.select_one("#sourceDrawer") is not None
    assert "运行追踪" in html
    assert "原文回溯" in html
    assert "/api/pipeline/dashboard" in html
    assert "/source" in html
