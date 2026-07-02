from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup


def _html() -> str:
    return Path("src/app/static/index.html").read_text(encoding="utf-8")


def test_static_frontend_contains_pipeline_dashboard_shell() -> None:
    html = _html()
    soup = BeautifulSoup(html, "html.parser")

    assert soup.select_one('[data-view="runs"]') is not None
    assert soup.select_one("#pipelineTopicGrid") is not None
    assert soup.select_one("#pipelineRunsBody") is not None
    assert soup.select_one("#sourceDrawer") is not None
    assert "运行追踪" in html
    assert "原文回溯" in html
    assert "/api/pipeline/dashboard" in html
    assert "/source" in html


def test_no_hardcoded_internal_research_api_calls() -> None:
    """The frontend must build project-aware URLs from activeProjectSlug rather than hardcoding internal-research."""
    html = _html()
    assert "activeProjectSlug" in html
    assert "?project_slug=internal-research" not in html
    assert "project_slug=internal-research&" not in html


def test_upload_topic_select_and_new_topic_inputs() -> None:
    html = _html()
    soup = BeautifulSoup(html, "html.parser")
    assert soup.select_one("#uploadTopicSelect") is not None
    assert soup.select_one("#newTopicSlug") is not None
    assert soup.select_one("#newTopicName") is not None


def test_chat_session_apis_referenced() -> None:
    html = _html()
    assert "/api/agent/sessions" in html
    assert "/api/agent/sessions/" in html


def test_source_markdown_body_referenced() -> None:
    html = _html()
    assert "data.markdown" in html or ".markdown" in html


def test_active_project_state_exists() -> None:
    html = _html()
    assert "activeProjectSlug" in html
    assert "activeProjectName" in html
    assert "activeProjectLabel" in html


def test_upload_refreshes_project_list_after_new_topic() -> None:
    html = _html()
    assert "projects.unshift" in html
    assert "Failed to refresh projects after upload" in html
    assert "return loadProjects().catch" in html
