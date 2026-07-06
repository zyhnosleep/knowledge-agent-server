from __future__ import annotations

import re
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


def test_topic_card_chat_action_referenced() -> None:
    html = _html()
    assert "enterTopicChat" in html
    assert "进入对话" in html
    assert "chatContextLabel" in html
    assert 'setView("chat", { skipSessionLoad: true })' in html


def test_select_session_includes_project_slug() -> None:
    html = _html()
    assert '/turns?project_slug=" + encodeURIComponent(activeProjectSlug)' in html


def test_session_list_uses_preview() -> None:
    html = _html()
    assert "session.preview" in html


def test_upload_refreshes_project_list_after_new_topic() -> None:
    html = _html()
    assert "projects.unshift" in html
    assert "Failed to refresh projects after upload" in html
    assert "return loadProjects().catch" in html


# New coverage for the dark-themed chat redesign.


def test_dark_theme_colors_present() -> None:
    html = _html()
    assert "#0b0f17" in html or "#111827" in html


def test_session_list_is_in_sidebar_not_chat_page() -> None:
    html = _html()
    soup = BeautifulSoup(html, "html.parser")
    sidebar = soup.select_one(".sidebar")
    assert sidebar is not None
    assert sidebar.select_one("#sessionList") is not None
    chat_view = soup.select_one('[data-view="chat"]')
    assert chat_view is not None
    assert chat_view.select_one("#sessionList") is None


def test_chat_page_has_wide_central_surface_and_composer() -> None:
    html = _html()
    soup = BeautifulSoup(html, "html.parser")
    chat_view = soup.select_one('[data-view="chat"]')
    assert chat_view is not None
    assert chat_view.select_one(".chat-surface") is not None
    assert chat_view.select_one(".composer") is not None
    assert chat_view.select_one(".recents") is None
    assert "height: calc(100dvh - 64px)" in html
    assert "overflow: hidden" in html
    assert "min-height: 150px" in html
    assert "min-height: 94px" in html
    assert "overflow-wrap: break-word" in html


def test_composer_hint_text() -> None:
    html = _html()
    assert "问点难的，让我多想一步" in html


def test_attachment_upload_and_delete_endpoints_referenced() -> None:
    html = _html()
    assert "/api/agent/sessions/" in html
    assert "/attachments?project_slug=" in html
    assert "/attachments/" in html
    assert "attachment_id" in html


def test_plus_button_opens_hidden_file_input() -> None:
    html = _html()
    soup = BeautifulSoup(html, "html.parser")
    assert soup.select_one("#attachmentButton") is not None
    file_input = soup.select_one("#attachmentInput")
    assert file_input is not None
    assert file_input.get("type") == "file"
    assert file_input.get("multiple") is not None
    assert file_input.get("hidden") is not None
    assert "files.forEach(uploadAttachment)" in html


def test_enter_sends_and_shift_enter_newline_and_ime_safe() -> None:
    html = _html()
    soup = BeautifulSoup(html, "html.parser")
    question_input = soup.select_one("#questionInput")
    assert question_input is not None
    assert question_input.get("aria-label") == "输入问题"
    assert 'event.key === "Enter"' in html
    assert "!event.shiftKey" in html
    assert "!event.isComposing" in html


def test_thinking_steps_ui_referenced() -> None:
    html = _html()
    assert "thinkingFeed" in html
    assert "appendThinking" in html
    assert "clearThinking" in html
    assert "thinking-step" in html
    assert "Array.isArray(data.steps)" in html


def test_start_new_chat_clears_state() -> None:
    html = _html()
    assert "function startNewChat" in html
    assert "activeSessionId = null" in html
    assert "activeAttachments = []" in html
    assert "clearThinking" in html
    assert "renderAttachments" in html


def test_select_session_loads_turns_and_attachments() -> None:
    html = _html()
    assert '/turns?project_slug=" + encodeURIComponent(activeProjectSlug)' in html
    assert "/attachments?project_slug=" in html
    assert "function loadSessionAttachments" in html
    assert "loadSessionAttachments()" in html
    assert "function selectSession" in html


def test_topic_switch_clears_stale_session_and_attachments() -> None:
    html = _html()
    match = re.search(r"function setActiveProject\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "activeSessionId = null" in body
    assert "activeAttachments = []" in body
    assert "renderAttachments()" in body


def test_enter_topic_chat_clears_session_and_attachments() -> None:
    html = _html()
    match = re.search(r"function enterTopicChat\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "activeSessionId = null" in body
    assert "activeAttachments = []" in body


def test_load_projects_clears_session_when_switching_topic() -> None:
    html = _html()
    match = re.search(
        r"if \(projects\.length && !projects\.find\(.*?\}\)\) \{(.*?)\}",
        html,
        re.S,
    )
    assert match is not None
    body = match.group(1)
    assert "activeSessionId = null" in body
    assert "activeAttachments = []" in body


def test_touched_views_use_readable_chinese_strings() -> None:
    html = _html()
    labels = [
        "运行追踪",
        "原文回溯",
        "暂无运行记录",
        "暂无文档",
        "正在读取原文",
        "暂无可预览原文",
        "上传文档",
        "提交",
        "新聊天",
        "历史会话",
        "输入问题",
        "发送",
        "添加临时文件",
        "问点难的，让我多想一步",
    ]
    for label in labels:
        assert label in html, f"Missing readable Chinese label: {label}"


def test_no_common_mojibake_fragments_in_touched_labels() -> None:
    html = _html()
    mojibake_fragments = [
        "Ã©",
        "Ã¨",
        "Ã ",
        "Ã¢",
        "Ã§",
        "Ã¯",
        "Ã¥",
        "Ã¤",
        "Ã¶",
        "Ã¼",
        "Ã±",
        "Â",
        "�",
    ]
    for fragment in mojibake_fragments:
        assert fragment not in html, f"Found mojibake fragment {fragment!r}"
