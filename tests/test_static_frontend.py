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


def test_inline_script_is_valid_javascript() -> None:
    html = _html()
    script_match = re.search(r"<script>([\s\S]*?)</script>", html)
    assert script_match is not None
    script = script_match.group(1)
    # Keep this cheap: it catches missing function wrappers/braces that make
    # the whole dashboard non-interactive in browsers.
    import subprocess

    result = subprocess.run(
        ["node", "--check", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
    assert "source_file_url" in html
    assert "pdf-frame" in html
    assert "#page=" in html
    assert "新窗口打开" in html


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
    assert '"/turns?" + scopeQueryParams()' in html
    assert '"project_slug=" + encodeURIComponent(activeProjectSlug)' in html


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
    assert '"/attachments?" + scopeQueryParams()' in html
    assert "/attachments/" in html
    assert "attachment_id" in html


def test_hard_delete_controls_and_endpoints_referenced() -> None:
    html = _html()
    assert "deleteProject" in html
    assert "deleteDocument" in html
    assert "deleteSession" in html
    assert 'method: "DELETE"' in html
    assert "/api/projects/" in html
    assert "/api/documents/" in html
    assert "/api/agent/sessions/" in html
    assert "confirm_slug" in html
    assert "delete-doc-action" in html
    assert "topic-delete" in html
    assert "recent-session" in html


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
    assert "var preserveDocumentScope = !!options.preserveDocumentScope" in html
    assert "activeDocumentId = preserveDocumentScope ? documentId : null" in html
    assert "clearThinking" in html
    assert "renderAttachments" in html


def test_select_session_loads_turns_and_attachments() -> None:
    html = _html()
    assert '"/turns?" + scopeQueryParams()' in html
    assert '"/attachments?" + scopeQueryParams()' in html
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


# Document-scoped chat coverage.


def test_active_document_state_exists() -> None:
    html = _html()
    assert "activeDocumentId" in html
    assert "activeDocumentTitle" in html


def test_document_chat_action_in_library() -> None:
    html = _html()
    assert "enterDocumentChat" in html
    assert "class=\"table-action chat-action\"" in html
    assert "对话" in html
    assert "doc.id" in html or "run.document_id" in html


def test_document_chat_action_in_run_table() -> None:
    html = _html()
    assert "class=\"table-action chat-action\"" in html
    assert "对话" in html


def test_enter_document_chat_sets_state() -> None:
    html = _html()
    match = re.search(r"function enterDocumentChat\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "activeDocumentId =" in body
    assert "activeDocumentTitle =" in body
    assert "updateChatContextLabel" in body


def test_scope_query_params_includes_document_id() -> None:
    html = _html()
    match = re.search(r"function scopeQueryParams\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "project_slug=" in body
    assert "document_id=" in body
    assert "activeDocumentId" in body


def test_project_chat_clears_document_state() -> None:
    html = _html()
    match = re.search(r"function setActiveProject\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "activeDocumentId = null" in body
    assert "activeDocumentTitle = null" in body


def test_session_restore_validates_scope() -> None:
    html = _html()
    assert "function restoreSessionScope" in html
    assert "scope_type" in html
    assert "document_id" in html
    assert "currentSessions.find" in html


def test_citation_markers_stripped() -> None:
    html = _html()
    assert "function stripCitationMarkers" in html
    assert "replace(/\\[\\d+" in html


def test_source_list_rendering() -> None:
    html = _html()
    assert "来源文献" in html
    assert "function renderSources" in html
    assert "function deduplicateSources" in html
    assert "getSourceTitle" in html
    assert "getSourceUrl" in html


def test_document_scope_in_rag_payload() -> None:
    html = _html()
    assert "function runRag" in html
    assert "payload.document_id = activeDocumentId" in html
    assert "/api/query" in html


def test_document_scope_in_agent_payload() -> None:
    html = _html()
    assert "function runAgent" in html
    assert "payload.document_id = activeDocumentId" in html
    assert "/api/agent/query/stream" in html


def test_session_list_uses_document_id_when_document_scope() -> None:
    html = _html()
    match = re.search(r"function loadAgentSessions\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "scopeQueryParams()" in body
    assert "/api/agent/sessions?" in body


def test_delete_session_uses_document_scope() -> None:
    html = _html()
    match = re.search(r"function deleteSession\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert "scopeQueryParams()" in body
    assert "preserveDocumentScope: isDocumentScope()" in html


def test_document_scoped_new_chat_preserves_document_scope() -> None:
    html = _html()
    assert html.count("startNewChat({ preserveDocumentScope: isDocumentScope() });") >= 2


def test_preserved_source_and_delete_actions() -> None:
    html = _html()
    assert "openSource" in html
    assert "deleteDocument" in html
    assert "deleteProject" in html
    assert "class=\"table-action source-action\"" in html
    assert "class=\"table-action chat-action\"" in html
    assert "class=\"icon-button danger-action delete-doc-action\"" in html


def test_document_scope_label_shows_real_title() -> None:
    html = _html()
    assert "单篇文献" in html
    assert "activeDocumentTitle || activeDocumentId" in html


def test_answer_meta_does_not_expose_technical_route_labels() -> None:
    html = _html()
    match = re.search(r"function runAgent\([^)]*\)\s*\{(.*?)\}", html, re.S)
    assert match is not None
    body = match.group(1)
    assert 'showAnswer(["Agent"' not in body
    assert 'data.route.route' not in body


def test_no_raw_html_injection_for_sources() -> None:
    html = _html()
    assert "link.textContent = title" in html
    assert "item.textContent = title" in html
    assert "link.href = url" in html
    assert "link.target = \"_blank\"" in html
    assert "link.rel = \"noopener\"" in html


def test_attachment_requests_preserve_project_scope() -> None:
    html = _html()
    assert '"/attachments?" + scopeQueryParams()' in html
    assert "activeProjectSlug" in html
    assert "activeDocumentId" in html
