"""
query.py —— 文本问答链路（T3）：@ 提问 → agent query（固定项目）→ 答案推送
=========================================================================

``Answerer`` 把问题发到 ``feishu-inbox`` 项目（不混入其他项目语料），
返回最终答案与去重后的引用文档标题；超时/异常/无实质内容均给明确结果。
答案按飞书消息长度限制截断（4000 字符），保证可读。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.feishu_bot.inbox import ServerApi

ANSWER_LIMIT = 4000
TRUNCATED_SUFFIX = "…（内容过长已截断）"


@dataclass
class QueryResult:
    """一次问答的结果。"""

    ok: bool
    answer: str | None = None
    citations: list[str] = None  # type: ignore[assignment]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.citations is None:
            self.citations = []


def truncate_answer(text: str, limit: int = ANSWER_LIMIT) -> str:
    """按飞书消息长度限制截断答案。"""
    if len(text) <= limit:
        return text
    return text[: limit - len(TRUNCATED_SUFFIX)] + TRUNCATED_SUFFIX


class Answerer:
    """文本问答编排：发问题 → 取答案与引用 → 结果。"""

    def __init__(self, api: ServerApi, *, project_slug: str) -> None:
        self._api = api
        self._project_slug = project_slug

    def answer(self, question: str) -> QueryResult:
        question = question.strip()
        if not question:
            return QueryResult(ok=False, error="问题内容为空")
        try:
            payload = self._api.query(self._project_slug, question)
        except Exception as exc:
            return QueryResult(ok=False, error=f"问答服务异常：{exc}")
        status = payload.get("status")
        if status != "completed":
            if status == "timeout":
                return QueryResult(ok=False, error="问答超时")
            return QueryResult(ok=False, error=f"问答失败（{status or 'unknown'}）")
        answer = payload.get("final_answer") or ""
        citations = _unique_titles(payload.get("citations") or [])
        return QueryResult(ok=True, answer=truncate_answer(answer), citations=citations)


def _unique_titles(citations: list[dict]) -> list[str]:
    """按出现顺序去重引用文档标题。"""
    seen: set[str] = set()
    titles: list[str] = []
    for citation in citations:
        title = (citation or {}).get("document_title")
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles
