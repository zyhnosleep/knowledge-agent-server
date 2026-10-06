"""RAG 适配层模块。

本模块提供 :class:`RAGAdapter`,它是 Agent(智能体)与既有
:class:`~app.services.search.QueryService` 之间的薄封装层。

设计目标:
- 让 Agent 以只读工具的方式调用 RAG,使 RAG 保持为独立的证据提供方;
- 隐藏 QueryService 的内部细节,避免 Agent 直接触及检索服务内部;
- 对"项目不存在"等异常做降级处理,保证调用方总能拿到合法响应,
  便于 Agent 记录步骤并向调用者暴露错误。

对外主要入口:
- :meth:`RAGAdapter.answer`:完整的"检索 + 回答"调用;
- :meth:`RAGAdapter.retrieve_evidence`:仅检索,返回证据包;
- :meth:`RAGAdapter.answer_is_insufficient_evidence`:判断答案是否
  表明检索证据不足。
"""

from __future__ import annotations

import logging
import re
import time

from sqlalchemy.orm import Session

from app.schemas.common import QueryResponse
from app.schemas.agent import EvidencePack
from app.services.search import PreparedEvidence, QueryService

logger = logging.getLogger(__name__)

# 用于识别"检索证据不足"答案的正则:命中即表示模型没有找到可支撑
# 答案的检索证据,调用方(Agent)可据此决定是否采取其他兜底策略。
_INSUFFICIENT_EVIDENCE_RE = re.compile(
    r"insufficient evidence|no supporting evidence was found|does not contain information",
    re.IGNORECASE,
)


class RAGAdapter:
    """围绕既有 QueryService 的薄封装。

    Agent 层把 :meth:`RAGAdapter.answer` 当作只读工具来调用,
    使 RAG 保持为独立的证据提供方,同时避免 Agent 直接触碰
    QueryService 的内部实现细节。
    """

    def answer(
        self,
        db: Session,
        project_slug: str,
        question: str,
        document_id: str | None = None,
        *,
        parse_version_map: dict[str, str] | None = None,
        prepared_evidence: PreparedEvidence | None = None,
        evidence_pack: EvidencePack | None = None,
        conversation_summary: str = '',
        visual_intent: bool | None = None,
        trusted_attachment_ids: frozenset[str] = frozenset(),
    ) -> QueryResponse:
        """执行一次 RAG 查询并返回完整响应。

        强制 ``save_answer=False``:这样 Agent 的结果才是权威记录,
        避免每个中间工具调用都污染逐问问答(qa)数据表。
        """
        # 记录耗时起点,便于调试阶段统计检索性能。
        t0 = time.monotonic()
        try:
            # 委托给 QueryService 完成实际的检索与回答生成。
            service = QueryService(
                db,
                parse_version_map=parse_version_map,
            )
            if prepared_evidence is not None:
                if evidence_pack is not None:
                    prepared_evidence = service.merge_prepared_evidence(
                        prepared_evidence, evidence_pack,
                        trusted_attachment_ids=trusted_attachment_ids,
                    )
                result = service.answer_from_evidence(
                    project_slug, question, prepared_evidence,
                    conversation_summary=conversation_summary, visual_intent=visual_intent,
                )
            else:
                result = service.answer(project_slug, question, save_answer=False, document_id=document_id)
        except ValueError:
            # 项目不存在时 QueryService 会抛 ValueError,这里返回一个
            # 空但合法的响应,让 Agent 可以记录该步骤并向调用方暴露错误。
            logger.warning(
                "RAGAdapter.answer: project_slug=%r not found", project_slug,
            )
            result = QueryResponse(
                answer_markdown="",
                citations=[],
                verification_status="project_not_found",
            )
        # 计算总耗时(毫秒)并输出 debug 日志。
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug("RAGAdapter.answer took %d ms", elapsed_ms)
        return result

    def prepare_evidence(
        self, db: Session, project_slug: str, question: str, limit: int = 15,
        document_id: str | None = None, *, parse_version_map: dict[str, str] | None = None,
    ) -> PreparedEvidence:
        try:
            return QueryService(db, parse_version_map=parse_version_map).prepare_evidence(
                project_slug, question, limit=limit, document_id=document_id,
            )
        except ValueError:
            return PreparedEvidence(
                project_id=None, project_slug=project_slug, retrieval_question=question,
                document_id=document_id, parse_version_map={}, contexts=[],
                pack=EvidencePack(status='project_not_found', items=[]),
            )

    def retrieve_evidence(
        self,
        db: Session,
        project_slug: str,
        question: str,
        limit: int = 15,
        document_id: str | None = None,
        *,
        parse_version_map: dict[str, str] | None = None,
    ) -> EvidencePack:
        """仅检索的 RAG:返回证据包(EvidencePack),不生成回答。

        ``limit`` 控制返回证据条目的最大数量。
        当项目不存在时,返回 ``status="project_not_found"`` 的空
        证据包,而不是让调用方崩溃。
        """
        # 记录耗时起点,便于调试阶段统计"仅检索"路径的性能。
        t0 = time.monotonic()
        try:
            # 委托给 QueryService 完成向量检索与证据组装。
            result = QueryService(
                db,
                parse_version_map=parse_version_map,
            ).retrieve_evidence(
                project_slug, question, limit=limit, document_id=document_id
            )
        except ValueError:
            # 项目不存在时返回带状态标记的空证据包,供上层判断。
            logger.warning(
                "RAGAdapter.retrieve_evidence: project_slug=%r not found",
                project_slug,
            )
            result = EvidencePack(status="project_not_found", items=[])
        # 计算耗时(毫秒)并输出 debug 日志。
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.debug("RAGAdapter.retrieve_evidence took %d ms", elapsed_ms)
        return result

    @staticmethod
    def answer_is_insufficient_evidence(answer_text: str) -> bool:
        """当回答文本表明检索证据不足以支撑有依据的答案时返回 True。

        通过匹配 _INSUFFICIENT_EVIDENCE_RE 中的关键短语来判断,
        供 Agent 决定是否需要补充检索或转向兜底回答。
        """
        return bool(_INSUFFICIENT_EVIDENCE_RE.search(answer_text))
