from __future__ import annotations

"""
Agent Synthesizer：基于检索证据生成最终答案。

为什么需要这个模块？
RAG（检索增强生成）返回的答案通常只是“把检索到的片段拼接/改写”，存在以下问题：
1. 可能重复、啰嗦、语言不一致；
2. 可能遗漏关键证据（尤其是数值、指标、对比类信息）；
3. 引用格式不统一，难以追踪；
4. 没有根据 query route（如 evidence_required、table_or_metric）做针对性整合。

AgentSynthesizer 的职责：
- 接收 RAG 初稿、引用片段、evidence_pack；
- 让模型“综合”而非“拼接”证据，生成简洁、 grounded、带引用的最终答案；
- 支持本地 Ollama 和外部 API 两种 provider；
- 对 evidence-heavy 的 route，检测是否遗漏关键证据锚点，必要时 retry；
- 失败时安全回退到 RAG 初稿。
"""

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.ai import DeepSeekClient, OllamaClient
from app.services.agent_model_router import InferenceTarget

logger = logging.getLogger(__name__)
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.ai import DeepSeekClient, OllamaClient
from app.services.agent_model_router import InferenceTarget

logger = logging.getLogger(__name__)

# 需求侧解析常量（9.8/任务9）：列/术语需求提取与表号、行号引用剔除。
# 枚举分隔符支持中文顿号/逗号、英文逗号、中文"和"与英文 "and"。
_ENUMERATION_SEPARATORS = re.compile(r"[、，,]|\band\b|和")
_COLUMN_STOPWORDS = frozenset(
    {
        "what", "which", "why", "when", "where", "who", "how",
        "is", "are", "was", "were", "does", "do", "did",
        "the", "a", "an", "of", "for", "with", "about", "from", "into",
        "and", "or", "not",
        "row", "column", "table",
        "this", "that", "these", "those", "its",
        "much", "many",
    }
)

# 需求侧行标签别名（任务9）：问题中的 "X row" 自然语言形式（total row /
# average row / mean row / overall row）映射到 canonical row label，与
# fact 侧 _fact_identity_keys 的 label: 展开使用同一规范化口径（_norm_column）。
# 只匹配 "X row" 形式，避免把修饰指标的 total/average 形容词误判为行需求。
_ROW_LABEL_ALIASES = ("total", "average", "mean", "overall")
_ROW_LABEL_PATTERN = re.compile(
    r"\b(" + "|".join(_ROW_LABEL_ALIASES) + r")\s+row\b",
    re.IGNORECASE,
)


class SynthesisPayload(BaseModel):
    """Response shape for evidence synthesis."""
    answer_markdown: str
    cited_indexes: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class AgentSynthesizer:
    """Evidence synthesis using Ollama, DeepSeek, or safe fallback.

    Allowed synthesis providers are ``auto``, ``deepseek``, ``external_api``,
    and ``local``. ``external_api`` remains for backwards compatibility.

    External synthesis prompts the model to synthesize from evidence
    (not concatenate paragraphs) and to cite only from provided citations.
    """

    # 合法的 synthesis provider：auto 会根据配置自动选择。
    VALID_PROVIDERS = {"auto", "deepseek", "external_api", "local", "ollama"}

    # evidence_pack 在 prompt 中的上限：最多 10 条，每条摘录最多 300 字符。
    MAX_EVIDENCE_PACK_ITEMS = 10
    MAX_EXCERPT_CHARS = 300
    MAX_TABLE_FACTS = 128

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        self._settings = get_settings()
        self._ollama = ollama_client

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str = "",
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        target: InferenceTarget | None = None,
        narrow_context: bool = False,
    ) -> dict[str, Any]:
        """Synthesize a final answer from RAG evidence.

        根据配置选择 provider：
        - local：使用本地 Ollama 重新综合生成。
        - external_api：调用外部 API 进行综合。
        - auto：外部 API 可用且有 key 时优先外部，否则回退 local。

        ``narrow_context``（9.3.1）：跨轮引用场景为 True 时，prompt 只含
        conversation_summary 事实、RAG answer、citations 与结构化
        table_facts，不含 evidence_pack items 摘录（证据全集）。

        Returns a dict with at least:
        ``answer_markdown``, ``cited_indexes``, ``warnings``,
        ``confidence``, ``provider``, and ``model``.
        """
        provider = self._resolve_provider()

        if provider == "local":
            result = self._local_synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                target=target or self._default_target(),
                narrow_context=narrow_context,
            )
        elif provider == "ollama":
            result = self._ollama_synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                narrow_context=narrow_context,
            )
        elif provider == "deepseek":
            result = self._deepseek_synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                narrow_context=narrow_context,
            )
        else:
            # external_api path
            result = self._external_synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                narrow_context=narrow_context,
            )

        # ---- generic fidelity guards (provider-uniform) ----
        # Never return a synthesis that loses draft-covered anchors, invents
        # numbers absent from the evidence, or drops evidence table labels.
        guard_meta: dict[str, Any] = {}
        guard = self._apply_fidelity_guards(
            question=query,
            rag_answer=rag_answer,
            citations=citations,
            evidence_pack=evidence_pack,
            synthesized_answer=result["answer_markdown"],
            warnings=result.get("warnings", []),
            fallback_provider=(
                result.get("provider")
                if result.get("provider") == "deepseek"
                else None
            ),
            fallback_model=(
                result.get("model")
                if result.get("provider") == "deepseek"
                else None
            ),
            guard_meta=guard_meta,
        )
        # 9.8：需求侧期望状态随结果透出（直通 trace 可审计）
        if guard is not None:
            # In a remote-only deployment the RAG layer intentionally skips
            # local Ollama generation and returns a degraded evidence marker.
            # Falling back to that marker plus raw excerpts after a fidelity
            # rejection is worse than asking DeepSeek once to repair the
            # rejected answer. Keep this retry scoped to the remote degraded
            # path; local synthesis retains the historical hard fallback.
            if (
                result.get("provider") == "deepseek"
                and self._is_degraded_rag_answer(rag_answer)
            ):
                revised = self._revise_deepseek_after_guard(
                    query=query,
                    route=route,
                    conversation_summary=conversation_summary,
                    rag_answer=rag_answer,
                    citations=citations,
                    evidence_pack=evidence_pack,
                    narrow_context=narrow_context,
                    rejected_answer=str(result.get("answer_markdown") or ""),
                    guard_warnings=guard.get("warnings", []),
                )
                if revised is not None:
                    revised_guard_meta: dict[str, Any] = {}
                    revised_guard = self._apply_fidelity_guards(
                        question=query,
                        rag_answer=rag_answer,
                        citations=citations,
                        evidence_pack=evidence_pack,
                        synthesized_answer=revised["answer_markdown"],
                        warnings=revised.get("warnings", []),
                        fallback_provider=revised.get("provider"),
                        fallback_model=revised.get("model"),
                        guard_meta=revised_guard_meta,
                    )
                    if revised_guard is None:
                        revised.setdefault("warnings", []).append(
                            "DeepSeek synthesis was revised once after a fidelity guard."
                        )
                        revised.setdefault(
                            "expected_facts_status",
                            revised_guard_meta.get("expected_facts_status"),
                        )
                        return revised
            guard.setdefault(
                "expected_facts_status", guard_meta.get("expected_facts_status")
            )
            return guard
        result.setdefault(
            "expected_facts_status", guard_meta.get("expected_facts_status")
        )
        return result

    def synthesize_stream(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str = "",
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        target: InferenceTarget | None = None,
        event_sink: Callable[[str, dict[str, Any]], None],
        cancel_event: threading.Event | None = None,
        narrow_context: bool = False,
    ) -> dict[str, Any]:
        """流式生成版本：本地 Ollama 逐 token 返回，同时组装标准结果结构。

        如果当前 provider 不是 local，则退回到非流式 synthesize()。
        """
        resolved_target = target or self._default_target()
        if self._resolve_provider() != "local":
            result = self.synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                target=resolved_target,
                narrow_context=narrow_context,
            )
            self._emit_citations(result, citations, event_sink)
            return result

        if not citations and not (evidence_pack or {}).get("items"):
            return self._local_fallback(rag_answer, citations)

        messages = self._build_local_messages(
            query=query,
            route=route,
            conversation_summary=conversation_summary,
            rag_answer=rag_answer,
            citations=citations,
            evidence_pack=evidence_pack,
            narrow_context=narrow_context,
        )
        ollama = self._ollama or OllamaClient(
            base_url=resolved_target.base_url,
            embedding_base_url=self._settings.ollama_embedding_base_url,
        )
        chunks: list[str] = []
        model = resolved_target.model
        generation_started = time.monotonic()
        first_token_ms: int | None = None
        last_generated: dict[str, Any] = {}
        try:
            for generated in ollama.stream_chat(
                messages=messages,
                model=resolved_target.model,
                context_length=resolved_target.context_length,
                max_output_tokens=768,
                cancel_event=cancel_event,
            ):
                last_generated = generated
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Local Ollama synthesis cancelled")
                delta = str(generated.get("content") or "")
                model = str(generated.get("model") or model)
                if delta:
                    if first_token_ms is None:
                        first_token_ms = int(
                            (time.monotonic() - generation_started) * 1000
                        )
                    chunks.append(delta)
                    event_sink("token", {"delta": delta, "model": model})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Streaming local Ollama synthesis failed: %s", exc)
            result = self._comparison_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
            )
            result["warnings"].append(
                f"Local Ollama synthesis failed; using evidence fallback: {exc}"
            )
            return result

        answer_markdown = "".join(chunks).strip()
        if not answer_markdown:
            result = self._local_fallback(rag_answer, citations)
            result["warnings"].append(
                "Local Ollama synthesis returned an empty answer; using evidence fallback."
            )
            return result

        answer_markdown = self._sanitize_inline_citations(
            answer_markdown, len(citations)
        )
        warnings: list[str] = []

        # Bounded coverage retry after the stream completes.  The streamed
        # tokens are a preview; the returned result selects the higher-fidelity
        # synthesis (first pass or one anchored retry) exactly like the
        # non-stream local path.
        if answer_markdown and self._should_retry_for_coverage(
            route=route,
            evidence_pack=evidence_pack,
            citations=citations,
            answer_text=answer_markdown,
        ):
            retry_result = self._retry_local_with_anchors(
                client=ollama,
                model=model,
                context_length=resolved_target.context_length,
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                first_answer=answer_markdown,
                narrow_context=narrow_context,
            )
            if retry_result is not None:
                answer_markdown = retry_result["answer_markdown"]
                warnings = list(retry_result["warnings"])
            else:
                warnings.append(
                    "Evidence coverage retry did not produce a usable local streaming "
                    "result; using first synthesis."
                )
            answer_markdown = self._sanitize_inline_citations(
                answer_markdown, len(citations)
            )

        result = {
            "answer_markdown": answer_markdown,
            "cited_indexes": self._extract_cited_indexes(
                answer_markdown, len(citations)
            ),
            "warnings": warnings,
            "confidence": 1.0,
            "provider": "local",
            "model": model,
            "performance": self._performance_metadata(
                last_generated, first_token_ms=first_token_ms
            ),
        }

        # Generic fidelity guards for the streaming selection path.
        guard_meta: dict[str, Any] = {}
        guard = self._apply_fidelity_guards(
            question=query,
            rag_answer=rag_answer,
            citations=citations,
            evidence_pack=evidence_pack,
            synthesized_answer=answer_markdown,
            warnings=warnings,
            guard_meta=guard_meta,
        )
        # 9.8：需求侧期望状态随结果透出（直通 trace 可审计）
        if guard is not None:
            guard.setdefault(
                "expected_facts_status", guard_meta.get("expected_facts_status")
            )
            self._emit_citations(guard, citations, event_sink)
            return guard
        result.setdefault(
            "expected_facts_status", guard_meta.get("expected_facts_status")
        )
        self._emit_citations(result, citations, event_sink)
        return result

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> str:
        """根据 settings 决定使用哪个 provider。"""
        configured = self._settings.agent_synthesis_provider
        if configured not in self.VALID_PROVIDERS:
            logger.warning(
                "Unknown AGENT_SYNTHESIS_PROVIDER=%r, falling back to 'auto'",
                configured,
            )
            configured = "auto"

        if configured == "local":
            return "local"

        if configured == "ollama":
            return "ollama"

        if configured == "external_api":
            return "external_api"

        if configured == "deepseek":
            return "deepseek"

        # "auto": prefer explicitly configured DeepSeek, then the legacy
        # external API path, otherwise use the existing local Ollama path.
        generation_provider = str(
            getattr(self._settings, "generation_provider", "ollama")
        ).strip().lower()
        deepseek_key = getattr(self._settings, "deepseek_api_key", None)
        if generation_provider == "deepseek" and self._has_secret(deepseek_key):
            return "deepseek"
        if self._settings.external_api_enabled and self._has_secret(
            self._settings.external_api_key
        ):
            return "external_api"
        return "local"

    @staticmethod
    def _has_secret(value: str | None) -> bool:
        """Treat example placeholders as missing credentials."""
        return bool(value and value.strip() and value.strip().upper() not in {
            "CHANGE_ME",
            "YOUR_API_KEY",
        })

    @staticmethod
    def _fallback_label(evidence_pack: dict[str, Any] | None) -> str:
        """Return an auditable name for the fallback actually being used."""
        if isinstance((evidence_pack or {}).get("comparison"), dict):
            return "structured comparison matrix fallback"
        return "grounded RAG fallback"

    @staticmethod
    def _is_degraded_rag_answer(answer: str) -> bool:
        """Whether the RAG layer returned a remote-only failure marker.

        A degraded marker is not a semantic draft: it is followed by copied
        evidence excerpts and may contain unrelated anchors or table labels.
        It is therefore suitable as audit context, but not as a fidelity
        baseline for deciding whether a remote synthesis must be discarded.
        """
        degraded_markers = (
            "llm 生成暂时失败",
            "llm generation temporarily failed",
            "以下为原始检索证据",
            "following is raw retrieval evidence",
            "根据表格证据，下面逐项列出",
            "以下内容可直接作为答案依据",
        )
        folded = str(answer or "").casefold()
        return any(marker in folded for marker in degraded_markers)

    def _local_fallback(
        self,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """安全回退：当本地/外部综合失败或没有证据时，直接返回 RAG 原答案。

        这不是“agent 再生成一遍”的路径，而是兜底，确保用户始终有答案可看。
        """
        answer = str(rag_answer or "")
        # A comparison fallback must *always* remain structured.  Reusing the
        # RAG answer is unsafe here: ``rag.answer`` is allowed to return a
        # legacy flat draft (or a very large table dump), and that draft can
        # silently mix the two paper sides after a DeepSeek parse/transport
        # failure.  The matrix is the authoritative, side-aware fallback and
        # already carries the missing/conflict markers required by v1.
        comparison = (evidence_pack or {}).get("comparison")
        if comparison:
            try:
                from app.services.comparison import ComparisonService

                items = (evidence_pack or {}).get("items") or []
                structured = ComparisonService.render_draft(comparison, items)
                if structured.strip():
                    answer = structured
            except Exception:  # pragma: no cover - defensive fallback only
                logger.exception("Failed to render comparison fallback")
        return {
            "answer_markdown": answer,
            "cited_indexes": list(range(len(citations))),
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        }

    def _comparison_fallback(
        self,
        *,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        warning: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        result = self._local_fallback(rag_answer, citations, evidence_pack)
        if provider:
            result["provider"] = provider
        if model:
            result["model"] = model
        if warning:
            result.setdefault("warnings", []).append(warning)
        return result

    def _local_synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        target: InferenceTarget,
        narrow_context: bool = False,
    ) -> dict[str, Any]:
        """使用本地 Ollama 模型对证据进行综合，生成带引用的最终答案。"""
        if not citations and not (evidence_pack or {}).get("items"):
            return self._local_fallback(rag_answer, citations)

        messages = self._build_local_messages(
            query=query,
            route=route,
            conversation_summary=conversation_summary,
            rag_answer=rag_answer,
            citations=citations,
            evidence_pack=evidence_pack,
            narrow_context=narrow_context,
        )
        ollama = self._ollama or OllamaClient(
            base_url=target.base_url,
            embedding_base_url=self._settings.ollama_embedding_base_url,
        )

        try:
            generated = ollama.generate_chat(
                messages=messages,
                model=target.model,
                context_length=target.context_length,
                max_output_tokens=768,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local Ollama synthesis failed: %s", exc)
            result = self._local_fallback(rag_answer, citations)
            result["warnings"].append(
                f"Local Ollama synthesis failed; using evidence fallback: {exc}"
            )
            return result

        answer_markdown = str(generated.get("content") or "").strip()
        if not answer_markdown:
            result = self._local_fallback(rag_answer, citations)
            result["warnings"].append(
                "Local Ollama synthesis returned an empty answer; using evidence fallback."
            )
            return result

        answer_markdown = self._sanitize_inline_citations(
            answer_markdown, len(citations)
        )
        warnings: list[str] = []

        # ---- bounded evidence-anchor coverage retry (evidence-heavy routes) ----
        # At most one retry per request; the retry prompt lists only generic
        # anchors extracted from the supplied evidence (never benchmark terms).
        if answer_markdown and self._should_retry_for_coverage(
            route=route,
            evidence_pack=evidence_pack,
            citations=citations,
            answer_text=answer_markdown,
        ):
            retry_result = self._retry_local_with_anchors(
                client=ollama,
                model=str(generated.get("model") or target.model),
                context_length=target.context_length,
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                first_answer=answer_markdown,
                narrow_context=narrow_context,
            )
            if retry_result is not None:
                answer_markdown = retry_result["answer_markdown"]
                warnings = list(retry_result["warnings"])
            else:
                warnings.append(
                    "Evidence coverage retry did not produce a usable local result; "
                    "using first synthesis."
                )
            answer_markdown = self._sanitize_inline_citations(
                answer_markdown, len(citations)
            )

        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": self._extract_cited_indexes(
                answer_markdown, len(citations)
            ),
            "warnings": warnings,
            "confidence": 1.0,
            "provider": "local",
            "model": str(generated.get("model") or target.model),
            "performance": self._performance_metadata(generated),
        }

    def _build_local_messages(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        narrow_context: bool = False,
    ) -> list[dict[str, str]]:
        """构建本地 Ollama 的 system + user prompt。

        Prompt 设计要点：
        1. 明确告诉模型“综合证据，而不是重复 RAG 草稿”。
        2. 强制使用与问题相同的语言（中文问题用中文回答）。
        3. 只能引用提供的 [index] 引用，不能编造事实。
        4. evidence_pack 里的标签是检索元数据，不是引用索引，避免误引用。
        5. ``narrow_context``（9.3.1）：跨轮场景为 True 时只保留
           conversation_summary 事实、RAG answer、citations 与结构化
           table_facts，不含 evidence_pack items 摘录（证据全集）。
        """
        evidence_parts: list[str] = []
        for index, citation in enumerate(citations):
            excerpt = str(citation.get("excerpt") or "").strip()
            title = citation.get("page_title") or citation.get("document_id") or "untitled"
            if excerpt:
                evidence_parts.append(f"[{index}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) or "(no citation excerpts)"
        evidence_pack_text = ""
        if not narrow_context:
            evidence_pack_text = self._format_evidence_pack_section(evidence_pack)
            evidence_pack_text = re.sub(
                r"(?m)^  \[(\d+)\]", r"  evidence-item-\1", evidence_pack_text
            )
        table_facts_text = self._format_table_facts_section(evidence_pack)
        precision_rules = self._precision_rules(route)
        comparison_text = self._format_comparison_section(evidence_pack, citations)
        comparison_rules = self._comparison_rules(evidence_pack)

        system_prompt = (
            "You are an evidence-grounded knowledge-base assistant. Answer the user's "
            "question by synthesizing the supplied evidence. The final answer MUST use the same language "
            "as the user's question. When the question contains Chinese, write the answer in Chinese "
            "except for necessary proper names, formulas, and quoted technical terms. "
            "Do not merely repeat the RAG draft. Do not add facts that are absent from "
            "the evidence. Cite supported statements with the supplied zero-based "
            "citation indexes, such as [0]. If evidence is insufficient, say exactly "
            "what cannot be established. Evidence-pack item labels are retrieval metadata, "
            "not citation indexes; cite only indexes listed under Citation excerpts.\n"
            + precision_rules
            + comparison_rules
        )
        user_prompt = (
            f"User question: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG draft:\n{rag_answer}\n\n"
            f"Citation excerpts:\n{evidence_text}\n\n"
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + (f"{comparison_text}\n\n" if comparison_text else "")
            + precision_rules
            + "\nReturn a concise final answer grounded only in this evidence, preserving every "
            "supported exact value and term, and match the question's language."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_cited_indexes(answer_markdown: str, max_index: int) -> list[int]:
        return list(
            dict.fromkeys(
                int(index)
                for index in re.findall(r"\[(\d+)\]", answer_markdown)
                if 0 <= int(index) < max_index
            )
        )

    @staticmethod
    def _sanitize_inline_citations(answer_markdown: str, max_index: int) -> str:
        """Remove inline numeric markers that cannot resolve to a citation."""
        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            return match.group(0) if 0 <= index < max_index else ""

        return re.sub(r"\[(-?\d+)\]", replace, answer_markdown)

    @staticmethod
    def _emit_citations(
        result: dict[str, Any],
        citations: list[dict[str, Any]],
        event_sink: Callable[[str, dict[str, Any]], None],
    ) -> None:
        for index in result.get("cited_indexes", []):
            if isinstance(index, int) and 0 <= index < len(citations):
                event_sink("citation", {"index": index, **citations[index]})

    @staticmethod
    def _performance_metadata(
        generated: dict[str, Any], *, first_token_ms: int | None = None
    ) -> dict[str, Any]:
        prompt_tokens = int(
            generated.get("prompt_eval_count")
            or generated.get("prompt_tokens")
            or 0
        )
        completion_tokens = int(
            generated.get("eval_count")
            or generated.get("completion_tokens")
            or 0
        )
        eval_duration = int(generated.get("eval_duration") or 0)
        tokens_per_second = (
            round(completion_tokens / (eval_duration / 1_000_000_000), 2)
            if completion_tokens and eval_duration
            else 0.0
        )
        return {
            "first_token_ms": first_token_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tokens_per_second,
            "load_ms": round(int(generated.get("load_duration") or 0) / 1_000_000, 1),
            "total_ms": round(int(generated.get("total_duration") or 0) / 1_000_000, 1),
        }

    def _default_target(self) -> InferenceTarget:
        """Provide the configured generation target for direct callers."""
        return InferenceTarget(
            profile="generation",
            base_url=self._settings.ollama_generation_base_url.rstrip("/"),
            model=self._settings.ollama_generation_model,
            context_length=self._settings.ollama_generation_context_length,
            reason="Default local synthesis target",
        )

    @staticmethod
    def _format_evidence_pack_section(
        evidence_pack: dict[str, Any] | None,
    ) -> str:
        """Build a compact, bounded evidence-pack prompt section.

        Each included line preserves: item index, document_id, evidence_kind,
        source_stage, support_hint, and excerpt text.

        The section is explicitly capped:
        - At most ``MAX_EVIDENCE_PACK_ITEMS`` items.
        - Each excerpt truncated to ``MAX_EXCERPT_CHARS`` characters.

        Returns an empty string when *evidence_pack* is missing or empty.
        """
        if not evidence_pack or not evidence_pack.get("items"):
            return ""

        items: list[dict[str, Any]] = evidence_pack["items"]
        max_items = AgentSynthesizer.MAX_EVIDENCE_PACK_ITEMS
        max_chars = AgentSynthesizer.MAX_EXCERPT_CHARS
        emitted = 0
        total = len(items)

        lines: list[str] = []
        for item in items:
            if emitted >= max_items:
                break
            excerpt = str(item.get("excerpt", ""))
            if not excerpt:
                continue
            # Truncate long excerpts
            if len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars] + "…"
            idx = item.get("index", "?")
            doc_id = item.get("document_id", "?")
            kind = item.get("evidence_kind", "")
            stage = item.get("source_stage", "")
            hint = item.get("support_hint", "")
            lines.append(
                f"  [{idx}] doc={doc_id} kind={kind} stage={stage} "
                f"hint={hint}: {excerpt}"
            )
            emitted += 1

        if not lines:
            return ""

        header = f"Evidence pack items ({emitted} of {total}):"
        suffix = ""
        if emitted < total:
            suffix = f"\n  … ({total - emitted} more items omitted)"
        return header + "\n" + "\n".join(lines) + suffix

    @staticmethod
    def _format_comparison_section(
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]] | None = None,
    ) -> str:
        """Render the side-aware comparison matrix for synthesis prompts.

        The ordinary evidence-pack section is intentionally flat for legacy
        RAG calls.  Comparison synthesis receives this additional bounded
        section so a model can never mistake evidence from the left paper for
        evidence from the right paper.  Cell statuses are authoritative:
        ``missing`` and ``conflict`` must be reflected rather than filled in
        from model priors.
        """
        if not evidence_pack:
            return ""
        comparison = evidence_pack.get("comparison")
        if not isinstance(comparison, dict):
            return ""
        papers = comparison.get("papers") or []
        cells = comparison.get("cells") or []
        items = evidence_pack.get("items") or []
        citations = citations or []

        paper_labels: dict[str, str] = {}
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("document_id") or "")
            if paper_id:
                paper_labels[paper_id] = str(
                    paper.get("title") or paper_id
                )

        def item_for(index: Any) -> dict[str, Any] | None:
            try:
                numeric = int(index)
            except (TypeError, ValueError):
                return None
            if 0 <= numeric < len(items) and isinstance(items[numeric], dict):
                return items[numeric]
            # Some older EvidencePack producers expose one-based ``index``
            # values.  Resolve that representation without changing the wire
            # schema used by the current producer.
            for candidate in items:
                if isinstance(candidate, dict) and candidate.get("index") == numeric:
                    return candidate
            return None

        def citation_indexes_for(cell: dict[str, Any], refs: list[Any]) -> list[int]:
            values: list[int] = []
            raw = cell.get("citation_indexes") or []
            for value in raw:
                try:
                    candidate = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= candidate < len(citations) and candidate not in values:
                    values.append(candidate)
            # Fall back to source identity matching when the matrix was built
            # before citation indexes were materialized.
            paper_id = str(cell.get("paper_id") or "")
            for ref in refs:
                item = item_for(ref)
                if not item:
                    continue
                chunk_id = item.get("chunk_id")
                for idx, citation in enumerate(citations):
                    if idx in values:
                        continue
                    if (
                        str(citation.get("document_id") or "") == paper_id
                        and (not chunk_id or str(citation.get("chunk_id") or "") == str(chunk_id))
                    ):
                        values.append(idx)
            return values

        lines = [
            "Comparison evidence matrix (authoritative; page-level evidence only):",
            f"Mode: {comparison.get('mode', 'cross_paper')}; "
            f"scope={'explicit' if comparison.get('explicit_scope') else 'auto-routed'}",
        ]
        if paper_labels:
            lines.append(
                "Papers: "
                + " | ".join(
                    f"{index + 1}. {title} (paper_id={paper_id})"
                    for index, (paper_id, title) in enumerate(paper_labels.items())
                )
            )
        if not cells:
            if comparison.get("status") == "needs_selection":
                lines.append(
                    "Status: needs_selection — no cross-paper conclusion may be generated."
                )
            return "\n".join(lines)

        lines.append("Cells (one row per paper × dimension):")
        for cell in cells[:60]:
            if not isinstance(cell, dict):
                continue
            paper_id = str(cell.get("paper_id") or "")
            paper_title = str(
                cell.get("paper_title") or paper_labels.get(paper_id) or paper_id
            )
            dimension = str(cell.get("dimension") or "")
            status = str(cell.get("status") or "missing")
            refs = list(cell.get("evidence_indexes") or [])
            citation_indexes = citation_indexes_for(cell, refs)
            citation_hint = (
                ", ".join(f"citation[{idx}]" for idx in citation_indexes)
                if citation_indexes
                else "no citation"
            )
            lines.append(
                f"- paper={paper_id} ({paper_title}); dimension={dimension}; "
                f"status={status}; {citation_hint}"
            )
            if status in {"missing", "conflict"}:
                notes = "; ".join(str(note) for note in (cell.get("notes") or []))
                lines.append(
                    f"  -> {status.upper()}: {notes or 'do not infer or merge across papers.'}"
                )
                continue
            # Include at most two short excerpts per cell.  The citation
            # excerpt block remains the canonical text; this line is a side /
            # dimension binding, not a second citation namespace.
            for ref in refs[:2]:
                item = item_for(ref)
                if not item:
                    continue
                excerpt = re.sub(r"\s+", " ", str(item.get("excerpt") or "")).strip()
                if len(excerpt) > 240:
                    excerpt = excerpt[:240] + "…"
                if excerpt:
                    page = item.get("page_label") or item.get("page_title") or "page ?"
                    lines.append(
                        f"  -> evidence-item={ref}; page={page}; "
                        f"citation={citation_hint}; excerpt={excerpt}"
                    )

        missing = comparison.get("missing_cells") or []
        conflicts = comparison.get("conflict_cells") or []
        if missing:
            lines.append("Missing cells (must be stated explicitly): " + ", ".join(map(str, missing)))
        if conflicts:
            lines.append("Conflict cells (report both claims and citations): " + ", ".join(map(str, conflicts)))
        lines.append(
            "Comparison output contract: organize the answer by dimension, keep each "
            "paper's evidence on its own side, cite the correct paper for every claim, "
            "and write '证据缺失/insufficient evidence' for missing cells. Never infer "
            "a value or resolve a conflict without page evidence."
        )
        return "\n".join(lines)

    @staticmethod
    def _comparison_rules(evidence_pack: dict[str, Any] | None) -> str:
        if not isinstance((evidence_pack or {}).get("comparison"), dict):
            return ""
        return (
            "Comparison rules (mandatory): treat the comparison evidence matrix as "
            "authoritative; cover every listed dimension for every listed paper; "
            "do not use Paper A evidence as Paper B evidence; preserve page citations; "
            "mark missing cells and conflicts explicitly instead of guessing. "
            "Return a structured comparison by dimension, not a flat evidence dump.\n"
        )

    @staticmethod
    def _format_table_facts_section(
        evidence_pack: dict[str, Any] | None,
    ) -> str:
        """Render exact table facts without the generic excerpt cap."""

        if not evidence_pack or not evidence_pack.get("table_facts"):
            return ""
        facts = evidence_pack.get("table_facts")
        if not isinstance(facts, list):
            return ""

        # ``TableFactEvidence`` deliberately stores the stable canonical
        # ``table_id`` rather than duplicating a caption.  That identity is
        # sufficient for storage and deduplication, but it is opaque to the
        # synthesis model (for example, ``table-2ead...`` does not tell it
        # that a fact came from Table 2).  Recover the human-readable label
        # from the exact citation excerpt that belongs to the same table.
        # Keeping this mapping at prompt-render time avoids changing the
        # public evidence schema or weakening the canonical identity rules.
        table_labels: dict[str, str] = {}
        table_captions: dict[str, str] = {}
        table_label_re = re.compile(
            r"\btable\s+(?:s\s*)?(?:\d+|[ivxlcdm]+)\b",
            re.IGNORECASE,
        )
        items = evidence_pack.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                table_id = str(item.get("table_id") or "").strip()
                excerpt = str(item.get("excerpt") or "")
                if not table_id or not excerpt:
                    continue
                match = table_label_re.search(excerpt)
                if match:
                    table_labels.setdefault(table_id, match.group(0))
                    # Keep the first caption line as a semantic disambiguator
                    # for repeated labels such as two different ``Table 7``s.
                    # Captions are display metadata only; table identity stays
                    # bound to the canonical table_id.
                    caption = excerpt[match.end() :].lstrip(" .:\t")
                    caption = caption.splitlines()[0].split("|", 1)[0].strip()
                    caption = re.sub(r"<[^>]+>", "", caption)
                    caption = re.sub(r"\s+", " ", caption).strip(" .")
                    if caption:
                        table_captions.setdefault(table_id, caption[:240])

        lines: list[str] = []
        legend_lines = ["Table fact source legend:"]
        for table_id, table_label in table_labels.items():
            caption = table_captions.get(table_id)
            if caption:
                legend_lines.append(
                    f"  table={table_id} label={table_label} caption={caption}"
                )
        for fact in facts[: AgentSynthesizer.MAX_TABLE_FACTS]:
            if not isinstance(fact, dict):
                continue
            value = str(fact.get("value") or "").strip()
            if not value:
                continue
            table_id = str(fact.get("table_id") or "?")
            table_label = table_labels.get(table_id, "unknown")
            # 9.3.2：渲染 fact_id / unit / term 结构化字段（有值才带上），
            # 使期望 facts 清单与模型可见字段一致。
            unit = str(fact.get("unit") or "").strip()
            term = str(fact.get("term") or "").strip()
            fact_id = str(fact.get("fact_id") or "").strip()
            suffix = ""
            if unit or term:
                suffix = f" unit={unit or '?'} term={term or '?'}"
            if fact_id:
                suffix += f" fact_id={fact_id}"
            lines.append(
                "  table={table} label={label} row={row} column={column} "
                "value={value}{suffix}".format(
                    table=table_id,
                    label=table_label,
                    row=fact.get("row_label", "?"),
                    column=fact.get("column", "?"),
                    value=value,
                    suffix=suffix,
                )
            )
        if not lines:
            return ""
        prefix = "\n".join(legend_lines) if len(legend_lines) > 1 else ""
        return (
            "Canonical table facts (exact source values; preserve every value "
            "and keep each fact with its labeled source table):\n"
            + (prefix + "\n" if prefix else "")
            + "\n".join(lines)
        )

    def _ollama_synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        narrow_context: bool = False,
    ) -> dict[str, Any]:
        """Synthesize evidence using local Ollama.

        ``narrow_context``（9.3.1）与 local/external 对齐：本路径天然不含
        evidence_pack items 摘录（只用 citations + table_facts），参数仅为
        接口一致性，行为不变。
        """
        model = self._settings.ollama_synthesis_model or self._settings.ollama_generation_model
        client = self._ollama or OllamaClient(
            base_url=self._settings.ollama_generation_base_url,
            embedding_base_url=self._settings.ollama_embedding_base_url,
        )

        evidence_parts: list[str] = []
        for i, c in enumerate(citations):
            excerpt = c.get("excerpt", "")
            title = c.get("page_title") or c.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{i}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"
        table_facts_text = self._format_table_facts_section(evidence_pack)
        precision_rules = self._precision_rules(route)
        comparison_text = self._format_comparison_section(evidence_pack, citations)
        comparison_rules = self._comparison_rules(evidence_pack)

        system_prompt = (
            "You are an evidence synthesis assistant. Your task is to synthesize "
            "a final answer from the provided RAG answer and evidence excerpts. "
            "You MUST synthesize - do NOT simply concatenate paragraphs. "
            "Only cite sources that are present in the provided evidence.\n"
            + precision_rules
            + comparison_rules
            + "\n"
            + self._answer_rules(query)
        )

        user_prompt = (
            f"Original query: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + (f"{comparison_text}\n\n" if comparison_text else "")
            + precision_rules
            + "\nSynthesize a final answer from the above evidence. "
            "Return a JSON object with: answer_markdown, cited_indexes, warnings, confidence."
        )

        try:
            result = client.generate_structured(
                SynthesisPayload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
            )
        except Exception as exc:
            logger.warning("Ollama synthesis failed: %s", exc)
            fallback = self._local_fallback(rag_answer, citations)
            fallback["warnings"].append(
                f"Ollama synthesis failed ({exc}); using local fallback."
            )
            return fallback

        max_idx = len(citations)
        sanitized = [
            i for i in result.cited_indexes
            if isinstance(i, int) and 0 <= i < max_idx
        ]

        answer_markdown = str(result.answer_markdown or "").strip()
        warnings = list(result.warnings)
        if answer_markdown and self._should_retry_for_coverage(
            route=route,
            evidence_pack=evidence_pack,
            citations=citations,
            answer_text=answer_markdown,
        ):
            retry_result = self._retry_ollama_with_anchors(
                client=client,
                model=model,
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                first_answer=answer_markdown,
                narrow_context=narrow_context,
            )
            if retry_result is not None:
                return retry_result
            warnings.append(
                "Evidence coverage retry did not produce a usable Ollama result; "
                "using first synthesis."
            )

        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": sanitized,
            "warnings": warnings,
            "confidence": float(result.confidence),
            "provider": "ollama",
            "model": model,
        }

    @staticmethod
    def _answer_rules(query: str) -> str:
        """T3 生成约束：语言跟随 / 元数据禁止 / 假设标记。

        2026-08-12 锁定回归实测失败形态：
        - R29/R38：中文问题，综合答案却输出英文 → 语言必须跟随用户；
        - R36：答案泄漏内部元数据（"关联地址为 21201 及 48824"、
          "相关文献编号为 2013"）→ 元数据禁止输出；
        - R26：基于证据的推断（作者取舍原因）未与事实区分 → 推断需显式标记。

        文本来自共享模块 ``prompt_rules``（与 search draft 同源，防漂移）。
        """
        from app.services.prompt_rules import answer_rules

        return answer_rules(query)

    @staticmethod
    def _precision_rules(route: str) -> str:
        """Return first-pass fidelity rules for evidence-heavy synthesis."""
        if route in {"table_or_metric", "evidence_required", "multi_source_compare"}:
            return (
                "Precision rules (mandatory for this evidence-heavy route):\n"
                "1. Preserve every supported exact numeric value and unit verbatim; do not replace a value with only a trend or qualitative summary.\n"
                "2. Preserve technical abbreviations, model names, dataset names, table labels, row names, and column names exactly as shown in the evidence.\n"
                "3. For table questions, use Canonical table facts as the primary source, report the requested row/column values, and keep the Table N label with each fact.\n"
                "4. Do not summarize away exact values or omit a requested evidence item; if it is absent, say so explicitly.\n"
            )
        return (
            "Preserve supported technical terms, table labels, exact numeric values, and units verbatim when they are relevant to the question.\n"
        )

    def _retry_ollama_with_anchors(
        self,
        *,
        client: OllamaClient,
        model: str,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        first_answer: str,
        narrow_context: bool = False,
    ) -> dict[str, Any] | None:
        """Retry local synthesis once with exact evidence anchors."""
        anchors = self._extract_evidence_anchors(evidence_pack, citations)
        missing = [anchor for anchor in anchors if anchor.casefold() not in first_answer.casefold()]
        if not missing:
            return None

        evidence_parts = []
        for index, citation in enumerate(citations):
            excerpt = citation.get("excerpt", "")
            title = citation.get("page_title") or citation.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{index}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"
        evidence_pack_text = (
            "" if narrow_context else self._format_evidence_pack_section(evidence_pack)
        )
        table_facts_text = self._format_table_facts_section(evidence_pack)
        comparison_text = self._format_comparison_section(evidence_pack, citations)
        comparison_rules = self._comparison_rules(evidence_pack)
        missing_list = ", ".join(missing[:12])
        system_prompt = (
            "You are an evidence synthesis assistant revising an incomplete answer. "
            "Keep the synthesis concise, but explicitly include every supported exact "
            f"evidence anchor in this list: {missing_list}. Preserve acronyms, table labels, "
            "technical terms, and numeric values verbatim; do not translate or drop them. "
            "Only cite sources present in the evidence. Return JSON with answer_markdown, "
            "cited_indexes, warnings, and confidence.\n"
            + self._answer_rules(query)
        )
        user_prompt = (
            f"Original query: {query}\nRoute type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"First synthesis: {first_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + f"The first synthesis omitted these exact anchors: {missing_list}. "
            "Rewrite the answer and include them wherever supported by the evidence."
        )
        try:
            retry = client.generate_structured(
                SynthesisPayload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama evidence coverage retry failed: %s", exc)
            return None

        answer_markdown = str(retry.answer_markdown or "").strip()
        if not answer_markdown:
            return None
        max_idx = len(citations)
        cited_indexes = [
            index
            for index in retry.cited_indexes
            if isinstance(index, int) and 0 <= index < max_idx
        ]
        warnings = list(retry.warnings)
        warnings.append("Evidence coverage retry performed for Ollama synthesis.")
        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": cited_indexes,
            "warnings": warnings,
            "confidence": float(retry.confidence),
            "provider": "ollama",
            "model": model,
        }

    def _retry_local_with_anchors(
        self,
        *,
        client: OllamaClient,
        model: str,
        context_length: int,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        first_answer: str,
        narrow_context: bool = False,
    ) -> dict[str, Any] | None:
        """Retry local synthesis once with exact evidence anchors.

        Mirrors ``_retry_ollama_with_anchors`` but uses the plain
        ``generate_chat`` channel that the local provider already uses, so a
        retry never forces structured JSON on the local model.  The anchor list
        comes from the generic ``_extract_evidence_anchors`` rules (numeric
        values, units, abbreviations) and contains no benchmark-specific terms.
        """
        anchors = self._extract_evidence_anchors(evidence_pack, citations)
        missing = [
            anchor
            for anchor in anchors
            if anchor.casefold() not in first_answer.casefold()
        ]
        if not missing:
            return None

        evidence_parts = []
        for index, citation in enumerate(citations):
            excerpt = citation.get("excerpt", "")
            title = citation.get("page_title") or citation.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{index}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"
        evidence_pack_text = (
            "" if narrow_context else self._format_evidence_pack_section(evidence_pack)
        )
        table_facts_text = self._format_table_facts_section(evidence_pack)
        missing_list = ", ".join(missing[:12])
        system_prompt = (
            "You are an evidence synthesis assistant revising an incomplete answer. "
            "Keep the synthesis concise, but explicitly include every supported exact "
            f"evidence anchor in this list: {missing_list}. Preserve acronyms, table labels, "
            "technical terms, and numeric values verbatim; do not translate or drop them. "
            "Only cite sources present in the evidence."
        )
        user_prompt = (
            f"User question: {query}\nRoute type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG draft:\n{rag_answer}\n\n"
            f"First synthesis:\n{first_answer}\n\n"
            f"Citation excerpts:\n{evidence_text}\n\n"
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + f"The first synthesis omitted these exact anchors: {missing_list}. "
            "Rewrite the answer and include them wherever supported by the evidence."
        )
        try:
            retry = client.generate_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                context_length=context_length,
                max_output_tokens=768,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local evidence coverage retry failed: %s", exc)
            return None

        answer_markdown = str(retry.get("content") or "").strip()
        if not answer_markdown:
            return None
        answer_markdown = self._sanitize_inline_citations(
            answer_markdown, len(citations)
        )
        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": self._extract_cited_indexes(
                answer_markdown, len(citations)
            ),
            "warnings": ["Evidence coverage retry performed for local synthesis."],
            "confidence": 1.0,
            "provider": "local",
            "model": str(retry.get("model") or model),
        }

    def _apply_fidelity_guards(
        self,
        *,
        question: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        synthesized_answer: str,
        warnings: list[str],
        fallback_provider: str | None = None,
        fallback_model: str | None = None,
        guard_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """依次执行 fidelity 守卫，任一失败返回 RAG-draft 回退，全部通过返回 None。

        守卫（provider-uniform，synthesize 与 synthesize_stream 共用）：
        1. draft-fidelity：合成答案丢失 RAG 草稿覆盖的 evidence-anchor；
        2. 新数字打回（9.3.3）：合成答案引入证据不支持的数值（模型编造）；
        3. 表号保留（9.3.3）：合成答案删除证据中的表号（如 Table 7）；
        4. 期望 facts 遗漏（9.3.2b）：draft 覆盖的结构化字段在合成中缺失。

        ``guard_meta``（可选，供调用方做 trace）：写入 9.8 的需求侧期望
        facts 三元状态（complete / missing / unknown），任一守卫失败也
        保留，保证审计可见。
        """
        # 9.8：先计算需求侧期望状态（inventory × 已返回 facts 的 exact
        # join，不依赖任一守卫是否触发）
        expected_status, _ = self._expected_facts_status(question, evidence_pack)
        if guard_meta is not None:
            guard_meta["expected_facts_status"] = expected_status
        comparison_failures = self._comparison_guard_failures(
            evidence_pack=evidence_pack,
            citations=citations,
            synthesized_answer=synthesized_answer,
        )
        if comparison_failures:
            return self._fidelity_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                warnings=warnings,
                provider=fallback_provider,
                model=fallback_model,
                reason=(
                    "Comparison fidelity guard failed; returned the structured evidence matrix: "
                    + "; ".join(comparison_failures)
                ),
            )
        # A comparison draft is an evidence matrix, not a prose answer.  Its
        # excerpts intentionally contain bibliography, OCR fragments, and
        # repeated anchors that a good synthesis should compress rather than
        # repeat verbatim.  The side-aware comparison guard above already
        # enforces paper/citation coverage; applying the generic draft-anchor
        # guard here would reject grounded paraphrases and return the matrix
        # unnecessarily.  Keep the stricter anchor check for legacy routes.
        if isinstance((evidence_pack or {}).get("comparison"), dict):
            guard = None
        else:
            guard = self._draft_fidelity_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                synthesized_answer=synthesized_answer,
                warnings=warnings,
                fallback_provider=fallback_provider,
                fallback_model=fallback_model,
            )
        if guard is not None:
            return guard
        unsupported = self._synthesis_unsupported_numbers(
            synthesized_answer,
            evidence_pack=evidence_pack,
            citations=citations,
        )
        if unsupported:
            return self._fidelity_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                warnings=warnings,
                provider=fallback_provider,
                model=fallback_model,
                reason=(
                    "Synthesized answer introduced numbers not supported by the evidence "
                    f"({', '.join(sorted(unsupported))}); using the RAG draft to avoid "
                    "fabricated values."
                ),
            )
        dropped_labels = self._synthesis_dropped_table_labels(
            synthesized_answer,
            evidence_pack=evidence_pack,
            citations=citations,
        )
        remote_degraded_draft = (
            fallback_provider == "deepseek"
            and self._is_degraded_rag_answer(rag_answer)
        )
        comparison_route = isinstance((evidence_pack or {}).get("comparison"), dict)
        if dropped_labels and not remote_degraded_draft and not comparison_route:
            return self._fidelity_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                warnings=warnings,
                provider=fallback_provider,
                model=fallback_model,
                reason=(
                    "Synthesized answer dropped evidence table labels "
                    f"({', '.join(sorted(dropped_labels))}); using the RAG draft to "
                    "preserve the table references."
                ),
            )
        if dropped_labels and (remote_degraded_draft or comparison_route):
            # The remote-only RAG placeholder often contains bibliography and
            # unrelated table labels. Requiring a synthesis to repeat every
            # one would reject a focused answer for the wrong reason. The same
            # applies to a comparison matrix: page citations and the
            # side-aware guard are authoritative; numeric and structured-fact
            # guards below still enforce grounding.
            warnings.append(
                "Comparison/remote draft table labels were not used as a synthesis requirement."
            )
        # 9.3.2b 期望 facts 遗漏校验：draft 覆盖的 table_facts 字段
        # （value/term/unit）在合成答案中缺失 → 回退。对照结构化 facts
        # （而非通用 anchor）。
        missing_fields, _ = self._synthesis_missing_fact_fields(
            question=question,
            rag_answer=rag_answer,
            synthesized_answer=synthesized_answer,
            evidence_pack=evidence_pack,
        )
        if missing_fields:
            return self._fidelity_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                warnings=warnings,
                provider=fallback_provider,
                model=fallback_model,
                reason=(
                    "Synthesized answer omitted expected fact fields "
                    f"({', '.join(sorted(missing_fields))}); using the RAG draft to "
                    "preserve the exact facts."
                ),
            )
        return None

    @staticmethod
    def _comparison_guard_failures(
        *,
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
        synthesized_answer: str,
    ) -> list[str]:
        """Validate the non-negotiable side/coverage invariants for v1.

        This guard intentionally does not judge scientific truth.  It only
        checks that the synthesized response still has a citation path for
        every supported paper-side cell and that missing/conflict cells are
        not silently presented as facts.  If the model omitted inline markers
        altogether, the existing citation-array contract remains authoritative
        and we do not reject the answer solely for formatting.
        """
        comparison = (evidence_pack or {}).get("comparison")
        if not isinstance(comparison, dict):
            return []
        items = evidence_pack.get("items") or []
        failures: list[str] = []
        paper_ids = {
            str(paper.get("document_id") or "")
            for paper in (comparison.get("papers") or [])
            if isinstance(paper, dict) and paper.get("document_id")
        }

        def source_for_ref(ref: Any) -> str:
            try:
                idx = int(ref)
            except (TypeError, ValueError):
                return ""
            item = None
            if 0 <= idx < len(items) and isinstance(items[idx], dict):
                item = items[idx]
            if item is None:
                item = next(
                    (
                        candidate
                        for candidate in items
                        if isinstance(candidate, dict) and candidate.get("index") == idx
                    ),
                    None,
                )
            return str((item or {}).get("document_id") or "")

        for cell in comparison.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            paper_id = str(cell.get("paper_id") or "")
            dimension = str(cell.get("dimension") or "")
            status = str(cell.get("status") or "missing")
            refs = list(cell.get("citation_indexes") or cell.get("evidence_indexes") or [])
            if status == "supported" and refs:
                if not any(
                    source_for_ref(ref) == paper_id
                    or (
                        isinstance(ref, int)
                        and 0 <= ref < len(citations)
                        and str(citations[ref].get("document_id") or "") == paper_id
                    )
                    for ref in refs
                ):
                    failures.append(f"{paper_id}:{dimension} has no same-paper citation")
            elif status == "supported" and not refs:
                failures.append(f"{paper_id}:{dimension} has no evidence reference")

        answer_lower = str(synthesized_answer or "").casefold()
        if comparison.get("missing_cells"):
            missing_tokens = ("missing", "insufficient evidence", "证据缺失", "缺少证据", "无法确定")
            if not any(token in answer_lower for token in missing_tokens):
                failures.append("missing cells were not acknowledged")
        if comparison.get("conflict_cells"):
            conflict_tokens = ("conflict", "contradict", "冲突", "不一致", "差异")
            if not any(token in answer_lower for token in conflict_tokens):
                failures.append("conflict cells were not acknowledged")

        # When inline markers are present, every selected paper must have at
        # least one marker resolving to that paper.  This catches the common
        # failure where the model answers both sides but cites only Paper A.
        marker_indexes = [
            int(value)
            for value in re.findall(r"\[(\d+)\]", str(synthesized_answer or ""))
            if 0 <= int(value) < len(citations)
        ]
        if marker_indexes and len(paper_ids) >= 2:
            cited_papers = {
                str(citations[index].get("document_id") or "")
                for index in marker_indexes
            }
            for paper_id in sorted(paper_ids):
                if paper_id not in cited_papers:
                    failures.append(f"paper {paper_id} has no inline citation")
        return failures[:8]

    def _fidelity_fallback(
        self,
        *,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        warnings: list[str],
        reason: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """构造 RAG-draft 回退结果，保留原 warnings 并追加守卫失败原因。"""
        fallback = self._local_fallback(rag_answer, citations, evidence_pack)
        # A rejected DeepSeek synthesis is still a DeepSeek request whose
        # grounded draft was retained.  Do not report it as an Ollama/local
        # model fallback: that makes provider regressions impossible to audit
        # and incorrectly suggests the local model was called.
        if provider:
            fallback["provider"] = provider
            fallback["model"] = model or fallback["model"]
        fallback["warnings"] = list(warnings)
        fallback["warnings"].append(reason)
        if provider == "deepseek":
            fallback["warnings"].append(
                "DeepSeek synthesis was rejected by a fidelity guard; returned the "
                + self._fallback_label(evidence_pack)
                + "."
            )
        return fallback

    @staticmethod
    def _table_labels(text: str) -> set[str]:
        """提取文本中的表号（Table 7 / Table S3 等）。"""
        return set(
            re.findall(r"\bTable\s*(?:S\s*)?\d+(?:\.\d+)?\b", text, re.IGNORECASE)
        )

    def _expected_facts_from_question(
        self,
        question: str,
        inventory: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """9.8/任务9：需求侧期望解析，返回 ``(status, targets)``。

        ``targets`` 是由"问题 + typed inventory"解析出的需求目标，不是
        已召回 facts；每个目标携带 exact join 身份（document_id +
        parse_version + table_id + row_index/row_label + column/term）。
        行维度按显式行号（"row 3"/"第 3 行"）优先，其次自然语言行标签
        （"total row"/"average row" 等，见 _question_row_label）；两者都
        未指定时行维度为 any。无法从 inventory 唯一解析目标表/版本时
        返回 ``("unknown", [])``，绝不回退"全部已召回 facts"（原
        ``return matched or dict_facts`` 已移除，避免多表问题被无关表
        全量校验误回退）。解析成功返回 ``("resolved", targets)``，由
        调用方与已返回 table_facts 做 exact join 得到 complete / missing。
        """
        if not isinstance(inventory, list) or not inventory:
            return "unknown", []
        entries = [entry for entry in inventory if isinstance(entry, dict)]
        if not entries:
            return "unknown", []
        table_digits = self._question_table_digits(question)
        if table_digits:
            bindings = self._resolve_table_bindings(table_digits, entries)
            if bindings is None:
                return "unknown", []
        else:
            binding = self._unique_inventory_binding(entries)
            if binding is None:
                return "unknown", []
            bindings = [binding]
        columns = self._question_column_demands(question)
        if table_digits and not columns:
            # 只显式指定表 → 列通配（该表任一已返回 fact 均可满足）
            columns = [""]
        if not columns:
            return "unknown", []
        row_index = self._question_row_index(question)
        row_label = self._question_row_label(question)
        table_ref = self._question_table_ref(question)
        targets: list[dict[str, Any]] = []
        for document_id, parse_version, table_id in bindings:
            for column in columns:
                targets.append(
                    {
                        "table_ref": table_ref,
                        "table_id": table_id,
                        "document_id": document_id,
                        "parse_version": parse_version,
                        "row_index": row_index,
                        "row_label": row_label,
                        "column": column,
                        "identity": self._target_identity(
                            document_id=document_id,
                            parse_version=parse_version,
                            table_id=table_id,
                            row_index=row_index,
                            row_label=row_label,
                            column=column,
                        ),
                    }
                )
        return "resolved", targets

    def _expected_facts_status(
        self,
        question: str,
        evidence_pack: dict[str, Any] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """9.8/任务9：需求侧目标 × 已返回 table_facts 的 exact join。

        匹配顺序固定为 document_id + parse_version + table_id +
        row_index/row_label + column/term；任一需求目标未被已返回
        facts 精确满足即 ``missing``，全部满足才 ``complete``。
        ``unknown`` 只表示需求不可判定（不表示覆盖），不参与
        complete/missing 判定。
        """
        inventory = (evidence_pack or {}).get("inventory")
        status, targets = self._expected_facts_from_question(question, inventory)
        if status == "unknown":
            return "unknown", []
        returned = self._index_facts_by_identity(evidence_pack)
        missing = [
            target for target in targets if target["identity"] not in returned
        ]
        return ("complete" if not missing else "missing"), targets

    @staticmethod
    def _question_table_digits(question: str) -> set[float]:
        """提取问题中的显式表号（Table 7 / Table S3），返回数字段集合。"""
        return {
            float(digits)
            for digits in re.findall(
                r"\bTable\s*(?:S\s*)?(\d+(?:\.\d+)?)\b", question, re.IGNORECASE
            )
        }

    @staticmethod
    def _question_table_ref(question: str) -> str:
        """返回问题中第一个显式表引用原文（如 "Table 7"），无则空串。"""
        match = re.search(
            r"\bTable\s*(?:S\s*)?\d+(?:\.\d+)?\b", question, re.IGNORECASE
        )
        return match.group(0) if match else ""

    @staticmethod
    def _question_row_index(question: str) -> int | None:
        """从问题解析显式行号（"row 3" / "第 3 行"），无则返回 None。"""
        match = re.search(r"\brow\s+(\d+)\b", question, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"第\s*(\d+)\s*行", question)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _question_row_label(question: str) -> str:
        """从问题解析自然语言行标签需求（"the total row" / "average row"）。

        返回规范化（小写）后的 canonical row label；无显式 "X row" 行
        标签引用返回空串。只匹配 "X row" 形式——它是不含歧义的行引用，
        不会把 "total binding energy" 这类修饰指标的形容词误判为行需求。
        """
        match = _ROW_LABEL_PATTERN.search(question)
        return match.group(1).casefold() if match else ""

    def _question_column_demands(self, question: str) -> list[str]:
        """从问题解析列/术语需求（需求侧，不依赖已召回 facts）。

        含枚举分隔（、，,，和，and）时按条目拆分：短 token（≥2 字符）
        也视为独立列需求（如 "Asp、C6、exptl"），条目内 value/values
        视为量词移除；无枚举时整句取 ≥4 字符 token 合并为单一列需求。
        表号/行号引用先从 token 流剔除；行标签短语（"total row" 等）同样
        剔除，避免行引用污染列/术语需求。无可解析列返回空列表（调用方
        在指定表时回退列通配）。
        """
        cleaned = re.sub(
            r"\bTable\s*(?:S\s*)?\d+(?:\.\d+)?\b", " ", question, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\brow\s+\d+\b|\b第\s*\d+\s*行\b", " ", cleaned)
        cleaned = _ROW_LABEL_PATTERN.sub(" ", cleaned)
        items = _ENUMERATION_SEPARATORS.split(cleaned)
        demands: list[str] = []
        if len(items) > 1:
            for item in items:
                tokens = [
                    token.casefold()
                    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", item)
                    if token.casefold() not in _COLUMN_STOPWORDS
                ]
                while tokens and tokens[-1] in ("value", "values"):
                    tokens.pop()
                if tokens:
                    demands.append(self._norm_column(" ".join(tokens)))
        else:
            tokens = [
                token.casefold()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", cleaned)
                if token.casefold() not in _COLUMN_STOPWORDS
            ]
            if tokens:
                demands.append(self._norm_column(" ".join(tokens)))
        return demands

    @staticmethod
    def _norm_column(text: str) -> str:
        """列/术语规范化：小写 + 朴素单数化（values → value），供 exact join。"""
        token = text.strip().casefold()
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        return token

    @classmethod
    def _target_identity(
        cls,
        document_id: str,
        parse_version: str,
        table_id: str,
        row_index: int | None,
        column: str,
        row_label: str = "",
    ) -> str:
        """构造需求目标的 exact join 身份键。

        维度顺序固定为 document_id + parse_version + table_id +
        row_index/row_label + column/term；行号优先于行标签（既有
        "row N"/"第 N 行" 语义不变），行标签与 fact 侧
        _fact_identity_keys 的 label: 展开同口径规范化，行未指定用
        any，列通配用 any。
        """
        row_key = f"index:{row_index}" if row_index is not None else "any"
        if row_index is None and row_label:
            row_key = f"label:{cls._norm_column(row_label)}"
        col = cls._norm_column(column) or "any"
        return f"{document_id}|{parse_version}|{table_id}|{row_key}|{col}"

    @classmethod
    def _fact_identity_keys(cls, fact: dict[str, Any]) -> set[str]:
        """把已返回 fact 展开为 exact join 身份键集合。

        每个 fact 按 行号/行标签/任意行 × 列/术语/任意列 组合生成
        身份键，供需求目标做 set 成员判定；facts 只提供已返回事实的
        value/unit/fact_id，不定义期望集合。
        """
        document_id = str(fact.get("document_id") or "")
        parse_version = str(fact.get("parse_version") or "")
        table_id = str(fact.get("table_id") or "")
        columns = {cls._norm_column(str(fact.get("term") or ""))}
        columns.add(cls._norm_column(str(fact.get("column") or "")))
        columns.discard("")
        if not columns:
            columns = {"any"}
        row_keys = {f"index:{int(fact.get('row_index') or 0)}"}
        row_label = str(fact.get("row_label") or "").strip()
        if row_label:
            row_keys.add(f"label:{cls._norm_column(row_label)}")
        row_keys.add("any")
        return {
            f"{document_id}|{parse_version}|{table_id}|{row_key}|{column}"
            for row_key in row_keys
            for column in columns
        }

    def _index_facts_by_identity(
        self, evidence_pack: dict[str, Any] | None
    ) -> dict[str, list[dict[str, Any]]]:
        """以 exact identity key 为已返回 table_facts 建立索引。

        每个 fact 展开多组身份键后挂入索引；同一 fact 可能命中多个
        需求目标，由调用方按对象身份去重。
        """
        indexed: dict[str, list[dict[str, Any]]] = {}
        facts = (evidence_pack or {}).get("table_facts") or []
        if not isinstance(facts, list):
            return indexed
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            for key in self._fact_identity_keys(fact):
                indexed.setdefault(key, []).append(fact)
        return indexed

    def _resolve_table_bindings(
        self,
        table_digits: set[float],
        entries: list[dict[str, Any]],
    ) -> list[tuple[str, str, str]] | None:
        """把问题中的每个 Table N 唯一解析到 inventory 条目。

        返回 ``(document_id, parse_version, table_id)`` 列表；同名表号
        出现在多个文档/版本（无法唯一解析）或 inventory 无对应表时
        返回 None（调用方转 unknown）。
        """
        bindings: list[tuple[str, str, str]] = []
        for digits in sorted(table_digits):
            matched = {
                (
                    str(entry.get("document_id") or ""),
                    str(entry.get("parse_version") or ""),
                    str(entry.get("table_id") or ""),
                )
                for entry in entries
                if self._fact_table_digits(entry) == {digits}
            }
            if len(matched) != 1:
                return None
            bindings.append(next(iter(matched)))
        return bindings

    @staticmethod
    def _unique_inventory_binding(
        entries: list[dict[str, Any]],
    ) -> tuple[str, str, str] | None:
        """无表号问题时，把列需求唯一绑定到 inventory 中的单张表。

        多张表（或同一表多版本）无法唯一解析时返回 None。
        """
        unique = {
            (
                str(entry.get("document_id") or ""),
                str(entry.get("parse_version") or ""),
                str(entry.get("table_id") or ""),
            )
            for entry in entries
        }
        return next(iter(unique)) if len(unique) == 1 else None

    @staticmethod
    def _fact_table_digits(fact: dict[str, Any]) -> set[float]:
        """从 fact 的 table_id 提取数字段（"table-7" → {7.0}），供 Table N 对照。"""
        table_id = str(fact.get("table_id") or "")
        digits = re.sub(r"\D", "", table_id)
        return {float(digits)} if digits else set()

    def _synthesis_missing_fact_fields(
        self,
        *,
        question: str,
        rag_answer: str,
        synthesized_answer: str,
        evidence_pack: dict[str, Any] | None,
    ) -> tuple[set[str], str]:
        """返回 ``(遗漏字段集合, expected_facts_status)``。

        9.3.2b/9.8/任务9：期望集合由"问题 + inventory"解析出的需求目标
        与已返回 table_facts 做 exact join；只有"已返回且 draft 覆盖、
        synthesis 遗漏"的目标字段才触发遗漏回退（基准是 RAG draft——
        draft 有而 synthesis 没有的字段才是 synthesis 的遗漏；draft
        本身缺失的字段不触发）。需求目标在 inventory 中存在但 facts
        未返回时状态为 missing，由 coverage/证据不足路径处理，不能把
        供给缺失误报为 synthesize 遗漏。unknown（需求不可判定）与
        missing（供给未返回）都返回空集合、不触发回退，但状态照常
        透出；draft-fidelity/表号/新数字锚点守卫不受影响。value 用
        数值等价比较（21==21.0），term/unit 用子串匹配（draft 含而
        synthesis 不含视为删除）。
        """
        status, targets = self._expected_facts_status(question, evidence_pack)
        if status == "unknown" or not targets:
            return set(), status
        returned = self._index_facts_by_identity(evidence_pack)
        draft_numbers = self._answer_numbers(rag_answer)
        synth_numbers = self._answer_numbers(synthesized_answer)
        missing: set[str] = set()
        checked: set[int] = set()
        for target in targets:
            for fact in returned.get(target["identity"], []):
                if id(fact) in checked:
                    continue
                checked.add(id(fact))
                value = str(fact.get("value") or "").strip()
                term = str(fact.get("term") or "").strip()
                unit = str(fact.get("unit") or "").strip()
                if (
                    value
                    and any(
                        self._number_equivalent(value, candidate)
                        for candidate in draft_numbers
                    )
                    and not any(
                        self._number_equivalent(value, candidate)
                        for candidate in synth_numbers
                    )
                ):
                    missing.add(f"value={value}")
                if term and term in rag_answer and term not in synthesized_answer:
                    missing.add(f"term={term}")
                if unit and unit in rag_answer and unit not in synthesized_answer:
                    missing.add(f"unit={unit}")
        return missing, status

    def _evidence_excerpt_parts(
        self,
        *,
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
        include_table_facts: bool,
    ) -> str:
        """拼接证据文本：citations excerpts +（可选 table_facts values）+ items excerpts。

        供 `_synthesis_unsupported_numbers` 与 `_synthesis_dropped_table_labels`
        共用，避免重复收集循环。
        """
        evidence_parts: list[str] = [
            str(citation.get("excerpt") or "")
            for citation in citations
        ]
        if evidence_pack:
            if include_table_facts:
                table_facts = evidence_pack.get("table_facts") or []
                if isinstance(table_facts, list):
                    evidence_parts.extend(
                        str(fact.get("value") or "")
                        for fact in table_facts
                        if isinstance(fact, dict)
                    )
            items = evidence_pack.get("items") or []
            if isinstance(items, list):
                evidence_parts.extend(
                    str(item.get("excerpt") or "")
                    for item in items
                    if isinstance(item, dict)
                )
        return " ".join(part for part in evidence_parts if part)

    def _synthesis_dropped_table_labels(
        self,
        synthesized_answer: str,
        *,
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
    ) -> set[str]:
        """返回证据中存在但合成答案删除的表号（如 Table 7）。

        canonical table_facts 的 table_id 是哈希（table-xxxx），表号只能从
        excerpts/captions 提取；证据中有表号而答案完全没有时视为删除，
        触发回退以保留表级引用。
        """
        evidence_text = self._evidence_excerpt_parts(
            evidence_pack=evidence_pack,
            citations=citations,
            include_table_facts=False,
        )
        evidence_labels = self._table_labels(evidence_text)
        answer_labels = self._table_labels(synthesized_answer)
        return evidence_labels - answer_labels

    @staticmethod
    def _answer_numbers(answer_markdown: str) -> set[str]:
        """提取数值，排除 citation 编号（[N]）、表号（Table 7 的 7）与词内数字
        （OPLS4 不提取 4）。右侧用 ``(?![A-Za-z0-9_.])`` 防止把 ``80`` 从
        ``80.5`` 中单独拆出。表号剔除口径与 ``_table_labels`` 一致
        （``\\s*`` 支持 "Table7"，``(?:\\d+(?:\\.\\d+)?)`` 覆盖 "Table 7.5"）。
        """
        text = re.sub(r"\[\s*\d+\s*\]", " ", answer_markdown)
        # Paper identifiers such as ``arXiv:2401.15884`` are metadata, not
        # scientific measurements.  The suffix would otherwise be parsed as a
        # fabricated number (``15884``) when a synthesis repeats a paper title.
        # Allow a title stem/underscore immediately before the identifier
        # (e.g. ``selfrag_2310.11511``) as well as the usual ``arXiv:`` form.
        text = re.sub(r"\d{4}\.\d{4,6}", " ", text)
        text = re.sub(
            r"\btable\s*(?:s\s*)?\d+(?:\.\d+)?\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        return set(
            re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?![A-Za-z0-9_.])", text)
        )

    @staticmethod
    def _number_equivalent(a: str, b: str) -> bool:
        """数值等价比较："21" 与 "21.0" 等价；"80" 与 "80.5" 不等价。"""
        try:
            return float(a) == float(b)
        except ValueError:
            return a == b

    def _synthesis_unsupported_numbers(
        self,
        synthesized_answer: str,
        *,
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
    ) -> set[str]:
        """返回合成答案中证据不支持的数值（模型新编的数字）。

        支持集合来自三处证据：citations excerpts、evidence_pack.table_facts
        values、evidence_pack.items excerpts。数值用 float 等价比较——
        "21" 与 "21.0" 视为同一数值（不误伤格式差异），"80" 与 "80.5"
        视为不同（取整编造被捕获）。答案中不在支持集合的数值视为编造，
        触发回退。
        """
        evidence_text = self._evidence_excerpt_parts(
            evidence_pack=evidence_pack,
            citations=citations,
            include_table_facts=True,
        )
        evidence_numbers = self._answer_numbers(evidence_text)
        answer_numbers = self._answer_numbers(synthesized_answer)
        return {
            number
            for number in answer_numbers
            if not any(
                self._number_equivalent(number, candidate)
                for candidate in evidence_numbers
            )
        }

    def _draft_fidelity_fallback(
        self,
        *,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        synthesized_answer: str,
        warnings: list[str],
        fallback_provider: str | None = None,
        fallback_model: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a RAG-draft fallback when a synthesis drops draft-covered anchors.

        The guarantee is deliberately generic: the Agent final answer must never
        lose evidence-anchor coverage that was already present in the same-request
        RAG draft.  Anchors come from the generic extraction rules, so this
        cannot hard-code benchmark case ids or required terms.  When no anchors
        are extractable, or the synthesis covers at least as many as the draft,
        ``None`` is returned and the synthesized answer is kept.
        """
        # In remote-only deployments the RAG layer may intentionally return a
        # raw-evidence placeholder instead of attempting local Ollama.  That
        # placeholder is not a semantic draft: its copied evidence can contain
        # unrelated anchors (for example bibliography terms) that a valid,
        # concise DeepSeek answer should not be forced to repeat.  Keep the
        # provider-uniform numeric/table guards below, but do not reject the
        # remote synthesis solely because it does not echo those placeholder
        # anchors.
        if self._is_degraded_rag_answer(rag_answer):
            return None
        anchors = self._extract_evidence_anchors(evidence_pack, citations)
        if not anchors:
            return None
        synth_lower = synthesized_answer.casefold()
        draft_lower = rag_answer.casefold()
        synth_covered = sum(1 for a in anchors if a.casefold() in synth_lower)
        draft_covered = sum(1 for a in anchors if a.casefold() in draft_lower)
        if synth_covered >= draft_covered:
            return None
        return self._fidelity_fallback(
            rag_answer=rag_answer,
            citations=citations,
            evidence_pack=evidence_pack,
            warnings=warnings,
            provider=fallback_provider,
            model=fallback_model,
            reason=(
                "Synthesized answer lost evidence-anchor coverage present in the RAG draft "
                f"(draft covered {draft_covered}/{len(anchors)} anchors, synthesis covered "
                f"{synth_covered}/{len(anchors)}); using the RAG draft to preserve exact "
                "evidence terms."
            ),
        )


    def _revise_deepseek_after_guard(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        narrow_context: bool,
        rejected_answer: str,
        guard_warnings: Any,
    ) -> dict[str, Any] | None:
        """Ask DeepSeek once to repair a rejected remote synthesis.

        The normal DeepSeek path already has a grounded JSON contract. This
        helper only adds the concrete fidelity failure (unsupported number,
        dropped table label, or missing fact) and the previous answer to the
        prompt. It intentionally reuses ``_deepseek_synthesize`` so parsing,
        citation-index normalisation, and provider accounting stay identical.
        """
        reasons = guard_warnings if isinstance(guard_warnings, list) else [guard_warnings]
        reason_text = "\n".join(
            f"- {str(reason).strip()}" for reason in reasons if str(reason).strip()
        )
        revision_draft = (
            "The remote RAG draft is unavailable; the following answer was a first "
            "DeepSeek synthesis that failed a groundedness guard. Rewrite it from "
            "the evidence and return a complete answer to the original question. "
            "Do not mention this repair process in the answer.\n\n"
            f"First synthesis:\n{rejected_answer}\n\n"
            f"Guard findings:\n{reason_text or '- fidelity guard rejected the first synthesis'}\n\n"
            "Rules for this revision: preserve every supported numeric value and "
            "table label needed by the question; remove unsupported numbers; cite "
            "only the supplied evidence indexes; if a required fact is absent, "
            "state that limitation explicitly instead of guessing."
        )
        try:
            revised = self._deepseek_synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=revision_draft,
                citations=citations,
                evidence_pack=evidence_pack,
                narrow_context=narrow_context,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the original guard fallback
            logger.warning("DeepSeek fidelity revision failed: %s", exc)
            return None
        answer = str(revised.get("answer_markdown") or "").strip()
        if not answer or answer == revision_draft:
            return None
        revised.setdefault("warnings", []).append(
            "DeepSeek first synthesis failed a fidelity guard; a grounded revision was requested."
        )
        return revised

    def _deepseek_synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        narrow_context: bool = False,
    ) -> dict[str, Any]:
        """Synthesize through the configured DeepSeek API.

        DeepSeek is kept separate from the legacy ``external_api`` path so its
        retry budget, model identity, and usage accounting are explicit.  A
        failed/empty/malformed response always returns the grounded RAG draft
        with ``provider=deepseek`` and an actionable warning.
        """
        api_key = getattr(self._settings, "deepseek_api_key", None)
        if not self._has_secret(api_key):
            result = self._comparison_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                provider="deepseek",
                model=getattr(self._settings, "deepseek_model", "deepseek-chat"),
            )
            result["warnings"].append(
                "DeepSeek synthesis requested but DEEPSEEK_API_KEY is not configured; "
                f"using {self._fallback_label(evidence_pack)}."
            )
            return result

        evidence_parts: list[str] = []
        for index, citation in enumerate(citations):
            excerpt = citation.get("excerpt", "")
            title = citation.get("page_title") or citation.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{index}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"
        evidence_pack_text = (
            "" if narrow_context else self._format_evidence_pack_section(evidence_pack)
        )
        table_facts_text = self._format_table_facts_section(evidence_pack)
        comparison_text = self._format_comparison_section(evidence_pack, citations)
        comparison_rules = self._comparison_rules(evidence_pack)
        system_prompt = (
            "You are an evidence synthesis assistant. Synthesize a final answer "
            "from the provided RAG answer and evidence excerpts. Do not invent "
            "facts or citations. Return a JSON object with keys: "
            "answer_markdown (string), cited_indexes (array of valid integers), "
            "warnings (array of strings), confidence (number 0.0-1.0).\n"
            + comparison_rules
            + self._answer_rules(query)
        )
        user_prompt = (
            f"Original query: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + (f"{comparison_text}\n\n" if comparison_text else "")
            + "Return only the JSON object."
        )
        client = DeepSeekClient(
            base_url=getattr(self._settings, "deepseek_base_url", None),
            api_key=api_key,
            model=getattr(self._settings, "deepseek_model", "deepseek-chat"),
            timeout=getattr(self._settings, "generation_timeout_seconds", 90),
            max_retries=getattr(self._settings, "generation_max_retries", 1),
            retry_backoff_seconds=getattr(
                self._settings, "generation_retry_backoff_seconds", 0.5
            ),
        )
        try:
            generated = client.generate_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=getattr(
                    self._settings, "generation_max_output_tokens", 2048
                ),
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek synthesis API call failed: %s", exc)
            result = self._comparison_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                provider="deepseek",
                model=client.model,
            )
            result["warnings"].append(
                f"DeepSeek synthesis failed ({type(exc).__name__}), using "
                f"{self._fallback_label(evidence_pack)}: {exc}"
            )
            return result

        content = str(generated.get("content") or "").strip()
        parse_retry_used = False
        try:
            parsed = self._parse_deepseek_json(content)
            # A few OpenAI-compatible gateways wrap the requested object in a
            # singleton JSON array even when ``response_format=json_object``
            # is set.  Accept that lossless wrapper, but keep rejecting arrays
            # containing multiple/ non-object values because they do not have
            # an unambiguous synthesis contract.
            if (
                isinstance(parsed, list)
                and len(parsed) == 1
                and isinstance(parsed[0], dict)
            ):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                raise ValueError("DeepSeek synthesis JSON must be an object")
        except Exception as exc:  # noqa: BLE001
            # JSON-mode gateways occasionally return a prose-wrapped array or
            # otherwise malformed payload even though the transport succeeded.
            # Retry the same bounded request once with an explicit single-object
            # reminder before falling back to the structured matrix.
            try:
                retry_generated = client.generate_chat(
                    messages=[
                        {"role": "system", "content": system_prompt + "\nReturn exactly one JSON object, not an array."},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=getattr(
                        self._settings, "generation_max_output_tokens", 2048
                    ),
                    response_format={"type": "json_object"},
                )
                retry_content = str(retry_generated.get("content") or "").strip()
                retry_parsed = self._parse_deepseek_json(retry_content)
                if (
                    isinstance(retry_parsed, list)
                    and len(retry_parsed) == 1
                    and isinstance(retry_parsed[0], dict)
                ):
                    retry_parsed = retry_parsed[0]
                if not isinstance(retry_parsed, dict):
                    raise ValueError("DeepSeek synthesis JSON must be an object")
                generated = retry_generated
                parsed = retry_parsed
                parse_retry_used = True
            except Exception as retry_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to parse DeepSeek synthesis response after retry: %s",
                    retry_exc,
                )
                result = self._comparison_fallback(
                    rag_answer=rag_answer,
                    citations=citations,
                    evidence_pack=evidence_pack,
                    provider="deepseek",
                    model=str(generated.get("model") or client.model),
                )
                result["warnings"].append(
                    f"DeepSeek returned an invalid synthesis response, using "
                    f"{self._fallback_label(evidence_pack)}: {exc}"
                )
                return result

        max_index = len(citations)
        raw_indexes = parsed.get("cited_indexes", [])
        cited_indexes: list[int] = []
        if isinstance(raw_indexes, list):
            for index in raw_indexes:
                try:
                    normalized_index = int(index)
                except (TypeError, ValueError):
                    continue
                if 0 <= normalized_index < max_index and normalized_index not in cited_indexes:
                    cited_indexes.append(normalized_index)
        answer_markdown = str(parsed.get("answer_markdown") or "").strip()
        if not answer_markdown:
            # A few OpenAI-compatible gateways rename this field to ``answer``
            # when they post-process JSON mode.  Accept the alias without
            # weakening the grounded fallback for a genuinely empty response.
            answer_markdown = str(parsed.get("answer") or parsed.get("content") or "").strip()
        warnings = [str(item) for item in (parsed.get("warnings") or [])]
        if isinstance(parsed.get("warnings"), str):
            warnings = [str(parsed["warnings"])]
        if parse_retry_used:
            warnings.append("DeepSeek JSON parse retry performed.")
        if not answer_markdown:
            answer_markdown = (
                self._local_fallback(rag_answer, citations, evidence_pack)["answer_markdown"]
                if isinstance((evidence_pack or {}).get("comparison"), dict)
                else rag_answer
            )
            warnings.append(
                "DeepSeek returned an empty answer; using "
                f"{self._fallback_label(evidence_pack)}."
            )
        if not cited_indexes and max_index:
            # Do not discard all source citations just because the gateway
            # omitted the optional array.  Prefer explicit inline markers;
            # otherwise select citations whose excerpts share a supported
            # evidence anchor with the answer.  The executor still treats an
            # empty list as "keep all" for the final response.
            cited_indexes = self._extract_cited_indexes(answer_markdown, max_index)
            if not cited_indexes:
                anchors = self._extract_evidence_anchors(evidence_pack, citations)
                cited_indexes = [
                    index
                    for index, citation in enumerate(citations)
                    if any(
                        anchor.casefold() in answer_markdown.casefold()
                        and anchor.casefold() in str(citation.get("excerpt") or "").casefold()
                        for anchor in anchors
                    )
                ]
        try:
            confidence = float(parsed.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": cited_indexes,
            "warnings": warnings,
            "confidence": confidence,
            "provider": "deepseek",
            "model": str(generated.get("model") or client.model),
            "usage": generated.get("usage"),
            "usage_source": generated.get("usage_source", "unknown"),
        }

    @staticmethod
    def _parse_deepseek_json(content: str) -> Any:
        """Parse JSON returned by DeepSeek/OpenAI-compatible gateways.

        JSON mode is not consistently preserved by every proxy: some return a
        fenced object, prepend a short explanation or ``<think>`` block, and
        a few double-encode the object as a JSON string.  Keep parsing local
        and deterministic; a malformed response still follows the grounded
        DeepSeek fallback path in ``_deepseek_synthesize``.
        """
        cleaned = str(content or "").strip()
        if not cleaned:
            raise ValueError("DeepSeek synthesis response was empty")
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        candidates = [cleaned]
        if cleaned.startswith("```"):
            unfenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
            candidates.insert(0, unfenced)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    pass
            if isinstance(parsed, (dict, list)):
                return parsed
        # Reuse the robust balanced-value scanner already used by the local
        # Ollama client for prose-wrapped and fenced responses.
        parsed = OllamaClient._extract_first_json_value(cleaned)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed


    def _external_synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
        narrow_context: bool = False,
    ) -> dict[str, Any]:
        """调用外部 API 进行综合；失败时安全回退，绝不抛异常。

        ``narrow_context``（9.3.1）：跨轮场景为 True 时不含 evidence_pack
        items 摘录，只保留 citations 与结构化 table_facts。
        """
        if not self._settings.external_api_enabled or not bool(self._settings.external_api_key):
            result = self._comparison_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
            )
            result["warnings"].append(
                "External synthesis requested but API is not configured; "
                "using local fallback."
            )
            return result

        # Build evidence context
        evidence_parts: list[str] = []
        for i, c in enumerate(citations):
            excerpt = c.get("excerpt", "")
            title = c.get("page_title") or c.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{i}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"

        # Include bounded evidence pack section when available (narrow 时省略)
        evidence_pack_text = (
            "" if narrow_context else self._format_evidence_pack_section(evidence_pack)
        )
        table_facts_text = self._format_table_facts_section(evidence_pack)
        comparison_text = self._format_comparison_section(evidence_pack, citations)
        comparison_rules = self._comparison_rules(evidence_pack)

        system_prompt = (
            "You are an evidence synthesis assistant. Your task is to synthesize "
            "a final answer from the provided RAG answer and evidence excerpts. "
            "You MUST synthesize — do NOT simply concatenate paragraphs. "
            "Only cite sources that are present in the provided evidence. "
            "Return a JSON object with keys: answer_markdown (string), "
            "cited_indexes (array of integers, only valid indexes from the evidence), "
            "warnings (array of strings), confidence (number 0.0-1.0).\n"
            + comparison_rules
            + self._answer_rules(query)
        )

        user_prompt = (
            f"Original query: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + (f"{comparison_text}\n\n" if comparison_text else "")
            + "Synthesize a final answer from the above evidence. "
            "Return only a JSON object."
        )

        headers = {
            "Authorization": f"Bearer {self._settings.external_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._settings.external_api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            with httpx.Client(timeout=self._settings.external_api_timeout) as client:
                response = client.post(
                    f"{self._settings.external_api_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("External synthesis API call failed: %s", exc)
            result = self._comparison_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                provider="external_api",
            )
            result["warnings"].append(
                f"External synthesis API call failed, using local fallback: {exc}"
            )
            return result

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, json.JSONDecodeError, IndexError) as exc:
            logger.warning("Failed to parse external synthesis response: %s", exc)
            result = self._comparison_fallback(
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                provider="external_api",
            )
            result["warnings"].append(
                f"Failed to parse external synthesis response, using local fallback: {exc}"
            )
            return result

        # Sanitize cited_indexes
        max_idx = len(citations)
        raw_indexes = parsed.get("cited_indexes", [])
        if isinstance(raw_indexes, list):
            sanitized = [i for i in raw_indexes if isinstance(i, int) and 0 <= i < max_idx]
        else:
            sanitized = []

        answer_markdown = parsed.get("answer_markdown", rag_answer)
        answer_markdown = str(answer_markdown or "").strip()
        warnings = [str(w) for w in (parsed.get("warnings") or [])]
        if not answer_markdown:
            if rag_answer.strip():
                answer_markdown = rag_answer
                warnings.append(
                    "External synthesis returned an empty answer; using RAG fallback."
                )
            else:
                answer_markdown = (
                    "The current knowledge base did not return enough relevant "
                    "evidence to answer this question reliably."
                )
                warnings.append(
                    "External synthesis returned an empty answer and no RAG "
                    "fallback was available."
                )

        # ---- 覆盖度重试（Phase 3） ----
        # 对 evidence-heavy 的 route，如果首次综合遗漏了关键证据锚点，
        # 则用显式锚点列表再次调用 API，要求模型把这些证据包含进去。
        if answer_markdown and self._should_retry_for_coverage(
            route=route,
            evidence_pack=evidence_pack,
            citations=citations,
            answer_text=answer_markdown,
        ):
            retry_result = self._retry_with_anchors(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                first_answer=answer_markdown,
                headers=headers,
                narrow_context=narrow_context,
            )
            if retry_result is not None:
                return retry_result
            # Retry failed — fall through to original result
            warnings.append(
                "Evidence coverage retry did not produce a usable result; "
                "using first synthesis."
            )

        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": sanitized,
            "warnings": warnings,
            "confidence": float(parsed.get("confidence", 1.0)),
            "provider": "external_api",
            "model": self._settings.external_api_model,
        }

    # ------------------------------------------------------------------
    # 覆盖度重试辅助方法（Phase 3）
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_evidence_anchors(
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
    ) -> list[str]:
        """Extract high-signal anchor terms from evidence using generic rules.

        Anchors come from two sources:
        1. Evidence pack item excerpts (numeric patterns, abbreviations).
        2. Citation excerpts (numeric patterns, abbreviations).

        Rules are generic — no hard-coded domain terms.
        Returns at most 10 deduplicated anchors.
        """
        excerpts: list[str] = []

        # Collect excerpts from evidence pack
        if evidence_pack and evidence_pack.get("items"):
            for item in evidence_pack["items"]:
                ex = item.get("excerpt", "")
                if ex:
                    excerpts.append(str(ex))

        # Collect excerpts from citations
        for c in citations:
            ex = c.get("excerpt", "")
            if ex:
                excerpts.append(str(ex))

        if not excerpts:
            return []

        all_text = " ".join(excerpts)
        anchors: list[str] = []

        # Rule 1: numeric patterns with optional units
        numeric_pattern = re.compile(
            r"(?:^|[\s(])"
            r"[+-]?\d+\.?\d*"
            r"(?:\s*(?:kcal/mol|kJ/mol|kcal|kJ|nm|Å|eV|kcal·mol|kJ·mol|%|°C|K))?"
            r"(?:$|[\s,.);])",
        )
        for m in numeric_pattern.finditer(all_text.lower()):
            val = m.group().strip(" ,.();")
            if len(val) >= 2:
                anchors.append(val)

        # Rule 2: uppercase abbreviations (2–5 uppercase letters, whole-word)
        abbrev_pattern = re.compile(r"\b[A-Z]{2,5}\b")
        for m in abbrev_pattern.finditer(all_text):
            val = m.group()
            # Skip very common noise abbreviations
            if val in ("THE", "AND", "FOR", "ARE", "WAS", "HAS", "THIS", "THAT", "WITH", "FROM"):
                continue
            anchors.append(val)

        # Deduplicate preserving order, case-insensitive
        seen: set[str] = set()
        unique: list[str] = []
        for a in anchors:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique[:10]

    @staticmethod
    def _should_retry_for_coverage(
        *,
        route: str,
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
        answer_text: str,
    ) -> bool:
        """Decide whether a coverage retry is warranted.

        Returns True only when ALL of:
        - Route is evidence-heavy (``evidence_required``, ``table_or_metric``,
          ``multi_source_compare``).
        - Evidence pack is present and non-empty.
        - At least some extracted anchors are missing from the answer text.
        - Retry has not already been attempted (caller enforces this).
        """
        # Only retry on evidence-heavy routes
        if route not in ("evidence_required", "table_or_metric", "multi_source_compare"):
            return False

        # Must have a non-empty evidence pack
        if not evidence_pack or not evidence_pack.get("items"):
            return False

        anchors = AgentSynthesizer._extract_evidence_anchors(
            evidence_pack, citations
        )
        if not anchors:
            return False

        # Check coverage: count how many anchors appear in answer (case-insensitive)
        answer_lower = answer_text.lower()
        covered = sum(1 for a in anchors if a.lower() in answer_lower)

        # Retry if fewer than half of the anchors are covered
        return covered < max(1, len(anchors) // 2)

    def _retry_with_anchors(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        first_answer: str,
        headers: dict[str, str],
        narrow_context: bool = False,
    ) -> dict[str, Any] | None:
        """Retry external synthesis once with explicit missing-anchor hints.

        Calls the external API a second time with an augmented prompt that
        lists the missing evidence anchors and instructs the model to
        include them.  Returns ``None`` on any failure so the caller can
        fall back to the first synthesis result.
        """
        anchors = self._extract_evidence_anchors(evidence_pack, citations)
        if not anchors:
            return None

        answer_lower = first_answer.lower()
        missing = [a for a in anchors if a.lower() not in answer_lower]
        if not missing:
            return None  # all covered — no need to retry

        # Rebuild evidence context (same as first call)
        evidence_parts: list[str] = []
        for i, c in enumerate(citations):
            excerpt = c.get("excerpt", "")
            title = c.get("page_title") or c.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{i}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"

        evidence_pack_text = (
            "" if narrow_context else self._format_evidence_pack_section(evidence_pack)
        )
        # 与 local/ollama retry 保持一致：结构化 table_facts 即使收窄也保留
        table_facts_text = self._format_table_facts_section(evidence_pack)

        missing_list = ", ".join(missing[:8])

        system_prompt = (
            "You are an evidence synthesis assistant. Your task is to synthesize "
            "a final answer from the provided RAG answer and evidence excerpts. "
            "You MUST synthesize — do NOT simply concatenate paragraphs. "
            "IMPORTANT: The first answer omitted key evidence anchors. "
            "Your revised answer MUST explicitly include these evidence terms: "
            f"{missing_list}. "
            "Only cite sources that are present in the provided evidence. "
            "Return a JSON object with keys: answer_markdown (string), "
            "cited_indexes (array of integers, only valid indexes from the evidence), "
            "warnings (array of strings), confidence (number 0.0-1.0)."
        )

        user_prompt = (
            f"Original query: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + (f"{table_facts_text}\n\n" if table_facts_text else "")
            + f"The first synthesis omitted these evidence anchors: {missing_list}\n"
            "Please produce a revised synthesis that explicitly includes "
            "these evidence terms where they are supported by the evidence. "
            "Return only a JSON object."
        )

        payload: dict[str, Any] = {
            "model": self._settings.external_api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            with httpx.Client(timeout=self._settings.external_api_timeout) as client:
                response = client.post(
                    f"{self._settings.external_api_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("Coverage retry API call failed: %s", exc)
            return None

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, json.JSONDecodeError, IndexError) as exc:
            logger.warning("Failed to parse coverage retry response: %s", exc)
            return None

        # Sanitize cited_indexes
        max_idx = len(citations)
        raw_indexes = parsed.get("cited_indexes", [])
        if isinstance(raw_indexes, list):
            sanitized = [i for i in raw_indexes if isinstance(i, int) and 0 <= i < max_idx]
        else:
            sanitized = []

        answer_markdown = parsed.get("answer_markdown", "")
        answer_markdown = str(answer_markdown or "").strip()
        warnings = [str(w) for w in (parsed.get("warnings") or [])]
        warnings.append(
            "Coverage retry performed — first answer omitted evidence anchors"
        )

        if not answer_markdown:
            return None  # fall back to first answer

        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": sanitized,
            "warnings": warnings,
            "confidence": float(parsed.get("confidence", 1.0)),
            "provider": "external_api",
            "model": self._settings.external_api_model,
        }
