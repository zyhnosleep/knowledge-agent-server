from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import DocumentChunk, PageKind, Project, QuestionAnswer, WikiPage
from app.schemas.common import Citation, QueryResponse
from app.services.ai import QueryAnswerPayload, VerificationPayload, cosine_similarity, safe_model_call
from app.services.ai import ExternalVerifier, OllamaClient
from app.services.filesystem import InvalidStoragePathError, safe_project_slug, slugify, strip_upload_prefix
from app.services.table_extraction import summarize_ablation_table, table_metric_values
from app.services.table_normalization import normalize_table_text
from app.services.wiki import WikiRenderer

settings = get_settings()
WIKI_PRIMARY_SCORE_THRESHOLD = 6.0
MIN_WIKI_OVERLAP_SCORE = 1
MIN_CONTEXT_SCORE = 2.5
CONTEXT_SCORE_RATIO = 0.40
MAX_CONTEXTS = 8
TABLE_CONTEXT_SCORE_BOOST = 40.0


@dataclass
class RetrievedContext:
    citation: Citation
    prompt_text: str
    score: float


@dataclass
class PageMatch:
    page: WikiPage
    score: float


@dataclass
class ExtractedMetric:
    context_index: int
    table_label: str | None
    dataset: str
    values: dict[str, str]


class QueryService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.ollama = OllamaClient()
        self.verifier = ExternalVerifier()

    def answer(self, project_slug: str, question: str, save_answer: bool = True) -> QueryResponse:
        project = self.db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            raise ValueError(f"Project '{project_slug}' not found")

        index_context = self._load_index_context(project.slug)
        page_matches = self._search_wiki_pages(question, project.id)
        contexts = self._build_contexts(question, project.id, page_matches)
        answer_payload = self._draft_answer(question, index_context, contexts)
        verification_status = "local-only"

        if self._is_high_risk(question):
            verification = self._verify_answer(answer_payload.answer_markdown, contexts)
            verification_status = verification.verdict
            if verification.notes:
                answer_payload.answer_markdown += f"\n\n> Verification note: {verification.notes}"

        answer_payload.answer_markdown = self._normalize_answer_citation_markup(answer_payload.answer_markdown)
        chosen_indexes = self._choose_citation_indexes(question, answer_payload, contexts)
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        answer_payload = self._repair_unsupported_numeric_answer(question, index_context, contexts, answer_payload, chosen_indexes)
        answer_payload = self._repair_missing_table_answer(question, index_context, contexts, answer_payload)
        answer_payload.answer_markdown = self._normalize_answer_citation_markup(answer_payload.answer_markdown)
        chosen_indexes = self._choose_citation_indexes(question, answer_payload, contexts)
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        chosen_indexes = self._select_citation_indexes(contexts, chosen_indexes)
        citations = self._select_citations(contexts, chosen_indexes)
        if not self._needs_source_evidence(question):
            citations = self._prefer_wiki_citations(citations, page_matches, project.id, question=question)
        answer_markdown = self._renumber_answer_citations(answer_payload.answer_markdown, chosen_indexes)
        answer_markdown = self._drop_unreturned_citation_markers(answer_markdown, len(citations))
        if not citations:
            answer_markdown = self._strip_answer_citation_markers(answer_markdown)
        response = QueryResponse(answer_markdown=answer_markdown, citations=citations, verification_status=verification_status)

        if save_answer:
            record = QuestionAnswer(
                project_id=project.id,
                question=question,
                answer_markdown=response.answer_markdown,
                citations=[citation.model_dump() for citation in citations],
                risk_level=answer_payload.risk_level,
                verification_status=verification_status,
            )
            self.db.add(record)
            self._save_query_page(project, question, response, citations)
            self.db.commit()
        return response

    def _search_wiki_pages(self, question: str, project_id: str, limit: int = 4) -> list[PageMatch]:
        query_terms = self._tokenize(question)
        pages = self.db.scalars(
            select(WikiPage).where(WikiPage.project_id == project_id, WikiPage.kind != PageKind.query_answer.value)
        ).all()
        matches: list[PageMatch] = []
        lowered_question = question.lower()
        for page in pages:
            metadata = page.metadata_json or {}
            title_terms = self._tokenize(page.title)
            clean_title = strip_upload_prefix(page.title)
            title_terms = self._tokenize(clean_title)
            slug_terms = self._tokenize(page.slug.replace("/", " "))
            body_text = self._strip_frontmatter(page.markdown_content)
            body_terms = self._tokenize(body_text)
            key_terms = metadata.get("key_terms", [])
            key_term_terms = self._tokenize(" ".join(str(term) for term in key_terms)) if isinstance(key_terms, list) else set()
            overlap = len(query_terms & body_terms)
            title_overlap = len(query_terms & title_terms)
            slug_overlap = len(query_terms & slug_terms)
            key_term_overlap = len(query_terms & key_term_terms)
            verified_count = int(metadata.get("verified_claim_count") or 0)
            score = overlap + title_overlap * 4 + slug_overlap * 2 + key_term_overlap * 3 + min(verified_count, 5) * 0.6
            content_signal = overlap + title_overlap + slug_overlap + key_term_overlap
            if lowered_question in body_text.lower():
                score += 8
                content_signal += 2
            if lowered_question in clean_title.lower():
                score += 10
                content_signal += 2
            if content_signal < MIN_WIKI_OVERLAP_SCORE:
                continue
            if page.kind == PageKind.source_summary.value:
                score += 1.5
            if page.kind == PageKind.entity.value:
                score += 0.5
            if page.kind == PageKind.entity.value and verified_count <= 0:
                continue
            if "No claims yet." in body_text or "No summary available." in body_text:
                continue
            if score <= 0:
                continue
            matches.append(PageMatch(page=page, score=float(score)))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def _search_source_chunks(self, question: str, project_id: str, document_ids: list[str], limit: int = 3) -> list[RetrievedContext]:
        statement = select(DocumentChunk).join(DocumentChunk.document).where(DocumentChunk.document.has(project_id=project_id))
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))
        chunks = self.db.scalars(statement).all()

        question_vector = safe_model_call(lambda: self.ollama.embed([question])[0], [])
        query_terms = self._tokenize(question)
        scored: list[RetrievedContext] = []
        for chunk in chunks:
            score = 0.0
            if question_vector and chunk.embedding:
                score = cosine_similarity(question_vector, chunk.embedding)
            else:
                chunk_terms = self._tokenize(chunk.text)
                overlap = len(query_terms & chunk_terms)
                if overlap:
                    score = min(0.3 + overlap * 0.1, 0.85)
            if score <= 0:
                continue
            prompt_text = self._window_text(chunk.text, query_terms, max_chars=1600, question=question)
            scored.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=prompt_text[:280],
                    ),
                    prompt_text=prompt_text,
                    score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def _build_contexts(self, question: str, project_id: str, page_matches: list[PageMatch]) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        fallback_contexts: list[RetrievedContext] = []
        matched_doc_ids: list[str] = []
        query_terms = self._tokenize(question)
        query_facets = self._extract_query_facets(question)
        top_score = page_matches[0].score if page_matches else 0.0
        min_score = max(MIN_CONTEXT_SCORE, top_score * CONTEXT_SCORE_RATIO) if top_score else MIN_CONTEXT_SCORE
        is_figure_query = self._is_figure_query(question)
        is_table_query = self._is_table_query(question)
        is_metric_query = self._is_metric_query(question)
        needs_table_first = is_table_query or is_metric_query
        has_table_context = False

        for match in page_matches:
            if match.score < min_score:
                continue
            page = match.page
            page_body = self._strip_frontmatter(page.markdown_content)

            # For figure/table/metric queries, extract the relevant blocks first
            # so the context window centers on the Figure Note / Table block
            # instead of the page start.
            if is_figure_query:
                figure_blocks = self._extract_figure_blocks(page_body)
                for block, block_score in self._rank_blocks(question, figure_blocks)[:4]:
                    contexts.append(
                        RetrievedContext(
                            citation=Citation(
                                page_slug=page.slug,
                                page_title=strip_upload_prefix(page.title),
                                page_kind=page.kind,
                                score=match.score + 2.0 + block_score,
                                page_label=self._extract_page_label_from_block(block),
                                excerpt=block[:280],
                            ),
                            prompt_text=block[:2000],
                            score=match.score + 2.0 + block_score,
                        )
                    )

            if is_table_query or is_metric_query:
                table_blocks = self._extract_table_blocks(page_body)
                for block, block_score in self._rank_blocks(question, table_blocks)[:5]:
                    if not self._context_has_table_data(block):
                        continue
                    has_table_context = True
                    score = match.score + TABLE_CONTEXT_SCORE_BOOST + block_score
                    contexts.append(
                        RetrievedContext(
                            citation=Citation(
                                page_slug=page.slug,
                                page_title=strip_upload_prefix(page.title),
                                page_kind=page.kind,
                                score=score,
                                page_label=self._extract_page_label_from_block(block),
                                excerpt=self._table_block_excerpt(block, question),
                            ),
                            prompt_text=block[:2000],
                            score=score,
                        )
                    )

            for facet in query_facets:
                facet_context = self._context_for_facet(page, page_body, facet, query_terms)
                if facet_context is not None:
                    if needs_table_first:
                        fallback_contexts.append(facet_context)
                    else:
                        contexts.append(facet_context)

            # Always include the main page context (windowed around query terms).
            prompt_text = self._window_text(page_body, query_terms, max_chars=4000, question=question)
            page_context = RetrievedContext(
                citation=Citation(
                    page_slug=page.slug,
                    page_title=strip_upload_prefix(page.title),
                    page_kind=page.kind,
                    score=match.score,
                    page_label=self._extract_page_label_from_wiki_content(page_body, prompt_text),
                    excerpt=self._window_text(page_body, query_terms, max_chars=280, question=question),
                ),
                prompt_text=prompt_text,
                score=match.score,
            )
            if needs_table_first:
                fallback_contexts.append(page_context)
            else:
                contexts.append(page_context)
            matched_doc_ids.extend(page.source_document_ids)

        if needs_table_first and has_table_context:
            return self._finalize_contexts(contexts)
        contexts.extend(fallback_contexts)
        if self._should_use_wiki_only(question, page_matches):
            return self._finalize_contexts(contexts)
        if matched_doc_ids:
            contexts.extend(self._search_source_chunks(question, project_id, sorted(set(matched_doc_ids))))
        if contexts:
            return self._finalize_contexts(contexts)
        return self._search_source_chunks(question, project_id, [], limit=5)

    def _draft_answer(self, question: str, index_context: str | None, contexts: list[RetrievedContext]) -> QueryAnswerPayload:
        if not contexts:
            return QueryAnswerPayload(
                answer_markdown="No supporting evidence was found yet. Please ingest relevant sources first.",
                citations=[],
                risk_level="normal",
            )

        prompt_sections: list[str] = []
        if index_context:
            prompt_sections.append("Index overview:\n" + index_context)
        prompt_sections.extend(f"[{index}] {context.prompt_text}" for index, context in enumerate(contexts))
        context_text = "\n\n".join(prompt_sections)

        # Build figure/table/dataset-aware guardrails.
        constraints = self._build_answer_constraints(question, contexts)

        fallback = QueryAnswerPayload(
            answer_markdown="\n".join(
                [
                    "## Answer",
                    "The answer below is based on the currently retrieved wiki pages and source evidence. Please verify against the cited materials when needed.",
                    "",
                    context_text[:1400],
                ]
            ),
            citations=list(range(len(contexts))),
            risk_level="high" if self._is_high_risk(question) else "normal",
        )
        prompt = "\n\n".join(
            [
                f"Question: {question}",
                (
                    "Answer using the retrieved wiki pages first, and use source evidence only when it adds precision. "
                    "If the question contains multiple entities, datasets, metrics, tables, figures, or components, "
                    "answer each requested item explicitly. Return citation indexes that directly support each claim."
                ),
                constraints,
                context_text,
            ]
        )
        return safe_model_call(
            lambda: self.ollama.generate_structured(
                QueryAnswerPayload,
                system_prompt="You are answering against a maintained wiki. Prefer synthesized wiki pages, cite supporting context indexes, and do not claim facts that are absent from the provided material.",
                user_prompt=prompt,
            ),
            fallback,
        )

    def _build_answer_constraints(self, question: str, contexts: list[RetrievedContext]) -> str:
        """Build guardrail instructions based on the question type."""
        parts: list[str] = []
        lowered = question.lower()

        has_figure_context = any(
            "figure" in ctx.prompt_text.lower() or "fig." in ctx.prompt_text.lower()
            for ctx in contexts
        )
        has_table_context = any(
            "table" in ctx.prompt_text.lower() or "|" in ctx.prompt_text
            for ctx in contexts
        )

        # Figure/Table constraint: if context has them, don't say "not included".
        if self._is_figure_query(question):
            if has_figure_context:
                parts.append(
                    "IMPORTANT: The context includes Figure descriptions. "
                    "You MUST describe what the figure shows using the provided Figure Notes. "
                    "Cite the specific figure number and page. "
                    "Do NOT say the figure is 'not included' or 'not available' in the context."
                )
            else:
                parts.append(
                    "Note: No specific figure descriptions were found in context. "
                    "If the context has relevant visual descriptions, reference them. "
                    "Otherwise state that the figure is not described in the available materials."
                )

        if self._is_table_query(question) or self._is_metric_query(question):
            if has_table_context:
                parts.append(
                    "IMPORTANT: The context includes Table data. "
                    "You MUST extract and report specific numbers/metrics from the tables. "
                    "Cite the table number (e.g., Table 2, Table 5) and page. "
                    "Do NOT say metrics are 'not available' when tables are present in context. "
                    "Every numeric metric in your answer MUST appear verbatim in one of the cited contexts."
                )
            else:
                parts.append(
                    "Note: No table data was found in the retrieved context. "
                    "If the context contains relevant metrics, report them with citations."
                )

        # Ablation study constraint.
        if "ablation" in lowered:
            parts.append(
                "IMPORTANT: If the context mentions ablation studies or component analysis, "
                "you MUST report the specific conclusions (which components matter most, "
                "how much performance drops when removing each component). "
                "Cite the table or section where ablation results appear."
            )

        # Dataset classification constraint.
        if any(word in lowered for word in ("数据集", "dataset", "benchmark", "基准")):
            parts.append(
                "IMPORTANT: Distinguish between:\n"
                "- Benchmark datasets (standard evaluation sets like OIE2016, NYT, PENN, WEB)\n"
                "- Case study categories (domain-specific groupings used for qualitative analysis)\n"
                "- Domain corpora (training or retrieval corpora, not evaluation datasets)\n"
                "Label each clearly and do not conflate them. "
                "Only report a dataset as a 'main dataset' if it was used for standardized evaluation."
            )

        facets = self._extract_query_facets(question)
        if facets:
            parts.append(
                "IMPORTANT: The question asks about these specific items: "
                + ", ".join(facets)
                + ". Address each item explicitly. If evidence for an item is missing, say so instead of answering only the first item."
            )

        parts.append(
            "CRITICAL: Only report specific numbers, percentages, scores, F1, AUC, Precision, Recall, or dataset metrics "
            "that appear verbatim in the provided context. If a number is not in the cited context, do not include it."
        )

        return "\n".join(parts) if parts else ""

    def _repair_unsupported_numeric_answer(
        self,
        question: str,
        index_context: str | None,
        contexts: list[RetrievedContext],
        answer_payload: QueryAnswerPayload,
        chosen_indexes: list[int],
    ) -> QueryAnswerPayload:
        unsupported = self._unsupported_answer_numbers(answer_payload.answer_markdown, contexts, chosen_indexes)
        if not unsupported:
            return answer_payload
        supported_pairs = [(index, contexts[index]) for index in chosen_indexes if 0 <= index < len(contexts)]
        if not supported_pairs:
            supported_pairs = list(enumerate(contexts[: min(3, len(contexts))]))
        constrained_context = "\n\n".join(f"[{index}] {ctx.prompt_text}" for index, ctx in supported_pairs)
        if index_context:
            constrained_context = "Index overview:\n" + index_context + "\n\n" + constrained_context
        fallback = QueryAnswerPayload(
            answer_markdown=(
                "The retrieved evidence did not support the specific numeric values in the first draft. "
                "Please re-run the query after ingesting stronger table evidence."
            ),
            citations=[index for index, _ in supported_pairs],
            risk_level=answer_payload.risk_level,
        )
        prompt = "\n\n".join(
            [
                f"Question: {question}",
                "The previous draft included unsupported numeric values: " + ", ".join(sorted(unsupported)),
                "Rewrite the answer using ONLY the evidence below. Do not include any number unless it appears verbatim in the evidence. If a requested metric is absent, say it is absent from the retrieved materials.",
                self._build_answer_constraints(question, [context for _, context in supported_pairs]),
                constrained_context,
            ]
        )
        repaired = safe_model_call(
            lambda: self.ollama.generate_structured(
                QueryAnswerPayload,
                system_prompt="You repair answers by removing unsupported numeric claims and citing only provided evidence.",
                user_prompt=prompt,
            ),
            fallback,
        )
        allowed_indexes = {index for index, _ in supported_pairs}
        repaired.citations = [index for index in repaired.citations if index in allowed_indexes] or [index for index, _ in supported_pairs]
        return QueryAnswerPayload(
            answer_markdown=repaired.answer_markdown,
            citations=repaired.citations,
            risk_level=repaired.risk_level,
        )

    def _repair_missing_table_answer(
        self,
        question: str,
        index_context: str | None,
        contexts: list[RetrievedContext],
        answer_payload: QueryAnswerPayload,
    ) -> QueryAnswerPayload:
        table_indexes = self._table_citation_indexes(question, contexts)
        if not table_indexes:
            return answer_payload
        extracted_metrics = self._extract_requested_metric_values(question, contexts, table_indexes)
        answer_missing = self._answer_claims_table_data_missing(answer_payload.answer_markdown)
        answer_lacks_metrics = bool(extracted_metrics) and not self._answer_contains_extracted_metrics(
            answer_payload.answer_markdown,
            extracted_metrics,
        )
        if not answer_missing and not answer_lacks_metrics:
            return answer_payload

        prompt_sections: list[str] = []
        if index_context:
            prompt_sections.append("Index overview:\n" + index_context)
        prompt_sections.extend(f"[{index}] {contexts[index].prompt_text}" for index in table_indexes[:4])
        context_text = "\n\n".join(prompt_sections)
        fallback = self._deterministic_table_answer(question, contexts, table_indexes, answer_payload.risk_level)
        prompt = "\n\n".join(
            [
                f"Question: {question}",
                (
                    "The previous draft incorrectly said the requested table data was absent. "
                    "The context below DOES contain relevant table or metric data. "
                    "Answer using only these table contexts. Report the specific values that appear verbatim. "
                    "Do not say the values are absent unless none of the requested table/dataset/metric values appear below. "
                    "Return citation indexes exactly as shown in square brackets."
                ),
                self._build_answer_constraints(question, [contexts[index] for index in table_indexes[:4]]),
                context_text,
            ]
        )
        repaired = safe_model_call(
            lambda: self.ollama.generate_structured(
                QueryAnswerPayload,
                system_prompt="You answer table and metric questions against retrieved wiki table contexts. Use only provided values and cite the supporting context indexes.",
                user_prompt=prompt,
            ),
            fallback,
        )
        repaired.answer_markdown = self._normalize_answer_citation_markup(repaired.answer_markdown)
        repaired.citations = [index for index in repaired.citations if index in table_indexes] or table_indexes[:2]
        repaired_lacks_metrics = bool(extracted_metrics) and not self._answer_contains_extracted_metrics(
            repaired.answer_markdown,
            extracted_metrics,
        )
        if self._answer_claims_table_data_missing(repaired.answer_markdown) or repaired_lacks_metrics:
            return fallback
        return repaired

    def _supported_citation_indexes(self, answer_markdown: str, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> list[int]:
        if not contexts:
            return []
        indexes = [index for index in chosen_indexes if 0 <= index < len(contexts)]
        if not indexes:
            indexes = list(range(min(2, len(contexts))))
        unsupported = self._unsupported_answer_numbers(answer_markdown, contexts, indexes)
        if not unsupported:
            return indexes
        numeric_context_indexes = [
            index
            for index, context in enumerate(contexts)
            if any(number in context.prompt_text for number in self._answer_numbers(answer_markdown))
        ]
        return numeric_context_indexes or indexes

    def _choose_citation_indexes(self, question: str, answer_payload: QueryAnswerPayload, contexts: list[RetrievedContext]) -> list[int]:
        indexes: list[int] = []
        for index in answer_payload.citations:
            if 0 <= index < len(contexts) and index not in indexes:
                indexes.append(index)
        for index in self._infer_citation_indexes(answer_payload.answer_markdown, len(contexts)):
            if index not in indexes:
                indexes.append(index)
        for index in self._table_citation_indexes(question, contexts):
            if index not in indexes:
                indexes.append(index)
        for index in self._coverage_citation_indexes(question, contexts):
            if index not in indexes:
                indexes.append(index)
        return indexes or list(range(min(2, len(contexts))))

    def _table_citation_indexes(self, question: str, contexts: list[RetrievedContext]) -> list[int]:
        if not (self._is_table_query(question) or self._is_metric_query(question)):
            return []
        scored: list[tuple[float, int]] = []
        for index, context in enumerate(contexts):
            text = context.prompt_text
            if not self._context_has_table_data(text):
                continue
            score = context.score
            for facet in self._extract_query_facets(question):
                if facet.lower() in text.lower():
                    score += 6.0
            for anchor in self._query_priority_anchors(question)["figure_table"]:
                if anchor.lower() in text.lower():
                    score += 8.0
            for anchor in self._query_priority_anchors(question)["dataset"]:
                if anchor.lower() in text.lower():
                    score += 8.0
            if any(metric in text.lower() for metric in ("f1", "auc", "precision", "recall", "score")):
                score += 3.0
            scored.append((score, index))
        return [index for _, index in sorted(scored, reverse=True)[:3]]

    @staticmethod
    def _context_has_table_data(text: str) -> bool:
        lowered = text.lower()
        has_table_marker = "|" in text or "<table" in lowered or re.search(r"\btable\s*\d+", lowered)
        has_number = bool(re.search(r"\d+(?:\.\d+)?", text))
        return bool(has_table_marker and has_number)

    @staticmethod
    def _answer_claims_table_data_missing(answer_markdown: str) -> bool:
        lowered = answer_markdown.lower()
        markers = (
            "not included",
            "not available",
            "absent",
            "missing",
            "no table data",
            "exact numeric",
            "specific table numbers are absent",
            "does not contain",
            "not present",
            "not included in the provided text",
            "not included in the provided context",
            "cannot determine from the provided",
            "cannot answer from the provided",
            "未包含",
            "未提供",
            "缺少",
            "缺失",
            "无法报告",
            "无法直接引用",
            "无法获取",
            "没有 table",
            "没有表",
        )
        chinese_markers = (
            "未包含",
            "不包含",
            "未提供",
            "不存在",
            "缺失",
            "无法基于现有材料",
            "无法根据现有材料",
            "无法从提供的材料",
        )
        return any(marker in lowered or marker in answer_markdown for marker in markers) or any(
            marker in answer_markdown for marker in chinese_markers
        )

    @classmethod
    def _answer_contains_extracted_metrics(cls, answer_markdown: str, metrics: list[ExtractedMetric]) -> bool:
        lowered = answer_markdown.lower()
        for metric in metrics:
            if metric.dataset.lower() not in lowered:
                return False
            for value in metric.values.values():
                if value not in answer_markdown:
                    return False
        return True

    def _deterministic_table_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
        risk_level: str,
    ) -> QueryAnswerPayload:
        if not self._is_metric_query(question):
            ablation_answer = self._deterministic_ablation_answer(question, contexts, table_indexes, risk_level)
            if ablation_answer is not None:
                return ablation_answer

        metrics = self._extract_requested_metric_values(question, contexts, table_indexes)
        citations: list[int] = []
        if metrics:
            parts: list[str] = []
            for metric in metrics:
                if metric.context_index not in citations:
                    citations.append(metric.context_index)
                values = " / ".join(f"{name} {value}" for name, value in metric.values.items())
                label = f"{metric.table_label} " if metric.table_label else ""
                parts.append(f"{label}{metric.dataset} {values}".strip())
            citation_marker = f" [{citations[0]}]" if citations else ""
            if self._is_chinese_question(question):
                answer = "已在表格证据中找到相关指标：" + "；".join(parts) + citation_marker
            else:
                answer = "The table evidence contains the requested metrics: " + "; ".join(parts) + citation_marker
            return QueryAnswerPayload(answer_markdown=answer, citations=citations or table_indexes[:1], risk_level=risk_level)

        for index in table_indexes:
            findings = summarize_ablation_table(contexts[index].prompt_text)
            if findings:
                citation_marker = f" [{index}]"
                table_label = self._extract_table_label(contexts[index].prompt_text)
                if self._is_chinese_question(question):
                    subject = f"{table_label} 的消融结果" if table_label else "消融表结果"
                    answer = subject + "显示：" + " ".join(findings) + citation_marker
                else:
                    subject = table_label or "The ablation table"
                    answer = f"{subject} shows: " + " ".join(findings) + citation_marker
                return QueryAnswerPayload(answer_markdown=answer, citations=[index], risk_level=risk_level)

        first_index = table_indexes[0]
        snippet = contexts[first_index].citation.excerpt or contexts[first_index].prompt_text[:1200]
        if self._is_chinese_question(question):
            answer = f"已找到相关表格证据，不能判定为缺失。相关片段如下： [{first_index}]\n\n{snippet}"
        else:
            answer = f"Relevant table evidence was found, so it should not be treated as missing. [{first_index}]\n\n{snippet}"
        return QueryAnswerPayload(answer_markdown=answer, citations=table_indexes[:1], risk_level=risk_level)

    def _deterministic_ablation_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
        risk_level: str,
    ) -> QueryAnswerPayload | None:
        for index in table_indexes:
            findings = summarize_ablation_table(contexts[index].prompt_text)
            if not findings:
                continue
            citation_marker = f" [{index}]"
            table_label = self._extract_table_label(contexts[index].prompt_text)
            if self._is_chinese_question(question):
                subject = f"{table_label} 的消融结果" if table_label else "消融表结果"
                answer = subject + "显示：" + " ".join(findings) + citation_marker
            else:
                subject = table_label or "The ablation table"
                answer = f"{subject} shows: " + " ".join(findings) + citation_marker
            return QueryAnswerPayload(answer_markdown=answer, citations=[index], risk_level=risk_level)
        return None

    def _extract_requested_metric_values(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
    ) -> list[ExtractedMetric]:
        row_selectors = self._question_row_selectors(question)
        results: list[ExtractedMetric] = []
        for index in table_indexes:
            text = contexts[index].prompt_text
            table_label = self._extract_table_label(text)
            requested = self._requested_datasets_for_table(question, text)
            table_row_selectors = [
                selector
                for selector in row_selectors
                if self._normalize_selector(selector) not in {self._normalize_selector(dataset) for dataset in requested}
            ]
            structured_metrics = table_metric_values(text, requested, row_selectors=table_row_selectors)
            if structured_metrics:
                for item in structured_metrics:
                    values = item.get("values") or {}
                    dataset = str(item.get("dataset") or "")
                    if dataset and values:
                        results.append(ExtractedMetric(index, str(item.get("table_label") or table_label or "") or None, dataset, dict(values)))
                continue
            parsed = self._extract_metric_values_from_markdown_table(text, index, table_label, requested, table_row_selectors)
            if not parsed:
                parsed = self._extract_inline_metric_values(text, index, table_label, requested)
            results.extend(parsed)

        requested_all = self._requested_datasets_for_contexts(question, contexts, table_indexes)
        filtered = [item for item in results if not requested_all or item.dataset.upper() in requested_all]
        ordered: list[ExtractedMetric] = []
        seen: set[tuple[int, str, tuple[tuple[str, str], ...]]] = set()
        for item in filtered:
            key = (item.context_index, item.dataset.upper(), tuple(item.values.items()))
            if key in seen:
                continue
            ordered.append(item)
            seen.add(key)
            if len(ordered) >= 8:
                break
        return ordered

    @classmethod
    def _requested_datasets_for_contexts(
        cls,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
    ) -> list[str]:
        requested: list[str] = []
        for anchor in cls._query_priority_anchors(question)["dataset"]:
            key = anchor.upper()
            if key not in requested:
                requested.append(key)
        for index in table_indexes:
            for dataset in cls._requested_datasets_for_table(question, contexts[index].prompt_text):
                if dataset.upper() not in requested:
                    requested.append(dataset.upper())
        return requested

    @classmethod
    def _requested_datasets_for_table(cls, question: str, table_text: str) -> list[str]:
        normalized_question = cls._normalize_selector(question)
        requested: list[str] = []
        for item in table_metric_values(table_text):
            dataset = str(item.get("dataset") or "")
            if dataset and cls._normalize_selector(dataset) in normalized_question and dataset.upper() not in requested:
                requested.append(dataset.upper())
        for anchor in cls._query_priority_anchors(question)["dataset"]:
            if anchor.upper() not in requested:
                requested.append(anchor.upper())
        return requested

    @classmethod
    def _extract_metric_values_from_markdown_table(
        cls,
        text: str,
        context_index: int,
        table_label: str | None,
        requested: list[str],
        row_selectors: list[str] | None = None,
    ) -> list[ExtractedMetric]:
        rows = cls._markdown_table_rows(text)
        if len(rows) < 2:
            return []
        simple = cls._extract_dataset_row_metrics(rows, context_index, table_label, requested)
        if simple:
            return simple
        return cls._extract_dataset_column_metrics(rows, context_index, table_label, requested, row_selectors or [])

    @classmethod
    def _extract_dataset_row_metrics(
        cls,
        rows: list[list[str]],
        context_index: int,
        table_label: str | None,
        requested: list[str],
    ) -> list[ExtractedMetric]:
        header = rows[0]
        metric_cols = {
            column: metric
            for column, cell in enumerate(header)
            if (metric := cls._normalize_metric_name(cell)) is not None
        }
        if not metric_cols:
            return []
        dataset_col = 0
        for column, cell in enumerate(header):
            if "dataset" in cell.lower():
                dataset_col = column
                break
        results: list[ExtractedMetric] = []
        for row in rows[1:]:
            if dataset_col >= len(row):
                continue
            dataset = cls._dataset_name_from_cell(row[dataset_col])
            if not dataset:
                continue
            if not cls._dataset_requested(dataset, requested):
                continue
            values = {
                metric: row[column].strip()
                for column, metric in metric_cols.items()
                if column < len(row) and re.search(r"\d+(?:\.\d+)?", row[column])
            }
            if values:
                results.append(ExtractedMetric(context_index, table_label, dataset, values))
        return results

    @classmethod
    def _extract_dataset_column_metrics(
        cls,
        rows: list[list[str]],
        context_index: int,
        table_label: str | None,
        requested: list[str],
        row_selectors: list[str],
    ) -> list[ExtractedMetric]:
        dataset_header_index = next(
            (index for index, row in enumerate(rows) if cls._row_has_dataset_header(row)),
            None,
        )
        if dataset_header_index is None:
            return []
        metric_header_index = next(
            (
                index
                for index in range(dataset_header_index + 1, min(len(rows), dataset_header_index + 4))
                if any(cls._normalize_metric_name(cell) for cell in rows[index])
            ),
            None,
        )
        if metric_header_index is None:
            return []

        dataset_header = rows[dataset_header_index]
        metric_header = rows[metric_header_index]
        width = max(len(dataset_header), len(metric_header), *(len(row) for row in rows[metric_header_index + 1 :]))
        dataset_by_col: dict[int, str] = {}
        current_dataset: str | None = None
        for column in range(width):
            cell = dataset_header[column].strip() if column < len(dataset_header) else ""
            dataset_name = cls._dataset_name_from_cell(cell)
            if dataset_name:
                current_dataset = dataset_name
            elif cell and column == 0:
                current_dataset = None
            if current_dataset:
                dataset_by_col[column] = current_dataset

        metric_by_col = {
            column: metric
            for column in range(width)
            if column < len(metric_header) and (metric := cls._normalize_metric_name(metric_header[column])) is not None
        }
        data_rows = rows[metric_header_index + 1 :]
        preferred_rows = cls._select_metric_rows(data_rows, row_selectors)

        results: list[ExtractedMetric] = []
        for row in preferred_rows:
            values_by_dataset: dict[str, dict[str, str]] = {}
            for column, dataset in dataset_by_col.items():
                if not cls._dataset_requested(dataset, requested):
                    continue
                metric = metric_by_col.get(column)
                if not metric or column >= len(row):
                    continue
                value = row[column].strip()
                if not re.search(r"\d+(?:\.\d+)?", value):
                    continue
                values_by_dataset.setdefault(dataset, {})[metric] = value
            for dataset, values in values_by_dataset.items():
                if values:
                    results.append(ExtractedMetric(context_index, table_label, dataset, values))
        return results

    @classmethod
    def _select_metric_rows(cls, rows: list[list[str]], row_selectors: list[str], limit: int = 3) -> list[list[str]]:
        data_rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not data_rows:
            return []
        selector_keys = [cls._normalize_selector(selector) for selector in row_selectors if selector]
        if selector_keys:
            matched = [
                row
                for row in data_rows
                if any(selector and selector in cls._normalize_selector(row[0] if row else "") for selector in selector_keys)
            ]
            if matched:
                return matched[:limit]
        numeric_rows = [row for row in data_rows if sum(1 for cell in row[1:] if re.search(r"\d+(?:\.\d+)?", cell)) >= 1]
        return numeric_rows[:limit]

    @classmethod
    def _question_row_selectors(cls, question: str) -> list[str]:
        selectors: list[str] = []
        dataset_keys = {cls._normalize_selector(anchor) for anchor in cls._query_priority_anchors(question)["dataset"]}
        metric_words = {"f1", "auc", "precision", "recall", "accuracy", "score", "metric", "metrics", "performance"}
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*){0,2}\b", question):
            value = match.group(0).strip()
            key = cls._normalize_selector(value)
            if len(key) < 3 or key in dataset_keys or key.lower() in metric_words:
                continue
            if value.lower() in {"what", "table"}:
                continue
            selectors.append(value)
        for facet in cls._extract_query_facets(question):
            key = cls._normalize_selector(facet)
            if len(key) >= 3 and key not in dataset_keys and facet not in selectors:
                selectors.append(facet)
        ordered: list[str] = []
        seen: set[str] = set()
        for selector in selectors:
            key = cls._normalize_selector(selector)
            if key and key not in seen:
                ordered.append(selector)
                seen.add(key)
        return ordered[:4]

    @staticmethod
    def _normalize_selector(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @classmethod
    def _extract_inline_metric_values(
        cls,
        text: str,
        context_index: int,
        table_label: str | None,
        requested: list[str],
    ) -> list[ExtractedMetric]:
        datasets = requested or [match.group(0).upper() for match in cls._DATASET_NAME_RE.finditer(text)]
        results: list[ExtractedMetric] = []
        lowered = text.lower()
        for dataset in dict.fromkeys(datasets):
            position = lowered.find(dataset.lower())
            if position < 0:
                continue
            window = text[max(0, position - 120) : min(len(text), position + 260)]
            values: dict[str, str] = {}
            for metric in ("F1", "AUC", "Precision", "Recall"):
                match = re.search(rf"\b{re.escape(metric)}\b(?:\s*score)?\s*(?:=|:|is|of)?\s*(\d+(?:\.\d+)?)", window, re.IGNORECASE)
                if match:
                    values[metric] = match.group(1)
            if values:
                results.append(ExtractedMetric(context_index, table_label, dataset, values))
        return results

    @staticmethod
    def _markdown_table_rows(text: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|") or "|" not in stripped[1:]:
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", cell or "---") for cell in cells):
                continue
            rows.append(cells)
        return rows

    @staticmethod
    def _normalize_metric_name(value: str) -> str | None:
        lowered = value.lower()
        if re.search(r"\bf\s*1\b", lowered) or "f1" in lowered:
            return "F1"
        if "auc" in lowered:
            return "AUC"
        if "precision" in lowered:
            return "Precision"
        if "recall" in lowered:
            return "Recall"
        if "accuracy" in lowered:
            return "Accuracy"
        return None

    @classmethod
    def _row_has_dataset_header(cls, row: list[str]) -> bool:
        names = [cls._dataset_name_from_cell(cell) for cell in row[1:]]
        return bool([name for name in names if name])

    @classmethod
    def _dataset_name_from_cell(cls, value: str) -> str:
        clean = re.sub(r"\s+", " ", str(value or "").strip())
        if (
            not clean
            or cls._normalize_metric_name(clean)
            or re.fullmatch(r"-?\d+(?:\.\d+)?%?", clean)
            or re.fullmatch(r":?-{3,}:?", clean)
        ):
            return ""
        match = cls._DATASET_NAME_RE.search(clean)
        return match.group(0).upper() if match else clean

    @classmethod
    def _dataset_requested(cls, dataset: str, requested: list[str]) -> bool:
        if not requested:
            return True
        dataset_key = cls._normalize_selector(dataset)
        return dataset_key in {cls._normalize_selector(item) for item in requested}

    @staticmethod
    def _extract_table_label(text: str) -> str | None:
        match = re.search(r"\bTable\s*\d+\b", text, re.IGNORECASE)
        return match.group(0) if match else None

    @staticmethod
    def _is_chinese_question(question: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", question))

    def _coverage_citation_indexes(self, question: str, contexts: list[RetrievedContext]) -> list[int]:
        indexes: list[int] = []
        facets = self._extract_query_facets(question)
        for facet in facets:
            lowered_facet = facet.lower()
            for index, context in enumerate(contexts):
                if lowered_facet in context.prompt_text.lower():
                    indexes.append(index)
                    break
        return indexes

    @staticmethod
    def _normalize_answer_citation_markup(answer_markdown: str) -> str:
        answer_markdown = re.sub(r"\[\[(\d+)\]\]", r"[\1]", answer_markdown)
        answer_markdown = re.sub(r"\[\[([^\]]+)\]\([^)]+\)\]", r"\1", answer_markdown)
        answer_markdown = re.sub(
            r"\[((?:\d+\s*,\s*)+\d+)\]",
            lambda match: "".join(f"[{part.strip()}]" for part in match.group(1).split(",")),
            answer_markdown,
        )

        def replace_wiki_link(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if re.fullmatch(r"(?:sources|entities|queries)/[^\s]+(?:\.md)?", inner):
                return ""
            return inner

        answer_markdown = re.sub(r"\[\[([^\]]+)\]\]", replace_wiki_link, answer_markdown)

        def strip_unresolved_label(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if re.fullmatch(r"\d+", inner):
                return match.group(0)
            return ""

        return re.sub(r"\[([^\]\n]+)\](?!\()", strip_unresolved_label, answer_markdown)

    @staticmethod
    def _renumber_answer_citations(answer_markdown: str, selected_indexes: list[int]) -> str:
        answer_markdown = QueryService._normalize_answer_citation_markup(answer_markdown)
        index_map = {context_index: output_index for output_index, context_index in enumerate(selected_indexes)}

        def replace(match: re.Match[str]) -> str:
            original = int(match.group(1))
            if original not in index_map:
                return ""
            return f"[{index_map[original]}]"

        return re.sub(r"\[(\d+)\]", replace, answer_markdown)

    @staticmethod
    def _strip_answer_citation_markers(answer_markdown: str) -> str:
        answer_markdown = QueryService._normalize_answer_citation_markup(answer_markdown)
        return re.sub(r"\[(\d+)\]", "", answer_markdown)

    @staticmethod
    def _drop_unreturned_citation_markers(answer_markdown: str, citation_count: int) -> str:
        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            return match.group(0) if index < citation_count else ""

        return re.sub(r"\[(\d+)\]", replace, answer_markdown)

    @classmethod
    def _unsupported_answer_numbers(cls, answer_markdown: str, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> set[str]:
        numbers = cls._answer_numbers(answer_markdown)
        if not numbers:
            return set()
        evidence = "\n".join(contexts[index].prompt_text for index in chosen_indexes if 0 <= index < len(contexts))
        return {number for number in numbers if number not in evidence}

    @staticmethod
    def _answer_numbers(answer_markdown: str) -> set[str]:
        numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?!\w)", answer_markdown))
        return {number for number in numbers if len(number) > 1 or "." in number or number.endswith("%")}

    def _verify_answer(self, answer_markdown: str, contexts: list[RetrievedContext]) -> VerificationPayload:
        claims = [
            {
                "subject": "answer",
                "predicate": "states",
                "object_text": answer_markdown,
                "evidence_excerpt": context.citation.excerpt,
                "confidence": context.score,
            }
            for context in contexts
        ]
        return self.verifier.verify_claims(answer_markdown, claims)

    def _select_citations(self, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> list[Citation]:
        return [citation for _, citation in self._select_citation_pairs(contexts, chosen_indexes)]

    def _select_citation_indexes(self, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> list[int]:
        return [index for index, _ in self._select_citation_pairs(contexts, chosen_indexes)]

    def _select_citation_pairs(self, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> list[tuple[int, Citation]]:
        selected: list[tuple[int, Citation]] = []
        seen_keys: set[str] = set()
        indexes = chosen_indexes or list(range(min(2, len(contexts))))
        for index in indexes:
            if not 0 <= index < len(contexts):
                continue
            citation = contexts[index].citation
            # Dedup by full content key.
            full_key = json.dumps(citation.model_dump(), sort_keys=True, ensure_ascii=False)
            if full_key in seen_keys:
                continue
            # Dedup by (page_slug, excerpt) to avoid same-page-same-excerpt repeats.
            if citation.page_slug:
                excerpt_key = f"{citation.page_slug}::{citation.excerpt}"
                if excerpt_key in seen_keys:
                    continue
                seen_keys.add(excerpt_key)
            seen_keys.add(full_key)
            selected.append((index, citation))

        # Cap same source page to at most 2 citations, keeping highest scores
        # and preferring different excerpts. Process once at the end.
        by_page: dict[str, list[tuple[int, Citation]]] = {}
        for index, citation in selected:
            if citation.page_slug:
                by_page.setdefault(citation.page_slug, []).append((index, citation))

        capped: list[tuple[int, Citation]] = []
        processed_pages: set[str] = set()
        for pair in selected:
            _, citation = pair
            if citation.page_slug and citation.page_slug in processed_pages:
                continue  # already added the capped set for this page
            if citation.page_slug and len(by_page.get(citation.page_slug, [])) > 2:
                kept = self._dedup_page_citation_pairs(by_page[citation.page_slug])
                capped.extend(kept)
                processed_pages.add(citation.page_slug)
            else:
                capped.append(pair)
        return capped

    @staticmethod
    def _dedup_page_citations(citations: list[Citation]) -> list[Citation]:
        """Keep at most 2 citations per page, preferring higher scores and distinct excerpts."""
        return [citation for _, citation in QueryService._dedup_page_citation_pairs(list(enumerate(citations)))]

    @staticmethod
    def _dedup_page_citation_pairs(pairs: list[tuple[int, Citation]]) -> list[tuple[int, Citation]]:
        sorted_pairs = sorted(pairs, key=lambda pair: (QueryService._citation_is_table_evidence(pair[1]), pair[1].score), reverse=True)
        kept: list[tuple[int, Citation]] = []
        seen_excerpts: set[str] = set()
        for index, citation in sorted_pairs:
            if len(kept) >= 2:
                break
            excerpt_normalized = citation.excerpt.strip()[:120]
            if excerpt_normalized in seen_excerpts:
                continue
            kept.append((index, citation))
            seen_excerpts.add(excerpt_normalized)
        return kept

    @staticmethod
    def _citation_is_table_evidence(citation: Citation) -> bool:
        excerpt = citation.excerpt or ""
        lowered = excerpt.lower()
        return bool("|" in excerpt or re.search(r"\btable\s*\d+", lowered))

    def _prefer_wiki_citations(
        self,
        citations: list[Citation],
        page_matches: list[PageMatch],
        project_id: str,
        question: str = "",
    ) -> list[Citation]:
        page_by_doc_id: dict[str, WikiPage] = {}
        for match in page_matches:
            page = match.page
            if page.kind != PageKind.source_summary.value:
                continue
            for document_id in page.source_document_ids:
                page_by_doc_id.setdefault(document_id, page)

        promoted: list[Citation] = []
        seen: set[str] = set()
        for citation in citations:
            replacement = citation
            if citation.document_id and citation.page_slug is None:
                page = page_by_doc_id.get(citation.document_id) or self._source_page_for_document(project_id, citation.document_id)
                if page is not None:
                    page_body = self._strip_frontmatter(page.markdown_content)
                    # Use the chunk excerpt as anchor text to find the relevant
                    # section in the wiki page, so the citation excerpt matches
                    # the actual evidence location instead of always the page start.
                    excerpt = self._promoted_wiki_excerpt(page_body, citation.excerpt, question)
                    replacement = Citation(
                        document_id=citation.document_id,
                        page_slug=page.slug,
                        page_title=strip_upload_prefix(page.title),
                        page_kind=page.kind,
                        score=citation.score,
                        page_label=citation.page_label,
                        excerpt=excerpt,
                    )
            key = json.dumps(replacement.model_dump(), sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            promoted.append(replacement)
            seen.add(key)
        # Also dedup by page_slug — keep at most 2 per page.
        return self._select_citations(
            [RetrievedContext(citation=c, prompt_text=c.excerpt, score=c.score) for c in promoted],
            list(range(len(promoted))),
        )

    def _promoted_wiki_excerpt(self, page_body: str, chunk_excerpt: str, question: str = "") -> str:
        if self._is_table_query(question) or self._is_metric_query(question):
            table_blocks = self._extract_table_blocks(page_body)
            ranked = self._rank_blocks(question or chunk_excerpt, table_blocks)
            for block, _ in ranked:
                if self._context_has_table_data(block):
                    return self._table_block_excerpt(block, question or chunk_excerpt)
        return self._window_text(page_body, self._tokenize(chunk_excerpt), max_chars=280, question=question)

    def _source_page_for_document(self, project_id: str, document_id: str) -> WikiPage | None:
        pages = self.db.scalars(
            select(WikiPage).where(
                WikiPage.project_id == project_id,
                WikiPage.kind == PageKind.source_summary.value,
            )
        ).all()
        for page in pages:
            if document_id in (page.source_document_ids or []):
                return page
        return None

    def _infer_citation_indexes(self, answer_markdown: str, context_count: int) -> list[int]:
        answer_markdown = self._normalize_answer_citation_markup(answer_markdown)
        indexes: list[int] = []
        for match in re.findall(r"\[(\d+)\]", answer_markdown):
            index = int(match)
            if 0 <= index < context_count and index not in indexes:
                indexes.append(index)
        return indexes

    def _load_index_context(self, project_slug: str) -> str | None:
        try:
            project_path_slug = safe_project_slug(project_slug)
        except InvalidStoragePathError:
            return None
        index_path = settings.wiki_dir / project_path_slug / "index.md"
        if not index_path.exists():
            return None
        return index_path.read_text(encoding="utf-8")[:4000]

    def _save_query_page(self, project: Project, question: str, response: QueryResponse, citations: list[Citation]) -> None:
        renderer = WikiRenderer(project)
        slug = f"queries/{datetime.utcnow():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}-{slugify(question)[:48]}"
        escaped_question = question.replace("\\", "\\\\").replace('"', '\\"')
        citation_lines: list[str] = []
        for citation in citations:
            if citation.page_slug:
                citation_lines.append(f"- Wiki: `{citation.page_slug}`")
            elif citation.document_id:
                label = citation.page_label or "n/a"
                citation_lines.append(f"- Source chunk: `{citation.document_id}` / `{citation.chunk_id}` (page={label})")

        markdown = "\n".join(
            [
                "---",
                f'title: "{escaped_question}"',
                'kind: "query_answer"',
                f"source_count: {len([citation for citation in citations if citation.document_id])}",
                f"verified_claim_count: {len(citations)}",
                "---",
                "",
                f"# {question}",
                "",
                "## Answer",
                response.answer_markdown,
                "",
                "## Citations",
                *(citation_lines or ["- None"]),
                "",
                "## Verification",
                response.verification_status,
            ]
        )
        path = renderer.write_page(slug, markdown)
        page = WikiPage(
            project_id=project.id,
            slug=slug,
            title=question[:255],
            kind=PageKind.query_answer.value,
            markdown_path=str(path),
            markdown_content=markdown,
            source_document_ids=sorted({citation.document_id for citation in citations if citation.document_id}),
        )
        self.db.add(page)
        self.db.flush()

        all_pages = self.db.scalars(select(WikiPage).where(WikiPage.project_id == project.id)).all()
        renderer.render_index(all_pages)
        self._append_query_log(renderer.paths["wiki_root"] / "log.md", question, slug, len(citations))

    @staticmethod
    def _append_query_log(log_path: Path, question: str, slug: str, citation_count: int) -> None:
        previous = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Log\n\n"
        entry = "\n".join(
            [
                f"## [{datetime.utcnow().isoformat()}] query | {question}",
                "",
                f"- Page: `{slug}`",
                f"- Citations: {citation_count}",
                "",
            ]
        )
        log_path.write_text(previous + entry, encoding="utf-8")

    @staticmethod
    def _strip_frontmatter(markdown: str) -> str:
        if markdown.startswith("---\n"):
            parts = markdown.split("\n---\n", 1)
            if len(parts) == 2:
                return parts[1]
        return markdown

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        lowered = text.lower()
        tokens: set[str] = set()
        for word in re.findall(r"[a-z0-9_]+", lowered):
            if len(word) > 1:
                tokens.add(word)
        for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
            if len(segment) == 1:
                tokens.add(segment)
                continue
            tokens.add(segment)
            for index in range(len(segment) - 1):
                tokens.add(segment[index : index + 2])
        return tokens

    # Patterns for prioritizing Figure/Table references in context windows.
    _FIGURE_TABLE_RE = re.compile(
        r"(Figure\s*\d+|Table\s*\d+|Fig\.\s*\d+|Appendix\s+[A-Z])",
        re.IGNORECASE,
    )
    _DATASET_NAME_RE = re.compile(
        r"\b(OIE2016|NYT|PENN|WEB|CoNLL|ACE|SemEval|WikiSQL|SQuAD|GLUE|SuperGLUE)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _window_text(cls, text: str, query_terms: set[str], max_chars: int = 1600, question: str = "") -> str:
        if len(text) <= max_chars:
            return text
        lowered = text.lower()

        question_anchors = cls._query_priority_anchors(question)

        # 1) Exact Figure/Table phrases from the question get highest priority.
        figure_table_positions: list[int] = []
        for anchor in question_anchors["figure_table"]:
            position = lowered.find(anchor.lower())
            if position >= 0:
                figure_table_positions.append(position)
        # 2) Dataset names from the question get second priority.
        dataset_positions: list[int] = []
        for anchor in question_anchors["dataset"]:
            position = lowered.find(anchor.lower())
            if position >= 0:
                dataset_positions.append(position)
        # 3) Generic query term positions.
        term_positions = [lowered.find(term) for term in query_terms if term and lowered.find(term) >= 0]

        # Pick the best anchor: prefer Figure/Table > dataset name > first term.
        anchor: int | None = None
        if figure_table_positions:
            # Pick the earliest Figure/Table mention as the anchor.
            anchor = min(figure_table_positions)
        elif dataset_positions:
            anchor = min(dataset_positions)
        elif term_positions:
            anchor = min(term_positions)

        if anchor is None:
            return text[:max_chars]

        # Center the window around the anchor, but bias toward showing content
        # *after* the anchor (captions, table data, metric rows).
        start = max(0, anchor - max_chars // 4)
        end = min(len(text), start + max_chars)
        start = max(0, end - max_chars)
        return text[start:end]

    def _context_for_facet(self, page: WikiPage, page_body: str, facet: str, query_terms: set[str]) -> RetrievedContext | None:
        lowered = page_body.lower()
        position = lowered.find(facet.lower())
        if position < 0:
            return None
        start = max(0, position - 500)
        end = min(len(page_body), position + 1600)
        excerpt = page_body[start:end]
        score = 4.0 + min(len(query_terms & self._tokenize(excerpt)), 8)
        return RetrievedContext(
            citation=Citation(
                page_slug=page.slug,
                page_title=strip_upload_prefix(page.title),
                page_kind=page.kind,
                score=score,
                page_label=self._extract_page_label_from_wiki_content(page_body, excerpt),
                excerpt=excerpt[:280],
            ),
            prompt_text=excerpt,
            score=score,
        )

    def _finalize_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        sorted_contexts = sorted(contexts, key=lambda item: item.score, reverse=True)
        finalized: list[RetrievedContext] = []
        seen_keys: set[str] = set()
        page_counts: dict[str, int] = {}
        for context in sorted_contexts:
            citation = context.citation
            normalized_excerpt = re.sub(r"\s+", " ", citation.excerpt.strip())[:180]
            key = f"{citation.page_slug or citation.document_id}:{citation.page_label}:{normalized_excerpt}"
            if key in seen_keys:
                continue
            if citation.page_slug:
                current_count = page_counts.get(citation.page_slug, 0)
                if current_count >= 3:
                    continue
                page_counts[citation.page_slug] = current_count + 1
            finalized.append(context)
            seen_keys.add(key)
            if len(finalized) >= MAX_CONTEXTS:
                break
        return finalized

    def _rank_blocks(self, question: str, blocks: list[str]) -> list[tuple[str, float]]:
        facets = [facet.lower() for facet in self._extract_query_facets(question)]
        query_terms = self._tokenize(question)
        ranked: list[tuple[str, float]] = []
        for index, block in enumerate(blocks):
            lowered = block.lower()
            block_terms = self._tokenize(block)
            score = float(len(query_terms & block_terms))
            for facet in facets:
                if facet and facet in lowered:
                    score += 6.0
            for anchor in self._query_priority_anchors(question)["figure_table"]:
                if anchor.lower() in lowered:
                    score += 6.0
            for anchor in self._query_priority_anchors(question)["dataset"]:
                if anchor.lower() in lowered:
                    score += 8.0
            if any(metric in lowered for metric in ("f1", "auc", "precision", "recall", "score", "指标")):
                score += 2.0
            if re.search(r"\d+(?:\.\d+)?", block):
                score += 1.5
            ranked.append((block, score - index * 0.01))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    @classmethod
    def _extract_query_facets(cls, question: str) -> list[str]:
        facets: list[str] = []
        for anchor in cls._query_priority_anchors(question)["figure_table"]:
            facets.append(anchor)
        for anchor in cls._query_priority_anchors(question)["dataset"]:
            facets.append(anchor)
        for match in re.finditer(r"\b(Generator|Verifier|Pruner|Retriever|OpenIE\s*6|Stanford\s*OIE|DeepEx|PIVE|ChatGPT|GPT-4|LLaMA|Qwen)\b", question, re.IGNORECASE):
            facets.append(match.group(0))
        chinese_component_map = {
            "生成器": "Generator",
            "验证器": "Verifier",
            "修剪器": "Pruner",
            "裁剪器": "Pruner",
            "检索器": "Retriever",
            "数据集": "dataset",
            "消融": "ablation",
            "指标": "metric",
        }
        for marker, facet in chinese_component_map.items():
            if marker in question:
                facets.append(facet)
        ordered: list[str] = []
        seen: set[str] = set()
        for facet in facets:
            normalized = facet.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            ordered.append(normalized)
            seen.add(key)
        return ordered

    @classmethod
    def _query_priority_anchors(cls, question: str) -> dict[str, list[str]]:
        if not question:
            return {"figure_table": [], "dataset": []}
        figure_table = [match.group(0) for match in cls._FIGURE_TABLE_RE.finditer(question)]
        for match in re.finditer(r"[图表]\s*\d+", question):
            figure_table.append(match.group(0))
        datasets = [match.group(0) for match in cls._DATASET_NAME_RE.finditer(question)]
        return {"figure_table": figure_table, "dataset": datasets}

    # ---- Figure / Table / Metric query helpers ----

    @staticmethod
    def _is_figure_query(question: str) -> bool:
        """Detect questions asking about specific figures or illustrations."""
        lowered = question.lower()
        return bool(
            re.search(r"figure\s*\d+", lowered)
            or re.search(r"fig\.?\s*\d+", lowered)
            or "图" in question
            or "illustration" in lowered
            or "diagram" in lowered
        )

    @staticmethod
    def _is_table_query(question: str) -> bool:
        """Detect questions asking about specific tables or tabular data."""
        lowered = question.lower()
        return bool(
            re.search(r"table\s*\d+", lowered)
            or "表" in question
            or "tabular" in lowered
        )

    @staticmethod
    def _is_metric_query(question: str) -> bool:
        """Detect questions asking about metrics, scores, or benchmark results."""
        lowered = question.lower()
        metric_markers = [
            "指标", "f1", "auc", "precision", "recall", "accuracy",
            "bleu", "rouge", "metric", "score", "performance",
            "oie2016", "nyt", "penn", "web",
        ]
        return any(marker in lowered for marker in metric_markers)

    @staticmethod
    def _extract_figure_blocks(markdown: str) -> list[str]:
        """Extract Figure Notes blocks from wiki markdown."""
        blocks: list[str] = []
        in_figure_section = False
        for line in markdown.split("\n"):
            if line.startswith("## Figure Notes"):
                in_figure_section = True
            elif in_figure_section and line.startswith("## ") and not line.startswith("## Figure Notes"):
                break
            elif in_figure_section and line.strip().startswith("- Page"):
                blocks.append(line.strip())
        if not blocks:
            # Fallback: search for "Figure" mentions anywhere.
            for match in re.finditer(r"(Figure\s*\d+|Fig\.\s*\d+)[:\-]?\s*(.+?)(?=Figure\s*\d+|Fig\.\s*\d+|$)", markdown, re.IGNORECASE):
                blocks.append(match.group(0).strip())
        return blocks

    @staticmethod
    def _extract_table_blocks(markdown: str) -> list[str]:
        """Extract Table blocks from wiki markdown."""
        blocks: list[str] = []
        in_table_section = False
        current_table: list[str] = []
        for line in markdown.split("\n"):
            if line.startswith("## Tables"):
                in_table_section = True
            elif in_table_section and line.startswith("## ") and not line.startswith("## Tables"):
                if current_table:
                    blocks.append("\n".join(current_table))
                    current_table = []
                in_table_section = False
            elif in_table_section and line.startswith("### Page"):
                if current_table:
                    blocks.append("\n".join(current_table))
                current_table = [line]
            elif in_table_section:
                current_table.append(line)
        if current_table:
            blocks.append("\n".join(current_table))
        if not blocks:
            # Fallback: search for markdown tables.
            table_re = re.compile(r"(\|.+\|[\s\S]*?(?=\n\n|\Z))", re.MULTILINE)
            for match in table_re.finditer(markdown):
                blocks.append(normalize_table_text(match.group(0).strip()))
        return [normalize_table_text(block) for block in blocks]

    @classmethod
    def _table_block_excerpt(cls, block: str, question: str = "", max_chars: int = 1200) -> str:
        block = normalize_table_text(block)
        lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            return block[:max_chars]

        start = 0
        for index, line in enumerate(lines):
            if re.search(r"\bTable\s*\d+\b", line, re.IGNORECASE) or line.strip().startswith("|"):
                start = index
                break
        lines = lines[start:]

        anchors = {cls._normalize_selector(anchor) for anchor in cls._query_priority_anchors(question)["dataset"]}
        anchors.update(cls._normalize_selector(facet) for facet in cls._extract_query_facets(question))
        anchors.update(cls._normalize_selector(selector) for selector in cls._question_row_selectors(question))
        anchors = {anchor for anchor in anchors if anchor}

        caption_lines = [line for line in lines if not line.strip().startswith("|")][:2]
        table_lines = [line for line in lines if line.strip().startswith("|")]
        header_lines = table_lines[:3]
        relevant_rows = [
            line
            for line in table_lines[3:]
            if any(anchor in cls._normalize_selector(line) for anchor in anchors)
        ]
        if not relevant_rows and table_lines:
            relevant_rows = table_lines[3:6]

        excerpt_lines: list[str] = []
        for line in [*caption_lines, *header_lines, *relevant_rows]:
            if line not in excerpt_lines:
                excerpt_lines.append(line)
        excerpt = "\n".join(excerpt_lines).strip() or "\n".join(lines).strip()
        return excerpt[:max_chars]

    @staticmethod
    def _extract_page_label_from_block(block: str) -> str | None:
        """Try to extract a page label from a table/figure block."""
        match = re.search(r"Page\s+(\d+)", block, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    @classmethod
    def _extract_page_label_from_wiki_content(cls, markdown: str, excerpt: str) -> str | None:
        if label := cls._extract_page_label_from_block(excerpt):
            return label
        position = markdown.find(excerpt[:80].strip()) if excerpt.strip() else -1
        if position < 0:
            return None
        prefix = markdown[:position]
        matches = list(re.finditer(r"(?:^|\n)(?:###?\s+)?Page\s+(\d+)", prefix, re.IGNORECASE))
        if matches:
            return matches[-1].group(1)
        return None

    # ---- End Figure / Table helpers ----

    @staticmethod
    def _is_high_risk(question: str) -> bool:
        markers = ("better", "best", "recommend", "advice", "which method")
        lowered = question.lower()
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _needs_source_evidence(question: str) -> bool:
        lowered = question.lower()
        markers = ("原文", "出处", "证据", "摘录", "quote", "quoted", "exact", "verbatim", "source")
        return any(marker in question or marker in lowered for marker in markers)

    def _should_use_wiki_only(self, question: str, page_matches: list[PageMatch]) -> bool:
        if not page_matches:
            return False
        if self._needs_source_evidence(question):
            return False
        return page_matches[0].score >= WIKI_PRIMARY_SCORE_THRESHOLD
