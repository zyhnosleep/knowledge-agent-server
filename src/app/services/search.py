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
from app.models.records import Document, DocumentChunk, PageKind, Project, QuestionAnswer, WikiPage
from app.schemas.common import Citation, QueryResponse
from app.services.ai import QueryAnswerPayload, VerificationPayload, cosine_similarity, safe_model_call
from app.services.ai import ExternalVerifier, OllamaClient
from app.services.filesystem import InvalidStoragePathError, safe_project_slug, slugify, strip_upload_prefix
from app.services.paper_profile import alias_in_text, paper_profile_data, paper_profile_text, source_fields_for_document
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
PAPER_ROUTE_MIN_SCORE = 2.0


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
class PaperMatch:
    document: Document
    score: float
    exact_alias: bool = False


@dataclass
class ExtractedMetric:
    context_index: int
    table_label: str | None
    dataset: str
    values: dict[str, str]


class QueryService:
    _TABLE_MODEL_TERM_RE = re.compile(r"^(?:amber|charmm|gaff|opls|c\d+|ff\d+)[a-z0-9]*$")
    _TABLE_BROAD_METRIC_TERM_KEYS = {
        "accuracy",
        "auc",
        "error",
        "errors",
        "f1",
        "mae",
        "metric",
        "metrics",
        "mse",
        "performance",
        "precision",
        "recall",
        "rmse",
        "score",
        "scores",
        "value",
        "values",
    }

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ollama = OllamaClient()
        self.verifier = ExternalVerifier()

    def answer(self, project_slug: str, question: str, save_answer: bool = True) -> QueryResponse:
        return self._answer_rag_first(project_slug, question, save_answer=save_answer)

    def _answer_wiki_first(self, project_slug: str, question: str, save_answer: bool = True) -> QueryResponse:
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
        chosen_indexes = self._table_evidence_indexes_only(question, contexts, chosen_indexes)
        chosen_indexes = self._select_citation_indexes(contexts, chosen_indexes)
        citations = self._select_citations(contexts, chosen_indexes)
        if not self._needs_source_evidence(question):
            citations = self._prefer_wiki_citations(citations, page_matches, project.id, question=question)
        answer_text_for_citations = self._retarget_table_answer_citations(
            question,
            answer_payload.answer_markdown,
            chosen_indexes,
        )
        answer_markdown = self._renumber_answer_citations(answer_text_for_citations, chosen_indexes)
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

    def _answer_rag_first(self, project_slug: str, question: str, save_answer: bool = True) -> QueryResponse:
        project = self.db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            raise ValueError(f"Project '{project_slug}' not found")

        paper_matches = self._route_papers(question, project.id)
        contexts = self._build_rag_contexts(question, project.id, paper_matches)
        if not contexts:
            contexts = self._search_source_chunks(question, project.id, [], limit=5)
        if not contexts:
            answer_payload = self._draft_answer(question, None, [])
            response = QueryResponse(answer_markdown=answer_payload.answer_markdown, citations=[], verification_status="local-only")
            if save_answer:
                record = QuestionAnswer(
                    project_id=project.id,
                    question=question,
                    answer_markdown=response.answer_markdown,
                    citations=[],
                    risk_level=answer_payload.risk_level,
                    verification_status=response.verification_status,
                )
                self.db.add(record)
                self.db.commit()
            return response

        answer_payload = self._deterministic_table_answer_if_supported(
            question,
            contexts,
            "high" if self._is_high_risk(question) else "normal",
        )
        if answer_payload is None:
            answer_payload = self._draft_answer(question, None, contexts)
        verification_status = "local-only"

        if self._is_high_risk(question):
            verification = self._verify_answer(answer_payload.answer_markdown, contexts)
            verification_status = verification.verdict
            if verification.notes:
                answer_payload.answer_markdown += f"\n\n> Verification note: {verification.notes}"

        answer_payload.answer_markdown = self._normalize_answer_citation_markup(answer_payload.answer_markdown)
        chosen_indexes = self._choose_citation_indexes(question, answer_payload, contexts)
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        answer_payload = self._repair_unsupported_numeric_answer(question, None, contexts, answer_payload, chosen_indexes)
        answer_payload = self._repair_missing_table_answer(question, None, contexts, answer_payload)
        if not (self._is_table_query(question) or self._is_metric_query(question)):
            answer_payload.answer_markdown = self._append_missing_supported_question_terms(
                question,
                answer_payload.answer_markdown,
                contexts,
            )
        answer_payload.answer_markdown = self._normalize_answer_citation_markup(answer_payload.answer_markdown)
        chosen_indexes = self._choose_citation_indexes(question, answer_payload, contexts)
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        chosen_indexes = self._table_evidence_indexes_only(question, contexts, chosen_indexes)
        chosen_indexes = self._select_citation_indexes(contexts, chosen_indexes)
        citations = self._select_citations(contexts, chosen_indexes)

        answer_text_for_citations = self._retarget_table_answer_citations(
            question,
            answer_payload.answer_markdown,
            chosen_indexes,
        )
        answer_markdown = self._renumber_answer_citations(answer_text_for_citations, chosen_indexes)
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
            if self._is_placeholder_wiki_page(body_text):
                continue
            if score <= 0:
                continue
            matches.append(PageMatch(page=page, score=float(score)))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:limit]

    def _route_papers(self, question: str, project_id: str, limit: int = 3) -> list[PaperMatch]:
        query_terms = self._tokenize(question)
        documents = self.db.scalars(select(Document).where(Document.project_id == project_id)).all()
        matches: list[PaperMatch] = []
        for document in documents:
            profile = paper_profile_data(document)
            profile_text = paper_profile_text(document)
            profile_terms = self._tokenize(profile_text)
            title_terms = self._tokenize(document.title)
            alias_values = [str(item) for item in profile.get("aliases") or []]
            key_values = [str(item) for item in profile.get("key_terms") or []]
            alias_terms = self._tokenize(" ".join(alias_values))
            key_terms = self._tokenize(" ".join(key_values))
            exact_alias = self._question_has_exact_alias(question, alias_values)
            score = (
                len(query_terms & profile_terms)
                + len(query_terms & title_terms) * 3
                + len(query_terms & alias_terms) * 5
                + len(query_terms & key_terms) * 2
            )
            if exact_alias:
                score += 18
            if score >= PAPER_ROUTE_MIN_SCORE:
                matches.append(PaperMatch(document=document, score=float(score), exact_alias=exact_alias))
        ranked = sorted(matches, key=lambda item: item.score, reverse=True)
        subject_locked = [
            match
            for match in ranked
            if self._question_locks_document_subject(question, [str(item) for item in (paper_profile_data(match.document).get("aliases") or [])])
        ]
        if subject_locked:
            return subject_locked[:1]
        if self._is_cross_paper_query(question):
            return ranked[: max(limit, 5)]
        exact_matches = [match for match in ranked if match.exact_alias]
        if exact_matches:
            return exact_matches[:limit]
        return ranked[:limit]

    @staticmethod
    def _question_locks_document_subject(question: str, aliases: list[str]) -> bool:
        for alias in aliases:
            alias = str(alias or "").strip()
            if not alias:
                continue
            escaped = re.escape(alias)
            if re.search(rf"(?<![A-Za-z0-9_/\-]){escaped}(?![A-Za-z0-9_/\-])\s*的\s*(?:表格|论文|文献)", question, re.IGNORECASE):
                return True
            if re.search(rf"(?<![A-Za-z0-9_/\-]){escaped}(?![A-Za-z0-9_/\-])['’]s\s+(?:table|paper|article)", question, re.IGNORECASE):
                return True
        return False

    def _build_rag_contexts(self, question: str, project_id: str, paper_matches: list[PaperMatch]) -> list[RetrievedContext]:
        document_ids = [match.document.id for match in paper_matches]
        contexts: list[RetrievedContext] = []
        if self._is_table_query(question) or self._is_metric_query(question):
            table_contexts = self._search_document_table_contexts(question, project_id, document_ids, limit=MAX_CONTEXTS)
            if not table_contexts and document_ids:
                table_contexts = self._search_document_table_contexts(question, project_id, [], limit=MAX_CONTEXTS)
            contexts.extend(table_contexts)
        if document_ids:
            contexts.extend(self._search_source_chunks(question, project_id, document_ids, limit=MAX_CONTEXTS))
        if not contexts and not document_ids:
            contexts.extend(self._search_source_chunks(question, project_id, [], limit=MAX_CONTEXTS))
        return self._finalize_contexts(contexts)

    @staticmethod
    def _question_has_exact_alias(question: str, aliases: list[str]) -> bool:
        return any(alias_in_text(alias, question) for alias in aliases if str(alias or "").strip())

    @staticmethod
    def _is_cross_paper_query(question: str) -> bool:
        lowered = question.lower()
        markers = (
            "compare",
            "comparison",
            "versus",
            " vs ",
            " v.s.",
            "between",
            "across papers",
            "multiple papers",
            "跨论文",
            "比较",
            "对比",
            "相比",
            "差异",
            "区别",
        )
        return any(marker in lowered or marker in question for marker in markers)

    def _search_document_table_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 5,
    ) -> list[RetrievedContext]:
        statement = select(Document).where(Document.project_id == project_id)
        if document_ids:
            statement = statement.where(Document.id.in_(document_ids))
        documents = self.db.scalars(statement).all()
        source_page_fields = self._source_page_fields_by_document_id(project_id, [document.id for document in documents])
        contexts: list[RetrievedContext] = []
        for document in documents:
            metadata = document.metadata_json or {}
            intelligence = metadata.get("document_intelligence") if isinstance(metadata.get("document_intelligence"), dict) else {}
            tables = intelligence.get("tables") if isinstance(intelligence, dict) else []
            if not isinstance(tables, list):
                continue
            for ordinal, table in enumerate(tables):
                if isinstance(table, dict):
                    markdown = str(table.get("markdown") or "")
                    page_label = str(table.get("page_label") or "").strip() or None
                else:
                    markdown = str(table or "")
                    page_label = None
                block = normalize_table_text(markdown)
                if not block or not self._context_has_table_data(block):
                    continue
                if not self._table_block_matches_query(question, block):
                    continue
                block_score = self._rank_blocks(question, [block])[0][1]
                score = TABLE_CONTEXT_SCORE_BOOST + block_score + max(0.0, 2.0 - ordinal * 0.01)
                excerpt = self._table_citation_excerpt(block, question)
                contexts.append(
                    RetrievedContext(
                        citation=Citation(
                            document_id=document.id,
                            **source_page_fields.get(document.id, {}),
                            score=score,
                            page_label=page_label,
                            excerpt=excerpt,
                        ),
                        prompt_text=block[:4000],
                        score=score,
                    )
                )
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    def _source_page_fields_by_document_id(self, project_id: str, document_ids: list[str]) -> dict[str, dict[str, str]]:
        wanted = set(document_ids)
        if not wanted:
            return {}
        documents = self.db.scalars(
            select(Document).where(
                Document.project_id == project_id,
                Document.id.in_(wanted),
            )
        ).all()
        fields: dict[str, dict[str, str]] = {}
        for document in documents:
            fields[document.id] = source_fields_for_document(document)
        return fields

    def _search_source_chunks(self, question: str, project_id: str, document_ids: list[str], limit: int = 3) -> list[RetrievedContext]:
        statement = select(DocumentChunk).join(DocumentChunk.document).where(DocumentChunk.document.has(project_id=project_id))
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))
        chunks = self.db.scalars(statement).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )

        question_vector = safe_model_call(lambda: self.ollama.embed([question])[0], [])
        query_terms = self._tokenize(question)
        is_table_query = self._is_table_query(question)
        needs_table_first = is_table_query or self._is_metric_query(question)
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
            has_table_data = self._context_has_table_data(chunk.text)
            if is_table_query and not has_table_data:
                continue
            if needs_table_first and not has_table_data and not self._context_has_metric_numbers(chunk.text):
                continue
            if needs_table_first and has_table_data:
                if not self._table_block_matches_query(question, chunk.text):
                    continue
                score += TABLE_CONTEXT_SCORE_BOOST + self._rank_blocks(question, [chunk.text])[0][1]
                prompt_text = self._table_citation_excerpt(chunk.text, question, max_chars=2400)
                excerpt = prompt_text
            else:
                prompt_text = self._window_text(chunk.text, query_terms, max_chars=1600, question=question)
                excerpt = prompt_text[:280]
            scored.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        **source_page_fields.get(chunk.document_id, {}),
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=excerpt,
                    ),
                    prompt_text=prompt_text,
                    score=score,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _context_has_metric_numbers(text: str) -> bool:
        lowered = text.lower()
        has_metric_word = bool(re.search(r"\b(f\s*1|auc|precision|recall|accuracy|rmse|mae|score|metric)\b", lowered))
        return has_metric_word and bool(re.search(r"\d+(?:\.\d+)?", text))

    @classmethod
    def _table_citation_excerpt(cls, block: str, question: str = "", max_chars: int = 2400) -> str:
        excerpt = cls._table_block_excerpt(block, question, max_chars=max_chars)
        if re.search(r"\btable\b", excerpt, re.IGNORECASE):
            return excerpt
        table_anchors = [
            anchor
            for anchor in cls._query_priority_anchors(question)["figure_table"]
            if re.match(r"table\s*\d+", anchor, re.IGNORECASE)
        ]
        label = table_anchors[0] if table_anchors else "Table evidence"
        return f"{label}: {excerpt}"

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
                    if not self._table_block_matches_query(question, block):
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
        if needs_table_first:
            global_table_contexts = self._search_wiki_table_contexts(question, project_id, page_matches)
            if global_table_contexts:
                contexts.extend(global_table_contexts)
                return self._finalize_contexts(contexts)
            if is_metric_query or is_table_query:
                return self._finalize_contexts(contexts)
        contexts.extend(fallback_contexts)
        if self._should_use_wiki_only(question, page_matches):
            return self._finalize_contexts(contexts)
        if matched_doc_ids:
            contexts.extend(self._search_source_chunks(question, project_id, sorted(set(matched_doc_ids))))
        if contexts:
            return self._finalize_contexts(contexts)
        return self._search_source_chunks(question, project_id, [], limit=5)

    def _search_wiki_table_contexts(
        self,
        question: str,
        project_id: str,
        page_matches: list[PageMatch],
        limit: int = 5,
    ) -> list[RetrievedContext]:
        matched_page_ids = {match.page.id for match in page_matches}
        pages = self.db.scalars(
            select(WikiPage).where(
                WikiPage.project_id == project_id,
                WikiPage.kind != PageKind.query_answer.value,
            )
        ).all()
        contexts: list[RetrievedContext] = []
        for page in pages:
            page_body = self._strip_frontmatter(page.markdown_content)
            table_blocks = self._extract_table_blocks(page_body)
            if not table_blocks:
                continue
            page_bonus = 8.0 if page.id in matched_page_ids else 0.0
            for block, block_score in self._rank_blocks(question, table_blocks)[:3]:
                if not self._context_has_table_data(block):
                    continue
                if not self._table_block_matches_query(question, block):
                    continue
                score = TABLE_CONTEXT_SCORE_BOOST + page_bonus + block_score
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
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    @classmethod
    def _table_block_matches_query(cls, question: str, block: str) -> bool:
        block_key = cls._normalize_selector(block)
        if not block_key:
            return False
        anchors = cls._query_priority_anchors(question)
        table_terms = [cls._normalize_selector(anchor) for anchor in anchors["figure_table"]]
        dataset_terms = [cls._normalize_selector(anchor) for anchor in anchors["dataset"]]
        generic_terms = [
            cls._normalize_selector(anchor)
            for anchor in [*cls._question_row_selectors(question), *cls._extract_generic_table_terms(question)]
        ]
        table_terms = [term for term in table_terms if len(term) >= 3]
        low_signal_terms = {
            "what",
            "which",
            "where",
            "when",
            "does",
            "please",
            "cite",
            "show",
            "tell",
            "give",
            "drawn",
            "conclusion",
            "conclusions",
            "report",
            "reports",
            "result",
            "results",
            "table",
            "figure",
            "metric",
            "metrics",
            "dataset",
            "datasets",
            "value",
            "values",
            "score",
            "scores",
            "performance",
            "accuracy",
            "precision",
            "recall",
            "ablation",
            "study",
        }
        non_table_terms = [
            term
            for term in [*dataset_terms, *generic_terms]
            if len(term) >= 3
            and term not in low_signal_terms
            and not re.fullmatch(r"(?:table|figure|fig)\d+", term)
        ]
        non_table_terms = list(dict.fromkeys(non_table_terms))
        if non_table_terms:
            matched_terms = {term for term in non_table_terms if term in block_key}
            required_matches = 2 if len(non_table_terms) >= 2 else 1
            return len(matched_terms) >= required_matches
        if table_terms:
            return any(term in block_key for term in table_terms)
        query_terms = cls._tokenize(question)
        block_terms = cls._tokenize(block)
        return len(query_terms & block_terms) >= 2

    def _draft_answer(self, question: str, index_context: str | None, contexts: list[RetrievedContext]) -> QueryAnswerPayload:
        if not contexts:
            return QueryAnswerPayload(
                answer_markdown="No supporting evidence was found yet. Please ingest relevant sources first.",
                citations=[],
                risk_level="normal",
            )

        prompt_sections: list[str] = []
        if index_context and not contexts:
            prompt_sections.append("Index overview:\n" + index_context)
        prompt_sections.extend(f"[{index}] {self._prompt_context_text(question, context)}" for index, context in enumerate(contexts))
        context_text = "\n\n".join(prompt_sections)

        # Build figure/table/dataset-aware guardrails.
        constraints = self._build_answer_constraints(question, contexts)

        fallback = QueryAnswerPayload(
            answer_markdown="\n".join(
                [
                    "## Answer",
                    "The answer below is based on the currently retrieved source evidence. Please verify against the cited materials when needed.",
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
                    "Answer using only the retrieved source evidence. "
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
                system_prompt="You are answering against a RAG evidence set. Use only retrieved source, table, and figure evidence; cite supporting context indexes and do not claim facts that are absent from the provided material.",
                user_prompt=prompt,
            ),
            fallback,
        )

    def _build_answer_constraints(self, question: str, contexts: list[RetrievedContext]) -> str:
        """Build guardrail instructions based on the question type."""
        parts: list[str] = []
        lowered = question.lower()

        has_figure_context = any(
            "figure" in self._context_evidence_text(ctx).lower() or "fig." in self._context_evidence_text(ctx).lower()
            for ctx in contexts
        )
        has_table_context = any(
            "table" in self._context_evidence_text(ctx).lower() or "|" in self._context_evidence_text(ctx)
            for ctx in contexts
        )
        evidence_acronyms = self._salient_evidence_acronyms(contexts)

        # Figure/Table constraint: if context has them, don't say "not included".
        if self._is_chinese_question(question):
            parts.append(
                "IMPORTANT: Answer in Chinese because the user's question is written in Chinese. "
                "Keep table labels, dataset names, model names, and metric names verbatim when needed."
            )
        if evidence_acronyms:
            parts.append(
                "IMPORTANT: Preserve these source acronyms/model or method names exactly when they are relevant: "
                + ", ".join(evidence_acronyms)
                + ". Do not replace an acronym only with an expanded translation."
            )

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
        constrained_context = "\n\n".join(f"[{index}] {self._prompt_context_text(question, ctx)}" for index, ctx in supported_pairs)
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

    def _deterministic_table_answer_if_supported(
        self,
        question: str,
        contexts: list[RetrievedContext],
        risk_level: str,
    ) -> QueryAnswerPayload | None:
        if not (self._is_table_query(question) or self._is_metric_query(question)):
            return None
        table_indexes = self._table_citation_indexes(question, contexts)
        if not table_indexes:
            return None
        answer = self._deterministic_table_answer(question, contexts, table_indexes, risk_level)
        if self._answer_claims_table_data_missing(answer.answer_markdown):
            return None
        if self._is_metric_query(question):
            metrics = self._extract_requested_metric_values(question, contexts, table_indexes)
            if metrics and self._answer_lacks_requested_metrics(question, answer.answer_markdown, metrics):
                return None
        evidence = "\n".join(self._context_table_evidence_text(contexts[index]) for index in table_indexes)
        if re.search(r"\d+(?:\.\d+)?", evidence) and not re.search(r"\d+(?:\.\d+)?", answer.answer_markdown):
            return None
        return answer

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
        answer_lacks_metrics = self._answer_lacks_requested_metrics(question, answer_payload.answer_markdown, extracted_metrics)
        if not answer_missing and not answer_lacks_metrics:
            return answer_payload

        prompt_sections: list[str] = []
        if index_context and not table_indexes:
            prompt_sections.append("Index overview:\n" + index_context)
        prompt_sections.extend(f"[{index}] {self._prompt_context_text(question, contexts[index])}" for index in table_indexes[:4])
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
                system_prompt="You answer table and metric questions against retrieved source table contexts. Use only provided values and cite the supporting context indexes.",
                user_prompt=prompt,
            ),
            fallback,
        )
        repaired.answer_markdown = self._normalize_answer_citation_markup(repaired.answer_markdown)
        repaired.citations = [index for index in repaired.citations if index in table_indexes] or table_indexes[:2]
        repaired_lacks_metrics = self._answer_lacks_requested_metrics(question, repaired.answer_markdown, extracted_metrics)
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
            if any(number in self._context_evidence_text(context) for number in self._answer_numbers(answer_markdown))
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

    def _table_evidence_indexes_only(
        self,
        question: str,
        contexts: list[RetrievedContext],
        indexes: list[int],
    ) -> list[int]:
        if not (self._is_table_query(question) or self._is_metric_query(question)):
            return indexes
        table_indexes = self._table_citation_indexes(question, contexts)
        if not table_indexes:
            return indexes
        allowed = set(table_indexes)
        return [index for index in indexes if index in allowed] or table_indexes

    def _table_citation_indexes(self, question: str, contexts: list[RetrievedContext]) -> list[int]:
        if not (self._is_table_query(question) or self._is_metric_query(question)):
            return []
        scored: list[tuple[float, int]] = []
        priority_terms = {
            self._normalize_selector(anchor)
            for anchor in self._query_priority_anchors(question)["dataset"]
            if len(self._normalize_selector(anchor)) >= 3
        }
        generic_terms = {
            self._normalize_selector(anchor)
            for anchor in self._extract_generic_table_terms(question)
            if len(self._normalize_selector(anchor)) >= 3
        }
        requested_table_terms = priority_terms or generic_terms
        requires_requested_terms = self._is_metric_query(question) and bool(requested_table_terms)
        for index, context in enumerate(contexts):
            text = self._context_table_evidence_text(context)
            if not self._context_has_table_data(text):
                continue
            text_key = self._normalize_selector(text)
            if requires_requested_terms and not any(term in text_key for term in requested_table_terms):
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
        return [index for _, index in sorted(scored, reverse=True)[:5]]

    @staticmethod
    def _context_has_table_data(text: str) -> bool:
        return QueryService._has_markdown_table_rows(text)

    @staticmethod
    def _has_markdown_table_rows(text: str) -> bool:
        rows: list[list[str]] = []
        for line in normalize_table_text(text).splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
                continue
            if QueryService._markdown_rows_have_data(rows):
                return True
            rows = []
        return QueryService._markdown_rows_have_data(rows)

    @staticmethod
    def _markdown_rows_have_data(rows: list[list[str]]) -> bool:
        if len(rows) < 2:
            return False
        has_separator = any(QueryService._is_markdown_separator_row(row) for row in rows)
        non_separator_rows = [row for row in rows if not QueryService._is_markdown_separator_row(row)]
        has_data_row = any(QueryService._is_markdown_data_row(row) for row in non_separator_rows[1:])
        if has_separator:
            return has_data_row
        has_header = QueryService._is_markdown_data_row(non_separator_rows[0])
        has_numeric_data_row = any(
            QueryService._is_markdown_data_row(row) and any(re.search(r"\d+(?:\.\d+)?", cell) for cell in row)
            for row in non_separator_rows[1:]
        )
        return has_header and has_numeric_data_row

    @staticmethod
    def _is_markdown_separator_row(row: list[str]) -> bool:
        cells = [cell.strip() for cell in row if cell.strip()]
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    @staticmethod
    def _is_markdown_data_row(row: list[str]) -> bool:
        return sum(1 for cell in row if cell.strip()) >= 2

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
            "cannot be extracted",
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

    @classmethod
    def _answer_lacks_requested_metrics(cls, question: str, answer_markdown: str, metrics: list[ExtractedMetric]) -> bool:
        if not cls._is_metric_query(question):
            return False
        return bool(metrics) and not cls._answer_contains_extracted_metrics(answer_markdown, metrics)

    def _deterministic_table_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
        risk_level: str,
    ) -> QueryAnswerPayload:
        is_ablation_query = "ablation" in question.lower() or "消融" in question
        if not self._is_metric_query(question):
            ablation_answer = self._deterministic_ablation_answer(question, contexts, table_indexes, risk_level)
            if ablation_answer is not None:
                return ablation_answer
            if is_ablation_query:
                first_index = table_indexes[0]
                if self._is_chinese_question(question):
                    answer = f"检索到的表格证据中没有可解析的结构化消融表。 [{first_index}]"
                else:
                    answer = f"The retrieved table evidence does not contain a structured ablation table. [{first_index}]"
                return QueryAnswerPayload(answer_markdown=answer, citations=table_indexes[:1], risk_level=risk_level)

        metrics = self._extract_requested_metric_values(question, contexts, table_indexes) if self._is_metric_query(question) else []
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
            findings = summarize_ablation_table(self._context_table_evidence_text(contexts[index]))
            if findings:
                citation_marker = f" [{index}]"
                table_label = self._extract_table_label(self._context_table_evidence_text(contexts[index]))
                if self._is_chinese_question(question):
                    subject = f"{table_label} 的消融结果" if table_label else "消融表结果"
                    answer = subject + "显示：" + " ".join(self._localize_ablation_findings(findings)) + citation_marker
                else:
                    subject = table_label or "The ablation table"
                    answer = f"{subject} shows: " + " ".join(findings) + citation_marker
                return QueryAnswerPayload(answer_markdown=answer, citations=[index], risk_level=risk_level)

        generic_answer = self._deterministic_generic_table_answer(question, contexts, table_indexes, risk_level)
        if generic_answer is not None:
            return generic_answer

        first_index = table_indexes[0]
        if self._is_metric_query(question):
            if self._is_chinese_question(question):
                answer = f"检索到的表格证据中没有可解析的请求指标值。 [{first_index}]"
            else:
                answer = f"The retrieved table evidence does not contain parseable requested metric values. [{first_index}]"
            return QueryAnswerPayload(answer_markdown=answer, citations=table_indexes[:1], risk_level=risk_level)

        snippet = contexts[first_index].citation.excerpt or contexts[first_index].prompt_text[:1200]
        if self._is_chinese_question(question):
            answer = f"已找到相关表格证据。相关片段如下： [{first_index}]\n\n{snippet}"
        else:
            answer = f"Relevant table evidence was found, so it should not be treated as missing. [{first_index}]\n\n{snippet}"
        return QueryAnswerPayload(answer_markdown=answer, citations=table_indexes[:1], risk_level=risk_level)

    def _deterministic_generic_table_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
        risk_level: str,
    ) -> QueryAnswerPayload | None:
        citations: list[int] = []
        parts: list[str] = []
        for index in table_indexes:
            text = self._context_table_evidence_text(contexts[index])
            rows = self._generic_table_value_rows(question, text)
            if not rows:
                continue
            if index not in citations:
                citations.append(index)
            table_label = self._extract_table_label(text)
            for row in rows[:8]:
                if len(parts) >= 8:
                    break
                prefix = f"{table_label} " if table_label else ""
                label = " - ".join(item for item in (row.get("group"), row.get("property")) if item)
                values = row.get("values", "")
                if label and values:
                    if self._is_chinese_question(question):
                        parts.append(f"{prefix}对于 {label}，各列对应的表格数值为：{values}".strip())
                    else:
                        parts.append(f"{prefix}{label}: {values}".strip())
            if len(parts) >= 8:
                break
        if not parts:
            return None
        citation_marker = f" [{citations[0]}]" if citations else ""
        if self._is_chinese_question(question):
            answer = (
                "根据表格证据，下面逐项列出与问题实体匹配的数值；每一项都来自同一表格行，"
                "英文模型名和数字按原表保留，便于和 citation 逐项核对。以下内容可直接作为答案依据："
                + "；".join(parts)
                + citation_marker
            )
        else:
            answer = "The relevant table values are: " + "; ".join(parts) + citation_marker
        return QueryAnswerPayload(answer_markdown=answer, citations=citations or table_indexes[:1], risk_level=risk_level)

    @classmethod
    def _generic_table_value_rows(cls, question: str, table_text: str) -> list[dict[str, str]]:
        excerpt = cls._table_block_excerpt(table_text, question, max_chars=2400)
        table_lines = [line for line in normalize_table_text(excerpt).splitlines() if cls._is_table_line(line)]
        if not table_lines:
            return []
        header_count = cls._table_header_line_count(table_lines)
        header_rows = [cls._markdown_table_line_cells(line) for line in table_lines[:header_count]]
        data_lines = table_lines[header_count:]
        if not header_rows or not data_lines:
            return []
        headers = cls._compose_display_headers(header_rows)
        selected_columns = cls._selected_table_value_columns(question, headers)
        include_all_numeric_columns = cls._is_comparison_or_difference_query(question)
        value_rows: list[dict[str, str]] = []
        fallback_value_rows: list[dict[str, str]] = []
        current_group = ""
        for ordinal, line in enumerate(data_lines):
            cells = cls._markdown_table_line_cells(line)
            if not cells or cls._is_markdown_separator_row(cells):
                continue
            padded = cells + [""] * max(0, len(headers) - len(cells))
            first_cell = padded[0].strip() if padded else ""
            numeric_columns = [
                column
                for column, cell in enumerate(padded[1:], start=1)
                if re.search(r"\d+(?:\.\d+)?", cell)
            ]
            if first_cell:
                current_group = first_cell
            if not numeric_columns:
                continue
            columns = [
                column
                for column in numeric_columns
                if include_all_numeric_columns or not selected_columns or column in selected_columns
            ]
            if not columns:
                columns = numeric_columns
            property_cell = padded[1].strip() if len(padded) > 1 else ""
            values: list[str] = []
            for column in columns:
                header = headers[column] if column < len(headers) else f"Column {column + 1}"
                value = padded[column].strip()
                if header and value:
                    values.append(f"{header} {value}")
            if values:
                score = cls._generic_table_row_relevance(question, current_group or first_cell, property_cell)
                row_payload = {
                    "group": current_group or first_cell,
                    "property": property_cell if not re.search(r"\d+(?:\.\d+)?", property_cell) else "",
                    "values": ", ".join(values),
                    "score": str(score),
                    "ordinal": str(ordinal),
                }
                if score > 0:
                    value_rows.append(row_payload)
                elif len(fallback_value_rows) < 4:
                    fallback_value_rows.append({**row_payload, "score": "0.1"})
        if not value_rows:
            value_rows = fallback_value_rows
        value_rows.sort(key=lambda row: (float(row.get("score") or 0), -float(row.get("ordinal") or 0)), reverse=True)
        return value_rows

    @classmethod
    def _generic_table_row_relevance(cls, question: str, group: str, property_cell: str) -> float:
        row_key = cls._normalize_selector(f"{group} {property_cell}")
        question_key = cls._normalize_selector(question)
        score = 0.0
        selector_values = [*cls._question_row_selectors(question), *cls._extract_generic_table_terms(question)]
        for selector in selector_values:
            if cls._selector_matches_text(selector, f"{group} {property_cell}", row_key):
                score += 5.0
        property_key = cls._normalize_selector(property_cell)
        if property_key and property_key in question_key:
            score += 6.0
        if "helix" in property_key and "helix" in question_key:
            score += 6.0
        normalized_property = property_key.replace("ppl", "ppi").replace("ppii", "ppi")
        normalized_question = question_key.replace("ppl", "ppi").replace("ppii", "ppi")
        if "ppi" in normalized_property and "ppi" in normalized_question:
            score += 6.0
        return score

    @staticmethod
    def _is_comparison_or_difference_query(question: str) -> bool:
        lowered = question.lower()
        return any(
            marker in lowered or marker in question
            for marker in (
                "compare",
                "comparison",
                "difference",
                "versus",
                " vs ",
                "相比",
                "差异",
                "对比",
                "比较",
                "变化",
                "改善",
                "降低",
                "提高",
                "一致",
                "从",
                "到",
            )
        )

    @classmethod
    def _compose_display_headers(cls, header_rows: list[list[str]]) -> list[str]:
        if not header_rows:
            return []
        width = max(len(row) for row in header_rows)
        headers: list[str] = []
        for column in range(width):
            pieces: list[str] = []
            for row in header_rows:
                cell = row[column].strip() if column < len(row) else ""
                if cell and not re.fullmatch(r":?-{3,}:?", cell) and cell not in pieces:
                    pieces.append(cell)
            headers.append(" ".join(pieces).strip() or f"Column {column + 1}")
        return headers

    @classmethod
    def _selected_table_value_columns(cls, question: str, headers: list[str]) -> set[int]:
        selectors = {
            cls._normalize_selector(selector)
            for selector in [
                *cls._scientific_identifier_selectors(question),
                *cls._query_priority_anchors(question)["dataset"],
            ]
            if len(cls._normalize_selector(selector)) >= 3
        }
        selected: set[int] = set()
        for column, header in enumerate(headers):
            header_key = cls._normalize_selector(header)
            if header_key and any(selector in header_key or header_key in selector for selector in selectors):
                selected.add(column)
        return selected

    def _deterministic_ablation_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
        risk_level: str,
    ) -> QueryAnswerPayload | None:
        for index in table_indexes:
            findings = summarize_ablation_table(self._context_table_evidence_text(contexts[index]))
            if not findings:
                continue
            citation_marker = f" [{index}]"
            table_label = self._extract_table_label(self._context_table_evidence_text(contexts[index]))
            if self._is_chinese_question(question):
                subject = f"{table_label} 的消融结果" if table_label else "消融表结果"
                answer = subject + "显示：" + " ".join(self._localize_ablation_findings(findings)) + citation_marker
            else:
                subject = table_label or "The ablation table"
                answer = f"{subject} shows: " + " ".join(findings) + citation_marker
            return QueryAnswerPayload(answer_markdown=answer, citations=[index], risk_level=risk_level)
        return None

    @staticmethod
    def _localize_ablation_findings(findings: list[str]) -> list[str]:
        localized: list[str] = []
        report_re = re.compile(
            r"^(?P<iteration>.+?): full (?P<model>.+?) reports (?P<parts>.+)\.$",
            re.IGNORECASE,
        )
        underperform_re = re.compile(
            r"^(?P<iteration>.+?): ablated variants underperform the full model, including (?P<models>.+)\.$",
            re.IGNORECASE,
        )
        for finding in findings:
            if match := report_re.match(finding):
                parts = match.group("parts")
                parts = re.sub(r"\brecalls\b", "召回数", parts, flags=re.IGNORECASE)
                parts = re.sub(r"\bprecision\b", "精确率", parts, flags=re.IGNORECASE)
                parts = re.sub(r"\bdomain specificity\b", "领域特异性", parts, flags=re.IGNORECASE)
                localized.append(f"{match.group('iteration')}：完整模型 {match.group('model')} 的{parts}。")
                continue
            if match := underperform_re.match(finding):
                localized.append(f"{match.group('iteration')}：消融变体整体弱于完整模型，包括 {match.group('models')}。")
                continue
            localized.append(finding)
        return localized

    def _extract_requested_metric_values(
        self,
        question: str,
        contexts: list[RetrievedContext],
        table_indexes: list[int],
    ) -> list[ExtractedMetric]:
        row_selectors = self._question_row_selectors(question)
        results: list[ExtractedMetric] = []
        for index in table_indexes:
            text = self._context_table_evidence_text(contexts[index])
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
            for dataset in cls._requested_datasets_for_table(question, cls._context_table_evidence_text(contexts[index])):
                if dataset.upper() not in requested:
                    requested.append(dataset.upper())
        return requested

    @staticmethod
    def _context_table_evidence_text(context: RetrievedContext) -> str:
        return QueryService._context_evidence_text(context)

    @classmethod
    def _prompt_context_text(cls, question: str, context: RetrievedContext) -> str:
        evidence = cls._context_evidence_text(context)
        max_chars = 2400 if (cls._is_table_query(question) or cls._is_metric_query(question)) else 1600
        if len(evidence) <= max_chars:
            return evidence
        return cls._window_text(evidence, cls._tokenize(question), max_chars=max_chars, question=question)

    @staticmethod
    def _context_evidence_text(context: RetrievedContext) -> str:
        parts: list[str] = []
        citation = getattr(context, "citation", None)
        excerpt = getattr(citation, "excerpt", "")
        for text in (getattr(context, "prompt_text", ""), excerpt):
            clean = (text or "").strip()
            if clean and clean not in parts:
                parts.append(clean)
        return "\n\n".join(parts)

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
        method_words = {"qm", "nmr", "md", "mm", "dft", "resp", "rna", "dna", "llm", "rag", "kg", "ai", "ml"}
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*){0,2}\b", question):
            value = match.group(0).strip()
            key = cls._normalize_selector(value)
            if len(key) < 3 or key in dataset_keys or key.lower() in metric_words or key.lower() in method_words:
                continue
            if value.lower() in {"what", "table"}:
                continue
            selectors.append(value)
        selectors.extend(cls._scientific_identifier_selectors(question))
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
        priority_keys = {cls._normalize_selector(selector) for selector in cls._scientific_identifier_selectors(question)}
        if priority_keys:
            ordered.sort(key=lambda selector: (0 if cls._normalize_selector(selector) in priority_keys else 1, -len(selector)))
        return ordered[:16]

    @staticmethod
    def _normalize_selector(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @classmethod
    def _scientific_identifier_selectors(cls, question: str) -> list[str]:
        patterns = (
            r"\b[A-Za-z]+-\([A-Za-z0-9]+\)[A-Za-z0-9-]*\b",
            r"\b[A-Za-z]+[0-9]+[A-Za-z0-9]*\b",
            r"\b[A-Z0-9]+(?:/[A-Z0-9]+)+\b",
            r"\b[A-Z]{2,}[A-Z0-9-]*\b",
        )
        selectors: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, question):
                value = match.group(0).strip()
                key = cls._normalize_selector(value)
                if len(key) < 3 or key in seen or key in {"qm", "nmr", "md", "mm", "dft", "resp", "rna", "dna", "llm", "rag", "kg", "ai", "ml"} or cls._normalize_metric_name(value):
                    continue
                selectors.append(value)
                seen.add(key)
        return selectors

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
        match = re.search(r"\bTable\s*(?:S\s*)?\d+\b", text, re.IGNORECASE)
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
                if lowered_facet in self._context_evidence_text(context).lower():
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

    @classmethod
    def _retarget_table_answer_citations(
        cls,
        question: str,
        answer_markdown: str,
        selected_indexes: list[int],
    ) -> str:
        if not selected_indexes or not (cls._is_table_query(question) or cls._is_metric_query(question)):
            return answer_markdown
        normalized = cls._normalize_answer_citation_markup(answer_markdown)
        markers = [int(match) for match in re.findall(r"\[(\d+)\]", normalized)]
        if not markers or any(marker in selected_indexes for marker in markers):
            return normalized
        first_selected = selected_indexes[0]
        return re.sub(r"\[(\d+)\]", f"[{first_selected}]", normalized)

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
        evidence = "\n".join(cls._context_evidence_text(contexts[index]) for index in chosen_indexes if 0 <= index < len(contexts))
        return {number for number in numbers if number not in evidence}

    @staticmethod
    def _answer_numbers(answer_markdown: str) -> set[str]:
        numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?!\w)", answer_markdown))
        return {number for number in numbers if len(number) > 1 or "." in number or number.endswith("%")}

    @classmethod
    def _salient_evidence_acronyms(cls, contexts: list[RetrievedContext], limit: int = 12) -> list[str]:
        seen: set[str] = set()
        acronyms: list[str] = []
        stopwords = {"AND", "THE", "FOR", "WITH", "FROM", "THIS", "THAT", "TABLE", "FIGURE", "PAGE"}
        pattern = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9]*(?:[-/][A-Z0-9]{2,})*|[A-Z]+[0-9]+[A-Z0-9]*(?:[-/][A-Z0-9]+)*)(?![A-Za-z0-9])")
        for context in sorted(contexts, key=lambda item: getattr(item, "score", 0.0), reverse=True)[:3]:
            for match in pattern.finditer(cls._context_evidence_text(context)):
                value = match.group(0).strip("-/")
                if value in stopwords or value.isdigit() or len(value) < 2:
                    continue
                if value not in seen:
                    acronyms.append(value)
                    seen.add(value)
                    if len(acronyms) >= limit:
                        return acronyms
        return acronyms

    @classmethod
    def _append_missing_supported_question_terms(
        cls,
        question: str,
        answer_markdown: str,
        contexts: list[RetrievedContext],
    ) -> str:
        evidence_parts: list[str] = []
        for context in contexts[:3]:
            evidence_parts.append(cls._context_evidence_text(context))
            citation = getattr(context, "citation", None)
            if citation is not None:
                evidence_parts.extend(
                    str(value or "")
                    for value in (
                        getattr(citation, "page_slug", ""),
                        getattr(citation, "page_title", ""),
                        getattr(citation, "page_kind", ""),
                    )
                )
        evidence = "\n".join(evidence_parts)
        candidate_terms = [
            *cls._scientific_identifier_selectors(question),
            *cls._salient_evidence_acronyms(contexts[:3], limit=8),
            *cls._salient_evidence_phrases(contexts[:3], limit=8),
        ]
        missing: list[str] = []
        for term in candidate_terms:
            if term in answer_markdown or term not in evidence:
                continue
            if term not in missing:
                missing.append(term)
            if len(missing) >= 8:
                break
        if not missing:
            return answer_markdown
        if cls._is_chinese_question(question):
            note = "证据中的关键术语还包括：" + "、".join(missing) + "。"
        else:
            note = "Key evidence terms also include: " + ", ".join(missing) + "."
        separator = "\n\n" if answer_markdown.strip() else ""
        return answer_markdown.rstrip() + separator + note

    @classmethod
    def _salient_evidence_phrases(cls, contexts: list[RetrievedContext], limit: int = 8) -> list[str]:
        text = "\n".join(cls._context_evidence_text(context) for context in contexts)
        patterns = (
            r"\b[a-z]+(?:-[a-z]+)+(?:\s+[a-z]+)?\b",
            r"\bradius of gyration\b",
            r"\bexplicit hydrogen\b",
            r"\bcharge transfer\b",
            r"\bside chain\b",
            r"\bbackbone\b",
            r"\bhelical propensity\b",
            r"\bLondon dispersion\b",
            r"\bmolten globule\b",
            r"\bMonte Carlo\b",
        )
        stopwords = {"The", "Table", "Figure", "Section", "Supporting Information"}
        phrases: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                phrase = re.sub(r"\s+", " ", match.group(0)).strip(" .,;:()[]")
                if len(phrase) < 5 or phrase in stopwords:
                    continue
                key = phrase.lower()
                if key in seen:
                    continue
                phrases.append(phrase)
                seen.add(key)
                if len(phrases) >= limit:
                    return phrases
        return phrases

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
        """Keep citations per source page, allowing multi-table evidence when needed."""
        return [citation for _, citation in QueryService._dedup_page_citation_pairs(list(enumerate(citations)))]

    @staticmethod
    def _dedup_page_citation_pairs(pairs: list[tuple[int, Citation]]) -> list[tuple[int, Citation]]:
        sorted_pairs = sorted(pairs, key=lambda pair: (QueryService._citation_is_table_evidence(pair[1]), pair[1].score), reverse=True)
        has_table_evidence = any(QueryService._citation_is_table_evidence(citation) for _, citation in pairs)
        limit = 5 if has_table_evidence else 2
        kept: list[tuple[int, Citation]] = []
        seen_excerpts: set[str] = set()
        for index, citation in sorted_pairs:
            if len(kept) >= limit:
                break
            if has_table_evidence and not QueryService._citation_is_table_evidence(citation) and len(kept) >= 2:
                continue
            excerpt_normalized = citation.excerpt.strip()[:120]
            if excerpt_normalized in seen_excerpts:
                continue
            kept.append((index, citation))
            seen_excerpts.add(excerpt_normalized)
        return kept

    @staticmethod
    def _citation_is_table_evidence(citation: Citation) -> bool:
        return QueryService._context_has_table_data(citation.excerpt or "")

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
            if citation.document_id and (citation.page_slug is None or citation.chunk_id is not None):
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
    def _is_placeholder_wiki_page(markdown: str) -> bool:
        lines = [line.strip() for line in markdown.splitlines() if line.strip()]
        non_heading_lines = [line for line in lines if not line.startswith("#")]
        placeholder_lines = {"No claims yet.", "No summary available."}
        meaningful_lines = [
            line
            for line in non_heading_lines
            if line.lstrip("-*0123456789. ").strip() not in placeholder_lines
        ]
        return bool(non_heading_lines) and not meaningful_lines

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
        r"(Figure\s*(?:S\s*)?\d+|Table\s*(?:S\s*)?\d+|Fig\.\s*(?:S\s*)?\d+|Appendix\s+[A-Z])",
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
                per_page_limit = 5 if self._context_has_table_data(citation.excerpt or context.prompt_text) else 3
                if current_count >= per_page_limit:
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
        generic_terms = self._extract_generic_table_terms(question)
        specific_anchor_terms = [term for term in generic_terms if self._is_specific_table_anchor(term)]
        ranked: list[tuple[str, float]] = []
        for index, block in enumerate(blocks):
            lowered = block.lower()
            block_key = self._normalize_selector(block)
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
            matched_specific_anchors = 0
            for term in generic_terms:
                if self._selector_matches_text(term, block, block_key):
                    score += self._generic_table_term_weight(term)
                    if self._is_specific_table_anchor(term):
                        matched_specific_anchors += 1
            if specific_anchor_terms:
                if matched_specific_anchors:
                    score += matched_specific_anchors * 4.0
                else:
                    score -= 2.5
            if any(metric in lowered for metric in ("f1", "auc", "precision", "recall", "score", "指标")):
                score += 2.0
            if re.search(r"\d+(?:\.\d+)?", block):
                score += 1.5
            ranked.append((block, score - index * 0.01))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    @classmethod
    def _generic_table_term_weight(cls, term: str) -> float:
        key = cls._normalize_selector(term)
        if cls._is_table_model_term_key(key):
            return 1.0
        if key in cls._TABLE_BROAD_METRIC_TERM_KEYS:
            return 1.5
        return 4.0

    @classmethod
    def _is_specific_table_anchor(cls, term: str) -> bool:
        key = cls._normalize_selector(term)
        return bool(key and not cls._is_table_model_term_key(key) and key not in cls._TABLE_BROAD_METRIC_TERM_KEYS)

    @classmethod
    def _is_table_model_term_key(cls, key: str) -> bool:
        return bool(key and cls._TABLE_MODEL_TERM_RE.fullmatch(key))

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
    def _extract_generic_table_terms(cls, question: str) -> list[str]:
        stopwords = {
            "what",
            "which",
            "where",
            "when",
            "does",
            "drawn",
            "from",
            "for",
            "table",
            "metrics",
            "metric",
            "values",
            "value",
            "score",
            "scores",
            "performance",
            "system",
            "systems",
            "accuracy",
            "precision",
            "recall",
            "ablation",
            "study",
            "results",
            "result",
            "how",
            "are",
            "the",
            "and",
            "or",
            "to",
            "into",
            "with",
            "between",
            "compared",
            "compare",
            "comparison",
            "improve",
            "improved",
            "improvement",
            "improvements",
            "change",
            "changes",
            "show",
            "shows",
            "report",
            "reports",
            "reported",
            "given",
            "experiment",
            "experimental",
            "consistency",
            "consistent",
        }
        terms: list[str] = []
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b", question):
            value = match.group(0).strip()
            normalized = cls._normalize_selector(value)
            if len(normalized) < 3 or normalized in stopwords:
                continue
            if cls._normalize_metric_name(value):
                continue
            terms.append(value)
        for match in re.finditer(r"\b\d+(?:[-_][A-Za-z0-9]+)+\b", question):
            terms.append(match.group(0).strip())
        chinese_aliases = {
            "误差": ["error", "RMSE", "MSE"],
            "改善": ["improvement"],
            "变化": ["change"],
            "降低": ["decrease"],
            "提高": ["increase"],
            "芳香": ["aromatic"],
            "盐桥": ["salt", "acetate", "guanidine", "guanidinium", "Acetate-guanidinium"],
            "结合": ["binding"],
            "水化": ["hydration", "HFE"],
            "构象能": ["relative", "energy", "energies"],
            "实验": ["exp", "exptl"],
        }
        for marker, aliases in chinese_aliases.items():
            if marker in question:
                terms.extend(aliases)
        for match in re.finditer(r"\b[A-Za-z0-9_-]*[Dd]ataset[-_\s]*[A-Za-z0-9_-]+\b", question):
            terms.append(match.group(0).strip())
        ordered: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = cls._normalize_selector(term)
            if key and key not in seen:
                ordered.append(term)
                seen.add(key)
        return ordered[:16]

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
            re.search(r"table\s*(?:s\s*)?\d+", lowered)
            or "\u8868" in question
            or "tabular" in lowered
        )

    @staticmethod
    def _is_metric_query(question: str) -> bool:
        """Detect questions asking about metrics, scores, or benchmark results."""
        lowered = question.lower()
        if any(marker in question for marker in ("\u6307\u6807", "\u5206\u6570", "\u5f97\u5206")):
            return True
        return bool(
            re.search(r"(?<![a-z0-9])f\s*1(?![a-z0-9])", lowered)
            or re.search(
                r"\b(auc|precision|recall|accuracy|bleu|rouge|rmse|mae|mse|hfe|pka|metric|score|performance|oie2016|nyt|penn|web)\b",
                lowered,
            )
        )

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
        if "ablation" in question.lower() or "消融" in question:
            max_chars = max(max_chars, 2400)

        start = 0
        table_caption_index: int | None = None
        first_table_index: int | None = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^(?:#+\s*)?Table\s*(?:S\s*)?\d+\b", stripped, re.IGNORECASE):
                table_caption_index = index
                break
            if first_table_index is None and (stripped.startswith("|") or stripped.lower().startswith("<table")):
                first_table_index = index
        if table_caption_index is not None:
            start = table_caption_index
        elif first_table_index is not None:
            has_page_heading = any(
                re.match(r"^#+\s*Page\s+\d+\b", line.strip(), re.IGNORECASE)
                for line in lines[:first_table_index]
            )
            start = 0 if has_page_heading else first_table_index
        lines = lines[start:]

        anchors = {
            str(anchor).strip()
            for anchor in [
                *cls._query_priority_anchors(question)["dataset"],
                *cls._extract_query_facets(question),
                *cls._question_row_selectors(question),
                *cls._extract_generic_table_terms(question),
            ]
            if cls._normalize_selector(anchor)
        }

        caption_lines = cls._table_caption_lines(lines)
        table_lines = [line for line in lines if cls._is_table_line(line)]
        header_line_count = cls._table_header_line_count(table_lines)
        header_lines = table_lines[:header_line_count]
        if "ablation" in question.lower() or "消融" in question:
            relevant_rows = table_lines[header_line_count:]
        else:
            relevant_rows = cls._table_relevant_rows_with_group_children(table_lines[header_line_count:], anchors)
        if not relevant_rows and table_lines:
            relevant_rows = table_lines[header_line_count : header_line_count + 4]

        excerpt_lines: list[str] = []
        for line in [*caption_lines, *header_lines, *relevant_rows]:
            if line not in excerpt_lines:
                excerpt_lines.append(line)
        excerpt = "\n".join(excerpt_lines).strip() or "\n".join(lines).strip()
        return excerpt[:max_chars]

    @classmethod
    def _table_caption_lines(cls, lines: list[str]) -> list[str]:
        caption_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if cls._is_table_line(stripped):
                break
            if re.match(r"^#+\s*Page\s+\d+\b", stripped, re.IGNORECASE):
                continue
            if stripped:
                caption_lines.append(line)
        caption_lines = caption_lines[:2]
        if not caption_lines:
            return []
        if any(re.search(r"\btable\b", line, re.IGNORECASE) for line in caption_lines):
            return caption_lines
        return [f"Table evidence: {caption_lines[0]}", *caption_lines[1:]]

    @classmethod
    def _table_relevant_rows_with_group_children(cls, table_lines: list[str], anchors: set[str]) -> list[str]:
        if not anchors:
            return []
        selected_indexes: set[int] = set()
        for index, line in enumerate(table_lines):
            line_key = cls._normalize_selector(line)
            if not any(cls._selector_matches_text(anchor, line, line_key) for anchor in anchors):
                continue
            if not cls._table_line_has_data_number(line) and not cls._table_line_needs_group_children(line):
                continue
            selected_indexes.add(index)
            if cls._table_line_needs_group_children(line) or (
                index + 1 < len(table_lines) and cls._table_line_is_group_child(table_lines[index + 1])
            ):
                for child_index in range(index + 1, len(table_lines)):
                    if not cls._table_line_is_group_child(table_lines[child_index]):
                        break
                    selected_indexes.add(child_index)
        return [line for index, line in enumerate(table_lines) if index in selected_indexes]

    @classmethod
    def _selector_matches_text(cls, selector: str, text: str, normalized_text: str | None = None) -> bool:
        selector_text = str(selector or "").strip()
        selector_key = cls._normalize_selector(selector_text)
        if not selector_key:
            return False
        if re.fullmatch(r"[a-z][a-z0-9]*", selector_text):
            return bool(re.search(rf"(?<![A-Za-z0-9-]){re.escape(selector_text)}(?![A-Za-z0-9-])", text, re.IGNORECASE))
        return selector_key in (normalized_text if normalized_text is not None else cls._normalize_selector(text))

    @classmethod
    def _table_header_line_count(cls, table_lines: list[str]) -> int:
        if not table_lines:
            return 0
        for index, line in enumerate(table_lines):
            cells = cls._markdown_table_line_cells(line)
            if not cls._is_markdown_separator_row(cells):
                continue
            header_count = index + 1
            if index + 1 < len(table_lines) and cls._table_line_looks_like_secondary_header(table_lines[index + 1]):
                header_count += 1
            return header_count
        return min(1, len(table_lines))

    @classmethod
    def _table_line_looks_like_secondary_header(cls, line: str) -> bool:
        cells = cls._markdown_table_line_cells(line)
        if not cells:
            return False
        non_empty = [cell for cell in cells if cell.strip()]
        if not non_empty:
            return False
        has_number = any(re.search(r"\d+(?:\.\d+)?", cell) for cell in non_empty)
        headerish = sum(
            1
            for cell in non_empty
            if cls._normalize_selector(cell) in {"calcd", "calc", "exptl", "exp", "experiment", "experimental"}
        )
        if headerish >= 2:
            return True
        if cells[0].strip() and headerish < 2:
            return False
        return not has_number

    @classmethod
    def _table_line_needs_group_children(cls, line: str) -> bool:
        cells = cls._markdown_table_line_cells(line)
        if not cells or cls._is_markdown_separator_row(cells):
            return False
        non_empty = [cell for cell in cells if cell.strip()]
        if len(non_empty) <= 1:
            return True
        numeric_cells = [cell for cell in non_empty if re.search(r"\d+(?:\.\d+)?", cell)]
        return not numeric_cells and len(non_empty) <= 2

    @classmethod
    def _table_line_has_data_number(cls, line: str) -> bool:
        cells = cls._markdown_table_line_cells(line)
        if not cells or cls._is_markdown_separator_row(cells):
            return False
        for cell in cells:
            key = cls._normalize_selector(cell)
            if cls._is_table_model_term_key(key):
                continue
            if re.search(r"\d+(?:\.\d+)?", cell):
                return True
        return False

    @classmethod
    def _table_line_is_group_child(cls, line: str) -> bool:
        cells = cls._markdown_table_line_cells(line)
        return bool(cells) and not cls._is_markdown_separator_row(cells) and not cells[0].strip()

    @staticmethod
    def _markdown_table_line_cells(line: str) -> list[str]:
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            return []
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") or stripped.lower().startswith("<table")

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
