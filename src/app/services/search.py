from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import DocumentChunk, PageKind, Project, QuestionAnswer, WikiPage
from app.schemas.common import Citation, QueryResponse
from app.services.ai import QueryAnswerPayload, VerificationPayload, cosine_similarity, safe_model_call
from app.services.ai import ExternalVerifier, OllamaClient
from app.services.filesystem import slugify, strip_upload_prefix
from app.services.wiki import WikiRenderer

settings = get_settings()
WIKI_PRIMARY_SCORE_THRESHOLD = 6.0
MIN_WIKI_OVERLAP_SCORE = 1
MIN_CONTEXT_SCORE = 2.0
CONTEXT_SCORE_RATIO = 0.35
MAX_CONTEXTS = 8


@dataclass
class RetrievedContext:
    citation: Citation
    prompt_text: str
    score: float


@dataclass
class PageMatch:
    page: WikiPage
    score: float


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

        chosen_indexes = answer_payload.citations or self._infer_citation_indexes(answer_payload.answer_markdown, len(contexts))
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        answer_payload = self._repair_unsupported_numeric_answer(question, index_context, contexts, answer_payload, chosen_indexes)
        chosen_indexes = answer_payload.citations or self._infer_citation_indexes(answer_payload.answer_markdown, len(contexts))
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        citations = self._select_citations(contexts, chosen_indexes)
        if not self._needs_source_evidence(question):
            citations = self._prefer_wiki_citations(citations, page_matches, project.id)
        response = QueryResponse(answer_markdown=answer_payload.answer_markdown, citations=citations, verification_status=verification_status)

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
        matched_doc_ids: list[str] = []
        query_terms = self._tokenize(question)
        query_facets = self._extract_query_facets(question)
        top_score = page_matches[0].score if page_matches else 0.0
        min_score = max(MIN_CONTEXT_SCORE, top_score * CONTEXT_SCORE_RATIO) if top_score else MIN_CONTEXT_SCORE
        is_figure_query = self._is_figure_query(question)
        is_table_query = self._is_table_query(question)
        is_metric_query = self._is_metric_query(question)

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

            for facet in query_facets:
                facet_context = self._context_for_facet(page, page_body, facet, query_terms)
                if facet_context is not None:
                    contexts.append(facet_context)

            # Always include the main page context (windowed around query terms).
            prompt_text = self._window_text(page_body, query_terms, max_chars=4000, question=question)
            contexts.append(
                RetrievedContext(
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
            )
            matched_doc_ids.extend(page.source_document_ids)

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
            citations=list(range(len(supported_contexts))),
            risk_level=answer_payload.risk_level,
        )
        prompt = "\n\n".join(
            [
                f"Question: {question}",
                "The previous draft included unsupported numeric values: " + ", ".join(sorted(unsupported)),
                "Rewrite the answer using ONLY the evidence below. Do not include any number unless it appears verbatim in the evidence. If a requested metric is absent, say it is absent from the retrieved materials.",
                self._build_answer_constraints(question, supported_contexts),
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

    @classmethod
    def _unsupported_answer_numbers(cls, answer_markdown: str, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> set[str]:
        numbers = cls._answer_numbers(answer_markdown)
        if not numbers:
            return set()
        evidence = "\n".join(contexts[index].prompt_text for index in chosen_indexes if 0 <= index < len(contexts))
        return {number for number in numbers if number not in evidence}

    @staticmethod
    def _answer_numbers(answer_markdown: str) -> set[str]:
        numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w.])", answer_markdown))
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
        selected: list[Citation] = []
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
            selected.append(citation)

        # Cap same source page to at most 2 citations, keeping highest scores
        # and preferring different excerpts. Process once at the end.
        by_page: dict[str, list[Citation]] = {}
        for citation in selected:
            if citation.page_slug:
                by_page.setdefault(citation.page_slug, []).append(citation)

        capped: list[Citation] = []
        processed_pages: set[str] = set()
        for citation in selected:
            if citation.page_slug and citation.page_slug in processed_pages:
                continue  # already added the capped set for this page
            if citation.page_slug and len(by_page.get(citation.page_slug, [])) > 2:
                kept = self._dedup_page_citations(by_page[citation.page_slug])
                capped.extend(kept)
                processed_pages.add(citation.page_slug)
            else:
                capped.append(citation)
        return capped

    @staticmethod
    def _dedup_page_citations(citations: list[Citation]) -> list[Citation]:
        """Keep at most 2 citations per page, preferring higher scores and distinct excerpts."""
        sorted_citations = sorted(citations, key=lambda c: c.score, reverse=True)
        kept: list[Citation] = []
        seen_excerpts: set[str] = set()
        for citation in sorted_citations:
            if len(kept) >= 2:
                break
            excerpt_normalized = citation.excerpt.strip()[:120]
            if excerpt_normalized in seen_excerpts:
                continue
            kept.append(citation)
            seen_excerpts.add(excerpt_normalized)
        return kept

    def _prefer_wiki_citations(self, citations: list[Citation], page_matches: list[PageMatch], project_id: str) -> list[Citation]:
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
                    excerpt = self._window_text(page_body, self._tokenize(citation.excerpt), max_chars=280)
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
        indexes: list[int] = []
        for match in re.findall(r"\[(\d+)\]", answer_markdown):
            index = int(match)
            if 0 <= index < context_count and index not in indexes:
                indexes.append(index)
        return indexes

    def _load_index_context(self, project_slug: str) -> str | None:
        index_path = settings.wiki_dir / project_slug / "index.md"
        if not index_path.exists():
            return None
        return index_path.read_text(encoding="utf-8")[:4000]

    def _save_query_page(self, project: Project, question: str, response: QueryResponse, citations: list[Citation]) -> None:
        renderer = WikiRenderer(project)
        slug = f"queries/{datetime.utcnow():%Y%m%d-%H%M%S}-{slugify(question)[:48]}"
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
                blocks.append(match.group(0).strip())
        return blocks

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
        markers = ("原文", "出处", "证据", "摘录", "引用", "quote", "quoted", "exact", "verbatim", "source")
        return any(marker in question or marker in lowered for marker in markers)

    def _should_use_wiki_only(self, question: str, page_matches: list[PageMatch]) -> bool:
        if not page_matches:
            return False
        if self._needs_source_evidence(question):
            return False
        return page_matches[0].score >= WIKI_PRIMARY_SCORE_THRESHOLD
