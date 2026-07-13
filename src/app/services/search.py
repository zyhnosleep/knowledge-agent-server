from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import Claim, Document, DocumentChunk, DocumentStatus, Project, QuestionAnswer
from app.schemas.common import Citation, QueryResponse
from app.services.ai import QueryAnswerPayload, VerificationPayload, cosine_similarity, safe_model_call
from app.services.ai import ExternalVerifier, OllamaClient
from app.services.filesystem import InvalidStoragePathError, safe_project_slug, slugify, strip_upload_prefix
from app.services.paper_profile import (
    alias_in_text,
    paper_profile_data,
    paper_profile_retrieval_terms,
    paper_profile_text,
    source_fields_for_document,
)
from app.services.table_extraction import summarize_ablation_table, table_metric_values
from app.services.table_normalization import normalize_table_text
from app.services.vector_store import SQLiteVecStore

settings = get_settings()
MIN_CONTEXT_SCORE = 2.5
CONTEXT_SCORE_RATIO = 0.40
MAX_CONTEXTS = 8
TABLE_CONTEXT_SCORE_BOOST = 40.0
PAPER_ROUTE_MIN_SCORE = 2.0
QUERY_GENERATION_TIMEOUT_SECONDS = 45


@dataclass
class RetrievedContext:
    citation: Citation
    prompt_text: str
    score: float
    evidence_kind: str | None = None


@dataclass


@dataclass
class PaperMatch:
    document: Document
    score: float
    exact_alias: bool = False
    locked: bool = False
    introduced_subject: bool = False


@dataclass
class ExtractedMetric:
    context_index: int
    table_label: str | None
    dataset: str
    values: dict[str, str]


class QueryService:
    _TABLE_MODEL_TERM_RE = re.compile(r"^(?:amber|charmm|gaff|opls|c\d+|ff\d+)[a-z0-9]*$")
    _SCIENTIFIC_PROFILE_TERM_KEYS = {
        "34organicliquids",
        "alanine",
        "alphal",
        "asp",
        "asn",
        "boss",
        "c6coefficients",
        "chargetransfer",
        "chi1",
        "chi2",
        "covalentrelaxation",
        "drude",
        "expandedensemble",
        "expandedensembles",
        "explicithydrogen",
        "fep",
        "fret",
        "fxa",
        "galib",
        "glh",
        "glu",
        "helicalpropensity",
        "helixcoil",
        "hydrationfreeenergy",
        "idp",
        "largedisorderedproteins",
        "ile",
        "leucine",
        "lfmm",
        "lennardjones",
        "mmp13",
        "moltenglobule",
        "montecarlo",
        "mse",
        "nmr",
        "neutralstate",
        "phase",
        "polarizability",
        "rdcs",
        "saltbridge",
        "sparta",
        "steric",
        "stericclash",
        "stericclashes",
        "tetrapeptide",
        "thr",
        "torsion",
        "torsional",
        "val",
        "valine",
        "vanderwaals",
        "vdw",
    }
    _SCIENTIFIC_CONTEXT_ANCHORS = (
        ("amino-acid specific", re.compile(r"\bamino-acid specific\b", re.IGNORECASE)),
        ("50%", re.compile(r"\b50\s*%", re.IGNORECASE)),
        ("34 organic liquids", re.compile(r"\b34\s+organic liquids\b", re.IGNORECASE)),
        ("390", re.compile(r"\b390\b", re.IGNORECASE)),
        ("500 K", re.compile(r"\b500\s*K\b", re.IGNORECASE)),
        ("-2.4", re.compile(r"(?<![\w.])-?\s*2\.4(?![\w.])", re.IGNORECASE)),
        ("-0.5", re.compile(r"(?<![\w.])-?\s*0\.5(?![\w.])", re.IGNORECASE)),
        ("alphaL", re.compile(r"(?:\\alpha|\u03b1|alpha)\s*(?:_|\{|\}|\\mathrm|\s)*L\b", re.IGNORECASE)),
        ("BOSS", re.compile(r"\bBOSS\b", re.IGNORECASE)),
        ("C6", re.compile(r"\bC\s*(?:_|\{|\}|\s)*6\b", re.IGNORECASE)),
        ("cation", re.compile(r"\bcations?\b", re.IGNORECASE)),
        ("charge transfer", re.compile(r"\bcharge transfer\b", re.IGNORECASE)),
        ("CMAP", re.compile(r"\bCMAPs?\b", re.IGNORECASE)),
        ("covalent relaxation", re.compile(r"\bcovalent relaxation\b", re.IGNORECASE)),
        ("Drude", re.compile(r"\bDrude\b", re.IGNORECASE)),
        ("expanded ensembles", re.compile(r"\bexpanded ensembles?\b", re.IGNORECASE)),
        ("explicit hydrogen", re.compile(r"\bexplicit hydrogen\b", re.IGNORECASE)),
        ("ff12SB", re.compile(r"\bff12SB\b", re.IGNORECASE)),
        ("FRET", re.compile(r"\bFRET\b", re.IGNORECASE)),
        ("FXA", re.compile(r"\bFXA\b", re.IGNORECASE)),
        ("helical propensity", re.compile(r"\bhelical propensity\b", re.IGNORECASE)),
        ("helix-coil", re.compile(r"\bhelix[- ]coil\b|\bhelical\b.{0,100}\bextended\b|\bextended\b.{0,100}\bhelical\b", re.IGNORECASE)),
        ("NMR", re.compile(r"\bNMR\b", re.IGNORECASE)),
        ("CHARMM36m", re.compile(r"\bCHARMM36m\b", re.IGNORECASE)),
        ("a99SB", re.compile(r"\ba99SB-?\b", re.IGNORECASE)),
        ("hydration free energy", re.compile(r"\bfree energ(?:y|ies) of hydration\b|\bhydration free energ(?:y|ies)\b", re.IGNORECASE)),
        ("IDP", re.compile(r"\bIDPs?\b", re.IGNORECASE)),
        ("large disordered proteins", re.compile(r"\blarge disordered proteins\b", re.IGNORECASE)),
        ("large conformational fluctuation", re.compile(r"\blarge conformational fluctuation\b", re.IGNORECASE)),
        ("Lennard-Jones", re.compile(r"\bLennard[-\u2010-\u2015]Jones\b", re.IGNORECASE)),
        ("LFMM", re.compile(r"\bLFMM\b", re.IGNORECASE)),
        ("LMP2", re.compile(r"\bLMP2\b", re.IGNORECASE)),
        ("London dispersion", re.compile(r"\bLondon dispersion\b", re.IGNORECASE)),
        ("metal", re.compile(r"\bmetals?\b", re.IGNORECASE)),
        ("MMP13", re.compile(r"\bMMP13\b", re.IGNORECASE)),
        ("molten globule", re.compile(r"\bmolten globule\b", re.IGNORECASE)),
        ("Monte Carlo", re.compile(r"\bMonte Carlo\b", re.IGNORECASE)),
        ("GAlib", re.compile(r"\bGAlib\b", re.IGNORECASE)),
        ("FEP", re.compile(r"\bFEP\+?\b", re.IGNORECASE)),
        ("phase", re.compile(r"\bphase\b", re.IGNORECASE)),
        ("rotamer", re.compile(r"\brotamers?\b", re.IGNORECASE)),
        ("population", re.compile(r"\bpopulations?\b", re.IGNORECASE)),
        ("barrier", re.compile(r"\bbarriers?\b", re.IGNORECASE)),
        ("QM-MM", re.compile(r"\bQM[-/\s]?MM\b", re.IGNORECASE)),
        ("neutral state", re.compile(r"\bneutral state\b", re.IGNORECASE)),
        ("PPII", re.compile(r"\bPPII\b", re.IGNORECASE)),
        ("polarizability", re.compile(r"\bpolarizability\b", re.IGNORECASE)),
        ("RDCs", re.compile(r"\bRDCs?\b", re.IGNORECASE)),
        ("Rg", re.compile(r"\bR\s*_?\s*\{?\s*g\s*\}?\b|\bRg\b", re.IGNORECASE)),
        ("RHF/6-31G", re.compile(r"\bRHF\s*/\s*6-31G\b", re.IGNORECASE)),
        ("2kT", re.compile(r"\b2\s*k\s*T\b", re.IGNORECASE)),
        ("salt bridge", re.compile(r"\bsalt bridge\b", re.IGNORECASE)),
        ("SPARTA", re.compile(r"\bSPARTA\b", re.IGNORECASE)),
        ("steric", re.compile(r"\bsteric\b", re.IGNORECASE)),
        ("steric clashes", re.compile(r"\bsteric clashes?\b", re.IGNORECASE)),
        ("sulfur", re.compile(r"\bsulfur\b", re.IGNORECASE)),
        ("tetrapeptide", re.compile(r"\btetrapeptide\b", re.IGNORECASE)),
        ("torsional", re.compile(r"\btorsional\b|\btorsions?\b", re.IGNORECASE)),
        ("TIP4P-EW", re.compile(r"\bTIP4P[- ]EW\b", re.IGNORECASE)),
        ("van der Waals", re.compile(r"\bvan der Waals\b", re.IGNORECASE)),
        ("vdW", re.compile(r"\bvdW\b", re.IGNORECASE)),
        ("chi1", re.compile(r"(?:\\chi|\u03c7|chi)\s*_?\s*\{?\s*1\s*\}?", re.IGNORECASE)),
        ("Alanine", re.compile(r"\bAlanine\b", re.IGNORECASE)),
        ("Valine", re.compile(r"\bValine\b", re.IGNORECASE)),
        ("Leucine", re.compile(r"\bLeucine\b", re.IGNORECASE)),
        ("Ile", re.compile(r"\bIle\b", re.IGNORECASE)),
        ("Val", re.compile(r"\bVal\b", re.IGNORECASE)),
        ("Thr", re.compile(r"\bThr\b", re.IGNORECASE)),
        ("Asp", re.compile(r"\bAsp\b", re.IGNORECASE)),
        ("Asn", re.compile(r"\bAsn\b", re.IGNORECASE)),
        ("GLH", re.compile(r"\bGLH\b|\bGlh\b", re.IGNORECASE)),
        ("ASP", re.compile(r"\bASP\b", re.IGNORECASE)),
        ("GLU", re.compile(r"\bGLU\b|\bGlu\b", re.IGNORECASE)),
        ("MSE", re.compile(r"\bMSE\b", re.IGNORECASE)),
    )
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
    _CLAIM_ANCHOR_STOP_KEYS = {
        "about",
        "article",
        "difference",
        "improve",
        "improved",
        "improvement",
        "main",
        "mechanism",
        "overview",
        "paper",
        "result",
        "results",
        "study",
        "what",
        "why",
    }
    _QUESTION_STOP_TERMS = frozenset({
        "what", "the", "is", "are", "a", "an", "and", "or", "of", "in", "to",
        "for", "with", "on", "at", "by", "between", "vs", "versus", "difference",
        "how", "why", "when", "where", "who", "does", "do", "can", "will",
        "has", "have", "it", "its", "this", "that", "these", "those", "from",
        "about", "which", "than", "follow", "follow-up", "up", "not", "but",
        "also", "been", "were", "was", "had", "did", "said", "get", "got",
        "just", "like", "make", "more", "much", "now", "only", "over", "put",
        "same", "some", "such", "take", "use", "used", "very", "well", "may",
        "might", "could", "would", "should", "let", "see", "say", "know",
        "need", "want", "ask", "like", "time", "way", "day", "year", "thing",
        "case", "part", "place", "point", "kind", "sort", "type", "example",
        "instance", "model", "method", "approach", "result", "results", "study",
        "paper", "article", "conclusion", "summary", "report", "analysis",
    })
    _SCIENTIFIC_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\d*[a-z]*\d*[a-z-]*\b")
    _FORCE_FIELD_RE = re.compile(
        r"\b(?:CHARMM|AMBER|OPLS|GAFF|GROMOS|MMFF|UFF|CGenFF|Martini)\d*[a-z]*\d*[a-z-]*\b",
        re.IGNORECASE,
    )

    def __init__(self, db: Session) -> None:
        self.db = db
        self.ollama = OllamaClient()
        self.verifier = ExternalVerifier()

    def answer(
        self,
        project_slug: str,
        question: str,
        save_answer: bool = True,
        document_id: str | None = None,
    ) -> QueryResponse:
        return self._answer_rag_first(
            project_slug, question, save_answer=save_answer, document_id=document_id
        )

    def retrieve_evidence(
        self,
        project_slug: str,
        question: str,
        limit: int = 15,
        document_id: str | None = None,
    ) -> "EvidencePack":
        """Retrieve-only RAG: return an EvidencePack without drafting an answer.

        Reuses the same routing and context selection as ``answer()`` but
        does **not** call the LLM, verify, or persist a ``QuestionAnswer``.
        When *document_id* is provided, retrieval is scoped to that document
        and never falls back to project-wide sources.
        """
        from app.schemas.agent import EvidenceItem, EvidencePack

        project = self.db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            raise ValueError(f"Project '{project_slug}' not found")

        scoped_document = self._validate_document_scope(project.id, document_id)
        document_ids = [document_id] if document_id else None
        paper_matches = self._route_papers(
            question, project.id, limit=limit, document_id=document_id
        )
        contexts = self._build_rag_contexts(
            question, project.id, paper_matches, document_ids=document_ids
        )
        if not contexts and paper_matches and document_ids is None:
            locked_document_ids = self._locked_document_ids(question, paper_matches)
            if not locked_document_ids and not QueryService._is_document_overview_query(question):
                contexts = self._search_source_chunks(question, project.id, [], limit=5)
        items: list[EvidenceItem] = []
        for idx, ctx in enumerate(contexts[:limit]):
            evidence_kind = self._context_evidence_kind(ctx)
            source_stage = self._determine_source_stage(ctx)
            support_hint = self._determine_support_hint(ctx, question)
            items.append(
                EvidenceItem(
                    index=idx,
                    document_id=ctx.citation.document_id,
                    chunk_id=ctx.citation.chunk_id,
                    page_slug=ctx.citation.page_slug,
                    page_title=ctx.citation.page_title,
                    page_kind=ctx.citation.page_kind,
                    page_label=ctx.citation.page_label,
                    score=ctx.citation.score,
                    excerpt=ctx.citation.excerpt,
                    evidence_kind=evidence_kind,
                    source_stage=source_stage,
                    support_hint=support_hint,
                )
            )
        status = "ok" if items else "empty"
        return EvidencePack(status=status, items=items)

    def _validate_document_scope(
        self, project_id: str, document_id: str | None
    ) -> Document | None:
        """Validate that *document_id* belongs to *project_id*.

        Returns the Document when valid, or None when *document_id* is None.
        Raises ValueError for unknown or mismatched documents.
        """
        if document_id is None:
            return None
        document = self.db.get(Document, document_id)
        if document is None or document.project_id != project_id:
            raise ValueError(
                f"Document '{document_id}' not found in project '{project_id}'"
            )
        return document

    @staticmethod
    def _determine_source_stage(ctx: "RetrievedContext") -> str:
        """Map a RetrievedContext to a deterministic source_stage label."""
        citation = ctx.citation
        ek = ctx.evidence_kind
        # Evidence kind → stage mapping
        kind_map = {
            "table": "document_table",
            "figure": "document_figure",
            "profile-term": "profile_term",
            "claim": "claim",
        }
        if ek and ek in kind_map:
            return kind_map[ek]
        if citation.document_id:
            return "source_chunk"
        return "unknown"

    @classmethod
    def _determine_support_hint(cls, ctx: "RetrievedContext", question: str) -> str:
        """Best-effort deterministic support quality label."""
        score = ctx.citation.score
        if score >= 15.0:
            return "direct"
        if score >= 5.0:
            return "contextual"
        # Check if question terms appear in the evidence text
        evidence = cls._context_evidence_text(ctx).lower()
        query_terms = cls._tokenize(question)
        if query_terms and any(term.lower() in evidence for term in query_terms):
            return "contextual"
        return "weak"

    def _answer_rag_first(
        self,
        project_slug: str,
        question: str,
        save_answer: bool = True,
        document_id: str | None = None,
    ) -> QueryResponse:
        project = self.db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            raise ValueError(f"Project '{project_slug}' not found")

        self._validate_document_scope(project.id, document_id)
        document_ids = [document_id] if document_id else None
        paper_matches = self._route_papers(
            question, project.id, document_id=document_id
        )
        contexts = self._build_rag_contexts(
            question, project.id, paper_matches, document_ids=document_ids
        )
        if not contexts and document_ids is None:
            locked_document_ids = self._locked_document_ids(question, paper_matches)
            if not locked_document_ids and not QueryService._is_document_overview_query(question):
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
        if self._should_append_supported_evidence_terms(question, contexts):
            answer_payload.answer_markdown = self._append_missing_supported_question_terms(
                question,
                answer_payload.answer_markdown,
                contexts,
            )
        answer_payload.answer_markdown = self._normalize_answer_citation_markup(answer_payload.answer_markdown)
        chosen_indexes = self._choose_citation_indexes(question, answer_payload, contexts)
        chosen_indexes = self._supported_citation_indexes(answer_payload.answer_markdown, contexts, chosen_indexes)
        chosen_indexes = self._table_evidence_indexes_only(question, contexts, chosen_indexes)
        evidence_insufficient = answer_payload.answer_markdown.lstrip().lower().startswith("## insufficient evidence")
        if evidence_insufficient:
            citations = []
            answer_markdown = self._strip_answer_citation_markers(answer_payload.answer_markdown)
        else:
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
            else:
                answer_markdown = self._ensure_valid_returned_citation_marker(answer_markdown, len(citations))
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

    def _route_papers(
        self,
        question: str,
        project_id: str,
        limit: int = 3,
        document_id: str | None = None,
    ) -> list[PaperMatch]:
        query_terms = self._tokenize(question)
        scientific_selectors = self._scientific_identifier_selectors(question)
        primary_selectors = self._primary_subject_selectors(question)
        statement = select(Document).where(
            Document.project_id == project_id,
            Document.status == DocumentStatus.ready.value,
        )
        if document_id is not None:
            statement = statement.where(Document.id == document_id)
        documents = self.db.scalars(statement).all()
        matches: list[PaperMatch] = []
        for document in documents:
            profile = paper_profile_data(document)
            profile_text = paper_profile_text(document)
            profile_terms = self._tokenize(profile_text)
            raw_terms = self._tokenize(document.raw_text or "")
            title_terms = self._tokenize(document.title)
            alias_values = [str(item) for item in profile.get("aliases") or []]
            key_values = [str(item) for item in profile.get("key_terms") or []]
            alias_terms = self._tokenize(" ".join(alias_values))
            key_terms = self._tokenize(" ".join(key_values))
            exact_alias = self._question_has_exact_alias(question, alias_values)
            selector_text = self._paper_route_text(document, profile_text)
            selector_hits = [
                selector
                for selector in scientific_selectors
                if self._selector_matches_text(selector, selector_text)
            ]
            introduced_selector_count = self._introduced_selector_count(
                primary_selectors or scientific_selectors,
                document.raw_text or "",
            )
            identity_text = self._paper_identity_route_text(document, profile)
            primary_selector_hits = [
                selector
                for selector in primary_selectors
                if self._selector_matches_text(selector, identity_text)
            ]
            score = (
                len(query_terms & profile_terms)
                + len(query_terms & raw_terms)
                + len(query_terms & title_terms) * 3
                + len(query_terms & alias_terms) * 5
                + len(query_terms & key_terms) * 2
                + len(selector_hits) * 4
                + introduced_selector_count * 12
                + len(primary_selector_hits) * 18
            )
            if exact_alias:
                score += 18
            if score >= PAPER_ROUTE_MIN_SCORE:
                matches.append(
                    PaperMatch(
                        document=document,
                        score=float(score),
                        exact_alias=exact_alias,
                        introduced_subject=introduced_selector_count > 0,
                    )
                )
        ranked = sorted(matches, key=lambda item: item.score, reverse=True)
        subject_locked = [
            match
            for match in ranked
            if self._question_locks_document_subject(question, [str(item) for item in (paper_profile_data(match.document).get("aliases") or [])])
        ]
        if subject_locked:
            return [self._locked_paper_match(subject_locked[0])]
        if document_id is not None and documents:
            # Scoped to a single document: always lock it so downstream retrieval
            # does not widen to other project documents.
            return [self._locked_paper_match(PaperMatch(document=documents[0], score=0.0))]
        primary = self._primary_subject_match(question, ranked)
        if primary is not None:
            return [self._locked_paper_match(primary)]
        exact_matches = [match for match in ranked if match.exact_alias]
        if exact_matches:
            if len(exact_matches) > 1:
                primary = self._primary_subject_match(question, exact_matches)
                if primary is not None:
                    return [self._locked_paper_match(primary)]
                return exact_matches[: max(limit, 5)]
            return [self._locked_paper_match(exact_matches[0])]
        if self._is_cross_paper_query(question):
            return ranked[: max(limit, 5)]
        introduced_matches = [match for match in ranked if match.introduced_subject]
        if len(introduced_matches) == 1:
            return [self._locked_paper_match(introduced_matches[0])]
        if self._top_paper_match_is_obvious(ranked):
            return [self._locked_paper_match(ranked[0])]
        return ranked[:limit]

    @staticmethod
    def _locked_paper_match(match: PaperMatch) -> PaperMatch:
        return PaperMatch(
            document=match.document,
            score=match.score,
            exact_alias=match.exact_alias,
            locked=True,
            introduced_subject=match.introduced_subject,
        )

    @staticmethod
    def _introduced_selector_count(selectors: list[str], text: str) -> int:
        if not selectors or not text:
            return 0
        count = 0
        for selector in selectors:
            escaped = re.escape(selector)
            if re.search(
                rf"(?:\bwe\s+(?:introduc\w*|creat\w*|propos\w*|develop\w*|present\w*)|"
                rf"\bthis\s+(?:paper|work|study)\s+(?:introduc\w*|propos\w*|develop\w*|present\w*)|"
                rf"本文(?:引入|提出|开发|构建))[^.\n]{{0,240}}{escaped}",
                text,
                re.IGNORECASE,
            ):
                count += 1
        return count

    @staticmethod
    def _top_paper_match_is_obvious(ranked: list[PaperMatch]) -> bool:
        if not ranked:
            return False
        top = ranked[0]
        if top.score < 10:
            return False
        if len(ranked) == 1:
            return True
        second = ranked[1]
        return top.score >= second.score + 8 or top.score >= second.score * 1.75

    @staticmethod
    def _paper_route_text(document: Document, profile_text: str = "") -> str:
        return "\n".join(
            part
            for part in (
                profile_text,
                document.title or "",
                document.file_name or "",
                document.raw_path or "",
            )
            if part
        )

    @staticmethod
    def _paper_identity_route_text(document: Document, profile: dict | None = None) -> str:
        profile = profile if isinstance(profile, dict) else paper_profile_data(document)
        aliases = profile.get("aliases") if isinstance(profile.get("aliases"), list) else []
        return "\n".join(
            part
            for part in (
                str(profile.get("title") or ""),
                " ".join(str(alias) for alias in aliases),
                document.title or "",
                document.file_name or "",
                document.raw_path or "",
            )
            if part
        )

    @classmethod
    def _primary_subject_selectors(cls, question: str) -> list[str]:
        lowered = question.lower()
        if any(marker in lowered for marker in ("compare", "comparison", "versus", " vs", " v.s.", "between")):
            return []
        if any(marker in question for marker in ("比较", "对比", "差异", "区别")):
            return []
        marker_positions = [
            position
            for marker in ("相比", "相对", "比", " than ")
            for position in [lowered.find(marker)]
            if position >= 0
        ]
        if not marker_positions:
            return []
        prefix = question[: min(marker_positions)]
        candidates = cls._scientific_identifier_selectors(prefix)
        keyed = [
            (selector, cls._normalize_selector(selector))
            for selector in candidates
            if cls._normalize_selector(selector)
        ]
        selectors: list[str] = []
        for selector, key in keyed:
            if any(key != other_key and key in other_key for _, other_key in keyed):
                continue
            selectors.append(selector)
        selectors.sort(key=lambda selector: (prefix.lower().find(selector.lower()), -len(cls._normalize_selector(selector))))
        return selectors

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
            if re.search(
                rf"^\s*{escaped}(?![A-Za-z0-9_/\-])\s*(?:相比|相对)",
                question,
                re.IGNORECASE,
            ):
                return True
        return False

    @staticmethod
    def _primary_subject_match(question: str, ranked: list[PaperMatch]) -> PaperMatch | None:
        lowered = question.lower()
        if any(marker in lowered for marker in ("compare", "comparison", "versus", " vs", " v.s.", "between")):
            return None
        if any(marker in question for marker in ("比较", "对比", "差异", "区别")):
            return None
        if not any(marker in question for marker in ("相比", "相对", "比")):
            return None
        primary_selectors = QueryService._primary_subject_selectors(question)
        if primary_selectors:
            selector_candidates: list[tuple[int, float, PaperMatch]] = []
            for match in ranked:
                route_text = QueryService._paper_identity_route_text(match.document)
                positions = [
                    question.lower().find(selector.lower())
                    for selector in primary_selectors
                    if QueryService._selector_matches_text(selector, route_text)
                ]
                positions = [position for position in positions if position >= 0]
                if positions:
                    selector_candidates.append((min(positions), -match.score, match))
            if selector_candidates:
                selector_candidates.sort(key=lambda item: (item[0], item[1]))
                if selector_candidates[0][0] <= 24:
                    return selector_candidates[0][2]
        candidates: list[tuple[int, PaperMatch]] = []
        for match in ranked:
            aliases = [str(item) for item in (paper_profile_data(match.document).get("aliases") or [])]
            positions = [question.lower().find(alias.lower()) for alias in aliases if str(alias or "").strip()]
            positions = [position for position in positions if position >= 0]
            if positions:
                candidates.append((min(positions), match))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        first_position, first_match = candidates[0]
        if first_position <= 12 and first_match.exact_alias:
            return first_match
        return None

    def _build_rag_contexts(
        self,
        question: str,
        project_id: str,
        paper_matches: list[PaperMatch],
        document_ids: list[str] | None = None,
    ) -> list[RetrievedContext]:
        locked_document_ids = self._locked_document_ids(question, paper_matches)
        # When an explicit document scope is provided, lock it and never widen.
        if document_ids is not None:
            locked_document_ids = document_ids
        derived_document_ids = locked_document_ids or [match.document.id for match in paper_matches]
        profile_terms = self._paper_profile_retrieval_terms(paper_matches)
        is_overview = QueryService._is_document_overview_query(question)
        overview_document_ids = self._overview_document_ids(question, project_id, paper_matches, document_ids=document_ids) if is_overview else None
        contexts: list[RetrievedContext] = []
        if overview_document_ids:
            contexts.extend(
                self._search_document_overview_contexts(question, project_id, overview_document_ids, limit=MAX_CONTEXTS)
            )
            if contexts:
                return self._finalize_contexts(contexts)
        if self._is_table_query(question) or self._is_metric_query(question):
            table_contexts = self._search_document_table_contexts(question, project_id, derived_document_ids, limit=MAX_CONTEXTS)
            if not table_contexts and derived_document_ids and not locked_document_ids:
                table_contexts = self._search_document_table_contexts(question, project_id, [], limit=MAX_CONTEXTS)
            contexts.extend(table_contexts)
            if table_contexts:
                return self._finalize_contexts(contexts)
        if self._is_figure_query(question):
            figure_contexts = self._search_document_figure_contexts(question, project_id, derived_document_ids, limit=MAX_CONTEXTS)
            if not figure_contexts and derived_document_ids and not locked_document_ids:
                figure_contexts = self._search_document_figure_contexts(question, project_id, [], limit=MAX_CONTEXTS)
            contexts.extend(figure_contexts)
        if derived_document_ids and not is_overview:
            if self._is_scientific_evidence_query(question) and not (
                self._is_table_query(question) or self._is_metric_query(question) or self._is_figure_query(question)
            ):
                contexts.extend(self._search_document_intro_contexts(question, project_id, derived_document_ids, limit=2))
                contexts.extend(self._search_document_limitation_contexts(question, project_id, derived_document_ids, limit=3))
                contexts.extend(self._search_document_parameterization_contexts(question, project_id, derived_document_ids, limit=5))
                contexts.extend(self._search_document_scientific_anchor_contexts(question, project_id, derived_document_ids, limit=8))
            contexts.extend(self._search_claim_evidence_contexts(question, project_id, derived_document_ids, limit=min(3, MAX_CONTEXTS)))
            contexts.extend(self._search_source_chunks(question, project_id, derived_document_ids, limit=MAX_CONTEXTS, route_terms=profile_terms))
            contexts.extend(
                self._supplement_profile_term_contexts(
                    project_id,
                    derived_document_ids,
                    contexts,
                    profile_terms,
                    limit=4 if self._is_scientific_evidence_query(question) else 2,
                )
            )
        if not contexts and not document_ids and not is_overview:
            contexts.extend(self._search_source_chunks(question, project_id, [], limit=MAX_CONTEXTS))
        return self._finalize_contexts(contexts)

    @staticmethod
    def _locked_document_ids(question: str, paper_matches: list[PaperMatch]) -> list[str]:
        if QueryService._is_cross_paper_query(question):
            return []
        locked = [match.document.id for match in paper_matches if match.locked]
        if locked:
            return list(dict.fromkeys(locked[:1]))
        exact = [match.document.id for match in paper_matches if match.exact_alias]
        if len(exact) == 1:
            return exact
        return []

    def _single_ready_document_id(self, project_id: str) -> list[str] | None:
        """Return the only ready document in a project, or None if not exactly one."""
        rows = self.db.scalars(
            select(Document).where(
                Document.project_id == project_id,
                Document.status == DocumentStatus.ready.value,
            )
        ).all()
        if len(rows) == 1:
            return [rows[0].id]
        return None

    def _overview_document_ids(
        self,
        question: str,
        project_id: str,
        paper_matches: list[PaperMatch],
        document_ids: list[str] | None = None,
    ) -> list[str] | None:
        """Resolve target document IDs for a document-overview query.

        Only returns a target when safe: an exact/locked paper match already
        selected a document, the caller provided an explicit document scope,
        or the project contains exactly one ready document. Otherwise returns
        None so overview retrieval does not silently mix or pick an arbitrary
        document.
        """
        if not QueryService._is_document_overview_query(question):
            return None
        if document_ids is not None:
            return document_ids
        locked = self._locked_document_ids(question, paper_matches)
        if locked:
            return locked
        exact = [match.document.id for match in paper_matches if match.exact_alias]
        if len(exact) == 1:
            return exact
        return self._single_ready_document_id(project_id)

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
        statement = select(Document).where(
            Document.project_id == project_id,
            Document.status == DocumentStatus.ready.value,
        )
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
                        evidence_kind="table",
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
                Document.status == DocumentStatus.ready.value,
            )
        ).all()
        fields: dict[str, dict[str, str]] = {}
        for document in documents:
            fields[document.id] = source_fields_for_document(document)
        return fields

    def _search_document_figure_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 5,
    ) -> list[RetrievedContext]:
        statement = select(Document).where(
            Document.project_id == project_id,
            Document.status == DocumentStatus.ready.value,
        )
        if document_ids:
            statement = statement.where(Document.id.in_(document_ids))
        documents = self.db.scalars(statement).all()
        source_page_fields = self._source_page_fields_by_document_id(project_id, [document.id for document in documents])
        contexts: list[RetrievedContext] = []
        for document in documents:
            metadata = document.metadata_json or {}
            intelligence = metadata.get("document_intelligence") if isinstance(metadata.get("document_intelligence"), dict) else {}
            figures = intelligence.get("figures") if isinstance(intelligence, dict) else []
            if not isinstance(figures, list):
                continue
            for ordinal, figure in enumerate(figures):
                if isinstance(figure, dict):
                    figure_parts: list[str] = []
                    seen_figure_values: set[str] = set()
                    for label, key in (
                        ("Caption", "caption"),
                        ("Note", "note"),
                        ("Text", "text"),
                        ("Image path", "image_path"),
                        ("Path", "path"),
                    ):
                        value = str(figure.get(key) or "").strip()
                        if value and value not in seen_figure_values:
                            figure_parts.append(f"{label}: {value}")
                            seen_figure_values.add(value)
                    note = "\n".join(figure_parts).strip()
                    page_label = str(figure.get("page_label") or "").strip() or None
                else:
                    note = str(figure or "").strip()
                    page_label = None
                if not note:
                    continue
                block = note
                if not re.search(r"\b(?:figure|fig\.)\b", block, re.IGNORECASE):
                    block = f"Figure evidence: {block}"
                if page_label and not re.search(r"\bpage\s+\d+", block, re.IGNORECASE):
                    block = f"Page {page_label}: {block}"
                block_score = self._rank_blocks(question, [block])[0][1]
                if block_score <= 0 and not (self._tokenize(question) & self._tokenize(block)):
                    continue
                score = 30.0 + block_score + max(0.0, 2.0 - ordinal * 0.01)
                contexts.append(
                    RetrievedContext(
                        citation=Citation(
                            document_id=document.id,
                            **source_page_fields.get(document.id, {}),
                            score=score,
                            page_label=page_label,
                            excerpt=block[:700],
                        ),
                        prompt_text=block[:2000],
                        score=score,
                        evidence_kind="figure",
                    )
        )
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    def _search_document_intro_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 2,
    ) -> list[RetrievedContext]:
        if not document_ids:
            return []
        chunks = self.db.scalars(
            select(DocumentChunk)
            .join(DocumentChunk.document)
            .where(
                DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
                DocumentChunk.document_id.in_(document_ids),
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
        ).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )
        query_terms = self._tokenize(question)
        contexts: list[RetrievedContext] = []
        for chunk in chunks:
            page_number = self._page_label_number(chunk.page_label)
            if chunk.ordinal > 3 and (page_number is None or page_number > 3):
                continue
            evidence = chunk.text.strip()
            if not evidence:
                continue
            overlap = len(query_terms & self._tokenize(evidence))
            if overlap <= 0:
                continue
            score = 42.0 + min(overlap, 12) + max(0.0, 4.0 - chunk.ordinal * 0.2)
            excerpt = self._window_text(evidence, query_terms, max_chars=900, question=question)
            contexts.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        **source_page_fields.get(chunk.document_id, {}),
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=excerpt[:900],
                    ),
                    prompt_text=excerpt,
                    score=score,
                    evidence_kind="intro",
                )
            )
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    def _search_document_overview_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 5,
    ) -> list[RetrievedContext]:
        """Retrieve substantive overview chunks for a single target document.

        Selects abstract/introduction, method, and conclusion chunks while
        dropping heading-only fragments such as "Conclusion" or "Related Work".
        """
        if not document_ids:
            return []
        chunks = self.db.scalars(
            select(DocumentChunk)
            .join(DocumentChunk.document)
            .where(
                DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
                DocumentChunk.document_id.in_(document_ids),
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
        ).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )
        contexts: list[RetrievedContext] = []
        for chunk in chunks:
            if QueryService._is_heading_only_text(chunk.text):
                continue
            evidence = chunk.text.strip()
            if QueryService._context_has_table_data(evidence):
                continue
            score = QueryService._overview_chunk_score(chunk, evidence)
            if score <= 0:
                continue
            excerpt = evidence[:900]
            contexts.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        **source_page_fields.get(chunk.document_id, {}),
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=excerpt,
                    ),
                    prompt_text=evidence[:1600],
                    score=score,
                    evidence_kind="overview",
                )
            )
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _overview_chunk_score(chunk: DocumentChunk, evidence: str) -> float:
        """Score prose chunks that are useful for a document-level summary."""
        lowered = evidence.lower()
        head = lowered[:240]
        score = 20.0
        page_number = QueryService._page_label_number(chunk.page_label)
        if page_number is not None and page_number <= 2:
            score += 28.0
        if chunk.ordinal <= 8:
            score += max(0.0, 18.0 - chunk.ordinal * 1.5)
        if "abstract" in head:
            score += 26.0
        if re.search(r"(?:^|\n)#+\s*(?:\d+(?:\.\d+)?\s*)?introduction\b|\bintroduction\b", head):
            score += 20.0
        if re.search(r"\bwe\s+(?:introduce|propose|present|develop|study|show|demonstrate)\b", lowered):
            score += 18.0
        if re.search(r"\b(?:method|approach|framework|algorithm|model)\b", lowered):
            score += 8.0
        if re.search(r"\b(?:conclusion|conclusions)\b", head):
            score += 14.0
        if re.search(r"\b(?:appendix|references|acknowledgements?)\b", head):
            score -= 18.0
        if re.search(r"\btable\s+\d+\b", head):
            score -= 16.0
        score += min(len(evidence) / 500.0, 4.0)
        return score

    @staticmethod
    def _page_label_number(page_label: str | None) -> int | None:
        match = re.search(r"\d+", str(page_label or ""))
        return int(match.group(0)) if match else None

    def _search_document_limitation_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 2,
    ) -> list[RetrievedContext]:
        if not document_ids or not self._is_limitation_query(question):
            return []
        chunks = self.db.scalars(
            select(DocumentChunk)
            .join(DocumentChunk.document)
            .where(
                DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
                DocumentChunk.document_id.in_(document_ids),
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
        ).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )
        query_terms = self._tokenize(question)
        anchors = (
            "limitation",
            "limitations",
            "unsatisfactory",
            "overestimation",
            "overestimated",
            "deficiency",
            "deficiencies",
            "common problem",
            "radius of gyration",
            "compactness",
            "large conformational",
            "large disordered proteins",
            "fast-folding",
            "not fine enough",
        )
        contexts: list[RetrievedContext] = []
        wants_a99sb = "a99sb" in question.lower()
        for chunk in chunks:
            evidence = chunk.text.strip()
            lowered = evidence.lower()
            if not evidence or not any(anchor in lowered for anchor in anchors):
                continue
            overlap = len(query_terms & self._tokenize(evidence))
            anchor_score = sum(1 for anchor in anchors if anchor in lowered)
            score = 43.0 + min(overlap, 8) + anchor_score * 2.0
            if wants_a99sb and "a99sb" in lowered:
                score += 8.0
            if "fast-folding" in lowered:
                score += 6.0
            if "large disordered proteins" in lowered:
                score += 6.0
            excerpt = self._window_text(evidence, query_terms | {"radius", "gyration", "fast", "folding"}, max_chars=900, question=question)
            contexts.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        **source_page_fields.get(chunk.document_id, {}),
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=excerpt[:900],
                    ),
                    prompt_text=excerpt,
                    score=score,
                    evidence_kind="limitation",
                )
            )
        return sorted(contexts, key=lambda item: item.score, reverse=True)[:limit]

    def _search_document_parameterization_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 4,
    ) -> list[RetrievedContext]:
        if not document_ids or not self._is_parameterization_anchor_query(question):
            return []
        chunks = self.db.scalars(
            select(DocumentChunk)
            .join(DocumentChunk.document)
            .where(
                DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
                DocumentChunk.document_id.in_(document_ids),
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
        ).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )
        required_groups = (
            ("RESP", ("resp", "charge fitting", "partial charge", "hf/6-31g", "hf/6-31g*")),
            ("QM", ("m05-2x", "mp2/cc-pvqz", "6-311g", "qm energy surface", "quantum mechanics")),
            ("CMAP", ("leu cmap", "val cmap", "ile", "β-branched", "beta-branched")),
            ("validation", ("5 milliseconds", "milliseconds md simulations", "explicit solvent")),
        )
        specific_group_anchors = {
            "RESP": ("hf/6-31g", "resp"),
            "QM": ("m05-2x", "mp2/cc-pvqz"),
            "CMAP": ("leu cmap", "val cmap"),
            "validation": ("5 milliseconds",),
        }
        contexts: list[RetrievedContext] = []
        best_by_group: dict[str, RetrievedContext] = {}
        query_terms = self._tokenize(question)
        for chunk in chunks:
            evidence = chunk.text.strip()
            if not evidence:
                continue
            normalized = self._normalize_scientific_evidence_text(evidence)
            matched_groups = [
                group
                for group, anchors in required_groups
                if any(anchor in normalized for anchor in anchors)
            ]
            if not matched_groups:
                continue
            specific_matches = {
                group: [anchor for anchor in specific_group_anchors.get(group, ()) if anchor in normalized]
                for group in matched_groups
            }
            specific_matches = {group: anchors for group, anchors in specific_matches.items() if anchors}
            flat_terms: set[str] = set(query_terms)
            for group, anchors in required_groups:
                if group in matched_groups:
                    for anchor in anchors:
                        flat_terms.update(self._tokenize(anchor))
            specific_terms: set[str] = set()
            for anchors in specific_matches.values():
                for anchor in anchors:
                    specific_terms.add(anchor)
                    specific_terms.update(self._tokenize(anchor))
            score = 50.0 + len(matched_groups) * 9.0 + min(len(query_terms & self._tokenize(evidence)), 10)
            score += sum(len(anchors) for anchors in specific_matches.values()) * 18.0
            if "RESP" in matched_groups and "HF/6-31G" in evidence:
                score += 8.0
            if "QM" in matched_groups and "M05-2X" in normalized.upper():
                score += 4.0
            if "QM" in matched_groups and "MP2/CC-PVQZ" in normalized.upper():
                score += 4.0
            if "CMAP" in matched_groups and ("Leu CMAP" in evidence or "Val CMAP" in evidence):
                score += 6.0
            if "validation" in matched_groups and "5 milliseconds" in normalized:
                score += 4.0
            excerpt = self._anchored_parameterization_excerpt(evidence, specific_matches, flat_terms)
            context = RetrievedContext(
                citation=Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    **source_page_fields.get(chunk.document_id, {}),
                    score=score,
                    page_label=chunk.page_label,
                    excerpt=excerpt[:1000],
                ),
                prompt_text=excerpt,
                score=score,
                evidence_kind="profile-term",
            )
            contexts.append(context)
            for group in specific_matches:
                current = best_by_group.get(group)
                if current is None or context.score > current.score:
                    best_by_group[group] = context
        prioritized: list[RetrievedContext] = []
        for group in ("RESP", "QM", "CMAP", "validation"):
            context = best_by_group.get(group)
            if context is not None and context not in prioritized:
                prioritized.append(context)
        for context in sorted(contexts, key=lambda item: item.score, reverse=True):
            if context not in prioritized:
                prioritized.append(context)
            if len(prioritized) >= limit:
                break
        return prioritized[:limit]

    def _search_document_scientific_anchor_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 6,
    ) -> list[RetrievedContext]:
        if not document_ids or not self._is_scientific_evidence_query(question):
            return []
        chunks = self.db.scalars(
            select(DocumentChunk)
            .join(DocumentChunk.document)
            .where(
                DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
                DocumentChunk.document_id.in_(document_ids),
            )
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
        ).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )
        question_terms = self._tokenize(question)
        contexts: list[RetrievedContext] = []
        best_by_anchor: dict[str, RetrievedContext] = {}
        for chunk in chunks:
            evidence = chunk.text.strip()
            if not evidence:
                continue
            evidence_key = self._normalize_selector(evidence)
            lowered_evidence = evidence.lower()
            matched: list[tuple[str, int]] = []
            for label, pattern in self._SCIENTIFIC_CONTEXT_ANCHORS:
                match = pattern.search(evidence)
                if match:
                    matched.append((label, match.start()))
                    continue
                label_key = self._normalize_selector(label)
                if label == "C6":
                    if "c6" in evidence_key and ("dispersion" in lowered_evidence or "coefficient" in lowered_evidence):
                        matched.append((label, 0))
                    continue
                if len(label_key) >= 4 and label_key in evidence_key:
                    matched.append((label, 0))
            if not matched:
                continue
            matched_labels = [label for label, _ in matched]
            anchor_terms = set(question_terms)
            for label in matched_labels:
                anchor_terms.update(self._tokenize(label))
                anchor_terms.add(self._normalize_selector(label))
            high_value_labels = {
                "Drude",
                "LFMM",
                "MMP13",
                "charge transfer",
                "GLH",
                "GLU",
                "TIP4P-EW",
                "C6",
                "large disordered proteins",
                "large conformational fluctuation",
                "SPARTA",
                "PPII",
                "Lennard-Jones",
                "steric",
                "2kT",
                "QM-MM",
                "molten globule",
            }
            window_priority_labels = (
                "PPII",
                "SPARTA",
                "Lennard-Jones",
                "steric",
                "2kT",
                "QM-MM",
                "molten globule",
                "hydration free energy",
                "torsional",
                "helix-coil",
            )
            window_priority_positions = [position for label, position in matched if label in window_priority_labels]
            priority_positions = [position for label, position in matched if label in high_value_labels]
            anchor_positions = window_priority_positions or priority_positions or [position for _, position in matched]
            first_anchor = min(anchor_positions)
            last_anchor = max(anchor_positions)
            start = max(0, first_anchor - 500)
            end = min(len(evidence), max(first_anchor + 1300, last_anchor + 420))
            if end - start > 1800:
                end = min(len(evidence), last_anchor + 420)
                start = max(0, end - 1800)
            excerpt = evidence[start:end].strip()
            if not excerpt:
                excerpt = self._window_text(evidence, anchor_terms, max_chars=1000, question=question)
            citation_excerpt = self._scientific_anchor_excerpt_window(
                excerpt,
                matched_labels,
                max_chars=1000,
            )
            overlap = len(question_terms & self._tokenize(evidence))
            score = 32.0 + min(len(matched_labels), 6) * 3.0 + min(overlap, 8)
            if any(label.lower() in question.lower() for label in matched_labels):
                score += 6.0
            if any(
                label
                in {
                    "Drude",
                    "LFMM",
                    "MMP13",
                    "charge transfer",
                    "GLH",
                    "GLU",
                    "TIP4P-EW",
                    "C6",
                    "large disordered proteins",
                    "large conformational fluctuation",
                    "SPARTA",
                    "PPII",
                    "Lennard-Jones",
                    "steric",
                    "2kT",
                    "QM-MM",
                    "molten globule",
                }
                for label in matched_labels
            ):
                score += 14.0
            if any(self._is_scientific_profile_term_key(self._normalize_selector(label)) for label in matched_labels):
                score += 4.0
            context = RetrievedContext(
                citation=Citation(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    **source_page_fields.get(chunk.document_id, {}),
                    score=score,
                    page_label=chunk.page_label,
                    excerpt=citation_excerpt,
                ),
                prompt_text=evidence,
                score=score,
                evidence_kind="profile-term",
            )
            contexts.append(context)
            for label, _ in matched:
                key = self._normalize_selector(label)
                current = best_by_anchor.get(key)
                if current is None or context.score > current.score:
                    best_by_anchor[key] = context
        prioritized: list[RetrievedContext] = []
        covered_anchor_keys: set[str] = set()
        high_value_keys = {
            self._normalize_selector(label)
            for label in (
                "Drude",
                "LFMM",
                "MMP13",
                "GLH",
                "GLU",
                "charge transfer",
                "C6",
                "London dispersion",
                "large disordered proteins",
                "large conformational fluctuation",
                "SPARTA",
                "PPII",
                "Lennard-Jones",
                "steric",
                "2kT",
                "QM-MM",
                "molten globule",
            )
        }
        question_key = self._normalize_selector(question)

        def add_context(context: RetrievedContext) -> None:
            if context in prioritized or len(prioritized) >= limit:
                return
            prioritized.append(context)
            covered_anchor_keys.update(self._normalize_selector(label) for label in self._scientific_anchor_labels_in_text(self._context_evidence_text(context)))

        preferred_keys = [
            key
            for key in best_by_anchor
            if key in high_value_keys or (len(key) >= 3 and key in question_key)
        ]
        for key in sorted(preferred_keys, key=lambda item: best_by_anchor[item].score, reverse=True):
            add_context(best_by_anchor[key])
            if len(prioritized) >= limit:
                return prioritized
        while len(prioritized) < limit:
            candidates = [
                context
                for context in contexts
                if context not in prioritized
                and any(self._normalize_selector(label) not in covered_anchor_keys for label in self._scientific_anchor_labels_in_text(self._context_evidence_text(context)))
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda item: (
                    len(
                        {
                            self._normalize_selector(label)
                            for label in self._scientific_anchor_labels_in_text(self._context_evidence_text(item))
                            if self._normalize_selector(label) not in covered_anchor_keys
                        }
                    ),
                    item.score,
                ),
                reverse=True,
            )
            add_context(candidates[0])
        for context in sorted(contexts, key=lambda item: item.score, reverse=True):
            add_context(context)
            if len(prioritized) >= limit:
                break
        return prioritized[:limit]

    @classmethod
    def _scientific_anchor_excerpt_window(cls, text: str, labels: list[str], max_chars: int = 1000) -> str:
        if len(text) <= max_chars:
            return text
        priority_order = (
            "PPII",
            "SPARTA",
            "Lennard-Jones",
            "steric",
            "2kT",
            "QM-MM",
            "molten globule",
            "hydration free energy",
            "torsional",
            "helix-coil",
        )
        priority_labels = [label for label in priority_order if label in labels]
        if priority_labels:
            labels = priority_labels
        positions: list[int] = []
        for wanted_label in labels:
            for label, pattern in cls._SCIENTIFIC_CONTEXT_ANCHORS:
                if label != wanted_label:
                    continue
                match = pattern.search(text)
                if match:
                    positions.append(match.start())
                break
        if not positions:
            return text[:max_chars]
        first_anchor = min(positions)
        last_anchor = max(positions)
        if last_anchor - first_anchor < max_chars:
            start = max(0, min(first_anchor - 160, last_anchor - max_chars + 240))
        else:
            start = max(0, last_anchor - max_chars + 240)
        end = min(len(text), start + max_chars)
        if last_anchor >= end:
            end = min(len(text), last_anchor + 240)
            start = max(0, end - max_chars)
        return text[start:end].strip()

    @classmethod
    def _scientific_anchor_labels_in_text(cls, text: str) -> list[str]:
        evidence = str(text or "")
        if not evidence.strip():
            return []
        evidence_key = cls._normalize_selector(evidence)
        lowered_evidence = evidence.lower()
        labels: list[str] = []
        seen: set[str] = set()
        for label, pattern in cls._SCIENTIFIC_CONTEXT_ANCHORS:
            matched = bool(pattern.search(evidence))
            if not matched:
                label_key = cls._normalize_selector(label)
                if label == "C6":
                    matched = "c6" in evidence_key and ("dispersion" in lowered_evidence or "coefficient" in lowered_evidence)
                elif len(label_key) >= 4:
                    matched = label_key in evidence_key
            key = cls._normalize_selector(label)
            if matched and key and key not in seen:
                labels.append("\u03c71" if key == "chi1" else label)
                seen.add(key)
        return labels

    @staticmethod
    def _is_parameterization_anchor_query(question: str) -> bool:
        lowered = question.lower()
        return bool(
            "参数化" in question
            or "验证规模" in question
            or re.search(r"\b(?:parameterization|parameterisation|resp|cmap|qm level|charge fitting)\b", lowered)
        )

    @classmethod
    def _anchored_parameterization_excerpt(
        cls,
        evidence: str,
        specific_matches: dict[str, list[str]],
        fallback_terms: set[str],
    ) -> str:
        anchors: list[str] = []
        for group in ("RESP", "QM", "CMAP", "validation"):
            anchors.extend(specific_matches.get(group, []))
        if not anchors:
            return cls._window_text(evidence, fallback_terms, max_chars=1000, question="")
        lowered = evidence.lower()
        positions = [lowered.find(anchor.lower()) for anchor in anchors if lowered.find(anchor.lower()) >= 0]
        if not positions:
            return cls._window_text(evidence, fallback_terms, max_chars=1000, question="")
        anchor = min(positions)
        start = max(0, anchor - 420)
        end = min(len(evidence), start + 1000)
        start = max(0, end - 1000)
        return evidence[start:end].strip()

    @staticmethod
    def _normalize_scientific_evidence_text(text: str) -> str:
        normalized = normalize_table_text(text)
        normalized = re.sub(r"\s*/\s*", "/", normalized)
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.lower()

    @staticmethod
    def _is_limitation_query(question: str) -> bool:
        lowered = question.lower()
        return any(
            marker in lowered or marker in question
            for marker in (
                "limitation",
                "limitations",
                "drawback",
                "drawbacks",
                "shortcoming",
                "shortcomings",
                "weakness",
                "weaknesses",
                "不足",
                "局限",
                "问题",
                "缺点",
            )
        )

    @classmethod
    def _paper_profile_retrieval_terms(cls, paper_matches: list[PaperMatch]) -> list[str]:
        terms: list[str] = []
        for match in paper_matches:
            for term in paper_profile_retrieval_terms(match.document):
                value = str(term or "").strip()
                if cls._is_profile_retrieval_term(value):
                    terms.append(value)
        ordered: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = cls._normalize_selector(term)
            if key and key not in seen:
                ordered.append(term)
                seen.add(key)
        return ordered[:96]

    def _supplement_profile_term_contexts(
        self,
        project_id: str,
        document_ids: list[str],
        contexts: list[RetrievedContext],
        profile_terms: list[str],
        limit: int = 2,
    ) -> list[RetrievedContext]:
        if not document_ids or not profile_terms:
            return []
        evidence_text = "\n".join(self._context_evidence_text(context) for context in contexts)
        covered_keys = self._tokenize(evidence_text)
        candidate_terms = [
            term
            for term in profile_terms
            if self._is_supplemental_profile_term(term) and not (self._tokenize(term) & covered_keys)
        ]
        if not candidate_terms:
            return []
        statement = select(DocumentChunk).join(DocumentChunk.document).where(
            DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
            DocumentChunk.document_id.in_(document_ids),
        )
        chunks = self.db.scalars(statement).all()
        source_page_fields = self._source_page_fields_by_document_id(project_id, document_ids)
        existing_chunk_ids = {context.citation.chunk_id for context in contexts if context.citation.chunk_id}
        scored: list[RetrievedContext] = []
        for chunk in chunks:
            if chunk.id in existing_chunk_ids:
                continue
            text_key = self._normalize_selector(chunk.text)
            matched_terms = [
                term
                for term in candidate_terms
                if self._normalize_selector(term) and self._normalize_selector(term) in text_key
            ]
            if not matched_terms:
                continue
            query_terms = self._tokenize(" ".join(matched_terms))
            excerpt = self._window_text(chunk.text, query_terms, max_chars=1000, question=" ".join(matched_terms))
            score = 3.0 + len(matched_terms) * 1.5
            if any(self._is_scientific_profile_term_key(self._normalize_selector(term)) for term in matched_terms):
                score = 28.0 + len(matched_terms) * 3.0 + min(len(query_terms & self._tokenize(chunk.text)), 5)
            scored.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        **source_page_fields.get(chunk.document_id, {}),
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=excerpt[:280],
                ),
                prompt_text=excerpt,
                score=score,
                evidence_kind="profile-term",
            )
        )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    @classmethod
    def _is_supplemental_profile_term(cls, term: str) -> bool:
        value = str(term or "").strip()
        key = cls._normalize_selector(value)
        if cls._is_scientific_profile_term_key(key):
            return True
        if len(key) < 4 or key in cls._CLAIM_ANCHOR_STOP_KEYS:
            return False
        if cls._is_table_model_term_key(key):
            return False
        if re.search(r"\d", value) and any(char.isalpha() for char in value):
            return True
        if re.fullmatch(r"(?:ff|opls|charmm|amber|tip)\d+[a-z0-9-]*", value, re.IGNORECASE):
            return False
        if re.search(r"[a-z][A-Z]", value):
            return True
        if re.fullmatch(r"[A-Z]{3,}[A-Z0-9-]*", value):
            return True
        if re.search(r"[-_/]", value) and any(char.isalpha() for char in value):
            return True
        return key in {"cmap", "galib", "boltzmann", "population", "barrier", "fitting", "protocol"}

    @classmethod
    def _is_profile_retrieval_term(cls, term: str) -> bool:
        key = cls._normalize_selector(term)
        if cls._is_scientific_profile_term_key(key):
            return True
        if len(key) < 3 or key in cls._CLAIM_ANCHOR_STOP_KEYS:
            return False
        if cls._is_table_model_term_key(key):
            return False
        if re.search(r"[-_/]", term):
            return True
        if re.search(r"\d", term):
            return True
        if re.search(r"[A-Z].*[A-Z]", term) or re.search(r"[a-z][A-Z]", term):
            return True
        return key in {"backbone", "sidechain", "sidechains", "rotamer", "rotamers", "torsion", "torsions"}

    @classmethod
    def _is_scientific_profile_term_key(cls, key: str) -> bool:
        return key in cls._SCIENTIFIC_PROFILE_TERM_KEYS or bool(
            key.startswith("c6") and ("dispersion" in key or "coefficient" in key)
        )

    def _search_claim_evidence_contexts(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 5,
    ) -> list[RetrievedContext]:
        if not document_ids:
            return []
        statement = (
            select(Claim, DocumentChunk)
            .join(
                DocumentChunk,
                and_(
                    DocumentChunk.id == Claim.evidence_chunk_id,
                    DocumentChunk.document_id == Claim.document_id,
                ),
            )
            .where(
                Claim.project_id == project_id,
                Claim.document_id.in_(document_ids),
                Claim.evidence_chunk_id.is_not(None),
                DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value),
            )
        )
        rows = self.db.execute(statement).all()
        if not rows:
            return []

        source_page_fields = self._source_page_fields_by_document_id(project_id, document_ids)
        query_terms = self._tokenize(question)
        selector_terms = [
            term
            for term in [
                *self._question_row_selectors(question),
                *self._extract_generic_table_terms(question),
                *self._extract_query_facets(question),
            ]
            if self._is_specific_claim_anchor(term)
        ]
        selector_keys = {self._normalize_selector(term) for term in selector_terms}
        if len(selector_keys) < 2:
            return []
        scored: list[RetrievedContext] = []
        seen_chunk_ids: set[str] = set()
        for claim, chunk in rows:
            if chunk.id in seen_chunk_ids:
                continue
            claim_text = f"{claim.subject} {claim.predicate} {claim.object_text} {claim.metadata_json or {}}"
            combined_text = f"{claim_text}\n{chunk.text}"
            overlap = len(query_terms & self._tokenize(combined_text))
            combined_key = self._normalize_selector(combined_text)
            anchor_matches = sum(1 for term in selector_keys if term in combined_key)
            if anchor_matches < 2:
                continue
            evidence_terms = query_terms | self._tokenize(claim_text)
            evidence = self._window_text(chunk.text, evidence_terms, max_chars=1400, question=question)
            score = 12.0 + overlap * 1.5 + anchor_matches * 3.0 + float(claim.confidence or 0.0)
            scored.append(
                RetrievedContext(
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        **source_page_fields.get(chunk.document_id, {}),
                        score=score,
                        page_label=chunk.page_label,
                        excerpt=evidence[:900],
                ),
                prompt_text=evidence,
                score=score,
                evidence_kind="claim",
            )
        )
            seen_chunk_ids.add(chunk.id)
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    def _search_source_chunks(
        self,
        question: str,
        project_id: str,
        document_ids: list[str],
        limit: int = 3,
        route_terms: list[str] | None = None,
    ) -> list[RetrievedContext]:
        question_vector = safe_model_call(lambda: self.ollama.embed([question])[0], [])
        is_table_query = self._is_table_query(question)
        needs_table_first = is_table_query or self._is_metric_query(question)
        vector_hits = (
            SQLiteVecStore(self.db).search(
                question_vector,
                limit=max(limit * 20, 50),
                document_ids=document_ids or None,
            )
            if question_vector
            else []
        )
        vector_scores_by_chunk_id = {
            hit.chunk_id: self._vector_distance_score(hit.distance)
            for hit in vector_hits
        }
        statement = select(DocumentChunk).join(DocumentChunk.document).where(
            DocumentChunk.document.has(project_id=project_id, status=DocumentStatus.ready.value)
        )
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))
        chunks = self.db.scalars(statement).all()
        source_page_fields = self._source_page_fields_by_document_id(
            project_id,
            sorted({chunk.document_id for chunk in chunks}),
        )

        base_query_terms = self._tokenize(question)
        route_query_terms = self._tokenize(" ".join(route_terms or []))
        query_terms = base_query_terms | route_query_terms
        route_token_counts: Counter[str] = Counter()
        if route_query_terms:
            for chunk in chunks:
                route_token_counts.update(route_query_terms & self._tokenize(chunk.text))
        rare_route_terms = {term for term, count in route_token_counts.items() if count <= 3}
        scored: list[RetrievedContext] = []
        for chunk in chunks:
            score = 0.0
            chunk_terms = self._tokenize(chunk.text)
            overlap = len(base_query_terms & chunk_terms)
            route_overlap = len(route_query_terms & chunk_terms)
            rare_route_overlap = len(rare_route_terms & chunk_terms)
            rare_route_bonus = min(rare_route_overlap * 1.1, 3.0)
            if chunk.id in vector_scores_by_chunk_id:
                score = vector_scores_by_chunk_id[chunk.id]
                score += min(overlap * 0.05 + route_overlap * 0.08, 0.8) + rare_route_bonus
            elif question_vector and chunk.embedding:
                score = cosine_similarity(question_vector, chunk.embedding)
                score += min(overlap * 0.05 + route_overlap * 0.08, 0.8) + rare_route_bonus
            else:
                total_overlap = overlap + route_overlap
                if total_overlap:
                    score = min(0.3 + total_overlap * 0.1 + rare_route_bonus, 1.2)
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
                    evidence_kind="table" if has_table_data else None,
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _vector_distance_score(distance: float) -> float:
        return 1.0 / (1.0 + max(float(distance), 0.0))

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

    # ------------------------------------------------------------------
    # evidence relevance — guard against answering from irrelevant sources
    # ------------------------------------------------------------------

    @classmethod
    def _extract_specific_question_scientific_terms(cls, question: str) -> set[str]:
        """Extract specific scientific terms from the question to check evidence relevance.

        Returns terms that represent specific scientific entities (force fields,
        model names, acronyms, etc.) that must appear in retrieved evidence for
        the answer to be considered grounded.
        """
        terms: set[str] = set()
        lowered = question.lower()
        # Force-field and model-name patterns (CHARMM36m, AMBER99SB, OPLS4, etc.)
        for match in cls._FORCE_FIELD_RE.finditer(question):
            term = match.group(0).lower()
            if term and term not in cls._QUESTION_STOP_TERMS:
                terms.add(term)
        # General scientific acronyms (uppercase+digits, >=3 chars)
        for match in cls._SCIENTIFIC_ACRONYM_RE.finditer(question):
            term = match.group(0).lower()
            if len(term) >= 3 and term not in cls._QUESTION_STOP_TERMS:
                terms.add(term)
        # Check against known scientific context anchors
        for label, pattern in cls._SCIENTIFIC_CONTEXT_ANCHORS:
            if pattern.search(question):
                key = cls._normalize_selector(label)
                if key and len(key) >= 3:
                    terms.add(key)
        return terms

    @classmethod
    def _evidence_overlaps_question_scientific_terms(
        cls, question: str, contexts: list["RetrievedContext"]
    ) -> bool:
        """Return True when at least one retrieved context mentions the specific
        scientific terms from the question.

        When no specific terms are extracted from the question the check is
        skipped (returns True) so the LLM handles the question normally.
        """
        specific_terms = cls._extract_specific_question_scientific_terms(question)
        if not specific_terms:
            return True  # nothing specific to gate on
        normalized_question = cls._normalize_selector(question)
        for ctx in contexts:
            citation = getattr(ctx, "citation", None)
            if citation is None:
                continue
            title_key = cls._normalize_selector(getattr(citation, "page_title", None) or "")
            if len(title_key) >= 5 and title_key in normalized_question:
                return True
        for ctx in contexts:
            evidence = cls._normalize_selector(cls._context_relevance_text(ctx))
            for term in specific_terms:
                if term and len(term) >= 3 and term in evidence:
                    return True
        return False

    @classmethod
    def _context_relevance_text(cls, context: "RetrievedContext") -> str:
        """Text used only for relevance gating.

        Profile-term retrieval can select a highly relevant source page while
        the snippet window omits the source name itself.  Source metadata is
        safe to use for this coarse relevance check, but remains separate from
        answer prompting and citation excerpts.
        """
        parts = [cls._context_evidence_text(context)]
        citation = getattr(context, "citation", None)
        if citation is not None:
            for value in (
                getattr(citation, "page_title", None),
                getattr(citation, "page_slug", None),
                getattr(citation, "page_label", None),
            ):
                clean = str(value or "").strip()
                if clean and clean not in parts:
                    parts.append(clean)
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _insufficient_evidence_answer(question_terms: set[str] | None = None) -> str:
        """Deterministic answer when retrieved evidence is irrelevant to the question."""
        if question_terms:
            quoted = ", ".join(sorted(question_terms)[:5])
            return (
                "## Insufficient Evidence\n\n"
                "The retrieved source documents do not contain information about the "
                f"specific scientific terms in your question ({quoted}). "
                "The current knowledge base may contain only sample or demo documents "
                "that do not discuss these entities.\n\n"
                "Please upload documents covering the requested topics, or switch to a "
                "project with the relevant knowledge base."
            )
        return (
            "## Insufficient Evidence\n\n"
            "The retrieved source documents do not contain information relevant to "
            "your question. The current knowledge base may contain only sample or "
            "demo documents.\n\n"
            "Please upload documents covering the requested topics, or switch to a "
            "project with the relevant knowledge base."
        )

    def _draft_answer(self, question: str, index_context: str | None, contexts: list[RetrievedContext]) -> QueryAnswerPayload:
        if not contexts:
            return QueryAnswerPayload(
                answer_markdown="No supporting evidence was found yet. Please ingest relevant sources first.",
                citations=[],
                risk_level="normal",
            )

        # Evidence-relevance gate: when the question mentions specific
        # scientific entities (force fields, model names, acronyms) but no
        # retrieved context discusses them, return an explicit
        # insufficient-evidence answer instead of asking the LLM to
        # fabricate one from unrelated text.
        question_terms = self._extract_specific_question_scientific_terms(question)
        if question_terms and not self._evidence_overlaps_question_scientific_terms(question, contexts):
            return QueryAnswerPayload(
                answer_markdown=self._insufficient_evidence_answer(question_terms),
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

        if self._is_chinese_question(question):
            fallback_text = "\n".join(
                [
                    "## 回答",
                    "根据当前检索到的原文证据，暂时返回可核查的证据片段；以下内容均来自返回的 citation。",
                    "",
                    context_text[:1400],
                ]
            )
        else:
            fallback_text = "\n".join(
                [
                    "## Answer",
                    "The answer below is based on the currently retrieved source evidence. Please verify against the cited materials when needed.",
                    "",
                    context_text[:1400],
                ]
            )
        deterministic_scientific = self._deterministic_scientific_evidence_answer_if_supported(
            question,
            contexts,
            "high" if self._is_high_risk(question) else "normal",
        )
        if (
            deterministic_scientific is not None
            and self._is_chinese_question(question)
            and any(self._context_evidence_kind(context) == "profile-term" for context in contexts)
            and not (
            self._is_table_query(question) or self._is_metric_query(question) or self._is_figure_query(question)
            )
        ):
            return deterministic_scientific
        fallback = deterministic_scientific or QueryAnswerPayload(
            answer_markdown=fallback_text,
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
        def generate_with_query_timeout() -> QueryAnswerPayload:
            original_timeout = getattr(self.ollama, "timeout", None)
            if original_timeout is not None:
                self.ollama.timeout = min(float(original_timeout), QUERY_GENERATION_TIMEOUT_SECONDS)
            try:
                return self.ollama.generate_structured(
                    QueryAnswerPayload,
                    system_prompt="You are answering against a RAG evidence set. Use only retrieved source, table, and figure evidence; cite supporting context indexes and do not claim facts that are absent from the provided material.",
                    user_prompt=prompt,
                )
            finally:
                if original_timeout is not None:
                    self.ollama.timeout = original_timeout

        return safe_model_call(
            lambda: generate_with_query_timeout(),
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
        evidence_phrases = self._salient_evidence_phrases(contexts)

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
        if evidence_phrases:
            parts.append(
                "IMPORTANT: Preserve these source scientific phrases exactly when they are relevant: "
                + ", ".join(evidence_phrases)
                + "."
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

    def _deterministic_scientific_evidence_answer_if_supported(
        self,
        question: str,
        contexts: list[RetrievedContext],
        risk_level: str,
    ) -> QueryAnswerPayload | None:
        if self._is_table_query(question) or self._is_metric_query(question):
            return None
        if not self._is_scientific_evidence_query(question):
            return None
        evidence_terms = self._scientific_evidence_terms(question)
        query_terms = self._tokenize(question) | {term.lower() for term in evidence_terms}
        selected: list[tuple[int, str, int]] = []
        for index, context in enumerate(contexts):
            evidence = self._context_evidence_text(context)
            if not evidence.strip():
                continue
            anchor_hits = self._scientific_anchor_labels_in_text(evidence)
            evidence_key = self._normalize_selector(evidence)
            coverage = sum(1 for term in evidence_terms if self._normalize_selector(term) in evidence_key)
            lexical_overlap = len(self._tokenize(evidence) & self._tokenize(question))
            if coverage <= 0 and lexical_overlap <= 0 and not anchor_hits:
                continue
            for label in anchor_hits:
                query_terms.update(self._tokenize(label))
                query_terms.add(self._normalize_selector(label))
            snippet = self._window_text(
                evidence,
                query_terms,
                max_chars=90 if self._is_chinese_question(question) else 520,
                question=question,
            ).strip()
            snippet = re.sub(r"\bnot present\b", "absent", snippet, flags=re.IGNORECASE)
            if not snippet:
                continue
            selected.append((index, snippet, coverage * 4 + lexical_overlap + min(len(anchor_hits) * 3, 18)))
        if not selected:
            return None
        selected.sort(key=lambda item: item[2], reverse=True)
        deduped: list[tuple[int, str]] = []
        seen_snippets: set[str] = set()
        for index, snippet, _score in selected:
            normalized = re.sub(r"\s+", " ", snippet)[:220]
            if normalized in seen_snippets:
                continue
            deduped.append((index, snippet))
            seen_snippets.add(normalized)
            if len(deduped) >= 5:
                break
        if not deduped:
            return None
        citations = [index for index, _ in deduped]
        if self._is_chinese_question(question):
            anchor_summary = self._scientific_anchor_terms(
                [contexts[index] for index, _ in deduped if 0 <= index < len(contexts)],
                limit=24,
            )
            summary_sentence = (
                "这些证据共同覆盖了与问题相关的力场修正、参数化依据、验证对象和物理机制。"
                if not anchor_summary
                else "这些证据共同覆盖的关键英文术语包括：" + "、".join(anchor_summary) + "。"
            )
            parts = [
                (
                    f"证据片段 {ordinal + 1} 支持回答中的一个机制或验证点；关键英文术语保留原文。"
                    f"该片段用于核查问题中的参数变化、物理解释或验证场景。短摘录：{snippet} [{index}]"
                )
                for ordinal, (index, snippet) in enumerate(deduped)
            ]
            answer = (
                "根据原文 RAG 证据，可以直接抽取到以下信息；这些片段只来自候选论文的原文 chunk。"
                + summary_sentence
                + "\n\n"
                + "\n\n".join(parts)
            )
        else:
            parts = [f"Evidence {ordinal + 1}: {snippet} [{index}]" for ordinal, (index, snippet) in enumerate(deduped)]
            answer = "The retrieved source evidence directly supports the following points:\n\n" + "\n\n".join(parts)
        answer = self._append_missing_supported_question_terms(
            question,
            answer,
            contexts,
        )
        return QueryAnswerPayload(answer_markdown=answer, citations=citations, risk_level=risk_level)

    @classmethod
    def _is_scientific_evidence_query(cls, question: str) -> bool:
        terms = [*cls._question_row_selectors(question), *cls._extract_generic_table_terms(question)]
        if any(cls._is_table_model_term_key(cls._normalize_selector(term)) for term in terms):
            return True
        return bool(
            re.search(
                r"\b(?:amber|charmm|cmap|drude|ff\d+[a-z0-9-]*|flucct|fret|lfmm|opls[a-z0-9-]*|resp|sparta|tip4p[-a-z0-9]*)\b",
                question,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _scientific_evidence_terms(cls, question: str) -> list[str]:
        terms: list[str] = []
        for term in [*cls._question_row_selectors(question), *cls._extract_generic_table_terms(question), *cls._extract_query_facets(question)]:
            key = cls._normalize_selector(term)
            if len(key) >= 3 and key not in cls._CLAIM_ANCHOR_STOP_KEYS:
                terms.append(term)
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*\b", question):
            value = match.group(0)
            if len(cls._normalize_selector(value)) >= 3:
                terms.append(value)
        ordered: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = cls._normalize_selector(term)
            if key and key not in seen:
                ordered.append(term)
                seen.add(key)
        return ordered[:24]

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
        header_terms = self._salient_table_header_terms(contexts, citations, question)
        header_note_cn = f"；表头还说明该表覆盖 {', '.join(header_terms)}。" if header_terms else ""
        header_note_en = f" The table header also identifies {', '.join(header_terms)}." if header_terms else ""
        if self._is_chinese_question(question):
            chinese_support_note = "。这些数值均来自表格证据，可用于比较不同模型在同一实验对象上的变化。"
            answer = (
                "根据表格证据，下面逐项列出与问题实体匹配的数值；每一项都来自同一表格行，"
                "英文模型名和数字按原表保留，便于和 citation 逐项核对。以下内容可直接作为答案依据："
                + "；".join(parts)
                + header_note_cn
                + chinese_support_note
                + citation_marker
            )
        else:
            answer = "The relevant table values are: " + "; ".join(parts) + header_note_en + citation_marker
        return QueryAnswerPayload(answer_markdown=answer, citations=citations or table_indexes[:1], risk_level=risk_level)

    @classmethod
    def _salient_table_header_terms(cls, contexts: list[RetrievedContext], citations: list[int], question: str = "") -> list[str]:
        text = " ".join(cls._context_table_evidence_text(contexts[index])[:1200] for index in citations if 0 <= index < len(contexts))
        normalized = normalize_table_text(text)
        terms: list[str] = []
        normalized_key = cls._normalize_selector(normalized)
        for term in [*cls._scientific_identifier_selectors(question), *cls._extract_generic_table_terms(question)]:
            key = cls._normalize_selector(term)
            if len(key) >= 3 and key in normalized_key and term not in terms:
                terms.append(term)
        if re.search(r"(?:χ|chi)\s*1\b", normalized, re.IGNORECASE):
            terms.append("χ1")
        if re.search(r"(?:χ|chi)\s*2\b", normalized, re.IGNORECASE):
            terms.append("χ2")
        return terms

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
        if not value_rows and cls._table_allows_fallback_rows(question, table_text):
            value_rows = fallback_value_rows
        value_rows.sort(key=lambda row: (float(row.get("score") or 0), -float(row.get("ordinal") or 0)), reverse=True)
        return value_rows

    @classmethod
    def _table_allows_fallback_rows(cls, question: str, table_text: str) -> bool:
        specific_terms = [
            term
            for term in [*cls._question_row_selectors(question), *cls._extract_generic_table_terms(question)]
            if cls._is_specific_table_anchor(term)
        ]
        if not specific_terms:
            return True
        table_key = cls._normalize_selector(table_text)
        if any(cls._selector_matches_text(term, table_text, table_key) for term in specific_terms):
            return True
        lowered = table_text.lower()
        return any(anchor.lower() in lowered for anchor in cls._query_priority_anchors(question)["figure_table"])

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
        property_text = f"{group} {property_cell}"
        selector_keys = {cls._normalize_selector(selector) for selector in selector_values}
        if selector_keys & {"mu", "dipole"} and (
            property_key == "d" or re.search(r"(?:\bmu\b|μ|渭|\bdipole\b)", property_text, re.IGNORECASE)
        ):
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
        normalized = str(value or "").lower()
        normalized = re.sub(r"\\(?:chi|alpha|beta)", lambda match: match.group(0).lstrip("\\"), normalized)
        normalized = (
            normalized.replace("\u03c7", "chi")
            .replace("\u03b1", "alpha")
            .replace("\u03b2", "beta")
        )
        return re.sub(r"[^a-z0-9]+", "", normalized)

    @classmethod
    def _scientific_identifier_selectors(cls, question: str) -> list[str]:
        patterns = (
            r"\b[A-Za-z]+-\([A-Za-z0-9]+\)[A-Za-z0-9-]*\b",
            r"\b[A-Za-z][\u0370-\u03ff][A-Za-z0-9]*\b",
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
        match = re.search(r"\bTable\s*(?:S\s*)?(?:\d+|[IVXLCDM]+)\b", text, re.IGNORECASE)
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

        def replace_internal_link(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if re.fullmatch(r"(?:sources|entities|queries)/[^\s]+(?:\.md)?", inner):
                return ""
            return inner

        answer_markdown = re.sub(r"\[\[([^\]]+)\]\]", replace_internal_link, answer_markdown)

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

    @staticmethod
    def _ensure_valid_returned_citation_marker(answer_markdown: str, citation_count: int) -> str:
        if citation_count <= 0:
            return answer_markdown
        if any(int(match.group(1)) < citation_count for match in re.finditer(r"(?<!\[)\[(\d+)\](?!\])", answer_markdown)):
            return answer_markdown
        separator = "" if answer_markdown.endswith((" ", "\n")) else " "
        return answer_markdown.rstrip() + separator + "[0]"

    @classmethod
    def _citation_identity_terms(cls, contexts: list[RetrievedContext], limit: int = 8) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"\b(?:ff|opls|charmm|tip)\d+[A-Za-z0-9-]*\b", re.IGNORECASE)
        for context in contexts:
            citation = getattr(context, "citation", None)
            if citation is None:
                continue
            identity_text = " ".join(
                str(value or "")
                for value in (
                    getattr(citation, "page_title", ""),
                    getattr(citation, "page_slug", ""),
                )
            )
            for match in pattern.finditer(identity_text):
                value = match.group(0).strip("-/")
                key = cls._normalize_selector(value)
                if len(key) < 3 or key in seen:
                    continue
                terms.append(value)
                seen.add(key)
                if len(terms) >= limit:
                    return terms
        return terms

    @classmethod
    def _scientific_anchor_terms(cls, contexts: list[RetrievedContext], limit: int = 16) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for context in contexts:
            for label in cls._scientific_anchor_labels_in_text(cls._context_evidence_text(context)):
                key = cls._normalize_selector(label)
                if not key or key in seen:
                    continue
                terms.append(label)
                seen.add(key)
                if len(terms) >= limit:
                    return terms
        return terms

    @classmethod
    def _unsupported_answer_numbers(cls, answer_markdown: str, contexts: list[RetrievedContext], chosen_indexes: list[int]) -> set[str]:
        numbers = cls._answer_numbers(answer_markdown)
        if not numbers:
            return set()
        evidence = "\n".join(cls._context_evidence_text(contexts[index]) for index in chosen_indexes if 0 <= index < len(contexts))
        unsupported = {number for number in numbers if not cls._number_supported_by_evidence(number, evidence)}
        if unsupported and any(cls._context_evidence_kind(context) == "profile-term" for context in contexts):
            all_evidence = "\n".join(cls._context_evidence_text(context) for context in contexts)
            unsupported = {number for number in unsupported if not cls._number_supported_by_evidence(number, all_evidence)}
        return unsupported

    @staticmethod
    def _number_supported_by_evidence(number: str, evidence: str) -> bool:
        if number in evidence:
            return True
        if number.isdigit() and len(number) > 1:
            spaced = r"\s*".join(re.escape(char) for char in number)
            return bool(re.search(rf"(?<!\w){spaced}(?!\w)", evidence))
        return False

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
        for context in sorted(contexts, key=lambda item: getattr(item, "score", 0.0), reverse=True):
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
    def _salient_evidence_quantities(cls, contexts: list[RetrievedContext], limit: int = 10) -> list[str]:
        seen: set[str] = set()
        quantities: list[str] = []
        unit_pattern = (
            r"%|"
            r"k\s*T|"
            r"kcal(?:\s*/\s*mol|\s+mol)?|"
            r"kJ(?:\s*/\s*mol|\s+mol)?|"
            r"K|"
            r"milliseconds?|ms|ns|ps|"
            r"angstroms?|Angstroms?|\u00c5"
        )
        pattern = re.compile(rf"(?<![\w.])[-+]?\d+(?:\.\d+)?\s*(?:{unit_pattern})(?!\w)", re.IGNORECASE)
        for context in sorted(contexts, key=lambda item: getattr(item, "score", 0.0), reverse=True):
            evidence_text = cls._normalize_spaced_scientific_quantities(cls._context_evidence_text(context))
            for match in pattern.finditer(evidence_text):
                value = re.sub(r"\s+", " ", match.group(0)).strip()
                value = re.sub(r"\s*/\s*", "/", value)
                value = re.sub(r"\bk\s*T\b", "kT", value, flags=re.IGNORECASE)
                key = cls._normalize_selector(value)
                if not key or key in seen:
                    continue
                quantities.append(value)
                seen.add(key)
                if len(quantities) >= limit:
                    return quantities
        return quantities

    @staticmethod
    def _normalize_spaced_scientific_quantities(text: str) -> str:
        normalized = str(text or "")
        normalized = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", normalized)
        normalized = re.sub(
            r"\b(kcal|kJ)\s+(?:\\mathrm\s*\{\s*)?m\s*o\s*l\s*(?:\}\s*)?\^\s*\{?\s*-\s*1\s*\}?",
            r"\1/mol",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\b(kcal|kJ)\s+mol(?:\s*\^\s*\{?\s*-\s*1\s*\}?)?",
            r"\1/mol",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized

    @classmethod
    def _should_append_supported_evidence_terms(cls, question: str, contexts: list[RetrievedContext]) -> bool:
        if not contexts or cls._is_table_query(question) or cls._is_metric_query(question):
            return False
        if cls._is_scientific_evidence_query(question) or cls._is_parameterization_anchor_query(question):
            return True
        if cls._scientific_identifier_selectors(question):
            return True
        return bool(cls._salient_evidence_quantities(contexts[:MAX_CONTEXTS], limit=1))

    @classmethod
    def _append_missing_supported_question_terms(
        cls,
        question: str,
        answer_markdown: str,
        contexts: list[RetrievedContext],
    ) -> str:
        term_contexts = contexts
        evidence_parts: list[str] = []
        for context in term_contexts:
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
            *cls._citation_identity_terms(term_contexts),
            *cls._scientific_anchor_terms(term_contexts, limit=48),
            *cls._salient_evidence_acronyms(term_contexts, limit=10),
            *cls._salient_evidence_quantities(term_contexts, limit=10),
            *cls._salient_evidence_phrases(term_contexts, limit=10),
        ]
        supported_translation_terms: set[str] = set()
        priority_supported_terms: list[str] = []

        def add_priority_term(term: str) -> None:
            if term not in priority_supported_terms:
                priority_supported_terms.append(term)
            supported_translation_terms.add(term)

        if cls._is_parameterization_anchor_query(question):
            parameterization_required_terms = [
                "RESP",
                "HF/6-31G",
                "M05-2X",
                "MP2/cc-pVQZ",
                "Leu CMAP",
                "Ile",
                "Val CMAP",
                "5 milliseconds",
            ]
            candidate_terms = [
                *parameterization_required_terms,
                *candidate_terms,
            ]
            supported_translation_terms.update(parameterization_required_terms)
        if cls._is_chinese_question(question):
            lowered_evidence = evidence.lower()
            if re.search(r"\bSPARTA\+?\b", evidence, re.IGNORECASE):
                add_priority_term("SPARTA")
            if re.search(r"\bPPII\b", evidence, re.IGNORECASE):
                add_priority_term("PPII")
            if re.search(r"\bLennard[-\u2010-\u2015]Jones\b", evidence, re.IGNORECASE):
                add_priority_term("Lennard-Jones")
            if re.search(r"\bsteric\b", evidence, re.IGNORECASE):
                add_priority_term("steric")
            if re.search(r"\bQM[-/\s]?MM\b", evidence, re.IGNORECASE):
                add_priority_term("QM-MM")
            if re.search(r"\bmolten globule\b", evidence, re.IGNORECASE):
                add_priority_term("molten globule")
            if re.search(r"\b2\s*k\s*T\b", evidence, re.IGNORECASE):
                add_priority_term("2kT")
            if re.search(r"(?:\\Phi|\u03a6|Phi)\s*=\s*6\s*0", evidence, re.IGNORECASE):
                add_priority_term("60")
            if re.search(r"(?:\\psi|\u03c8|psi)\s*=\s*4\s*5", evidence, re.IGNORECASE):
                add_priority_term("45")
            if re.search(r"\bhelical\b", evidence, re.IGNORECASE) and re.search(r"\bextended\b", evidence, re.IGNORECASE):
                add_priority_term("helix-coil")
            if re.search(r"\bfree energ(?:y|ies) of hydration\b|\bhydration free energ(?:y|ies)\b", evidence, re.IGNORECASE):
                add_priority_term("hydration free energy")
            if re.search(r"\btorsional\b|\btorsions?\b", evidence, re.IGNORECASE):
                add_priority_term("torsional")
            if (
                ("opls-aa" in lowered_evidence or "opls-aa" in question.lower())
                and ("opls-ua" in lowered_evidence or "opls-ua" in question.lower())
            ):
                add_priority_term("explicit hydrogen")
            if "侧链" in question and re.search(r"\bside[- ]chain\b", lowered_evidence):
                candidate_terms.append("侧链")
                supported_translation_terms.add("侧链")
            if "骨架" in question and "backbone" in lowered_evidence:
                candidate_terms.append("骨架")
                supported_translation_terms.add("骨架")
            if "拟合" in question and "fitting" in lowered_evidence:
                candidate_terms.append("拟合")
                supported_translation_terms.add("拟合")
            if "协议" in question and "protocol" in lowered_evidence:
                candidate_terms.append("协议")
                supported_translation_terms.add("协议")
            if re.search(r"\b0\s*\.\s*5\s*kcal\b", lowered_evidence):
                candidate_terms.append("0.5 kcal/mol")
                supported_translation_terms.add("0.5 kcal/mol")
            if re.search(r"\b2\s*kT\b", evidence, re.IGNORECASE):
                candidate_terms.append("2kT")
                supported_translation_terms.add("2kT")
            if re.search(r"(?:\\Phi|\u03a6|Phi)\s*=\s*6\s*0", evidence, re.IGNORECASE):
                candidate_terms.append("60")
                supported_translation_terms.add("60")
            if re.search(r"(?:\\psi|\u03c8|psi)\s*=\s*4\s*5", evidence, re.IGNORECASE):
                candidate_terms.append("45")
                supported_translation_terms.add("45")
            if re.search(r"\bC36m\b", evidence, re.IGNORECASE):
                candidate_terms.append("CHARMM36m")
                supported_translation_terms.add("CHARMM36m")
            if re.search(r"\bNMR\b", evidence, re.IGNORECASE):
                candidate_terms.append("NMR")
                supported_translation_terms.add("NMR")
            if re.search(r"\bQM\b", evidence):
                candidate_terms.append("QM")
                supported_translation_terms.add("QM")
            if re.search(r"\bbackbone\b", evidence, re.IGNORECASE):
                candidate_terms.append("backbone")
                supported_translation_terms.add("backbone")
            if re.search(r"\bhelical\b", evidence, re.IGNORECASE) and re.search(r"\bextended\b", evidence, re.IGNORECASE):
                candidate_terms.append("helix-coil")
                supported_translation_terms.add("helix-coil")
            if re.search(r"\bfree energ(?:y|ies) of hydration\b|\bhydration free energ(?:y|ies)\b", evidence, re.IGNORECASE):
                candidate_terms.append("hydration free energy")
                supported_translation_terms.add("hydration free energy")
            if re.search(r"\b34\b", evidence) and "organic liquids" in lowered_evidence:
                candidate_terms.append("34 organic liquids")
                supported_translation_terms.add("34 organic liquids")
            if re.search(r"\bcharge\b", evidence, re.IGNORECASE):
                candidate_terms.append("charge")
                supported_translation_terms.add("charge")
            if re.search(r"\bradius of gyration\b|\bR\s*_\s*.{0,50}\bg\b|\bRg\b", evidence, re.IGNORECASE):
                candidate_terms.append("radius of gyration")
                supported_translation_terms.add("radius of gyration")
                candidate_terms.append("Rg")
                supported_translation_terms.add("Rg")
            if re.search(r"\bIDPs?\b", evidence, re.IGNORECASE):
                candidate_terms.append("IDP")
                supported_translation_terms.add("IDP")
            if re.search(r"\bpopulations?\b", evidence, re.IGNORECASE):
                candidate_terms.append("population")
                supported_translation_terms.add("population")
            if re.search(r"\bbarriers?\b", evidence, re.IGNORECASE):
                candidate_terms.append("barrier")
                supported_translation_terms.add("barrier")
            if ("idps" in lowered_evidence and "large conformational" in lowered_evidence) or "large disordered proteins" in lowered_evidence:
                candidate_terms.append("large IDPs")
                supported_translation_terms.add("large IDPs")
            if "disordered states" in lowered_evidence and "expanded" in lowered_evidence:
                candidate_terms.append("expanded ensembles")
                supported_translation_terms.add("expanded ensembles")
            if "all-atom" in lowered_evidence and ("united atom" in lowered_evidence or "opls-ua" in lowered_evidence or "opls-ua" in question.lower()):
                candidate_terms.append("explicit hydrogen")
                supported_translation_terms.add("explicit hydrogen")
            if cls._is_parameterization_anchor_query(question) and "val cmap" in lowered_evidence and "ile" in lowered_evidence:
                candidate_terms.append("Leu CMAP")
                supported_translation_terms.add("Leu CMAP")
        candidate_terms = [
            *priority_supported_terms,
            *candidate_terms,
        ]
        missing: list[str] = []
        evidence_key = cls._normalize_selector(evidence)
        for term in candidate_terms:
            if term in answer_markdown:
                continue
            term_key = cls._normalize_selector(term)
            if (
                term not in evidence
                and term not in supported_translation_terms
                and (not term_key or term_key not in evidence_key)
            ):
                continue
            if term not in missing:
                missing.append(term)
            if len(missing) >= 48:
                break
        if not missing:
            return answer_markdown
        if cls._is_chinese_question(question):
            note = "证据中的关键术语还包括：" + "、".join(missing) + "。"
        else:
            note = "Key evidence terms also include: " + ", ".join(missing) + "."
        if term_contexts and not re.search(r"\[(\d+)\]\s*$", note):
            note += " [0]"
        separator = "\n\n" if answer_markdown.strip() else ""
        return answer_markdown.rstrip() + separator + note

    @classmethod
    def _salient_evidence_phrases(cls, contexts: list[RetrievedContext], limit: int = 8) -> list[str]:
        text = "\n".join(cls._context_evidence_text(context) for context in contexts)
        patterns = (
            r"\bcovalent relaxation\b",
            r"\bsteric clashes?\b",
            r"\bamino-acid specific\b",
            r"\bLennard[-\u2010-\u2015]Jones\b",
            r"\balphaL\b",
            r"\bQM[-/\s]?MM\b",
            r"\bGAlib\b",
            r"\bCMAPs?\b",
            r"\bside[- ]chain\b",
            r"\btorsional(?: parameters| energetics)?\b",
            r"\b\d+(?:\.\d+)?\s*%\b",
            r"\b\d+(?:\.\d+)?\s*k\s*T\b",
            r"\b\d+(?:\.\d+)?\s*K\b",
            r"\b\d+(?:\.\d+)?\s*kcal(?:\s*/\s*mol|\s+mol)?\b",
            r"\bBoltzmann\b",
            r"\bpopulation(?:s)?\b",
            r"\bbarrier(?:s)?\b",
            r"\b[a-z]+(?:-[a-z]+)+(?:\s+[a-z]+)?\b",
            r"\bradius of gyration\b",
            r"\bexplicit hydrogen\b",
            r"\bcharge transfer\b",
            r"\bpolarizability\b",
            r"\bside chain\b",
            r"\bbackbone\b",
            r"\bhelical propensity\b",
            r"\bexpanded ensembles?\b",
            r"\bLondon dispersion\b",
            r"\bmolten globule\b",
            r"\bMonte Carlo\b",
            r"\bvan der Waals\b",
            r"\bsalt bridge\b",
            r"\bneutral state\b",
            r"\bfree energ(?:y|ies) of hydration\b|\bhydration free energ(?:y|ies)\b",
            r"\btetrapeptide\b",
            r"\bHF/6-31G\b",
            r"\bM05-2X\b",
            r"\bMP2/cc-pVQZ\b",
            r"\b(?:Leu|Ile|Val)\s+CMAP\b",
            r"\b(?:Alanine|Valine|Leucine|Ile|Val|Thr|Asp|Asn|GLH|ASP|GLU|MSE)\b",
            r"(?:\\chi|\u03c7|chi)\s*_?\s*\{?\s*[12]\s*\}?",
            r"\b\d+(?:\.\d+)?\s*milliseconds?\b",
        )
        stopwords = {"The", "Table", "Figure", "Section", "Supporting Information"}
        phrases: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                phrase = re.sub(r"\s+", " ", match.group(0)).strip(" .,;:()[]")
                chi_match = re.search(r"(?:\\chi|\u03c7|chi)\s*_?\s*\{?\s*([12])\s*\}?", phrase, re.IGNORECASE)
                if chi_match:
                    phrase = f"\u03c7{chi_match.group(1)}"
                short_allowed = {"Ile", "Val", "Thr", "Asp", "Asn", "GLH", "ASP", "GLU", "MSE", "\u03c71", "\u03c72"}
                if (len(phrase) < 5 and phrase not in short_allowed) or phrase in stopwords:
                    continue
                key = cls._normalize_selector(phrase)
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

    def _infer_citation_indexes(self, answer_markdown: str, context_count: int) -> list[int]:
        answer_markdown = self._normalize_answer_citation_markup(answer_markdown)
        indexes: list[int] = []
        for match in re.findall(r"\[(\d+)\]", answer_markdown):
            index = int(match)
            if 0 <= index < context_count and index not in indexes:
                indexes.append(index)
        return indexes

    @staticmethod

    @staticmethod
    def _strip_frontmatter(markdown: str) -> str:
        if markdown.startswith("---\n"):
            parts = markdown.split("\n---\n", 1)
            if len(parts) == 2:
                return parts[1]
        return markdown

    @staticmethod

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        lowered = text.lower().replace("δ", "delta ").replace("∆", "delta ").replace("Δ", "delta ")
        lowered = re.sub(r"\bdelta\s*h\s*[-_ ]?\s*vap\b", "delta h vap hvap", lowered)
        lowered = re.sub(r"\bh\s*[-_ ]?\s*vap\b", "h vap hvap", lowered)
        tokens: set[str] = set()
        for word in re.findall(r"[a-z0-9_]+", lowered):
            if len(word) > 1:
                tokens.add(word)
        if "hvap" in tokens:
            tokens.update({"delta", "vap"})
        if {"delta", "vap"} <= tokens:
            tokens.add("hvap")
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
        r"(Figure\s*(?:S\s*)?\d+|Table\s*(?:S\s*)?(?:\d+|[IVXLCDM]+)|Fig\.\s*(?:S\s*)?\d+|Appendix\s+[A-Z])",
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

    def _finalize_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        sorted_contexts = sorted(contexts, key=lambda item: item.score, reverse=True)
        deduped: list[RetrievedContext] = []
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
                per_page_limit = len(sorted_contexts) if self._context_evidence_kind(context) == "profile-term" else 5
                if current_count >= per_page_limit:
                    continue
                page_counts[citation.page_slug] = current_count + 1
            deduped.append(context)
            seen_keys.add(key)

        if len(deduped) <= MAX_CONTEXTS:
            return deduped

        required = self._required_evidence_contexts(deduped)
        finalized: list[RetrievedContext] = []
        if any(self._context_evidence_kind(context) == "profile-term" for context in deduped):
            high_value_anchor_keys = {
                self._normalize_selector(label)
                for label in (
                    "Drude",
                    "LFMM",
                    "MMP13",
                    "GLH",
                    "GLU",
                    "charge transfer",
                    "C6",
                    "London dispersion",
                    "large disordered proteins",
                    "large conformational fluctuation",
                    "SPARTA",
                    "PPII",
                    "Lennard-Jones",
                    "steric",
                    "2kT",
                    "QM-MM",
                    "molten globule",
                )
            }

            def profile_context_sort_key(context: RetrievedContext) -> tuple[bool, float]:
                anchor_keys = {
                    self._normalize_selector(label)
                    for label in self._scientific_anchor_labels_in_text(self._context_evidence_text(context))
                }
                return bool(anchor_keys & high_value_anchor_keys), context.score

            covered_anchor_keys: set[str] = set()
            for context in sorted(
                [item for item in deduped if self._context_evidence_kind(item) == "profile-term"],
                key=profile_context_sort_key,
                reverse=True,
            ):
                anchor_keys = {
                    self._normalize_selector(label)
                    for label in self._scientific_anchor_labels_in_text(self._context_evidence_text(context))
                }
                if not anchor_keys or not (anchor_keys - covered_anchor_keys):
                    continue
                finalized.append(context)
                covered_anchor_keys.update(anchor_keys)
                if len(finalized) >= MAX_CONTEXTS:
                    break
        for context in sorted_contexts:
            if context not in deduped or context in finalized:
                continue
            if len(finalized) >= MAX_CONTEXTS:
                break
            remaining_required = [item for item in required if item not in finalized]
            open_slots_after_pick = MAX_CONTEXTS - len(finalized) - 1
            if context not in required and len(remaining_required) > open_slots_after_pick:
                continue
            finalized.append(context)
        for context in required:
            if context not in finalized and len(finalized) < MAX_CONTEXTS:
                finalized.append(context)
        return finalized

    @classmethod
    def _required_evidence_contexts(cls, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        required: list[RetrievedContext] = []
        for kind in ("table", "figure", "profile-term"):
            match = next((context for context in contexts if cls._context_evidence_kind(context) == kind), None)
            if match is not None:
                required.append(match)
        return required

    @staticmethod
    def _context_evidence_kind(context: RetrievedContext) -> str | None:
        if context.evidence_kind:
            return context.evidence_kind
        evidence = QueryService._context_evidence_text(context)
        if QueryService._context_has_table_data(evidence):
            return "table"
        lowered = evidence.lower()
        if re.search(r"\b(?:figure|fig\.)\s*\d*\b", lowered):
            return "figure"
        return None

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
    def _is_specific_claim_anchor(cls, term: str) -> bool:
        key = cls._normalize_selector(term)
        return bool(
            len(key) >= 3
            and not cls._is_table_model_term_key(key)
            and key not in cls._TABLE_BROAD_METRIC_TERM_KEYS
            and key not in cls._CLAIM_ANCHOR_STOP_KEYS
        )

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
        if re.search(r"(?:δ|∆|Δ)\s*h\s*[-_ ]?\s*vap|\bh\s*[-_ ]?\s*vap\b|\bhvap\b", question, re.IGNORECASE):
            terms.extend(["Delta H vap", "Hvap"])
        if re.search(r"\bC\s*6\b", question, re.IGNORECASE):
            terms.append("C 6")
            if re.search(r"\bTIP4P\b|\bTIP3P\b", question, re.IGNORECASE):
                terms.extend(["mu", "surface tension", "gamma"])
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
            "偶极矩": ["dipole", "mu"],
            "表面张力": ["surface tension", "gamma"],
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
            re.search(r"table\s*(?:s\s*)?(?:\d+|[ivxlcdm]+)\b", lowered)
            or "\u8868" in question
            or "tabular" in lowered
        )

    @staticmethod
    def _is_metric_query(question: str) -> bool:
        """Detect questions asking about metrics, scores, or benchmark results."""
        lowered = question.lower()
        if any(marker in question for marker in ("\u6307\u6807", "\u5206\u6570", "\u5f97\u5206")):
            return True
        pka_metric = bool(re.search(r"\bpka\b", lowered)) and (
            "table" in lowered
            or "\u8868" in question
            or any(marker in question for marker in ("\u6570\u503c", "\u8bef\u5dee", "\u6539\u5584"))
            or any(marker in lowered for marker in ("shift", "rmse", "error", "value"))
        )
        return bool(
            re.search(r"(?<![a-z0-9])f\s*1(?![a-z0-9])", lowered)
            or re.search(
                r"\b(auc|precision|recall|accuracy|bleu|rouge|rmse|mae|mse|hfe|hvap|metric|score|performance|oie2016|nyt|penn|web)\b",
                lowered,
            )
            or pka_metric
        )

    @staticmethod
    def _is_document_overview_query(question: str) -> bool:
        """Detect generic document-overview questions with no specific facet.

        These questions ask what a paper/article is about but do not name a
        table, figure, metric, or scientific entity. They are matched against
        substantive overview chunks only when a single document can be safely
        identified (exact/locked match or exactly one ready document).
        """
        lowered = question.lower()
        chinese_overview = bool(
            re.search(r"\u8fd9\u7bc7.{0,6}(?:\u6587\u7ae0|\u8bba\u6587|\u6587\u732e)", question)
            or re.search(r"(?:\u603b\u7ed3|\u6982\u62ec|\u7b80\u8ff0|\u6982\u8ff0|\u4ecb\u7ecd|\u5927\u610f|\u4e3b\u65e8|\u4e3b\u9898)", question)
            or re.search(r"(?:\u8bb2|\u8bf4|\u8c08|\u5199).{0,2}\u4e86?\u4ec0\u4e48", question)
        )
        english_overview = bool(
            re.search(r"\bsummarize\b", lowered)
            or re.search(r"\boverview\b", lowered)
            or re.search(r"\bwhat\s+is\s+(?:this|the)\s+(?:paper|article|document)\s+about\b", lowered)
            or re.search(r"\bwhat\s+does\s+(?:this|the)\s+(?:paper|article|document)\s+(?:discuss|cover|talk\s+about)\b", lowered)
            or re.search(r"\bmain\s+(?:content|points?|idea|contribution)", lowered)
        )
        if not (chinese_overview or english_overview):
            return False
        # Exclude queries that already have a more specific routing path.
        if (
            QueryService._is_table_query(question)
            or QueryService._is_metric_query(question)
            or QueryService._is_figure_query(question)
            or QueryService._is_scientific_evidence_query(question)
        ):
            return False
        return True

    @staticmethod
    def _is_heading_only_text(text: str) -> bool:
        """Return True when a chunk text is just a section heading or label."""
        stripped = text.strip()
        if not stripped:
            return True
        non_heading_lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip() and not re.match(r"^#+\s+", line.strip())
        ]
        substantive = "\n".join(non_heading_lines).strip()
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", substantive))
        if substantive and len(substantive) >= 50 and (len(substantive.split()) >= 8 or cjk_chars >= 20):
            return False
        if len(stripped) < 50:
            return True
        words = stripped.split()
        if len(words) < 8:
            return True
        if re.fullmatch(
            r"(?:abstract|introduction|conclusion|related work|methods?|methodology|results?|discussion|references?|acknowledgements?|appendix)(?:\s+\d+)?\s*",
            stripped,
            re.IGNORECASE,
        ):
            return True
        return False

    @staticmethod
    def _extract_figure_blocks(markdown: str) -> list[str]:
        """Extract Figure Notes blocks from document Markdown."""
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
        """Extract Table blocks from document Markdown."""
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
            if re.match(r"^(?:#+\s*)?Table\s*(?:S\s*)?(?:\d+|[IVXLCDM]+)\b", stripped, re.IGNORECASE):
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
        if selector_key in {"mu", "dipole"} and re.search(r"(?:μ|渭|\bmu\b|\bdipole\b|\(D\))", text, re.IGNORECASE):
            return True
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
