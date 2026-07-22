from __future__ import annotations

import logging
import json
import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import (
    Claim,
    Document,
    DocumentChunk,
    DocumentStatus,
    Entity,
    PipelineRun,
    Project,
    ReviewItem,
    ReviewSeverity,
    ReviewStatus,
    RunStatus,
    RunType,
)
from app.services.ai import (
    DocumentAnalysisPayload,
    DocumentExtraction,
    ExternalVerifier,
    ExtractedClaim,
    ExtractedEntity,
    GeneratedTriple,
    GrowthDecision,
    GrowthDecisionPayload,
    HeadAnalysisPayload,
    OllamaClient,
    cosine_similarity,
    safe_model_call,
)
from app.services.filesystem import compute_sha256, display_title_from_path, looks_like_internal_sample, readable_title_from_path, slugify, strip_upload_prefix
from app.services.paper_profile import ensure_paper_profile, ensure_source_identity
from app.services.parser import parse_document
from app.services.repositories import get_or_create_project
from app.services.storage import ObjectStorage
from app.services.vector_store import ChunkVector, get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

FACT_MARKERS = (
    "建议",
    "复查",
    "随访",
    "诊断",
    "用药",
    "剂量",
    "治疗",
    "检查",
    "结果",
    "结论",
    "风险",
    "时间",
    "recommended",
    "recommendation",
    "follow-up",
    "follow up",
    "diagnosis",
    "medication",
    "dose",
    "treatment",
    "result",
    "conclusion",
    "risk",
)
FACT_VALUE_PATTERN = re.compile(
    r"(\d+\s*个?\s*(天|周|月|年|小时|分钟|mg|g|ml|%|次|days?|weeks?|months?|years?|hours?)|"
    r"[一二三四五六七八九十两]+个?(天|周|月|年|小时|分钟))",
    re.IGNORECASE,
)
GROW_ENTITY_TYPES = {
    "condition",
    "component",
    "concept",
    "disease",
    "drug",
    "entity",
    "institution",
    "method",
    "model",
    "module",
    "organization",
    "person",
    "procedure",
    "test",
    "therapy",
    "treatment",
}
PRUNE_ENTITY_TYPES = {"date", "time", "dose", "dosage", "measurement", "number", "value"}
GENERIC_TRIPLE_EXAMPLES = [
    {"subject": "Disease", "predicate": "requires_follow_up", "object_text": "scheduled monitoring"},
    {"subject": "Medication", "predicate": "has_dosage", "object_text": "specific dose guidance"},
    {"subject": "Treatment Plan", "predicate": "includes", "object_text": "follow-up examination"},
    {"subject": "Clinical Finding", "predicate": "supports", "object_text": "diagnosis"},
    {"subject": "Project", "predicate": "documents", "object_text": "key recommendation"},
]
HEAD_MAX_COUNT = 10
HEAD_REPROMPT_ERROR_THRESHOLD = 3
HEAD_SNIPPET_LIMIT = 6
HEAD_CONTEXT_CHAR_BUDGET = 3600


class IngestionPipeline:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.ollama = OllamaClient()
        self.verifier = ExternalVerifier()
        self.storage = ObjectStorage()

    def register_document(self, project_slug: str, project_name: str, file_path: Path) -> tuple[Project, Document, PipelineRun]:
        project = get_or_create_project(self.db, slug=project_slug, name=project_name)
        sha256 = compute_sha256(file_path)
        existing = self.db.scalar(select(Document).where(Document.project_id == project.id, Document.sha256 == sha256))
        if existing:
            if existing.status == DocumentStatus.ready.value:
                run = PipelineRun(project_id=project.id, document_id=existing.id, run_type=RunType.ingest.value, status=RunStatus.completed.value, notes="Duplicate document skipped.")
                self.db.add(run)
                self.db.commit()
                self.db.refresh(run)
                return project, existing, run

            previous_status = existing.status
            existing.raw_path = str(file_path)
            existing.object_key = self.storage.upload(file_path, f"{project.slug}/{file_path.name}")
            existing.status = DocumentStatus.pending.value
            metadata = dict(existing.metadata_json or {})
            metadata["stored_file_name"] = file_path.name
            metadata["retry_reason"] = f"Duplicate upload requeued from status '{previous_status}'."
            existing.metadata_json = metadata
            run = PipelineRun(
                project_id=project.id,
                document_id=existing.id,
                run_type=RunType.ingest.value,
                status=RunStatus.queued.value,
                notes="Duplicate upload requeued because previous ingest was not ready.",
            )
            self.db.add(run)
            self.db.commit()
            self.db.refresh(existing)
            self.db.refresh(run)
            return project, existing, run

        object_key = self.storage.upload(file_path, f"{project.slug}/{file_path.name}")
        document = Document(
            project_id=project.id,
            title=display_title_from_path(file_path),
            file_name=strip_upload_prefix(file_path.name),
            sha256=sha256,
            raw_path=str(file_path),
            object_key=object_key,
            metadata_json={"stored_file_name": file_path.name},
            status=DocumentStatus.pending.value,
        )
        self.db.add(document)
        self.db.flush()

        run = PipelineRun(project_id=project.id, document_id=document.id, run_type=RunType.ingest.value, status=RunStatus.queued.value)
        self.db.add(run)
        self.db.commit()
        self.db.refresh(document)
        self.db.refresh(run)
        return project, document, run

    def process_document(self, document_id: str) -> PipelineRun:
        document = self.db.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document {document_id} not found")
        run = self.db.scalar(select(PipelineRun).where(PipelineRun.document_id == document_id).order_by(PipelineRun.created_at.desc()))
        if run is None:
            run = PipelineRun(project_id=document.project_id, document_id=document.id, run_type=RunType.ingest.value, status=RunStatus.queued.value)
            self.db.add(run)
            self.db.flush()

        try:
            run.status = RunStatus.running.value
            document.status = DocumentStatus.processing.value
            self._set_progress(run, 5, "started", "Worker accepted the ingest job.")
            self.db.commit()

            self._set_progress(run, 15, "parsing", "Parsing document and building page/snippet structure.")
            parsed = parse_document(Path(document.raw_path))
            document.title = self._resolve_document_title(parsed, document)
            document.raw_text = parsed.text
            merged_metadata = dict(document.metadata_json or {})
            merged_metadata.update(parsed.metadata)
            document.metadata_json = merged_metadata
            ensure_source_identity(document, document.title)
            ensure_paper_profile(document)
            merged_metadata = dict(document.metadata_json or {})
            canonical_metadata = parsed.metadata.get("canonical", {})
            canonical_quality = canonical_metadata.get("quality", {})
            canonical_quality_status = canonical_quality.get("status")
            canonical_quality_accepted = canonical_quality.get("accepted") is True
            activation_allowed = canonical_metadata.get(
                "table_activation_allowed", True
            )
            quality_accepted = (
                canonical_quality_accepted
                and canonical_quality_status
                in {"accepted", "accepted_with_warnings"}
                and activation_allowed is not False
            )
            quality_rejected = not quality_accepted
            if quality_accepted:
                quality_report_status = "ok"
            elif canonical_quality_status not in {
                "accepted",
                "accepted_with_warnings",
                None,
            }:
                quality_report_status = canonical_quality_status
            else:
                quality_report_status = "validation_failed"
            quality_report = {
                "status": quality_report_status,
                "document_id": document.id,
                "canonical_status": canonical_metadata.get("status"),
                "canonical_quality": canonical_quality,
                "table_activation_allowed": activation_allowed,
                "table_repair_requests": canonical_metadata.get(
                    "table_repair_requests", []
                ),
            }
            merged_metadata["ingest_quality"] = quality_report
            if quality_rejected:
                error = (
                    "Canonical structured evidence validation failed; "
                    "document activation was blocked before chunk persistence."
                )
                merged_metadata["ingest_error"] = error
                document.metadata_json = merged_metadata
                document.status = DocumentStatus.failed.value
                run.status = RunStatus.failed.value
                run.notes = error
                run.provider_report = {
                    **dict(run.provider_report or {}),
                    "ingest_quality": quality_report,
                    "error": error,
                }
                self._set_progress(run, 100, "failed", error)
                self.db.commit()
                return run
            document.metadata_json = merged_metadata
            self._set_progress(run, 30, "chunking", "Replacing document chunks and preparing embeddings.")
            self._replace_chunks(document, parsed.chunks)

            if not settings.sac_kg_enabled:
                self._set_progress(run, 82, "indexing_rag", "Skipping SAC-KG extraction; RAG chunks and vector index are ready.")
                document.status = DocumentStatus.ready.value
                run.status = RunStatus.completed.value
                run.provider_report = {
                    **dict(run.provider_report or {}),
                    "entities": 0,
                    "claims": 0,
                    "review_items": 0,
                    "ingest_quality": quality_report,
                    "sac_kg_enabled": False,
                }
                self._set_progress(run, 100, "completed", "RAG-only ingest completed successfully.")
                self.db.commit()
                return run

            self._set_progress(run, 45, "extracting", "Generating SAC-KG routing facts with Ollama.")
            extraction = self._extract_document(document, parsed.text)
            ensure_source_identity(document, extraction.title or document.title)
            ensure_paper_profile(document)
            self._set_progress(run, 65, "structuring", "Writing entities, claims, verifier metadata, and pruner decisions.")
            entities = self._upsert_entities(document.project_id, extraction)
            claims = self._create_claims(document, extraction)
            entity_decisions = self._decide_entity_growth(document, entities, claims)
            self._apply_claim_growth_decisions(claims, entity_decisions)
            entities = self._ensure_growing_entities(document.project_id, entities, claims, entity_decisions)
            self._set_progress(run, 82, "indexing_rag", "RAG and SAC-KG artifacts are ready.")
            self._set_progress(run, 94, "reviewing", "Creating review items and final provider report.")
            review_count = self._create_review_items(document, extraction, claims)

            document.status = DocumentStatus.ready.value
            run.status = RunStatus.completed.value
            run.provider_report = {
                **dict(run.provider_report or {}),
                "entities": len(entities),
                "claims": len(claims),
                "review_items": review_count,
                "ingest_quality": quality_report,
            }
            self._set_progress(run, 100, "completed", "Ingest completed successfully.")
            self.db.commit()
            return run
        except Exception as exc:  # noqa: BLE001
            logger.exception("Document processing failed")
            document_id = document.id
            run_id = run.id
            self.db.rollback()
            failed_document = self.db.get(Document, document_id)
            failed_run = self.db.get(PipelineRun, run_id)
            if failed_document is not None:
                self._clear_document_chunks(failed_document.id)
                failed_document.status = DocumentStatus.failed.value
            if failed_run is not None:
                failed_run.status = RunStatus.failed.value
                failed_run.notes = str(exc)
                self._set_progress(failed_run, 100, "failed", str(exc))
            self.db.commit()
            raise

    def _resolve_document_title(self, parsed, document: Document) -> str:
        """Choose the best display title for a document.

        Preference order:
        1. A non-empty title supplied by the parser (e.g. MinerU or metadata).
        2. A title found in the parsed metadata.
        3. The existing sanitized filename, unless it is just an internal
           UUID/hash sample name.
        4. A generic fallback.

        Internal UUID/hash/sample identifiers are never promoted to the
        primary display title when a human-readable alternative exists.
        """
        candidates: list[str] = []
        if parsed.title and not looks_like_internal_sample(parsed.title):
            candidates.append(parsed.title.strip())
        metadata_title = (parsed.metadata or {}).get("title")
        if metadata_title and not looks_like_internal_sample(metadata_title):
            candidates.append(str(metadata_title).strip())
        existing_title = document.title or readable_title_from_path(Path(document.raw_path))
        if existing_title and not looks_like_internal_sample(existing_title):
            candidates.append(existing_title.strip())
        if candidates:
            return candidates[0]
        # Final fallback: anything we have, stripped of the upload UUID prefix.
        fallback = strip_upload_prefix(
            document.title or document.file_name or Path(document.raw_path).stem or ""
        ).strip()
        if fallback and not looks_like_internal_sample(fallback):
            return fallback
        return "Untitled document"

    def _set_progress(self, run: PipelineRun, percent: int, stage: str, message: str) -> None:
        report = dict(run.provider_report or {})
        report["progress"] = {
            "percent": max(0, min(percent, 100)),
            "stage": stage,
            "message": message,
        }
        run.provider_report = report
        logger.info(
            "[ingest progress] %s %s%% %s document_id=%s run_id=%s - %s",
            self._progress_bar(percent),
            max(0, min(percent, 100)),
            stage,
            run.document_id or "-",
            run.id,
            message,
        )
        self.db.commit()

    @staticmethod
    def _progress_bar(percent: int, width: int = 24) -> str:
        normalized = max(0, min(percent, 100))
        filled = round(width * normalized / 100)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _replace_chunks(self, document: Document, parsed_chunks) -> None:
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
        texts = [chunk.text for chunk in parsed_chunks]
        embeddings = safe_model_call(lambda: self.ollama.embed(texts), [[] for _ in texts])
        records: list[DocumentChunk] = []
        for chunk, embedding in zip(parsed_chunks, embeddings, strict=False):
            record = DocumentChunk(
                document_id=document.id,
                ordinal=chunk.ordinal,
                heading=chunk.heading,
                page_label=chunk.page_label,
                text=chunk.text,
                token_estimate=max(1, len(chunk.text) // 4),
                embedding=embedding or None,
            )
            self.db.add(record)
            records.append(record)
        self.db.flush()
        get_vector_store(self.db).replace_document_chunks(
            document.id,
            [
                ChunkVector(chunk_id=record.id, document_id=document.id, embedding=record.embedding or [])
                for record in records
                if record.embedding
            ],
        )

    def _clear_document_chunks(self, document_id: str) -> None:
        get_vector_store(self.db).delete_document(document_id)
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

    def _extract_document(self, document: Document, full_text: str) -> DocumentExtraction:
        chunks = self.db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.ordinal)).all()
        contexts = self._select_generation_contexts(document, full_text, list(chunks))
        sentence_entries = self._build_sentence_entries(list(chunks))
        corpus_context = ""
        seed_analysis = self._seed_document_analysis(document, full_text, contexts, corpus_context)
        candidate_heads = self._collect_candidate_heads(document, full_text, seed_analysis)
        previous_claims = self._project_verified_claims(document.project_id)

        head_payloads: list[HeadAnalysisPayload] = []
        for head in candidate_heads[:HEAD_MAX_COUNT]:
            head_contexts = self._retrieve_head_contexts(head, sentence_entries)
            examples = self._open_kg_examples(document.project_id, head["name"])
            head_payloads.append(
                self._generate_head_analysis(
                    document=document,
                    head=head,
                    contexts=head_contexts,
                    open_kg_examples=examples,
                    corpus_context=corpus_context,
                    previous_claims=previous_claims,
                )
            )

        extraction = self._merge_document_analysis(document, seed_analysis, head_payloads)
        if extraction.claims:
            return extraction

        fallback = self._analysis_to_extraction(document, self._fallback_analysis(document, full_text, contexts))
        fallback.coverage_notes.append("Head-driven generation produced no verified triples; fallback extraction used.")
        return fallback

    def _seed_document_analysis(
        self,
        document: Document,
        full_text: str,
        contexts: list[dict],
        corpus_context: str,
    ) -> DocumentAnalysisPayload:
        fallback = self._fallback_analysis(document, full_text, contexts)
        prompt = "\n\n".join(
            [
                f"Document title: {document.title}",
                "Task: extract a document overview for RAG routing and SAC-KG-style evidence organization.",
                (
                    "Return a concise summary, key facts, entities, concepts, and draft triples. "
                    "Preserve exact dates, doses, diagnoses, and recommendations."
                ),
                "Current corpus context:\n" + (corpus_context or "No existing corpus context."),
                "Retrieved document contexts:\n" + json.dumps(contexts, ensure_ascii=False),
            ]
        )
        return safe_model_call(
            lambda: self.ollama.generate_structured(
                DocumentAnalysisPayload,
                system_prompt=(
                    "You are preparing a structured overview for an internal document knowledge base. "
                    "Focus on candidate heads, key facts, and entity-rich summaries."
                ),
                user_prompt=prompt,
                model=settings.ollama_batch_model,
            ),
            fallback,
        )

    def _collect_candidate_heads(self, document: Document, full_text: str, seed_analysis: DocumentAnalysisPayload) -> list[dict]:
        candidates: list[dict] = []
        seen: set[str] = set()

        def add_candidate(name: str, *, entity_type: str = "concept", aliases: list[str] | None = None, summary: str = "") -> None:
            cleaned = re.sub(r"\s+", " ", name).strip()
            if not cleaned or self._looks_like_transient_value(cleaned):
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                {
                    "name": cleaned,
                    "entity_type": entity_type or "concept",
                    "aliases": aliases or [],
                    "summary": summary,
                }
            )

        add_candidate(document.title, entity_type="document")
        for entity in seed_analysis.entities:
            add_candidate(entity.name, entity_type=entity.entity_type, aliases=entity.aliases, summary=entity.summary)
        for concept in seed_analysis.concepts:
            add_candidate(concept, entity_type="concept")
        for subject in self._project_verified_subjects(document.project_id):
            if subject and subject in full_text:
                add_candidate(subject, entity_type="concept")
        if len(candidates) == 1 and seed_analysis.key_facts:
            for fact in seed_analysis.key_facts[:3]:
                for segment in re.findall(r"[\u4e00-\u9fff]{2,12}|[A-Z][a-zA-Z0-9_-]{2,}", fact):
                    add_candidate(segment, entity_type="concept")
        return candidates or [{"name": document.title, "entity_type": "document", "aliases": [], "summary": ""}]

    def _build_sentence_entries(self, chunks: list[DocumentChunk]) -> list[dict]:
        entries: list[dict] = []
        for chunk in chunks:
            for sentence_index, snippet in enumerate(self._split_text_into_snippets(chunk.text)):
                entries.append(
                    {
                        "chunk_id": chunk.id,
                        "chunk_ordinal": chunk.ordinal,
                        "page_label": chunk.page_label,
                        "heading": chunk.heading,
                        "text": snippet,
                        "score": 0.0,
                        "sentence_ref": f"chunk-{chunk.ordinal}:sentence-{sentence_index}",
                        "embedding": chunk.embedding,
                    }
                )
        return entries

    def _split_text_into_snippets(self, text: str, max_chars: int = 320) -> list[str]:
        raw_parts = [part.strip() for part in re.split(r"(?<=[。！？!?\.])\s+|\n+", text) if part.strip()]
        snippets: list[str] = []
        buffer = ""
        for part in raw_parts:
            candidate = f"{buffer} {part}".strip() if buffer else part
            if len(candidate) > max_chars and buffer:
                snippets.append(buffer)
                buffer = part
            else:
                buffer = candidate
        if buffer:
            snippets.append(buffer)
        return snippets or [text[:max_chars]]

    def _retrieve_head_contexts(self, head: dict, sentence_entries: list[dict]) -> list[dict]:
        if not sentence_entries:
            return []
        head_terms = self._text_terms(head["name"])
        alias_terms = set().union(*(self._text_terms(alias) for alias in head.get("aliases", [])))
        query_vector = safe_model_call(lambda: self.ollama.embed([head["name"]])[0], [])
        scored: list[dict] = []
        for entry in sentence_entries:
            text = entry["text"]
            lowered = text.lower()
            exact_score = lowered.count(head["name"].lower()) * 6 if head["name"] else 0
            alias_score = sum(lowered.count(alias.lower()) for alias in head.get("aliases", []) if alias) * 4
            overlap_score = len(head_terms & self._text_terms(text)) * 2
            marker_score = sum(1 for marker in FACT_MARKERS if marker.lower() in lowered)
            value_score = 1 if FACT_VALUE_PATTERN.search(text) else 0
            heading_score = 2 if entry.get("heading") and head_terms & self._text_terms(entry["heading"]) else 0
            semantic_score = 0.0
            if query_vector and entry.get("embedding"):
                semantic_score = max(cosine_similarity(query_vector, entry["embedding"]), 0.0) * 4
            score = exact_score + alias_score + overlap_score + len(alias_terms & self._text_terms(text)) + marker_score + value_score + heading_score + semantic_score
            if score <= 0:
                continue
            scored.append({**entry, "score": round(score, 4)})
        if not scored:
            return [self._prompt_context_entry(entry) for entry in sentence_entries[: min(3, len(sentence_entries))]]
        ordered = sorted(scored, key=lambda item: item["score"], reverse=True)
        selected: list[dict] = []
        total_chars = 0
        for entry in ordered[:HEAD_SNIPPET_LIMIT * 2]:
            if total_chars + len(entry["text"]) > HEAD_CONTEXT_CHAR_BUDGET and selected:
                break
            selected.append(self._prompt_context_entry(entry))
            total_chars += len(entry["text"])
            if len(selected) >= HEAD_SNIPPET_LIMIT:
                break
        return selected

    @staticmethod
    def _prompt_context_entry(entry: dict) -> dict:
        compact = {
            "chunk_id": entry.get("chunk_id"),
            "chunk_ordinal": entry.get("chunk_ordinal"),
            "page_label": entry.get("page_label"),
            "heading": entry.get("heading"),
            "sentence_ref": entry.get("sentence_ref"),
            "score": entry.get("score"),
            "text": re.sub(r"\s+", " ", str(entry.get("text") or "")).strip()[:520],
        }
        return {key: value for key, value in compact.items() if value not in (None, "", [])}

    def _open_kg_examples(self, project_id: str, head_name: str, limit: int = 8) -> list[dict]:
        verified_claims = self._project_verified_claims(project_id)
        exact = [
            claim
            for claim in verified_claims
            if claim.subject.lower() == head_name.lower()
        ]
        if exact:
            return [self._claim_as_example(claim) for claim in exact[:limit]]

        tokens = [token for token in self._head_lookup_tokens(head_name) if len(token) > 1]
        fuzzy = [
            claim
            for claim in verified_claims
            if any(token in claim.subject.lower() or token in claim.object_text.lower() for token in tokens)
        ]
        if fuzzy:
            return [self._claim_as_example(claim) for claim in fuzzy[:limit]]

        return GENERIC_TRIPLE_EXAMPLES[:limit]

    def _generate_head_analysis(
        self,
        *,
        document: Document,
        head: dict,
        contexts: list[dict],
        open_kg_examples: list[dict],
        corpus_context: str,
        previous_claims: list[Claim],
    ) -> HeadAnalysisPayload:
        fallback = self._fallback_head_analysis(head, contexts)
        base_prompt = "\n\n".join(
            [
                f"Document title: {document.title}",
                f"Target head entity: {head['name']}",
                "Task: generate triples only for the target head entity.",
                (
                    "Every triple must use the target head as subject. "
                    "Preserve exact evidence, dates, doses, and follow-up guidance. "
                    "Return related entities and concepts only when they are grounded in the snippets."
                ),
                "Current corpus context:\n" + (corpus_context or "No existing corpus context."),
                "Open KG example triples:\n" + json.dumps(open_kg_examples, ensure_ascii=False),
                "Retrieved domain snippets:\n" + json.dumps(contexts, ensure_ascii=False),
            ]
        )
        analysis = safe_model_call(
            lambda: self.ollama.generate_structured(
                HeadAnalysisPayload,
                system_prompt=(
                    "You are the Generator in a SAC-KG-inspired document pipeline. "
                    "Return triples-first JSON for a single head entity."
                ),
                user_prompt=base_prompt,
                model=settings.ollama_batch_model,
            ),
            fallback,
        )
        analysis = self._normalize_head_analysis(head["name"], analysis)
        return self._verify_and_correct_head_analysis(
            head_name=head["name"],
            analysis=analysis,
            contexts=contexts,
            open_kg_examples=open_kg_examples,
            previous_claims=previous_claims,
            correction_prompt=base_prompt,
        )

    def _normalize_head_analysis(self, head_name: str, analysis: HeadAnalysisPayload) -> HeadAnalysisPayload:
        normalized_triples = [
            GeneratedTriple(
                subject=triple.subject.strip() or head_name,
                predicate=triple.predicate.strip(),
                object_text=triple.object_text.strip(),
                expected_head=head_name,
                evidence_excerpt=triple.evidence_excerpt.strip(),
                confidence=triple.confidence,
                source_chunk_ordinals=triple.source_chunk_ordinals,
                source_sentence_refs=triple.source_sentence_refs,
                relation_type=triple.relation_type,
            )
            for triple in analysis.triples
            if triple.predicate.strip() and triple.object_text.strip()
        ]
        return HeadAnalysisPayload(
            head_entity=head_name,
            summary=analysis.summary.strip(),
            key_facts=[fact.strip() for fact in analysis.key_facts if fact.strip()],
            triples=normalized_triples,
            related_entities=analysis.related_entities,
            related_concepts=[concept.strip() for concept in analysis.related_concepts if concept.strip()],
            coverage_notes=[note.strip() for note in analysis.coverage_notes if note.strip()],
        )

    def _fallback_head_analysis(self, head: dict, contexts: list[dict]) -> HeadAnalysisPayload:
        facts = [entry["text"][:240] for entry in contexts[:4]]
        triples = [
            GeneratedTriple(
                subject=head["name"],
                predicate="mentions",
                object_text=fact,
                expected_head=head["name"],
                evidence_excerpt=fact,
                confidence=0.62,
                source_chunk_ordinals=[entry["chunk_ordinal"]],
                source_sentence_refs=[entry["sentence_ref"]],
                relation_type="source_fact",
            )
            for entry, fact in zip(contexts[:4], facts, strict=False)
            if fact
        ]
        return HeadAnalysisPayload(
            head_entity=head["name"],
            summary=" ".join(facts[:2]).strip(),
            key_facts=facts[:4],
            triples=triples,
            related_entities=[],
            related_concepts=[],
            coverage_notes=["Fallback head analysis used because structured model generation was unavailable."],
        )

    def _verify_and_correct_head_analysis(
        self,
        *,
        head_name: str,
        analysis: HeadAnalysisPayload,
        contexts: list[dict],
        open_kg_examples: list[dict],
        previous_claims: list[Claim],
        correction_prompt: str,
    ) -> HeadAnalysisPayload:
        filtered, error_report, total_errors = self._filter_head_triples(head_name, analysis, contexts, previous_claims)
        if total_errors < HEAD_REPROMPT_ERROR_THRESHOLD:
            filtered.coverage_notes.extend(error_report)
            return filtered

        correction = safe_model_call(
            lambda: self.ollama.generate_structured(
                HeadAnalysisPayload,
                system_prompt=(
                    "You are the Verifier correction pass in a SAC-KG-inspired document pipeline. "
                    "Fix triple count, head-entity mismatches, format issues, contradictions, and missing evidence."
                ),
                user_prompt="\n\n".join(
                    [
                        correction_prompt,
                        "Verifier detected these issues:\n" + "\n".join(error_report),
                        "Please regenerate valid triples for the same head entity only.",
                        "Open KG example triples:\n" + json.dumps(open_kg_examples, ensure_ascii=False),
                    ]
                ),
                model=settings.ollama_batch_model,
            ),
            analysis,
        )
        corrected = self._normalize_head_analysis(head_name, correction)
        corrected_filtered, corrected_report, corrected_errors = self._filter_head_triples(head_name, corrected, contexts, previous_claims)
        if corrected_errors <= total_errors:
            corrected_filtered.coverage_notes.extend(["Verifier correction reprompt executed.", *corrected_report])
            return corrected_filtered
        filtered.coverage_notes.extend(["Verifier correction prompt did not improve the triples.", *error_report])
        return filtered

    def _filter_head_triples(
        self,
        head_name: str,
        analysis: HeadAnalysisPayload,
        contexts: list[dict],
        previous_claims: list[Claim],
    ) -> tuple[HeadAnalysisPayload, list[str], int]:
        reports: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        accepted: list[GeneratedTriple] = []
        total_errors = 0

        if len(analysis.triples) < 3:
            reports.append(f"[{head_name}] quantity_too_small")
            total_errors += 1

        for index, triple in enumerate(analysis.triples):
            matched_context = self._match_context_for_generated_triple(triple, contexts)
            errors = self._generated_triple_errors(
                head_name=head_name,
                triple=triple,
                matched_context=matched_context,
                previous_claims=previous_claims,
                seen=seen,
            )
            if errors:
                reports.append(f"[{head_name}] triple[{index}] -> {', '.join(errors)}")
                total_errors += len(errors)
            else:
                if matched_context is not None:
                    if not triple.evidence_excerpt:
                        triple.evidence_excerpt = matched_context["text"]
                    if not triple.source_chunk_ordinals:
                        triple.source_chunk_ordinals = [matched_context["chunk_ordinal"]]
                    if not triple.source_sentence_refs:
                        triple.source_sentence_refs = [matched_context["sentence_ref"]]
                accepted.append(triple)

        return (
            HeadAnalysisPayload(
                head_entity=head_name,
                summary=analysis.summary,
                key_facts=analysis.key_facts,
                triples=accepted,
                related_entities=analysis.related_entities,
                related_concepts=analysis.related_concepts,
                coverage_notes=analysis.coverage_notes,
            ),
            reports,
            total_errors,
        )

    def _match_context_for_generated_triple(self, triple: GeneratedTriple, contexts: list[dict]) -> dict | None:
        evidence = triple.evidence_excerpt.strip()
        object_text = triple.object_text.strip()
        best: dict | None = None
        best_score = -1
        for context in contexts:
            text = context["text"]
            score = 0
            if evidence and evidence[:80] in text:
                score += 5
            if object_text and object_text[:80] in text:
                score += 4
            if triple.subject and triple.subject in text:
                score += 2
            if triple.predicate and triple.predicate.lower() in text.lower():
                score += 1
            if score > best_score:
                best = context
                best_score = score
        return best if best_score > 0 else None

    def _generated_triple_errors(
        self,
        *,
        head_name: str,
        triple: GeneratedTriple,
        matched_context: dict | None,
        previous_claims: list[Claim],
        seen: set[tuple[str, str, str]],
    ) -> list[str]:
        errors: list[str] = []
        subject = triple.subject.strip()
        predicate = triple.predicate.strip()
        object_text = triple.object_text.strip()
        if not subject or not predicate or not object_text:
            errors.append("format_error")
            return errors
        if subject.lower() != head_name.lower():
            errors.append("head_entity_error")
        if subject.lower() == object_text.lower():
            errors.append("head_tail_contradiction")
        if matched_context is None:
            errors.append("missing_evidence")
        key = (subject.lower(), predicate.lower(), object_text.lower())
        if key in seen:
            errors.append("duplicate")
        else:
            seen.add(key)
        for previous in previous_claims:
            if previous.subject.lower() == subject.lower() and previous.predicate.lower() == predicate.lower() and previous.object_text.lower() != object_text.lower():
                errors.append("potential_conflict")
                break
        return sorted(set(errors))

    def _merge_document_analysis(
        self,
        document: Document,
        seed_analysis: DocumentAnalysisPayload,
        head_payloads: list[HeadAnalysisPayload],
    ) -> DocumentExtraction:
        key_facts = list(dict.fromkeys([*seed_analysis.key_facts, *(fact for payload in head_payloads for fact in payload.key_facts)]))
        flattened_keywords = [
            keyword
            for keyword in [
                *seed_analysis.keywords,
                *(keyword for payload in head_payloads for keyword in self._derive_keywords(payload.summary)),
            ]
            if str(keyword).strip()
        ]
        entities = list(seed_analysis.entities)
        entity_names = {entity.name.lower() for entity in entities}
        concepts = list(seed_analysis.concepts)
        for payload in head_payloads:
            if payload.head_entity.lower() not in entity_names:
                entities.append(ExtractedEntity(name=payload.head_entity, entity_type="concept", summary=payload.summary))
                entity_names.add(payload.head_entity.lower())
            for entity in payload.related_entities:
                if entity.name.lower() not in entity_names:
                    entities.append(entity)
                    entity_names.add(entity.name.lower())
            for concept in payload.related_concepts:
                if concept not in concepts:
                    concepts.append(concept)

        claims = [
            ExtractedClaim(
                subject=triple.subject,
                predicate=triple.predicate,
                object_text=triple.object_text,
                expected_head=triple.expected_head,
                evidence_excerpt=triple.evidence_excerpt,
                confidence=triple.confidence,
                source_chunk_ordinals=triple.source_chunk_ordinals,
                source_sentence_refs=triple.source_sentence_refs,
                relation_type=triple.relation_type,
            )
            for payload in head_payloads
            for triple in payload.triples
        ]
        coverage_notes = list(dict.fromkeys([*seed_analysis.coverage_notes, *(note for payload in head_payloads for note in payload.coverage_notes)]))
        summary_parts = [seed_analysis.summary.strip(), *(payload.summary.strip() for payload in head_payloads if payload.summary.strip())]
        summary = "\n".join(part for part in summary_parts if part).strip() or document.title
        if not claims and seed_analysis.triples:
            return self._analysis_to_extraction(document, seed_analysis)

        return DocumentExtraction(
            title=seed_analysis.title or document.title,
            summary=summary,
            keywords=list(dict.fromkeys(flattened_keywords))[:20],
            entities=entities,
            concepts=concepts,
            claims=claims,
            key_facts=key_facts[:20],
            coverage_notes=coverage_notes,
        )

    def _project_verified_claims(self, project_id: str) -> list[Claim]:
        return self.db.scalars(
            select(Claim)
            .where(Claim.project_id == project_id, Claim.verification_status == "verified")
            .order_by(Claim.created_at.desc())
        ).all()

    def _project_verified_subjects(self, project_id: str) -> list[str]:
        return list(dict.fromkeys(claim.subject for claim in self._project_verified_claims(project_id)))

    @staticmethod
    def _claim_as_example(claim: Claim) -> dict:
        return {
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object_text": claim.object_text,
            "evidence_excerpt": (claim.metadata_json or {}).get("evidence_excerpt", ""),
        }

    def _head_lookup_tokens(self, head_name: str) -> list[str]:
        tokens = list(self._text_terms(head_name))
        if not tokens:
            return [head_name.lower()]
        return [token.lower() for token in tokens]

    def _select_generation_contexts(self, document: Document, full_text: str, chunks: list[DocumentChunk]) -> list[dict]:
        if not chunks and full_text:
            return [{"chunk_ordinal": 0, "heading": None, "score": 1.0, "text": full_text[:2400]}]

        candidate_terms = self._candidate_terms(document, full_text, chunks)
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in chunks:
            lowered = chunk.text.lower()
            marker_score = sum(1 for marker in FACT_MARKERS if marker.lower() in lowered) * 4
            value_score = 3 if FACT_VALUE_PATTERN.search(chunk.text) else 0
            term_score = sum(1 for term in candidate_terms if term and term in lowered)
            heading_score = 2 if chunk.heading else 0
            early_score = 1.0 / (chunk.ordinal + 1)
            scored.append((marker_score + value_score + term_score + heading_score + early_score, chunk))

        if len(chunks) <= 8:
            selected = sorted(chunks, key=lambda item: item.ordinal)
        else:
            top_chunks = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:8]]
            if chunks[0] not in top_chunks:
                top_chunks[-1] = chunks[0]
            selected = sorted(top_chunks, key=lambda item: item.ordinal)

        return [
            {
                "chunk_ordinal": chunk.ordinal,
                "heading": chunk.heading,
                "page_label": chunk.page_label,
                "score": round(next((score for score, item in scored if item.id == chunk.id), 0.0), 3),
                "text": chunk.text[:2400],
            }
            for chunk in selected
        ]

    def _candidate_terms(self, document: Document, full_text: str, chunks: list[DocumentChunk]) -> set[str]:
        terms = self._text_terms(document.title)
        for chunk in chunks[:5]:
            if chunk.heading:
                terms.update(self._text_terms(chunk.heading))
        for marker in FACT_MARKERS:
            if marker.lower() in full_text.lower():
                terms.add(marker.lower())
        return terms

    def _local_triple_examples(self, project_id: str) -> list[dict]:
        examples: list[dict] = []
        claims = self.db.scalars(select(Claim).where(Claim.project_id == project_id).order_by(Claim.created_at.desc())).all()
        for claim in claims[:6]:
            examples.append(
                {
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object_text": claim.object_text,
                    "evidence_excerpt": (claim.metadata_json or {}).get("evidence_excerpt", ""),
                    "confidence": claim.confidence,
                }
            )
        return examples

    def _fallback_analysis(self, document: Document, full_text: str, contexts: list[dict]) -> DocumentAnalysisPayload:
        key_facts = self._extract_key_facts(full_text)
        triples = [
            GeneratedTriple(
                subject=document.title,
                predicate="states",
                object_text=fact,
                expected_head=document.title,
                evidence_excerpt=fact,
                confidence=0.65,
                source_chunk_ordinals=self._fact_chunk_ordinals(fact, contexts),
                relation_type="source_fact",
            )
            for fact in key_facts[:8]
        ]
        summary_parts = key_facts[:5] if key_facts else [full_text[:1200]]
        return DocumentAnalysisPayload(
            title=document.title,
            summary="\n".join(part for part in summary_parts if part).strip() or document.title,
            keywords=self._derive_keywords(full_text),
            key_facts=key_facts,
            entities=[],
            concepts=[],
            triples=triples,
            coverage_notes=["Fallback analysis used because structured model generation was unavailable."],
        )

    def _analysis_to_extraction(self, document: Document, analysis: DocumentAnalysisPayload) -> DocumentExtraction:
        claims = [
            ExtractedClaim(
                subject=triple.subject,
                predicate=triple.predicate,
                object_text=triple.object_text,
                expected_head=triple.expected_head,
                evidence_excerpt=triple.evidence_excerpt,
                confidence=triple.confidence,
                source_chunk_ordinals=triple.source_chunk_ordinals,
                source_sentence_refs=triple.source_sentence_refs,
                relation_type=triple.relation_type,
            )
            for triple in analysis.triples
            if triple.subject.strip() and triple.predicate.strip() and triple.object_text.strip()
        ]
        return DocumentExtraction(
            title=analysis.title or document.title,
            summary=analysis.summary,
            keywords=analysis.keywords,
            entities=analysis.entities,
            concepts=analysis.concepts,
            claims=claims,
            key_facts=analysis.key_facts,
            coverage_notes=analysis.coverage_notes,
        )

    def _extract_key_facts(self, text: str) -> list[str]:
        pieces = re.split(r"(?<=[。！？!?\.])\s*|\n+", text)
        facts: list[str] = []
        seen: set[str] = set()
        for piece in pieces:
            sentence = re.sub(r"\s+", " ", piece).strip()
            if len(sentence) < 6:
                continue
            lowered = sentence.lower()
            has_marker = any(marker.lower() in lowered for marker in FACT_MARKERS)
            has_value = bool(FACT_VALUE_PATTERN.search(sentence))
            if not has_marker and not has_value:
                continue
            fact = sentence[:600]
            key = fact.lower()
            if key in seen:
                continue
            facts.append(fact)
            seen.add(key)
            if len(facts) >= 12:
                break
        return facts

    def _derive_keywords(self, text: str) -> list[str]:
        keywords = [marker for marker in FACT_MARKERS if marker.lower() in text.lower()]
        english_terms = [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower()) if word not in keywords]
        for term in english_terms:
            if term not in keywords:
                keywords.append(term)
            if len(keywords) >= 12:
                break
        return keywords[:12]

    @staticmethod
    def _fact_chunk_ordinals(fact: str, contexts: list[dict]) -> list[int]:
        return [
            int(context["chunk_ordinal"])
            for context in contexts
            if fact and fact[:80] in str(context.get("text", ""))
        ][:3]

    @staticmethod
    def _text_terms(text: str) -> set[str]:
        lowered = text.lower()
        terms = {word for word in re.findall(r"[a-z0-9_]+", lowered) if len(word) > 1}
        for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
            if len(segment) <= 2:
                terms.add(segment)
                continue
            terms.add(segment)
            terms.update(segment[index : index + 2] for index in range(len(segment) - 1))
        return {term for term in terms if term}

    def _upsert_entities(self, project_id: str, extraction: DocumentExtraction) -> list[Entity]:
        entities: list[Entity] = []
        items = list(extraction.entities)
        existing_names = {item.name for item in items}
        for concept in extraction.concepts:
            if concept and concept not in existing_names:
                items.append(ExtractedEntity(name=concept, entity_type="concept", summary=""))
                existing_names.add(concept)

        unique_items: list[ExtractedEntity] = []
        seen_item_names: set[str] = set()
        for item in items:
            key = item.name.strip()
            if not key or key in seen_item_names:
                continue
            seen_item_names.add(key)
            unique_items.append(item)

        for item in unique_items:
            existing = self.db.scalar(select(Entity).where(Entity.project_id == project_id, Entity.name == item.name))
            if existing:
                existing.summary = item.summary or existing.summary
                existing.entity_type = item.entity_type or existing.entity_type
                merged_aliases = sorted(set(existing.aliases + item.aliases))
                existing.aliases = merged_aliases
                entities.append(existing)
                continue
            entity = Entity(
                project_id=project_id,
                name=item.name,
                entity_type=item.entity_type,
                aliases=item.aliases,
                summary=item.summary,
            )
            self.db.add(entity)
            entities.append(entity)
        try:
            self.db.commit()
            return entities
        except IntegrityError:
            self.db.rollback()
            return list(self.db.scalars(select(Entity).where(Entity.project_id == project_id, Entity.name.in_([item.name for item in unique_items]))).all())

    def _create_claims(self, document: Document, extraction: DocumentExtraction) -> list[Claim]:
        self.db.query(Claim).filter(Claim.document_id == document.id).delete()
        chunks = self.db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.ordinal)).all()
        previous_claims = self.db.scalars(select(Claim).where(Claim.project_id == document.project_id)).all()
        seen: set[tuple[str, str, str]] = set()
        claims: list[Claim] = []
        for item in extraction.claims:
            evidence_chunk = self._find_evidence_chunk(item, list(chunks))
            evidence_excerpt = item.evidence_excerpt.strip()
            if not evidence_excerpt and evidence_chunk is not None:
                evidence_excerpt = self._best_evidence_excerpt(item, evidence_chunk.text)
            verification_errors = sorted(
                set(item.verification_errors + self._verify_claim_locally(item, evidence_chunk, previous_claims, seen))
            )
            source_chunk_ids = [evidence_chunk.id] if evidence_chunk is not None else []
            claim = Claim(
                project_id=document.project_id,
                document_id=document.id,
                subject=item.subject,
                predicate=item.predicate,
                object_text=item.object_text,
                evidence_chunk_id=evidence_chunk.id if evidence_chunk is not None else None,
                confidence=item.confidence,
                verification_status="needs-review" if verification_errors else "verified",
                metadata_json={
                    "expected_head": item.expected_head,
                    "evidence_excerpt": evidence_excerpt,
                    "source_chunk_ids": source_chunk_ids,
                    "source_chunk_ordinals": item.source_chunk_ordinals,
                    "source_sentence_refs": item.source_sentence_refs,
                    "verification_errors": verification_errors,
                    "head_growth_decision": item.growth_decision,
                    "tail_growth_decision": "keep",
                    "relation_type": item.relation_type,
                },
            )
            self.db.add(claim)
            claims.append(claim)
        self.db.commit()
        return claims

    def _find_evidence_chunk(self, claim: ExtractedClaim, chunks: list[DocumentChunk]) -> DocumentChunk | None:
        for sentence_ref in claim.source_sentence_refs:
            match = re.match(r"chunk-(\d+):sentence-\d+", sentence_ref)
            if not match:
                continue
            chunk_ordinal = int(match.group(1))
            for chunk in chunks:
                if chunk.ordinal == chunk_ordinal:
                    return chunk
        for ordinal in claim.source_chunk_ordinals:
            for chunk in chunks:
                if chunk.ordinal == ordinal:
                    return chunk

        evidence = claim.evidence_excerpt.strip()
        if evidence:
            evidence_variants = [evidence, evidence[:160], evidence[:80]]
            for variant in evidence_variants:
                if len(variant) < 8:
                    continue
                for chunk in chunks:
                    if variant in chunk.text:
                        return chunk

        object_text = claim.object_text.strip()
        if object_text and len(object_text) >= 6:
            for chunk in chunks:
                if object_text[:120] in chunk.text or object_text in chunk.text:
                    return chunk

        subject = claim.subject.strip()
        if subject and len(subject) >= 2:
            for chunk in chunks:
                if subject in chunk.text:
                    return chunk
        return None

    def _best_evidence_excerpt(self, claim: ExtractedClaim, text: str, max_chars: int = 360) -> str:
        anchors = [claim.object_text.strip(), claim.subject.strip()]
        lowered = text.lower()
        for anchor in anchors:
            if not anchor:
                continue
            position = lowered.find(anchor.lower())
            if position < 0:
                continue
            start = max(0, position - max_chars // 3)
            end = min(len(text), start + max_chars)
            return text[start:end].strip()
        return text[:max_chars].strip()

    def _verify_claim_locally(
        self,
        claim: ExtractedClaim,
        evidence_chunk: DocumentChunk | None,
        previous_claims: list[Claim],
        seen: set[tuple[str, str, str]],
    ) -> list[str]:
        errors: list[str] = []
        subject = claim.subject.strip()
        predicate = claim.predicate.strip()
        object_text = claim.object_text.strip()
        is_source_fact = claim.relation_type == "source_fact"

        if not subject or not predicate or not object_text:
            errors.append("format_error")

        expected_head = claim.expected_head.strip()
        if expected_head and subject.lower() != expected_head.lower():
            errors.append("head_entity_error")

        if subject.lower() == object_text.lower():
            errors.append("head_tail_contradiction")

        # source_fact claims are fallback triples — relax confidence threshold
        # so they count as verified and improve downstream RAG scoring.
        confidence_threshold = 0.45 if is_source_fact else 0.6
        if claim.confidence < confidence_threshold:
            errors.append("low_confidence")

        # source_fact claims may not map to a single chunk — that's expected.
        if not is_source_fact and evidence_chunk is None:
            errors.append("missing_evidence")

        key = (subject.lower(), predicate.lower(), object_text.lower())
        if key in seen:
            errors.append("duplicate")
        else:
            seen.add(key)

        for previous in previous_claims:
            if previous.subject.lower() != subject.lower() or previous.predicate.lower() != predicate.lower():
                continue
            if previous.object_text.lower() != object_text.lower():
                errors.append("potential_conflict")
                break

        # Skip evidence-anchoring checks for source_fact — they are fallback triples.
        if not is_source_fact and evidence_chunk is not None and subject and len(subject) > 2 and predicate.lower() != "states":
            source_text = evidence_chunk.text
            if subject not in source_text and not self._has_term_overlap(subject, source_text):
                errors.append("subject_not_in_evidence")

        return sorted(set(errors))

    @staticmethod
    def _has_term_overlap(left: str, right: str) -> bool:
        left_terms = IngestionPipeline._text_terms(left)
        right_terms = IngestionPipeline._text_terms(right)
        return bool(left_terms & right_terms)

    def _decide_entity_growth(self, document: Document, entities: list[Entity], claims: list[Claim]) -> dict[str, GrowthDecision]:
        verified_claims = [claim for claim in claims if claim.verification_status == "verified"]
        candidates: dict[str, dict] = {}
        for entity in entities:
            candidates.setdefault(
                entity.name,
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type or "concept",
                    "summary": entity.summary or "",
                    "claim_count": sum(1 for claim in verified_claims if claim.subject == entity.name or claim.object_text == entity.name),
                    "item_type": "head",
                },
            )
        for claim in verified_claims:
            tail_name = claim.object_text.strip()
            if not tail_name or len(tail_name) > 80:
                continue
            candidates.setdefault(
                tail_name,
                {
                    "name": tail_name,
                    "entity_type": self._infer_tail_entity_type(tail_name, entities),
                    "summary": f"Referenced by {claim.subject} via {claim.predicate}.",
                    "claim_count": sum(1 for item in verified_claims if item.object_text == tail_name),
                    "item_type": "tail",
                },
            )

        decisions: dict[str, GrowthDecision] = {}
        uncertain: list[dict] = []
        for candidate in candidates.values():
            decision = self._rule_growth_decision(candidate)
            decisions[candidate["name"]] = decision
            if decision.decision == "keep":
                uncertain.append(candidate)

        if uncertain:
            prompt = "\n\n".join(
                [
                    f"Document title: {document.title}",
                    (
                        "Decide whether each candidate should continue growing in the SAC-KG routing graph. "
                        "Use grow for durable head or tail entities, keep for useful but not yet expanded items, "
                        "and prune for dates, doses, isolated numbers, or transient values."
                    ),
                    json.dumps(uncertain[:20], ensure_ascii=False),
                ]
            )
            fallback = GrowthDecisionPayload(decisions=[])
            ai_decisions = safe_model_call(
                lambda: self.ollama.generate_structured(
                    GrowthDecisionPayload,
                    system_prompt="You are the Pruner in a SAC-KG-inspired RAG pipeline. Return strict JSON decisions only.",
                    user_prompt=prompt,
                    model=settings.ollama_batch_model,
                ),
                fallback,
            )
            for item in ai_decisions.decisions:
                if item.name not in decisions:
                    continue
                normalized = self._normalize_growth_decision(item.decision)
                decisions[item.name] = GrowthDecision(
                    name=item.name,
                    item_type=item.item_type or "entity",
                    decision=normalized,
                    reason=item.reason or "Ollama pruner decision.",
                )
        return decisions

    def _rule_growth_decision(self, candidate: dict) -> GrowthDecision:
        entity_type = str(candidate.get("entity_type") or "concept").lower()
        name = str(candidate.get("name") or "").strip()
        claim_count = int(candidate.get("claim_count") or 0)
        item_type = str(candidate.get("item_type") or "entity")
        if entity_type in PRUNE_ENTITY_TYPES or self._looks_like_transient_value(name):
            return GrowthDecision(name=name, item_type=item_type, decision="prune", reason="Transient date, dose, or numeric value.")
        if entity_type in GROW_ENTITY_TYPES:
            if item_type == "tail" and claim_count <= 0:
                return GrowthDecision(name=name, item_type=item_type, decision="keep", reason="Durable type but no verified claim supports a standalone tail page yet.")
            return GrowthDecision(name=name, item_type=item_type, decision="grow", reason="Durable entity type suitable for continued growth.")
        if item_type == "tail" and claim_count == 1 and len(name) > 36:
            return GrowthDecision(name=name, item_type=item_type, decision="keep", reason="Tail value is descriptive but may be too broad for its own page.")
        if claim_count > 0:
            return GrowthDecision(name=name, item_type=item_type, decision="grow", reason="Supported by verified claims and suitable for RAG routing expansion.")
        return GrowthDecision(name=name, item_type=item_type, decision="keep", reason="Needs pruner judgment before creating a standalone page.")

    def _infer_tail_entity_type(self, tail_name: str, entities: list[Entity]) -> str:
        for entity in entities:
            if entity.name.lower() == tail_name.lower():
                return entity.entity_type or "concept"
        lowered = tail_name.lower()
        if FACT_VALUE_PATTERN.search(tail_name):
            return "value"
        if any(keyword in lowered for keyword in ("disease", "diagnosis", "syndrome", "病", "症")):
            return "disease"
        if any(keyword in lowered for keyword in ("drug", "tablet", "capsule", "胍", "药")):
            return "drug"
        if any(keyword in lowered for keyword in ("check", "exam", "scan", "检查", "复查")):
            return "test"
        return "concept"

    @staticmethod
    def _looks_like_transient_value(name: str) -> bool:
        stripped = name.strip()
        if not stripped:
            return True
        if FACT_VALUE_PATTERN.search(stripped):
            return True
        return bool(re.fullmatch(r"[\d\s.,:%/-]+", stripped))

    @staticmethod
    def _normalize_growth_decision(decision: str) -> str:
        lowered = decision.lower().strip()
        if lowered in {"grow", "keep", "prune"}:
            return lowered
        return "keep"

    @staticmethod
    def _apply_claim_growth_decisions(claims: list[Claim], entity_decisions: dict[str, GrowthDecision]) -> None:
        for claim in claims:
            metadata = dict(claim.metadata_json or {})
            head_decision = entity_decisions.get(claim.subject)
            tail_decision = entity_decisions.get(claim.object_text)
            metadata["head_growth_decision"] = head_decision.decision if head_decision else metadata.get("head_growth_decision", "keep")
            metadata["tail_growth_decision"] = tail_decision.decision if tail_decision else metadata.get("tail_growth_decision", "keep")
            metadata["growth_decision"] = metadata["tail_growth_decision"]
            claim.metadata_json = metadata

    def _ensure_growing_entities(
        self,
        project_id: str,
        entities: list[Entity],
        claims: list[Claim],
        entity_decisions: dict[str, GrowthDecision],
    ) -> list[Entity]:
        entity_map = {entity.name.lower(): entity for entity in entities}
        grow_names = [name for name, decision in entity_decisions.items() if decision.decision == "grow"]
        if grow_names:
            existing_entities = self.db.scalars(
                select(Entity).where(Entity.project_id == project_id, Entity.name.in_(grow_names))
            ).all()
            for entity in existing_entities:
                key = entity.name.lower()
                if key not in entity_map:
                    entities.append(entity)
                    entity_map[key] = entity
        for name, decision in entity_decisions.items():
            if decision.decision != "grow" or name.lower() in entity_map:
                continue
            related_claims = [claim for claim in claims if claim.object_text == name]
            if not related_claims:
                continue
            entity = Entity(
                project_id=project_id,
                name=name,
                entity_type=self._infer_tail_entity_type(name, entities),
                aliases=[],
                summary=" ".join(f"{claim.subject} {claim.predicate} {claim.object_text}" for claim in related_claims[:2])[:400],
            )
            self.db.add(entity)
            entities.append(entity)
            entity_map[name.lower()] = entity
        try:
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            return list(self.db.scalars(select(Entity).where(Entity.project_id == project_id, Entity.name.in_(grow_names))).all())
        return entities

    def _source_page_metadata(self, extraction: DocumentExtraction, entities: list[Entity], claims: list[Claim]) -> dict:
        key_terms = sorted(
            {
                *extraction.keywords,
                *extraction.concepts,
                *(entity.name for entity in entities),
                *(claim.subject for claim in claims if claim.verification_status == "verified"),
                *(
                    claim.object_text
                    for claim in claims
                    if claim.verification_status == "verified" and (claim.metadata_json or {}).get("tail_growth_decision") == "grow"
                ),
            }
        )
        # Ensure we always have at least some key_terms — fall back to
        # extraction title words and key facts so frontmatter is rarely empty.
        if not key_terms:
            fallback_text = " ".join([extraction.title, *extraction.key_facts[:5]])
            key_terms = sorted(
                term
                for term in self._text_terms(fallback_text)
                if term.lower() not in {"the", "and", "for", "with"}
            )
        verified_count = sum(1 for claim in claims if claim.verification_status == "verified")
        return {
            "summary": extraction.summary[:700],
            "key_terms": list(key_terms)[:30],
            "source_count": 1,
            "verified_claim_count": verified_count,
            "growth_decision": "grow",
        }

    @staticmethod
    def _entity_page_metadata(entity_name: str, decision: GrowthDecision, claims: list[Claim]) -> dict:
        entity_claims = [claim for claim in claims if claim.subject == entity_name or claim.object_text == entity_name]
        return {
            "summary": " ".join(
                claim.object_text if claim.subject == entity_name else f"{claim.subject} {claim.predicate}"
                for claim in entity_claims[:3]
            )[:700],
            "key_terms": sorted({entity_name, *(claim.predicate for claim in entity_claims), *(claim.subject for claim in entity_claims)})[:20],
            "source_count": len({claim.document_id for claim in entity_claims}),
            "verified_claim_count": sum(1 for claim in entity_claims if claim.verification_status == "verified"),
            "growth_decision": decision.decision,
            "growth_reason": decision.reason,
        }

    def _create_review_items(self, document: Document, extraction: DocumentExtraction, claims: list[Claim]) -> int:
        existing_items = self.db.scalars(select(ReviewItem).where(ReviewItem.document_id == document.id)).all()
        for item in existing_items:
            if item.status == ReviewStatus.pending.value and (item.payload or {}).get("generated_by") == "ingest":
                self.db.delete(item)
        count = 0
        if not claims and (document.raw_text or "").strip():
            self.db.add(
                ReviewItem(
                    project_id=document.project_id,
                    document_id=document.id,
                    title="Verifier quantity check: no claims extracted",
                    detail="The document contains text, but no structured claims were generated.",
                    severity=ReviewSeverity.medium.value,
                    payload={"generated_by": "ingest", "issue": "empty_claim_set", "check": "quantity"},
                )
            )
            count += 1

        head_groups: dict[str, list[Claim]] = {}
        for claim in claims:
            expected_head = ((claim.metadata_json or {}).get("expected_head") or claim.subject).strip()
            head_groups.setdefault(expected_head, []).append(claim)
        for head_name, head_claims in head_groups.items():
            verified_count = sum(1 for claim in head_claims if claim.verification_status == "verified")
            if verified_count < 3:
                self.db.add(
                    ReviewItem(
                        project_id=document.project_id,
                        document_id=document.id,
                        title=f"Verifier quantity check: {head_name}",
                        detail=f"Head '{head_name}' has only {verified_count} verified triples.",
                        severity=ReviewSeverity.low.value,
                        payload={"generated_by": "ingest", "issue": "quantity_too_small", "head": head_name, "verified_count": verified_count},
                    )
                )
                count += 1

        for claim in claims:
            metadata = claim.metadata_json or {}
            errors = metadata.get("verification_errors", [])
            if errors:
                high_risk_errors = {"format_error", "missing_evidence", "potential_conflict", "head_entity_error", "head_tail_contradiction"}
                self.db.add(
                    ReviewItem(
                        project_id=document.project_id,
                        document_id=document.id,
                        claim_id=claim.id,
                        title=f"Verifier flagged claim: {claim.subject}",
                        detail=f"{claim.subject} {claim.predicate} {claim.object_text}",
                        severity=ReviewSeverity.high.value if high_risk_errors & set(errors) else ReviewSeverity.medium.value,
                        payload={
                            "generated_by": "ingest",
                            "issues": errors,
                            "check": "local_verifier",
                            "confidence": claim.confidence,
                            "evidence_excerpt": metadata.get("evidence_excerpt", ""),
                            "source_chunk_ids": metadata.get("source_chunk_ids", []),
                        },
                    )
                )
                count += 1

        for note in extraction.coverage_notes:
            self.db.add(
                ReviewItem(
                    project_id=document.project_id,
                    document_id=document.id,
                    title="Verifier coverage note",
                    detail=note,
                    severity=ReviewSeverity.low.value,
                    payload={"generated_by": "ingest", "issue": "coverage_note", "note": note},
                )
            )
            count += 1

        verification = self.verifier.verify_claims(extraction.summary, extraction.claims)
        if verification.flagged_claim_indexes:
            for claim_index in verification.flagged_claim_indexes:
                if 0 <= claim_index < len(claims):
                    claims[claim_index].verification_status = verification.verdict
            self.db.add(
                ReviewItem(
                    project_id=document.project_id,
                    document_id=document.id,
                    title="External verification flagged claims",
                    detail=verification.notes,
                    severity=ReviewSeverity.high.value,
                    payload={"generated_by": "ingest", "flagged_claim_indexes": verification.flagged_claim_indexes, "verdict": verification.verdict},
                )
            )
            count += 1
        self.db.commit()
        return count
