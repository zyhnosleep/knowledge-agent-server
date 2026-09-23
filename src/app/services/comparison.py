"""Evidence-matrix and Graph-lite support for multi-paper comparisons.

The comparison layer deliberately sits on top of the existing QueryService:
vector, lexical and table retrieval stay the source of truth, while this
module keeps each ``paper x dimension`` cell separate.  SAC-KG is an optional
enrichment: when it is enabled, the small edge store is an auditable
extension of the existing entities/claims tables; when it is disabled, the
matrix remains fully usable and only the paper-pair ``compared_with`` edge is
recorded.  Neither mode replaces page-level citations.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import Claim, Document, DocumentStatus, Entity, KnowledgeEdge, Project
from app.schemas.agent import (
    COMPARISON_DIMENSIONS,
    ComparisonEdge,
    ComparisonEvidenceCell,
    ComparisonPack,
    ComparisonPaper,
    EvidenceItem,
    EvidencePack,
    TableFactEvidence,
)

logger = logging.getLogger(__name__)


class ComparisonService:
    """Build a bounded comparison pack from the existing RAG evidence paths."""

    MAX_PAPERS = 5
    MAX_DIMENSIONS = 8
    CELL_EVIDENCE_LIMIT = 2
    EXTRACTION_VERSION = "comparison-v1"

    _DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
        "method/architecture": (
            "method",
            "architecture",
            "mechanism",
            "方法",
            "架构",
            "机制",
        ),
        "retrieval unit/data representation": (
            "retrieval",
            "representation",
            "embedding",
            "retriever",
            "检索",
            "表示",
            "向量",
        ),
        "training objective": (
            "training",
            "objective",
            "loss",
            "训练",
            "目标",
            "损失",
        ),
        "datasets/tasks": (
            "dataset",
            "task",
            "benchmark",
            "数据集",
            "任务",
            "基准",
        ),
        "metrics/results": (
            "metric",
            "result",
            "accuracy",
            "score",
            "指标",
            "结果",
            "性能",
        ),
        "limitations/use cases": (
            "limitation",
            "use case",
            "应用",
            "局限",
            "适用",
        ),
    }

    def __init__(
        self,
        db: Session,
        *,
        parse_version_map: dict[str, str] | None = None,
        sac_kg_enabled: bool | None = None,
    ) -> None:
        self.db = db
        self.parse_version_map = parse_version_map
        # Keep the production switch authoritative, while allowing focused
        # tests/callers to exercise either mode without mutating global
        # settings.  In particular, SAC-KG must never be enabled implicitly
        # merely because comparison was requested.
        self.sac_kg_enabled = (
            bool(sac_kg_enabled)
            if sac_kg_enabled is not None
            else bool(get_settings().sac_kg_enabled)
        )

    def build(
        self,
        project_slug: str,
        question: str,
        *,
        document_ids: list[str] | None = None,
        dimensions: list[str] | None = None,
        limit: int = 15,
    ) -> EvidencePack:
        """Build an EvidencePack containing a structured comparison matrix.

        Explicit document IDs are strict.  Without them, the existing
        deterministic paper router chooses up to five candidates.  A one-paper
        result is retained as an explicitly labelled intra-paper comparison;
        an automatic zero/one-candidate result is marked ``needs_selection``.
        """
        from app.services.search import PaperMatch, QueryService

        project = self.db.scalar(select(Project).where(Project.slug == project_slug))
        if project is None:
            raise ValueError(f"Project '{project_slug}' not found")

        explicit = bool(document_ids)
        documents: list[Document] = []
        candidate_ids: list[str] = []
        qs = QueryService(self.db, parse_version_map=self.parse_version_map)

        if explicit:
            requested = list(dict.fromkeys(str(item) for item in document_ids or []))
            if not 1 <= len(requested) <= self.MAX_PAPERS:
                raise ValueError("compare_document_ids must contain between 1 and 5 IDs")
            rows = self.db.scalars(
                select(Document).where(
                    Document.project_id == project.id,
                    Document.status == DocumentStatus.ready.value,
                    Document.id.in_(requested),
                )
            ).all()
            by_id = {row.id: row for row in rows}
            missing = [item for item in requested if item not in by_id]
            if missing:
                raise ValueError(f"Comparison documents are not ready or not in project: {missing}")
            documents = [by_id[item] for item in requested]
        else:
            matches = qs._route_comparison_papers(
                question, project.id, limit=self.MAX_PAPERS
            )
            documents = [match.document for match in matches[: self.MAX_PAPERS]]
            candidate_ids = [document.id for document in documents]

        normalized_dimensions = self._normalize_dimensions(question, dimensions)
        mode = "cross_paper" if len(documents) >= 2 else "intra_paper"
        status = "ready" if documents else "needs_selection"
        if not explicit and len(documents) < 2:
            mode = "candidate_selection"
            status = "needs_selection"

        papers = [
            ComparisonPaper(document_id=document.id, title=document.title or document.file_name or document.id, ordinal=index)
            for index, document in enumerate(documents)
        ]
        if not documents:
            pack = ComparisonPack(
                mode="candidate_selection",
                explicit_scope=False,
                papers=[],
                dimensions=normalized_dimensions,
                candidate_document_ids=candidate_ids,
                status="needs_selection",
            )
            return EvidencePack(status="empty", items=[], comparison=pack)
        if not explicit and len(documents) < 2:
            # Automatic routing with insufficient confidence must ask the user
            # to confirm scope instead of producing a one-sided conclusion.
            pack = ComparisonPack(
                mode="candidate_selection",
                explicit_scope=False,
                papers=papers,
                dimensions=normalized_dimensions,
                candidate_document_ids=candidate_ids,
                status="needs_selection",
            )
            return EvidencePack(status="empty", items=[], comparison=pack)

        edges = self._ensure_graph_edges(project.id, documents)
        all_items: list[EvidenceItem] = []
        all_table_facts: list[TableFactEvidence] = []
        item_keys: dict[tuple[str | None, str | None], int] = {}
        cells: list[ComparisonEvidenceCell] = []

        for paper in papers:
            document = next(item for item in documents if item.id == paper.document_id)
            paper_match = PaperMatch(document=document, score=1.0, locked=True)
            for dimension in normalized_dimensions:
                dimension_query = self._dimension_query(question, dimension, document.title or "")
                try:
                    contexts = qs._build_rag_contexts(
                        dimension_query,
                        project.id,
                        [paper_match],
                        document_ids=[document.id],
                    )
                    cell_items, cell_facts = qs._evidence_items_and_facts(
                        contexts,
                        self.CELL_EVIDENCE_LIMIT,
                        question=dimension_query,
                    )
                except Exception as exc:
                    logger.warning(
                        "Comparison cell retrieval failed for %s/%s: %s",
                        document.id,
                        dimension,
                        exc,
                    )
                    cell_items, cell_facts = [], []

                evidence_indexes: list[int] = []
                for item in cell_items:
                    item = item.model_copy(
                        update={
                            "comparison_paper_id": document.id,
                            "comparison_dimension": dimension,
                        }
                    )
                    key = (item.document_id, item.chunk_id)
                    if key in item_keys:
                        index = item_keys[key]
                        # Keep the strongest cell association in metadata while
                        # retaining a single citation for the final answer.
                        existing = all_items[index]
                        if existing.comparison_dimension != dimension:
                            existing.comparison_dimension = f"{existing.comparison_dimension}; {dimension}"
                    else:
                        index = len(all_items)
                        item_keys[key] = index
                        all_items.append(item)
                    evidence_indexes.append(index)

                all_table_facts.extend(cell_facts)
                status_value = "supported" if evidence_indexes else "missing"
                confidence = min(1.0, 0.35 + 0.2 * len(evidence_indexes)) if evidence_indexes else 0.0
                cells.append(
                    ComparisonEvidenceCell(
                        paper_id=document.id,
                        paper_title=paper.title or document.file_name or document.id,
                        dimension=dimension,
                        status=status_value,
                        evidence_indexes=evidence_indexes,
                        citation_indexes=evidence_indexes,
                        confidence=confidence,
                        notes=[] if evidence_indexes else ["No page-level evidence matched this dimension."],
                    )
                )

        # Respect the caller's bounded evidence budget without allowing the
        # first paper to consume all slots.  Preserve at least one item per
        # selected paper, then fill remaining slots in retrieval order.  Cells
        # whose only references were trimmed become explicit ``missing`` cells.
        evidence_budget = max(len(documents), int(limit or 0))
        if evidence_budget and len(all_items) > evidence_budget:
            keep: list[int] = []
            for document in documents:
                first = next(
                    (
                        index
                        for index, item in enumerate(all_items)
                        if item.document_id == document.id
                    ),
                    None,
                )
                if first is not None and first not in keep:
                    keep.append(first)
            for index in range(len(all_items)):
                if len(keep) >= evidence_budget:
                    break
                if index not in keep:
                    keep.append(index)
            keep = sorted(keep[:evidence_budget])
            remap = {old: new for new, old in enumerate(keep)}
            all_items = [all_items[index].model_copy(update={"index": new}) for new, index in enumerate(keep)]
            for cell in cells:
                refs = [remap[index] for index in cell.evidence_indexes if index in remap]
                cell.evidence_indexes = refs
                cell.citation_indexes = [remap[index] for index in cell.citation_indexes if index in remap]
                if cell.status == "supported" and not refs:
                    cell.status = "missing"
                    cell.confidence = 0.0
                    cell.notes.append("Evidence budget trimmed this cell; retrieve it separately to verify.")

        missing_cells = [
            f"{cell.paper_id}:{cell.dimension}"
            for cell in cells
            if cell.status == "missing"
        ]
        conflict_cells = self._detect_conflict_cells(cells, all_items, documents)
        for cell in cells:
            if f"{cell.paper_id}:{cell.dimension}" in conflict_cells:
                cell.status = "conflict"
                cell.notes.append("Claims from the selected papers require explicit comparison.")

        pack = ComparisonPack(
            mode=mode,
            explicit_scope=explicit,
            papers=papers,
            dimensions=normalized_dimensions,
            cells=cells,
            edges=edges,
            missing_cells=missing_cells,
            conflict_cells=conflict_cells,
            candidate_document_ids=candidate_ids,
            status="partial" if missing_cells else status,
        )
        return EvidencePack(
            status="ok" if all_items else "empty",
            items=all_items,
            table_facts=all_table_facts,
            comparison=pack,
        )

    @classmethod
    def render_draft(cls, pack: ComparisonPack | dict[str, Any], items: list[EvidenceItem | dict[str, Any]]) -> str:
        """Render a grounded, structured fallback draft for the synthesizer."""
        raw_pack = pack.model_dump() if isinstance(pack, ComparisonPack) else pack
        raw_items = [item.model_dump() if isinstance(item, EvidenceItem) else item for item in items]
        lines = ["## Comparison evidence matrix"]
        if raw_pack.get("status") == "needs_selection":
            lines.append("Comparison scope needs confirmation; select at least two papers before generating a cross-paper conclusion.")
            candidates = raw_pack.get("candidate_document_ids") or [
                paper.get("document_id")
                for paper in raw_pack.get("papers", [])
                if isinstance(paper, dict)
            ]
            if candidates:
                lines.append("Candidate paper IDs: " + ", ".join(str(value) for value in candidates))
        for cell in raw_pack.get("cells", []):
            label = f"{cell.get('paper_title') or cell.get('paper_id')} — {cell.get('dimension')}"
            indexes = cell.get("evidence_indexes") or []
            if not indexes:
                lines.append(f"- {label}: evidence missing; do not infer.")
                continue
            lines.append(f"- {label} ({cell.get('status', 'supported')}):")
            for index in indexes:
                if 0 <= index < len(raw_items):
                    excerpt = str(raw_items[index].get("excerpt") or "").strip()
                    if excerpt:
                        lines.append(f"  - {excerpt} [{index}]")
        if raw_pack.get("missing_cells"):
            lines.append("\n## Unresolved cells")
            lines.extend(f"- {value}" for value in raw_pack["missing_cells"])
        return "\n".join(lines)

    @classmethod
    def _normalize_dimensions(cls, question: str, dimensions: list[str] | None) -> list[str]:
        requested = [str(item).strip() for item in dimensions or [] if str(item).strip()]
        result: list[str] = []
        for item in requested:
            result.append(cls._canonical_dimension(item))
        if not result:
            lowered = question.lower()
            # Keep the fixed base set stable.  Query-specific dimensions are
            # promoted first so the bounded per-cell retrieval spends its
            # budget on the user's wording.
            matched = [
                dimension
                for dimension in COMPARISON_DIMENSIONS
                if any(alias.lower() in lowered or alias in question for alias in cls._DIMENSION_ALIASES.get(dimension, ()))
            ]
            result = matched + [dimension for dimension in COMPARISON_DIMENSIONS if dimension not in matched]
        unique: list[str] = []
        for item in result:
            if item not in unique:
                unique.append(item)
        return unique[: cls.MAX_DIMENSIONS]

    @classmethod
    def _canonical_dimension(cls, value: str) -> str:
        key = re.sub(r"\s+", " ", value.strip().lower())
        for canonical, aliases in cls._DIMENSION_ALIASES.items():
            if key == canonical.lower() or any(alias.lower() == key for alias in aliases):
                return canonical
        return value.strip()[:120]

    @staticmethod
    def _dimension_query(question: str, dimension: str, title: str) -> str:
        return (
            f"{question}\nPaper under comparison: {title}\n"
            f"Comparison dimension: {dimension}. Retrieve exact page evidence for this dimension."
        )

    def _ensure_graph_edges(self, project_id: str, documents: list[Document]) -> list[ComparisonEdge]:
        """Persist pair metadata and optional SAC-KG-derived graph edges.

        ``compared_with`` is a comparison-session edge and is safe to create
        without SAC-KG.  All semantic edges (claims, entities, alignment and
        contradictions) are strictly gated by ``SAC_KG_ENABLED`` so a
        deployment that has never run SAC-KG does not issue Claim/Entity
        queries or present empty graph data as if it were meaningful.
        """
        edges: list[ComparisonEdge] = []

        def add_edge(
            *,
            source_type: str,
            source_id: str,
            relation_type: str,
            target_type: str,
            target_id: str,
            document_id: str | None = None,
            parse_version: str | None = None,
            evidence_chunk_id: str | None = None,
            confidence: float = 0.0,
        ) -> None:
            parse_key = parse_version or "legacy"
            try:
                existing = self.db.scalar(
                    select(KnowledgeEdge).where(
                        KnowledgeEdge.project_id == project_id,
                        KnowledgeEdge.source_type == source_type,
                        KnowledgeEdge.source_id == source_id,
                        KnowledgeEdge.relation_type == relation_type,
                        KnowledgeEdge.target_type == target_type,
                        KnowledgeEdge.target_id == target_id,
                        KnowledgeEdge.parse_version == parse_key,
                    )
                )
                if existing is None:
                    existing = KnowledgeEdge(
                        project_id=project_id,
                        source_type=source_type,
                        source_id=source_id,
                        relation_type=relation_type,
                        target_type=target_type,
                        target_id=target_id,
                        document_id=document_id,
                        parse_version=parse_key,
                        evidence_chunk_id=evidence_chunk_id,
                        confidence=confidence,
                        extraction_version=self.EXTRACTION_VERSION,
                        metadata_json={},
                    )
                    self.db.add(existing)
                    self.db.flush()
                edges.append(
                    ComparisonEdge(
                        id=existing.id,
                        source_type=source_type,
                        source_id=source_id,
                        relation_type=relation_type,
                        target_type=target_type,
                        target_id=target_id,
                        document_id=document_id,
                        parse_version=parse_key,
                        evidence_chunk_id=evidence_chunk_id,
                        confidence=float(existing.confidence or confidence),
                    )
                )
            except (OperationalError, IntegrityError) as exc:
                # A comparison must remain usable if an older deployment has
                # not migrated the additive edge table yet.
                self.db.rollback()
                logger.warning("Graph-lite edge persistence unavailable: %s", exc)
                edges.append(
                    ComparisonEdge(
                        source_type=source_type,
                        source_id=source_id,
                        relation_type=relation_type,
                        target_type=target_type,
                        target_id=target_id,
                        document_id=document_id,
                        parse_version=parse_key,
                        evidence_chunk_id=evidence_chunk_id,
                        confidence=confidence,
                    )
                )

        document_ids = [document.id for document in documents]
        document_versions = {
            document.id: (
                self.parse_version_map.get(document.id)
                if self.parse_version_map
                else document.active_parse_version
            )
            or "legacy"
            for document in documents
        }
        claims_by_document: dict[str, list[Claim]] = {}
        if self.sac_kg_enabled:
            claims = self.db.scalars(
                select(Claim).where(
                    Claim.project_id == project_id,
                    Claim.document_id.in_(document_ids),
                )
            ).all()
            for claim in claims:
                claims_by_document.setdefault(claim.document_id, []).append(claim)
                parse_version = document_versions.get(claim.document_id, "legacy")
                add_edge(
                    source_type="paper",
                    source_id=claim.document_id,
                    relation_type="reports",
                    target_type="claim",
                    target_id=claim.id,
                    document_id=claim.document_id,
                    parse_version=parse_version,
                    evidence_chunk_id=claim.evidence_chunk_id,
                    confidence=float(claim.confidence or 0.0),
                )
                if claim.evidence_chunk_id:
                    add_edge(
                        source_type="claim",
                        source_id=claim.id,
                        relation_type="supports",
                        target_type="chunk",
                        target_id=claim.evidence_chunk_id,
                        document_id=claim.document_id,
                        parse_version=parse_version,
                        evidence_chunk_id=claim.evidence_chunk_id,
                        confidence=float(claim.confidence or 0.0),
                    )
                # Claim predicates are not treated as facts by the graph
                # layer; these edges are only typed shortcuts for later
                # matrix retrieval.
                predicate = self._normalize_claim_text(claim.predicate)
                relation = None
                if any(token in predicate for token in ("use", "adopt", "employ", "method", "model")):
                    relation = "uses"
                elif any(token in predicate for token in ("evaluat", "benchmark", "dataset", "task", "test")):
                    relation = "evaluates_on"
                if relation:
                    target_name = claim.object_text or claim.subject
                    entity_id = self._entity_id_for_name(project_id, target_name)
                    if entity_id:
                        add_edge(
                            source_type="paper",
                            source_id=claim.document_id,
                            relation_type=relation,
                            target_type="entity",
                            target_id=entity_id,
                            document_id=claim.document_id,
                            parse_version=parse_version,
                            evidence_chunk_id=claim.evidence_chunk_id,
                            confidence=float(claim.confidence or 0.0),
                        )
        for left, right in combinations(documents, 2):
            pair_parse_version = self._pair_parse_version(
                document_versions.get(left.id, "legacy"),
                document_versions.get(right.id, "legacy"),
            )
            add_edge(
                source_type="paper",
                source_id=left.id,
                relation_type="compared_with",
                target_type="paper",
                target_id=right.id,
                document_id=left.id,
                parse_version=pair_parse_version,
                confidence=1.0,
            )

            if self.sac_kg_enabled:
                # Persist only conservative, auditable contradictions.  A pair
                # is contradictory when both claims are verified/high-confidence
                # and share the same normalized subject + predicate but report
                # different normalized objects.  The edge is tied to the left
                # claim's source chunk; the right claim remains the target so the
                # matrix can surface both page citations without treating a model
                # guess as a fact.
                left_claims = [
                    claim
                    for claim in claims_by_document.get(left.id, [])
                    if claim.evidence_chunk_id
                    and (
                        str(claim.verification_status or "").lower() == "verified"
                        or float(claim.confidence or 0.0) >= 0.8
                    )
                ]
                right_claims = [
                    claim
                    for claim in claims_by_document.get(right.id, [])
                    if claim.evidence_chunk_id
                    and (
                        str(claim.verification_status or "").lower() == "verified"
                        or float(claim.confidence or 0.0) >= 0.8
                    )
                ]
                for left_claim in left_claims:
                    for right_claim in right_claims:
                        if self._normalize_claim_text(left_claim.subject) != self._normalize_claim_text(right_claim.subject):
                            continue
                        if self._normalize_claim_text(left_claim.predicate) != self._normalize_claim_text(right_claim.predicate):
                            continue
                        if self._normalize_claim_text(left_claim.object_text) == self._normalize_claim_text(right_claim.object_text):
                            continue
                        add_edge(
                            source_type="claim",
                            source_id=left_claim.id,
                            relation_type="contradicts",
                            target_type="claim",
                            target_id=right_claim.id,
                            document_id=left.id,
                            parse_version=pair_parse_version,
                            evidence_chunk_id=left_claim.evidence_chunk_id,
                            confidence=min(
                                float(left_claim.confidence or 0.0),
                                float(right_claim.confidence or 0.0),
                            ),
                        )

                # Conservative entity alignment: only exact normalized names or
                # aliases that occur in claims from both selected papers qualify.
                # No embedding similarity or free-form LLM inference is used here.
                left_entities = self._entities_for_claims(project_id, claims_by_document.get(left.id, []))
                right_entities = self._entities_for_claims(project_id, claims_by_document.get(right.id, []))
                for normalized_name in sorted(set(left_entities) & set(right_entities)):
                    for left_entity_id in left_entities[normalized_name]:
                        for right_entity_id in right_entities[normalized_name]:
                            if left_entity_id == right_entity_id:
                                continue
                            add_edge(
                                source_type="entity",
                                source_id=left_entity_id,
                                relation_type="same_as",
                                target_type="entity",
                                target_id=right_entity_id,
                                document_id=left.id,
                                parse_version=pair_parse_version,
                                confidence=0.98,
                            )
        return edges

    def _detect_conflict_cells(
        self,
        cells: list[ComparisonEvidenceCell],
        items: list[EvidenceItem],
        documents: list[Document],
    ) -> list[str]:
        """Return cells backed by conservative cross-paper Claim conflicts.

        A conflict requires the same normalized subject + predicate, different
        normalized object values, verified/high-confidence claims, and an
        evidence chunk that is actually present in the corresponding matrix
        cell.  Generic prose disagreement is never promoted to a conflict.
        """
        if not self.sac_kg_enabled:
            return []
        del documents
        # Query only the selected documents; the lightweight helper keeps the
        # method usable in SQLite tests where a cell may contain no evidence.
        selected_document_ids = {
            str(item.document_id)
            for item in items
            if item.document_id
        }
        if not selected_document_ids:
            return []
        claims = self.db.scalars(
            select(Claim).where(Claim.document_id.in_(selected_document_ids))
        ).all()
        eligible = [
            claim
            for claim in claims
            if claim.evidence_chunk_id
            and (
                str(claim.verification_status or "").lower() == "verified"
                or float(claim.confidence or 0.0) >= 0.8
            )
        ]
        conflict_chunks: set[str] = set()
        for left, right in combinations(eligible, 2):
            if left.document_id == right.document_id:
                continue
            if self._normalize_claim_text(left.subject) != self._normalize_claim_text(right.subject):
                continue
            if self._normalize_claim_text(left.predicate) != self._normalize_claim_text(right.predicate):
                continue
            if self._normalize_claim_text(left.object_text) == self._normalize_claim_text(right.object_text):
                continue
            conflict_chunks.update(
                chunk for chunk in (left.evidence_chunk_id, right.evidence_chunk_id) if chunk
            )

        if not conflict_chunks:
            return []
        result: list[str] = []
        for cell in cells:
            for index in cell.evidence_indexes:
                if 0 <= index < len(items) and items[index].chunk_id in conflict_chunks:
                    result.append(f"{cell.paper_id}:{cell.dimension}")
                    break
        return sorted(set(result))

    @staticmethod
    def _normalize_claim_text(value: str | None) -> str:
        value = re.sub(r"[^\w\s]+", " ", str(value or "").casefold())
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _pair_parse_version(left: str, right: str) -> str:
        # Keep alignment edges isolated when either source is reparsed.
        value = f"compare:{left}|{right}"
        return value[:128]

    def _entity_id_for_name(self, project_id: str, value: str | None) -> str | None:
        normalized = self._normalize_claim_text(value)
        if not normalized:
            return None
        entities = self.db.scalars(select(Entity).where(Entity.project_id == project_id)).all()
        for entity in entities:
            names = [entity.name, *(entity.aliases or [])]
            if any(self._normalize_claim_text(name) == normalized for name in names):
                return entity.id
        return None

    def _entities_for_claims(
        self, project_id: str, claims: list[Claim]
    ) -> dict[str, list[str]]:
        entities = self.db.scalars(select(Entity).where(Entity.project_id == project_id)).all()
        result: dict[str, list[str]] = {}
        for entity in entities:
            names = [entity.name, *(entity.aliases or [])]
            normalized_names = {
                self._normalize_claim_text(name) for name in names if str(name or "").strip()
            }
            claim_text = {
                self._normalize_claim_text(claim.subject)
                for claim in claims
            }
            claim_text.update(self._normalize_claim_text(claim.object_text) for claim in claims)
            for name in sorted(normalized_names & claim_text):
                result.setdefault(name, []).append(entity.id)
        return result
