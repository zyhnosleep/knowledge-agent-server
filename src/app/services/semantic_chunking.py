from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.services.canonical_models import CanonicalBlock, CanonicalDocument, SourceSpan
from app.services.canonical_provenance import block_is_generated
from app.services.structured_evidence import (
    StructuredEvidenceBuilder,
    StructuredEvidenceChunk,
)


class ChunkDraft(BaseModel):
    """Serializable in-memory chunk awaiting persistence ID mapping."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    local_id: str
    parse_version: str
    chunk_role: Literal["parent", "child"]
    block_type: Literal["narrative", "table", "figure", "formula", "caption", "appendix"]
    text: str
    embedding_text: str
    token_count: int = Field(ge=0)
    parent_local_id: str | None = None
    previous_child_local_id: str | None = None
    next_child_local_id: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    ordinal: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticChunker:
    def __init__(
        self,
        embedder: Any,
        token_counter: Any,
        *,
        parent_min_tokens: int | None = None,
        parent_target_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        child_min_tokens: int | None = None,
        child_target_tokens: int | None = None,
        child_max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        break_percentile: int | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder
        self.token_counter = token_counter
        self.parent_token_limits = (
            settings.semantic_parent_min_tokens
            if parent_min_tokens is None
            else parent_min_tokens,
            settings.semantic_parent_target_tokens
            if parent_target_tokens is None
            else parent_target_tokens,
            settings.semantic_parent_max_tokens
            if parent_max_tokens is None
            else parent_max_tokens,
        )
        self.child_token_limits = (
            settings.semantic_child_min_tokens
            if child_min_tokens is None
            else child_min_tokens,
            settings.semantic_child_target_tokens
            if child_target_tokens is None
            else child_target_tokens,
            settings.semantic_child_max_tokens
            if child_max_tokens is None
            else child_max_tokens,
        )
        self.overlap_tokens = (
            settings.semantic_child_overlap_tokens
            if overlap_tokens is None
            else overlap_tokens
        )
        self.break_percentile = (
            settings.semantic_break_percentile
            if break_percentile is None
            else break_percentile
        )
        self._validate_configuration()
        self._structured_builder = StructuredEvidenceBuilder(token_counter=self._count_tokens)

    def build(self, document: CanonicalDocument) -> list[ChunkDraft]:
        entries, raw_units = self._document_entries(document)
        if raw_units:
            self._attach_embeddings(raw_units)

        drafts: list[ChunkDraft] = []
        for entry in entries:
            if isinstance(entry, _NarrativeSegment):
                self._append_narrative(entry, document.parse_version, drafts)
            else:
                self._append_structured(entry, document, drafts)
        return drafts

    def _validate_configuration(self) -> None:
        if not (
            0 < self.parent_token_limits[0]
            <= self.parent_token_limits[1]
            <= self.parent_token_limits[2]
        ):
            raise ValueError("parent token limits must satisfy 0 < min <= target <= max")
        if not (
            0 < self.child_token_limits[0]
            <= self.child_token_limits[1]
            <= self.child_token_limits[2]
        ):
            raise ValueError("child token limits must satisfy 0 < min <= target <= max")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if not 1 <= self.break_percentile <= 99:
            raise ValueError("break_percentile must be between 1 and 99")

    def _count_tokens(self, text: str) -> int:
        counter = self.token_counter
        if callable(counter):
            count = counter(text)
        elif hasattr(counter, "count") and callable(counter.count):
            count = counter.count(text)
        else:
            raise TypeError("token_counter must be callable or expose count(text)")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("token counter must return a non-negative integer")
        return count

    def _document_entries(
        self, document: CanonicalDocument
    ) -> tuple[list[_NarrativeSegment | _StructuredEntry], list[_RawUnit]]:
        entries: list[_NarrativeSegment | _StructuredEntry] = []
        raw_units: list[_RawUnit] = []
        current: _NarrativeSegment | None = None
        heading_path: list[str] = []

        def flush() -> None:
            nonlocal current
            if current is not None and current.units:
                entries.append(current)
                raw_units.extend(current.units)
            current = None

        for block in sorted(document.blocks, key=lambda item: item.reading_order):
            if block.block_type == "heading":
                flush()
                heading_path = list(block.section_path) or (
                    [block.text.strip()] if block.text.strip() else []
                )
                continue
            section_path = list(block.section_path) or list(heading_path)
            if not block.retrievable or self._is_reference_section(section_path):
                flush()
                continue
            if block.block_type in {"table", "figure", "formula"}:
                flush()
                entries.append(_StructuredEntry(block=block, section_path=section_path))
                continue
            if not self._is_retrievable_narrative(block, section_path):
                flush()
                continue

            format_kind = str(block.metadata.get("kind") or "text").strip().casefold()
            preformatted = format_kind in {
                "code",
                "code_block",
                "fenced_code",
                "html_pre",
                "pre",
                "preformatted",
            }
            key = (
                block.block_type,
                format_kind,
                block.block_id if preformatted else "",
                tuple(section_path),
            )
            if current is None or current.key != key:
                flush()
                current = _NarrativeSegment(
                    key=key,
                    block_type=block.block_type,
                    section_path=section_path,
                )
            raw_texts = (
                [block.text]
                if preformatted and block.text
                else self._sentences(block.text)
            )
            for sentence_index, sentence in enumerate(raw_texts):
                current.units.append(
                    _RawUnit(
                        text=sentence,
                        token_count=self._count_tokens(sentence),
                        block_id=block.block_id,
                        block_type=block.block_type,
                        section_path=section_path,
                        source_spans=list(block.source_spans),
                        sentence_id=f"{block.block_id}:{sentence_index}",
                        preformatted=preformatted,
                    )
                )
        flush()
        return entries, raw_units

    @staticmethod
    def _is_retrievable_narrative(
        block: CanonicalBlock, section_path: list[str]
    ) -> bool:
        if block.block_type not in {"narrative", "caption", "appendix"}:
            return False
        return not SemanticChunker._is_reference_section(section_path)

    @staticmethod
    def _is_reference_section(section_path: list[str]) -> bool:
        reference_names = {
            "reference",
            "references",
            "bibliography",
            "works cited",
            "参考文献",
            "引用文献",
        }
        for part in section_path:
            normalized = re.sub(
                r"^\s*(?:(?:\d+(?:\.\d+)*)|(?:[ivxlcdm]+)|(?:[一二三四五六七八九十]+))[.、)]?\s*",
                "",
                part.strip().casefold(),
            ).rstrip(":：")
            if normalized in reference_names:
                return True
        return False

    @staticmethod
    def _sentences(text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        pattern = re.compile(r".+?(?:[.!?。！？]+[\"'”’）)\]]*|$)", re.DOTALL)
        return [
            match.group(0).strip()
            for match in pattern.finditer(normalized)
            if match.group(0).strip()
        ]

    def _attach_embeddings(self, units: list[_RawUnit]) -> None:
        texts = [unit.text for unit in units]
        if hasattr(self.embedder, "embed") and callable(self.embedder.embed):
            response = self.embedder.embed(texts)
        elif callable(self.embedder):
            response = self.embedder(texts)
        else:
            raise TypeError("embedder must be callable or expose embed(list[str])")
        try:
            vectors = list(response)
        except TypeError as exc:
            raise ValueError("embedder must return one vector per raw unit") from exc
        if len(vectors) != len(units):
            raise ValueError("embedding batch length does not match raw unit count")

        expected_dimension: int | None = None
        for unit, raw_vector in zip(units, vectors, strict=True):
            try:
                vector = list(raw_vector)
            except TypeError as exc:
                raise ValueError("embedding vector must be an iterable of finite numbers") from exc
            if not vector:
                raise ValueError("embedding vector dimension must be positive")
            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ValueError("embedding vector dimension mismatch")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise ValueError("embedding vectors must contain finite numbers")
            unit.embedding = [float(value) for value in vector]
            unit.zero_vector = math.sqrt(sum(value * value for value in unit.embedding)) == 0.0

    def _append_narrative(
        self,
        segment: _NarrativeSegment,
        parse_version: str,
        drafts: list[ChunkDraft],
    ) -> None:
        expanded: list[_RawUnit] = []
        parent_max = self.parent_token_limits[2]
        for unit in segment.units:
            window_max = (
                min(parent_max, self.child_token_limits[2])
                if unit.preformatted
                else parent_max
            )
            if unit.token_count <= window_max:
                expanded.append(unit)
                continue
            windows = (
                self._lossless_token_windows(unit.text, window_max)
                if unit.preformatted
                else self._safe_token_windows(unit.text, window_max)
            )
            for window_index, text in enumerate(windows):
                expanded.append(
                    unit.copy_with(
                        text=text,
                        token_count=self._count_tokens(text),
                        token_window_split=True,
                        window_index=window_index,
                    )
                )

        parent_groups = self._parent_groups(expanded)
        structure_id = "narrative:" + hashlib.sha256(
            json.dumps(
                [segment.block_type, segment.section_path, self._source_ids(expanded)],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        for parent_index, units in enumerate(parent_groups):
            parent_text = self._join_units(units)
            parent_id = self._local_id(
                parse_version,
                self._source_ids(units),
                structure_id,
                "parent",
                parent_index,
                parent_text,
            )
            parent = self._draft(
                local_id=parent_id,
                parse_version=parse_version,
                chunk_role="parent",
                block_type=segment.block_type,
                text=parent_text,
                parent_local_id=None,
                source_block_ids=self._source_ids(units),
                source_spans=self._source_spans(units),
                section_path=segment.section_path,
                ordinal=len(drafts),
                metadata={
                    "semantic_zero_vector_fallback": any(unit.zero_vector for unit in units),
                    "token_window_split": any(unit.token_window_split for unit in units),
                    "parent_min_underflow": self._count_tokens(parent_text)
                    < self.parent_token_limits[0],
                    "source_unit_ids": [unit.unit_id for unit in units],
                },
            )
            drafts.append(parent)
            children = self._child_drafts(
                units,
                parent,
                structure_id=f"{structure_id}:parent:{parent_index}",
                start_ordinal=len(drafts),
            )
            drafts.extend(children)

    def _parent_groups(self, units: list[_RawUnit]) -> list[list[_RawUnit]]:
        if not units:
            return []
        minimum, target, maximum = self.parent_token_limits
        semantic_boundaries = self._low_similarity_boundaries(units)
        groups: list[list[_RawUnit]] = []
        current: list[_RawUnit] = []
        current_tokens = 0
        for index, unit in enumerate(units):
            if current and self._group_tokens([*current, unit]) > maximum:
                groups.append(current)
                current = []
            current.append(unit)
            current_tokens = self._group_tokens(current)
            if (
                index < len(units) - 1
                and current_tokens >= minimum
                and current_tokens >= target
                and index in semantic_boundaries
            ):
                groups.append(current)
                current = []
                current_tokens = 0
        if current:
            groups.append(current)
        if (
            len(groups) > 1
            and self._group_tokens(groups[-1]) < minimum
            and self._group_tokens([*groups[-2], *groups[-1]]) <= maximum
        ):
            groups[-2].extend(groups.pop())
        return groups

    def _low_similarity_boundaries(self, units: list[_RawUnit]) -> set[int]:
        scored: list[tuple[float, int]] = []
        for index in range(len(units) - 1):
            left = units[index]
            right = units[index + 1]
            if left.embedding is None or right.embedding is None:
                raise RuntimeError("raw unit embedding missing")
            scored.append((self._cosine(left.embedding, right.embedding), index))
        if not scored:
            return set()
        count = max(1, math.ceil(len(scored) * self.break_percentile / 100))
        ranked = sorted(scored, key=lambda item: (item[0], item[1]))[:count]
        return {index for _score, index in ranked}

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)

    def _safe_token_windows(self, text: str, maximum: int) -> list[str]:
        atoms = re.findall(r"\S+\s*", text)
        if not atoms:
            atoms = list(text)
        windows: list[str] = []
        current = ""
        for atom in atoms:
            candidate = current + atom
            if current and self._count_tokens(candidate.strip()) > maximum:
                windows.append(current.strip())
                current = ""
            if self._count_tokens(atom.strip()) <= maximum:
                current += atom
                continue
            if current:
                windows.append(current.strip())
                current = ""
            windows.extend(self._split_oversized_atom(atom.strip(), maximum))
        if current.strip():
            windows.append(current.strip())
        if not windows or any(self._count_tokens(item) > maximum for item in windows):
            raise ValueError("token counter cannot produce a lossless parent window within max")
        return windows

    def _split_oversized_atom(self, text: str, maximum: int) -> list[str]:
        windows: list[str] = []
        remaining = text
        while remaining:
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self._count_tokens(remaining[:middle]) <= maximum:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValueError("token counter reports one source character above parent max")
            windows.append(remaining[:best])
            remaining = remaining[best:]
        return windows

    def _lossless_token_windows(self, text: str, maximum: int) -> list[str]:
        windows: list[str] = []
        remaining = text
        while remaining:
            if self._count_tokens(remaining) <= maximum:
                windows.append(remaining)
                break
            low, high = 1, len(remaining)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self._count_tokens(remaining[:middle]) <= maximum:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValueError("token counter reports one source character above parent max")
            windows.append(remaining[:best])
            remaining = remaining[best:]
        return windows

    def _child_drafts(
        self,
        units: list[_RawUnit],
        parent: ChunkDraft,
        *,
        structure_id: str,
        start_ordinal: int,
    ) -> list[ChunkDraft]:
        minimum, target, maximum = self.child_token_limits
        base_groups: list[list[_RawUnit]] = []
        current: list[_RawUnit] = []
        current_tokens = 0
        for unit in units:
            if current and self._group_tokens([*current, unit]) > maximum:
                base_groups.append(current)
                current = []
            current.append(unit)
            current_tokens = self._group_tokens(current)
            if current_tokens >= target:
                base_groups.append(current)
                current = []
                current_tokens = 0
        if current:
            base_groups.append(current)
        if (
            len(base_groups) > 1
            and self._group_tokens(base_groups[-1]) < minimum
            and self._group_tokens([*base_groups[-2], *base_groups[-1]]) <= maximum
        ):
            base_groups[-2].extend(base_groups.pop())

        child_groups: list[tuple[list[_RawUnit], int]] = []
        for index, group in enumerate(base_groups):
            overlap: list[_RawUnit] = []
            if index > 0 and self.overlap_tokens:
                overlap = self._whole_sentence_overlap(
                    base_groups[index - 1],
                    group,
                    maximum,
                )
            child_groups.append(([*overlap, *group], len(overlap)))

        children: list[ChunkDraft] = []
        for child_index, (group, overlap_count) in enumerate(child_groups):
            text = self._join_units(group)
            source_ids = self._source_ids(group)
            local_id = self._local_id(
                parent.parse_version,
                source_ids,
                structure_id,
                "child",
                child_index,
                text,
            )
            base_group = base_groups[child_index]
            children.append(
                self._draft(
                    local_id=local_id,
                    parse_version=parent.parse_version,
                    chunk_role="child",
                    block_type=parent.block_type,
                    text=text,
                    parent_local_id=parent.local_id,
                    source_block_ids=source_ids,
                    source_spans=self._source_spans(group),
                    section_path=parent.section_path,
                    ordinal=start_ordinal + child_index,
                    metadata={
                        "overlap_sentence_count": overlap_count,
                        "overlap_tokens": self._group_tokens(group[:overlap_count]),
                        "single_sentence_overflow": len(base_group) == 1
                        and base_group[0].token_count > maximum,
                        "source_sentence_token_window": any(
                            item.token_window_split for item in group
                        ),
                        "child_min_underflow": self._count_tokens(text) < minimum,
                        "source_unit_ids": [item.unit_id for item in group],
                    },
                )
            )
        self._link_neighbors(children)
        return children

    def _whole_sentence_overlap(
        self,
        previous: list[_RawUnit],
        current: list[_RawUnit],
        maximum: int,
    ) -> list[_RawUnit]:
        selected: list[_RawUnit] = []
        for unit in reversed(previous):
            candidate = [unit, *reversed(selected)]
            if self._group_tokens(candidate) > self.overlap_tokens:
                break
            if self._group_tokens([*candidate, *current]) > maximum:
                break
            selected.append(unit)
        if (
            not selected
            and previous
            and self._group_tokens([previous[-1], *current]) <= maximum
        ):
            selected.append(previous[-1])
        selected.reverse()
        return selected

    def _append_structured(
        self,
        entry: _StructuredEntry,
        document: CanonicalDocument,
        drafts: list[ChunkDraft],
    ) -> None:
        block = entry.block
        source_parent: StructuredEvidenceChunk
        source_children: list[StructuredEvidenceChunk]
        structure_id: str
        structure_source_ids = [block.block_id]
        structure_source_spans: list[SourceSpan] = []
        if block.block_type == "table":
            table = next(
                (item for item in document.tables if item.table_id == block.table_id),
                None,
            )
            if table is None:
                raise ValueError(f"structured table {block.table_id!r} is missing")
            source_parent, source_children = self._structured_builder.table_chunks(
                table, max_tokens=self.child_token_limits[2]
            )
            structure_id = table.table_id
            structure_source_spans = self._deduplicate_spans(
                [*block.source_spans, *source_parent.source_spans]
            )
        elif block.block_type == "figure":
            figure = next(
                (item for item in document.figures if item.figure_id == block.figure_id),
                None,
            )
            if figure is None:
                raise ValueError(f"structured figure {block.figure_id!r} is missing")
            source_parent = self._structured_builder.figure_chunk(figure, document.blocks)
            source_children = [source_parent.model_copy(deep=True)]
            structure_id = figure.figure_id
            structure_source_ids, structure_source_spans = self._structured_sources(
                block,
                figure.nearby_block_ids,
                source_parent.source_spans,
                document.blocks,
            )
        elif block.block_type == "formula":
            formula = next(
                (item for item in document.formulas if item.formula_id == block.formula_id),
                None,
            )
            if formula is None:
                raise ValueError(f"structured formula {block.formula_id!r} is missing")
            source_parent = self._structured_builder.formula_chunk(formula, document.blocks)
            source_children = [source_parent.model_copy(deep=True)]
            structure_id = formula.formula_id
            structure_source_ids, structure_source_spans = self._structured_sources(
                block,
                formula.nearby_block_ids,
                source_parent.source_spans,
                document.blocks,
            )
        else:  # pragma: no cover - guarded by entry construction
            raise ValueError(f"unsupported structured block type: {block.block_type}")

        parent_id = self._local_id(
            document.parse_version,
            structure_source_ids,
            structure_id,
            "parent",
            0,
            source_parent.text,
        )
        parent = self._draft(
            local_id=parent_id,
            parse_version=document.parse_version,
            chunk_role="parent",
            block_type=block.block_type,
            text=source_parent.text,
            embedding_text=source_parent.embedding_text,
            parent_local_id=None,
            source_block_ids=structure_source_ids,
            source_spans=structure_source_spans,
            section_path=entry.section_path,
            ordinal=len(drafts),
            metadata={
                **source_parent.metadata,
                "structured_chunk_id": source_parent.chunk_id,
            },
        )
        drafts.append(parent)

        children: list[ChunkDraft] = []
        for child_index, source_child in enumerate(source_children):
            child_id = self._local_id(
                document.parse_version,
                structure_source_ids,
                structure_id,
                "child",
                child_index,
                source_child.text,
            )
            metadata = {
                **source_child.metadata,
                "structured_chunk_id": source_child.chunk_id,
            }
            if block.block_type in {"figure", "formula"}:
                metadata["source_faithful_single_child"] = True
            children.append(
                self._draft(
                    local_id=child_id,
                    parse_version=document.parse_version,
                    chunk_role="child",
                    block_type=block.block_type,
                    text=source_child.text,
                    embedding_text=source_child.embedding_text,
                    parent_local_id=parent_id,
                    source_block_ids=structure_source_ids,
                    source_spans=(
                        source_child.source_spans
                        if block.block_type == "table"
                        else structure_source_spans
                    ),
                    section_path=entry.section_path,
                    ordinal=len(drafts) + child_index,
                    metadata=metadata,
                )
            )
        self._link_neighbors(children)
        drafts.extend(children)

    @classmethod
    def _structured_sources(
        cls,
        structure_block: CanonicalBlock,
        nearby_block_ids: list[str],
        structure_spans: Iterable[SourceSpan],
        blocks: Iterable[CanonicalBlock],
    ) -> tuple[list[str], list[SourceSpan]]:
        wanted = set(nearby_block_ids)
        contributors = [
            block
            for block in blocks
            if block.block_id in wanted
            and block.block_type in {"narrative", "appendix"}
            and not block_is_generated(block)
            and block.text.strip()
        ]
        source_ids = [structure_block.block_id]
        for block in contributors:
            if block.block_id not in source_ids:
                source_ids.append(block.block_id)
        spans = cls._deduplicate_spans(
            [
                *structure_block.source_spans,
                *structure_spans,
                *(span for item in contributors for span in item.source_spans),
            ]
        )
        return source_ids, spans

    def _draft(
        self,
        *,
        local_id: str,
        parse_version: str,
        chunk_role: Literal["parent", "child"],
        block_type: str,
        text: str,
        parent_local_id: str | None,
        source_block_ids: list[str],
        source_spans: Iterable[SourceSpan],
        section_path: list[str],
        ordinal: int,
        metadata: dict[str, Any],
        embedding_text: str | None = None,
    ) -> ChunkDraft:
        return ChunkDraft(
            local_id=local_id,
            parse_version=parse_version,
            chunk_role=chunk_role,
            block_type=block_type,
            text=text,
            embedding_text=text if embedding_text is None else embedding_text,
            token_count=self._count_tokens(text if embedding_text is None else embedding_text),
            parent_local_id=parent_local_id,
            source_block_ids=source_block_ids,
            source_spans=self._deduplicate_spans(source_spans),
            section_path=section_path,
            ordinal=ordinal,
            metadata=metadata,
        )

    @staticmethod
    def _link_neighbors(children: list[ChunkDraft]) -> None:
        for index, child in enumerate(children):
            child.previous_child_local_id = children[index - 1].local_id if index else None
            child.next_child_local_id = (
                children[index + 1].local_id if index + 1 < len(children) else None
            )

    @staticmethod
    def _join_units(units: Iterable[_RawUnit]) -> str:
        values = list(units)
        if values and all(unit.preformatted for unit in values):
            return "".join(unit.text for unit in values)
        return " ".join(unit.text.strip() for unit in values if unit.text.strip())

    def _group_tokens(self, units: Iterable[_RawUnit]) -> int:
        return self._count_tokens(self._join_units(units))

    @staticmethod
    def _source_ids(units: Iterable[_RawUnit]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for unit in units:
            if unit.block_id not in seen:
                result.append(unit.block_id)
                seen.add(unit.block_id)
        return result

    @classmethod
    def _source_spans(cls, units: Iterable[_RawUnit]) -> list[SourceSpan]:
        return cls._deduplicate_spans(
            span for unit in units for span in unit.source_spans
        )

    @staticmethod
    def _deduplicate_spans(spans: Iterable[SourceSpan]) -> list[SourceSpan]:
        result: list[SourceSpan] = []
        seen: set[str] = set()
        for span in spans:
            key = json.dumps(
                span.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            if key not in seen:
                result.append(span)
                seen.add(key)
        return result

    @staticmethod
    def _local_id(
        parse_version: str,
        source_block_ids: list[str],
        structure_id: str,
        role: str,
        ordinal: int,
        text: str,
    ) -> str:
        payload = json.dumps(
            {
                "parse_version": parse_version,
                "source_block_ids": source_block_ids,
                "structure_id": structure_id,
                "role": role,
                "ordinal": ordinal,
                "source_fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"draft-{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass
class _RawUnit:
    text: str
    token_count: int
    block_id: str
    block_type: str
    section_path: list[str]
    source_spans: list[SourceSpan]
    sentence_id: str
    embedding: list[float] | None = None
    zero_vector: bool = False
    token_window_split: bool = False
    window_index: int | None = None
    preformatted: bool = False

    @property
    def unit_id(self) -> str:
        suffix = "" if self.window_index is None else f":window:{self.window_index}"
        return f"{self.sentence_id}{suffix}"

    def copy_with(self, **changes: Any) -> _RawUnit:
        values = {
            "text": self.text,
            "token_count": self.token_count,
            "block_id": self.block_id,
            "block_type": self.block_type,
            "section_path": list(self.section_path),
            "source_spans": list(self.source_spans),
            "sentence_id": self.sentence_id,
            "embedding": self.embedding,
            "zero_vector": self.zero_vector,
            "token_window_split": self.token_window_split,
            "window_index": self.window_index,
            "preformatted": self.preformatted,
        }
        values.update(changes)
        return _RawUnit(**values)


@dataclass
class _NarrativeSegment:
    key: tuple[str, str, str, tuple[str, ...]]
    block_type: str
    section_path: list[str]
    units: list[_RawUnit] = field(default_factory=list)


@dataclass
class _StructuredEntry:
    block: CanonicalBlock
    section_path: list[str]
