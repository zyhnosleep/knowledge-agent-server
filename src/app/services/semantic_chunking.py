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
    splitter_name: str
    splitter_version: str
    splitting_model: str
    semantic_boundary_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticChunker:
    def __init__(
        self,
        embedder: Any,
        token_counter: Any | None = None,
        *,
        parent_min_tokens: int | None = None,
        parent_target_tokens: int | None = None,
        parent_max_tokens: int | None = None,
        child_min_tokens: int | None = None,
        child_target_tokens: int | None = None,
        child_max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        break_percentile: int | None = None,
        splitter_name: str = "section_aware_semantic",
        splitter_version: str = "semantic-v1",
        splitting_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder
        self.token_counter = (
            StructuredEvidenceBuilder(
                tokenizer_name=settings.semantic_tokenizer_name
            ).estimate_tokens
            if token_counter is None
            else token_counter
        )
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
        self.splitter_name = splitter_name
        self.splitter_version = splitter_version
        embedder_model = next(
            (
                value
                for value in (
                    getattr(embedder, "model_name", None),
                    getattr(embedder, "model", None),
                    getattr(embedder, "name", None),
                )
                if isinstance(value, str) and value.strip()
            ),
            None,
        )
        self.splitting_model = (
            splitting_model
            or embedder_model
            or settings.semantic_splitting_model
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
                [(0, len(block.text), block.text)]
                if preformatted and block.text
                else self._sentence_ranges(block.text)
            )
            for sentence_index, (start, end, sentence) in enumerate(raw_texts):
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
                        block_char_start=start,
                        block_char_end=end,
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
            "literature cited",
            "cited literature",
            "sources",
            "参考文献",
            "引用文献",
            "文献",
            "参考资料",
            "引用资料",
        }
        for part in section_path:
            normalized = re.sub(
                r"^\s*(?:(?:\d+(?:\.\d+)*|[ivxlcdm]+|[一二三四五六七八九十]+)(?:[.、)]|\s+))\s*",
                "",
                part.strip().casefold(),
            ).rstrip(":：")
            if normalized in reference_names:
                return True
        return False

    @staticmethod
    def _sentence_ranges(text: str) -> list[tuple[int, int, str]]:
        if not text:
            return []
        nonterminal_abbreviations = {
            "dr",
            "mr",
            "mrs",
            "ms",
            "prof",
            "sr",
            "jr",
            "st",
            "vs",
            "fig",
            "eq",
            "sec",
            "ref",
        }
        terminal_capable_abbreviations = {"al", "etc", "e.g", "i.e"}
        closers = {'"', "'", "”", "’", "）", ")", "]"}
        ranges: list[tuple[int, int, str]] = []
        start = 0
        index = 0
        while index < len(text):
            character = text[index]
            boundary = character in "。！？"
            if character in "!?":
                probe = index + 1
                while probe < len(text) and text[probe] in closers:
                    probe += 1
                boundary = probe == len(text) or text[probe].isspace()
            elif character == ".":
                next_character = text[index + 1] if index + 1 < len(text) else ""
                previous_character = text[index - 1] if index else ""
                decimal = previous_character.isdigit() and next_character.isdigit()
                token_match = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$", text[start : index + 1])
                token = token_match.group(1).casefold() if token_match else ""
                initialism = "." in token and all(
                    1 <= len(part) <= 2 for part in token.split(".")
                )
                punctuation_context = (
                    not next_character
                    or next_character.isspace()
                    or next_character in closers
                )
                if token in nonterminal_abbreviations:
                    boundary = False
                elif token in terminal_capable_abbreviations or initialism:
                    probe = index + 1
                    while probe < len(text) and text[probe] in closers:
                        probe += 1
                    if probe < len(text) and not text[probe].isspace():
                        boundary = False
                    else:
                        while probe < len(text) and text[probe].isspace():
                            probe += 1
                        following = text[probe] if probe < len(text) else ""
                        boundary = punctuation_context and (
                            not following
                            or following.isupper()
                            or "\u3400" <= following <= "\u9fff"
                        )
                else:
                    boundary = not decimal and punctuation_context
            if not boundary:
                index += 1
                continue
            end = index + 1
            while end < len(text) and text[end] in closers:
                end += 1
            while end < len(text) and text[end].isspace():
                end += 1
            ranges.append((start, end, text[start:end]))
            start = end
            index = end
        if start < len(text):
            ranges.append((start, len(text), text[start:]))
        return ranges

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
            window_max = min(parent_max, self.child_token_limits[2])
            if unit.token_count <= window_max:
                expanded.append(unit)
                continue
            windows = self._lossless_token_windows(unit.text, window_max)
            for window_index, text in enumerate(windows):
                relative_start = sum(len(item) for item in windows[:window_index])
                token_count = self._count_tokens(text)
                expanded.append(
                    unit.copy_with(
                        text=text,
                        token_count=token_count,
                        token_window_split=True,
                        window_index=window_index,
                        block_char_start=unit.block_char_start + relative_start,
                        block_char_end=unit.block_char_start + relative_start + len(text),
                        unavoidable_overflow=token_count > window_max,
                        overflow_reason=(
                            "single_source_character_exceeds_max"
                            if token_count > window_max
                            else None
                        ),
                    )
                )

        parent_groups = self._semantic_groups(expanded, *self.parent_token_limits)
        structure_id = "narrative:" + hashlib.sha256(
            json.dumps(
                [segment.block_type, segment.section_path, self._source_ids(expanded)],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        structure_children: list[ChunkDraft] = []
        for parent_index, parent_group in enumerate(parent_groups):
            units = parent_group.units
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
                    "unavoidable_token_overflow": any(
                        unit.unavoidable_overflow for unit in units
                    ),
                    "overflow_reason": next(
                        (
                            unit.overflow_reason
                            for unit in units
                            if unit.overflow_reason is not None
                        ),
                        None,
                    ),
                    "parent_min_underflow": self._count_tokens(parent_text)
                    < self.parent_token_limits[0],
                    "undersized_reason": (
                        "section_too_short_or_max_prevents_rebalance"
                        if self._count_tokens(parent_text) < self.parent_token_limits[0]
                        else None
                    ),
                    "source_unit_ids": [unit.unit_id for unit in units],
                    "semantic_boundary_score": parent_group.boundary_score,
                    "boundary_reason": parent_group.boundary_reason,
                    "source_span_mapping": self._source_span_mapping(units),
                    "structural_separators": self._structural_separators(units),
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
            structure_children.extend(children)
        self._link_neighbors(structure_children)

    def _semantic_groups(
        self,
        units: list[_RawUnit],
        minimum: int,
        target: int,
        maximum: int,
    ) -> list[_ChunkGroup]:
        if not units:
            return []
        token_cache: dict[tuple[int, int], int] = {}

        def range_tokens(start: int, end: int) -> int:
            key = (start, end)
            if key not in token_cache:
                token_cache[key] = self._group_tokens(units[start:end])
            return token_cache[key]

        def largest_fitting_end(start: int) -> int:
            first_end = start + 1
            if range_tokens(start, first_end) > maximum:
                return first_end
            lower = first_end
            distance = 1
            while lower < len(units):
                distance *= 2
                probe = min(len(units), start + distance)
                if range_tokens(start, probe) > maximum:
                    upper = probe - 1
                    break
                lower = probe
            else:
                return lower
            while lower < upper:
                middle = (lower + upper + 1) // 2
                if range_tokens(start, middle) <= maximum:
                    lower = middle
                else:
                    upper = middle - 1
            return lower

        def first_end_reaching(start: int, end: int, threshold: int) -> int | None:
            if range_tokens(start, end) < threshold:
                return None
            lower = start + 1
            upper = end
            while lower < upper:
                middle = (lower + upper) // 2
                if range_tokens(start, middle) >= threshold:
                    upper = middle
                else:
                    lower = middle + 1
            return lower

        groups: list[list[_RawUnit]] = []
        boundary_audit: dict[str, tuple[float | None, str]] = {}
        start = 0
        while start < len(units):
            max_end = largest_fitting_end(start)

            candidates: list[tuple[float, int]] = []
            threshold = max(minimum, target)
            first_candidate = first_end_reaching(start, max_end, threshold)
            if first_candidate is not None:
                for end in range(first_candidate, min(max_end + 1, len(units))):
                    left = units[end - 1].embedding
                    right = units[end].embedding
                    if left is None or right is None:
                        raise RuntimeError("raw unit embedding missing")
                    candidates.append((self._cosine(left, right), end))

            if candidates:
                percentile_count = max(
                    1,
                    math.ceil(len(candidates) * self.break_percentile / 100),
                )
                bottom = sorted(candidates, key=lambda item: (item[0], item[1]))[
                    :percentile_count
                ]
                score, chosen_end = min(bottom, key=lambda item: item[1])
                chosen_tokens = range_tokens(start, chosen_end)
                max_tokens = range_tokens(start, max_end)
                if chosen_tokens > maximum or (
                    chosen_tokens < threshold <= max_tokens
                ):
                    chosen_end = max_end
                    score = None
                    reason = (
                        "section_end" if chosen_end >= len(units) else "max_tokens"
                    )
                else:
                    reason = "semantic_percentile"
            else:
                chosen_end = max_end
                score = None
                reason = "section_end" if chosen_end >= len(units) else "max_tokens"
            group = units[start:chosen_end]
            groups.append(group)
            boundary_audit[group[-1].unit_id] = (score, reason)
            start = chosen_end

        groups = self._rebalance_tail(groups, minimum, maximum)
        result: list[_ChunkGroup] = []
        for index, group in enumerate(groups):
            if index + 1 == len(groups):
                score, reason = None, "section_end"
            elif group[-1].unit_id in boundary_audit:
                score, reason = boundary_audit[group[-1].unit_id]
            else:
                left = group[-1].embedding
                right = groups[index + 1][0].embedding
                score = (
                    self._cosine(left, right)
                    if left is not None and right is not None
                    else None
                )
                reason = "min_rebalance"
            result.append(
                _ChunkGroup(
                    units=group,
                    boundary_score=score,
                    boundary_reason=reason,
                )
            )
        return result

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
                windows.append(remaining[0])
                remaining = remaining[1:]
                continue
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
                windows.append(remaining[0])
                remaining = remaining[1:]
                continue
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
        base_chunk_groups = self._semantic_groups(units, minimum, target, maximum)
        base_groups = [item.units for item in base_chunk_groups]

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
                        "unavoidable_token_overflow": any(
                            item.unavoidable_overflow for item in base_group
                        ),
                        "overflow_reason": next(
                            (
                                item.overflow_reason
                                for item in base_group
                                if item.overflow_reason is not None
                            ),
                            None,
                        ),
                        "source_sentence_token_window": any(
                            item.token_window_split for item in group
                        ),
                        "child_min_underflow": self._count_tokens(text) < minimum,
                        "undersized_reason": (
                            "structure_too_short_or_max_prevents_rebalance"
                            if self._count_tokens(text) < minimum
                            else None
                        ),
                        "source_unit_ids": [item.unit_id for item in group],
                        "semantic_boundary_score": base_chunk_groups[
                            child_index
                        ].boundary_score,
                        "boundary_reason": base_chunk_groups[
                            child_index
                        ].boundary_reason,
                        "source_span_mapping": self._source_span_mapping(group),
                        "structural_separators": self._structural_separators(group),
                    },
                )
            )
        self._link_neighbors(children)
        return children

    def _rebalance_tail(
        self,
        groups: list[list[_RawUnit]],
        minimum: int,
        maximum: int,
    ) -> list[list[_RawUnit]]:
        if len(groups) < 2 or self._group_tokens(groups[-1]) >= minimum:
            return groups
        previous = groups[-2]
        tail = groups[-1]
        if self._group_tokens([*previous, *tail]) <= maximum:
            previous.extend(groups.pop())
            return groups

        candidate_previous = list(previous)
        candidate_tail = list(tail)
        while self._group_tokens(candidate_tail) < minimum and len(candidate_previous) > 1:
            candidate_tail.insert(0, candidate_previous.pop())
            if (
                self._group_tokens(candidate_previous) > maximum
                or self._group_tokens(candidate_tail) > maximum
            ):
                return groups
        if (
            self._group_tokens(candidate_previous) >= minimum
            and self._group_tokens(candidate_tail) >= minimum
        ):
            groups[-2] = candidate_previous
            groups[-1] = candidate_tail
        return groups

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
            contributors = self._structured_contributors(
                figure.nearby_block_ids,
                document.blocks,
            )
            source_parent = self._structured_builder.figure_chunk(figure, contributors)
            source_children = [source_parent.model_copy(deep=True)]
            structure_id = figure.figure_id
            structure_source_ids, structure_source_spans = self._structured_sources(
                block,
                source_parent.source_spans,
                contributors,
            )
        elif block.block_type == "formula":
            formula = next(
                (item for item in document.formulas if item.formula_id == block.formula_id),
                None,
            )
            if formula is None:
                raise ValueError(f"structured formula {block.formula_id!r} is missing")
            contributors = self._structured_contributors(
                formula.nearby_block_ids,
                document.blocks,
            )
            source_parent = self._structured_builder.formula_chunk(formula, contributors)
            source_children = [source_parent.model_copy(deep=True)]
            structure_id = formula.formula_id
            structure_source_ids, structure_source_spans = self._structured_sources(
                block,
                source_parent.source_spans,
                contributors,
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
                "semantic_boundary_score": None,
                "boundary_reason": "structured_boundary",
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
                "semantic_boundary_score": None,
                "boundary_reason": "structured_boundary",
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
    def _structured_contributors(
        cls,
        nearby_block_ids: list[str],
        blocks: Iterable[CanonicalBlock],
    ) -> list[CanonicalBlock]:
        wanted = set(nearby_block_ids)
        return [
            block
            for block in blocks
            if block.block_id in wanted
            and block.retrievable
            and cls._is_retrievable_narrative(block, list(block.section_path))
            and not block_is_generated(block)
            and block.text.strip()
        ]

    @classmethod
    def _structured_sources(
        cls,
        structure_block: CanonicalBlock,
        structure_spans: Iterable[SourceSpan],
        contributors: Iterable[CanonicalBlock],
    ) -> tuple[list[str], list[SourceSpan]]:
        contributors = list(contributors)
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
            splitter_name=self.splitter_name,
            splitter_version=self.splitter_version,
            splitting_model=self.splitting_model,
            semantic_boundary_score=metadata.get("semantic_boundary_score"),
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
        if not values:
            return ""
        parts = [values[0].text]
        for previous, current in zip(values, values[1:]):
            if previous.block_id != current.block_id:
                parts.append("\n\n")
            parts.append(current.text)
        return "".join(parts)

    @staticmethod
    def _structural_separators(units: Iterable[_RawUnit]) -> list[dict[str, Any]]:
        values = list(units)
        return [
            {
                "text": "\n\n",
                "source_backed": False,
                "between_block_ids": [previous.block_id, current.block_id],
            }
            for previous, current in zip(values, values[1:])
            if previous.block_id != current.block_id
        ]

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
        mapped: list[SourceSpan] = []
        for unit in units:
            for span in unit.source_spans:
                if (
                    span.char_start is not None
                    and span.char_end is not None
                    and span.char_end - span.char_start >= unit.block_char_end
                ):
                    mapped.append(
                        span.model_copy(
                            update={
                                "char_start": span.char_start + unit.block_char_start,
                                "char_end": span.char_start + unit.block_char_end,
                            }
                        )
                    )
                else:
                    mapped.append(span)
        deduplicated = cls._deduplicate_spans(mapped)
        merged: list[SourceSpan] = []
        for span in deduplicated:
            if merged and cls._spans_are_adjacent(merged[-1], span):
                merged[-1] = merged[-1].model_copy(update={"char_end": span.char_end})
            else:
                merged.append(span)
        return merged

    @staticmethod
    def _source_span_mapping(units: Iterable[_RawUnit]) -> str:
        for unit in units:
            if not unit.source_spans:
                return "approximate"
            for span in unit.source_spans:
                if (
                    span.char_start is None
                    or span.char_end is None
                    or span.char_end - span.char_start < unit.block_char_end
                ):
                    return "approximate"
        return "exact"

    @staticmethod
    def _spans_are_adjacent(left: SourceSpan, right: SourceSpan) -> bool:
        if left.char_end is None or right.char_start is None:
            return False
        left_locator = left.model_dump(mode="json")
        right_locator = right.model_dump(mode="json")
        left_locator.pop("char_start", None)
        left_locator.pop("char_end", None)
        right_locator.pop("char_start", None)
        right_locator.pop("char_end", None)
        return left_locator == right_locator and left.char_end == right.char_start

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
    block_char_start: int = 0
    block_char_end: int = 0
    unavoidable_overflow: bool = False
    overflow_reason: str | None = None

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
            "block_char_start": self.block_char_start,
            "block_char_end": self.block_char_end,
            "unavoidable_overflow": self.unavoidable_overflow,
            "overflow_reason": self.overflow_reason,
        }
        values.update(changes)
        return _RawUnit(**values)


@dataclass
class _ChunkGroup:
    units: list[_RawUnit]
    boundary_score: float | None
    boundary_reason: str


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
