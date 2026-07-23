from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.services.ai import ContextualizationOllamaClient
from app.services.semantic_chunking import ChunkDraft


RETRIEVABLE_BLOCK_TYPES = frozenset(
    {"narrative", "table", "figure", "formula", "caption", "appendix"}
)
_COMMON_ENGLISH_TERMS = frozenset(
    {
        "AI",
        "API",
        "CPU",
        "GPU",
        "JSON",
        "LLM",
        "ML",
        "NLP",
        "OCR",
        "PDF",
        "RAG",
        "SOTA",
        "e.g.",
        "i.e.",
    }
)
_RELATION_MARKERS = (
    "关系",
    "说明",
    "对应",
    "用于",
    "属于",
    "部分",
    "章节",
    "论文",
    "研究",
    "方法",
    "模型",
    "数据集",
    "指标",
    "实验",
    "表",
    "图",
    "公式",
    "附录",
)
_NAMED_METRIC_MARKERS = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "bleu",
    "rouge",
    "auc",
    "map",
    "ndcg",
    "perplexity",
    "准确率",
    "精确率",
    "召回率",
    "特异度",
    "困惑度",
)


class DocumentContext(BaseModel):
    """Source document context supplied verbatim to contextualization."""

    model_config = ConfigDict(extra="forbid")

    title: str
    source_abstract: str | None = None
    section_outline: list[str] = Field(default_factory=list)


class ContextualPrefixItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_id: str
    prefix: str


class ContextualPrefixBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ContextualPrefixItem]


class ContextualizedChunk(ChunkDraft):
    """Contextualized child fields map directly to DocumentChunk columns."""

    contextual_prefix: str
    contextualization_model: str
    contextualization_version: str
    contextualization_prompt_version: str
    contextualized_at: datetime


class ContextualizationFailed(RuntimeError):
    def __init__(self, errors: Mapping[str, str]) -> None:
        self.errors = dict(errors)
        self.failed_child_ids = set(self.errors)
        details = "; ".join(f"{child_id}: {detail}" for child_id, detail in self.errors.items())
        super().__init__(f"contextualization failed for {sorted(self.failed_child_ids)}: {details}")


Checkpoint = Callable[[list[ContextualizedChunk]], None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContextualizationService:
    """Generate and validate contextual prefixes without weakening source evidence."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
        max_prefix_chars: int = 240,
        contextualization_version: str = "contextualization-v1",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        settings = get_settings()
        self.client = client or ContextualizationOllamaClient()
        self.batch_size = settings.contextualization_batch_size if batch_size is None else batch_size
        self.max_retries = (
            settings.contextualization_max_retries if max_retries is None else max_retries
        )
        self.max_prefix_chars = max_prefix_chars
        self.contextualization_model = str(getattr(self.client, "model", "")).strip()
        self.contextualization_version = contextualization_version.strip()
        self.sleep = sleep
        self.clock = clock
        self.prompt_version = self.client.prompt_version
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_prefix_chars <= 0:
            raise ValueError("max_prefix_chars must be positive")
        if not self.contextualization_model:
            raise ValueError("contextualization client must declare a non-empty model")
        if not self.contextualization_version:
            raise ValueError("contextualization_version must be non-empty")
        if not str(self.prompt_version).strip():
            raise ValueError("contextualization client must declare a non-empty prompt_version")

    def contextualize(
        self,
        *,
        document: DocumentContext,
        children: Sequence[ChunkDraft],
        parents: Mapping[str, ChunkDraft],
        checkpoint: Checkpoint | None = None,
    ) -> list[ContextualizedChunk]:
        ordered_children = list(children)
        resolved_parents = self._validate_inputs(ordered_children, parents)
        if not ordered_children:
            return []

        successes: dict[str, ContextualizedChunk] = {}
        errors: dict[str, str] = {}
        for offset in range(0, len(ordered_children), self.batch_size):
            batch = ordered_children[offset : offset + self.batch_size]
            batch_successes, batch_errors = self._recover_batch(
                document=document,
                children=batch,
                resolved_parents=resolved_parents,
                checkpoint=checkpoint,
            )
            successes.update(batch_successes)
            errors.update(batch_errors)

        if errors:
            raise ContextualizationFailed(errors)
        return [successes[child.local_id] for child in ordered_children]

    def _recover_batch(
        self,
        *,
        document: DocumentContext,
        children: list[ChunkDraft],
        resolved_parents: Mapping[str, ChunkDraft],
        checkpoint: Checkpoint | None,
    ) -> tuple[dict[str, ContextualizedChunk], dict[str, str]]:
        successes: dict[str, ContextualizedChunk] = {}
        pending = list(children)
        try:
            result, errors = self._attempt(
                document=document,
                children=pending,
                resolved_parents=resolved_parents,
                checkpoint=checkpoint,
            )
            successes.update(result)
            if not errors:
                return successes, {}
            pending = [child for child in pending if child.local_id in errors]
            last_errors = errors
        except Exception as exc:  # noqa: BLE001
            if self._is_capacity_error(exc):
                return self._shrink_for_capacity(
                    document=document,
                    children=pending,
                    resolved_parents=resolved_parents,
                    checkpoint=checkpoint,
                    error=exc,
                )
            last_errors = {
                child.local_id: self._error_detail(exc) for child in pending
            }

        if self.max_retries >= 1 and pending:
            self.sleep(2.0)
            try:
                result, errors = self._attempt(
                    document=document,
                    children=pending,
                    resolved_parents=resolved_parents,
                    checkpoint=checkpoint,
                    correction=self._errors_detail(last_errors),
                )
                successes.update(result)
                if not errors:
                    return successes, {}
                pending = [child for child in pending if child.local_id in errors]
                last_errors = errors
            except Exception as exc:  # noqa: BLE001
                if self._is_capacity_error(exc):
                    shrunk_successes, shrunk_errors = self._shrink_for_capacity(
                        document=document,
                        children=pending,
                        resolved_parents=resolved_parents,
                        checkpoint=checkpoint,
                        error=exc,
                    )
                    successes.update(shrunk_successes)
                    return successes, shrunk_errors
                last_errors = {
                    child.local_id: self._error_detail(exc) for child in pending
                }

        if self.max_retries >= 2 and len(pending) > 1:
            self.sleep(8.0)
            midpoint = len(pending) // 2
            errors: dict[str, str] = {}
            for half in (pending[:midpoint], pending[midpoint:]):
                try:
                    result, half_errors = self._attempt(
                        document=document,
                        children=half,
                        resolved_parents=resolved_parents,
                        checkpoint=checkpoint,
                        correction=self._errors_detail(last_errors),
                    )
                    successes.update(result)
                    errors.update(half_errors)
                except Exception as exc:  # noqa: BLE001
                    if self._is_capacity_error(exc):
                        shrunk_successes, shrunk_errors = self._shrink_for_capacity(
                            document=document,
                            children=half,
                            resolved_parents=resolved_parents,
                            checkpoint=checkpoint,
                            error=exc,
                        )
                        successes.update(shrunk_successes)
                        errors.update(shrunk_errors)
                    else:
                        detail = self._error_detail(exc)
                        errors.update({child.local_id: detail for child in half})
            return successes, errors

        return successes, last_errors

    def _shrink_for_capacity(
        self,
        *,
        document: DocumentContext,
        children: list[ChunkDraft],
        resolved_parents: Mapping[str, ChunkDraft],
        checkpoint: Checkpoint | None,
        error: Exception,
    ) -> tuple[dict[str, ContextualizedChunk], dict[str, str]]:
        if len(children) == 1:
            return {}, {children[0].local_id: self._error_detail(error)}
        midpoint = len(children) // 2
        successes: dict[str, ContextualizedChunk] = {}
        errors: dict[str, str] = {}
        for half in (children[:midpoint], children[midpoint:]):
            try:
                result, half_errors = self._attempt(
                    document=document,
                    children=half,
                    resolved_parents=resolved_parents,
                    checkpoint=checkpoint,
                    correction=self._error_detail(error),
                )
                successes.update(result)
                errors.update(half_errors)
            except Exception as exc:  # noqa: BLE001
                if self._is_capacity_error(exc):
                    half_successes, half_errors = self._shrink_for_capacity(
                        document=document,
                        children=half,
                        resolved_parents=resolved_parents,
                        checkpoint=checkpoint,
                        error=exc,
                    )
                    successes.update(half_successes)
                    errors.update(half_errors)
                else:
                    detail = self._error_detail(exc)
                    errors.update({child.local_id: detail for child in half})
        return successes, errors

    def _attempt(
        self,
        *,
        document: DocumentContext,
        children: list[ChunkDraft],
        resolved_parents: Mapping[str, ChunkDraft],
        checkpoint: Checkpoint | None,
        correction: str | None = None,
    ) -> tuple[dict[str, ContextualizedChunk], dict[str, str]]:
        system_prompt, user_prompt = self._prompts(
            document=document,
            children=children,
            resolved_parents=resolved_parents,
            correction=correction,
        )
        response = self.client.generate_contextualization(
            ContextualPrefixBatch,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        by_id, errors = self._validate_response(
            response, document, children, resolved_parents
        )
        contextualized = [
            self._apply_prefix(child, by_id[child.local_id])
            for child in children
            if child.local_id in by_id
        ]
        if checkpoint is not None:
            if contextualized:
                checkpoint(contextualized)
        return {item.local_id: item for item in contextualized}, errors

    def _prompts(
        self,
        *,
        document: DocumentContext,
        children: list[ChunkDraft],
        resolved_parents: Mapping[str, ChunkDraft],
        correction: str | None,
    ) -> tuple[str, str]:
        system_prompt = "\n".join(
            [
                f"Contextualization prompt version: {self.prompt_version}",
                "为每个 Child 生成 1-2 句中文关系说明，并严格返回 JSON schema。",
                "说明 Child 与论文、章节及 Parent 的关系；不得复制 Child 原文。",
                "论文名、方法名、模型名、数据集名、指标名及公式英文原名必须保持原样。",
                "除识别表格或指标关系确有必要外，不得生成或改写具体数值。",
                "不得遗漏、重复或增加 child_id。生成说明不是引文或来源证据。",
            ]
        )
        payload = {
            "document": document.model_dump(mode="json"),
            "children": [
                {
                    "child_id": child.local_id,
                    "block_type": child.block_type,
                    "section_path": child.section_path,
                    "parent_text": resolved_parents[child.local_id].text,
                    "original_text": child.text,
                }
                for child in children
            ],
        }
        correction_text = ""
        if correction:
            correction_text = (
                "CORRECTION_FEEDBACK:\n"
                f"上一批输出未通过严格校验：{correction}\n"
                "修正上述具体错误，并只输出本批全部 child_id。\n\n"
            )
        user_prompt = (
            correction_text
            + "请依据以下原始上下文生成关系说明。\n\nINPUT_JSON:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        return system_prompt, user_prompt

    def _validate_response(
        self,
        response: ContextualPrefixBatch,
        document: DocumentContext,
        children: list[ChunkDraft],
        resolved_parents: Mapping[str, ChunkDraft],
    ) -> tuple[dict[str, str], dict[str, str]]:
        expected_ids = [child.local_id for child in children]
        actual_ids = [item.child_id for item in response.items]
        duplicate_ids = {
            child_id for child_id, count in Counter(actual_ids).items() if count > 1
        }
        missing_ids = set(expected_ids) - set(actual_ids)
        extra_ids = set(actual_ids) - set(expected_ids)
        errors: dict[str, str] = {}
        if extra_ids:
            detail = f"extra child IDs: {sorted(extra_ids)}"
            return {}, {child_id: detail for child_id in expected_ids}
        for child_id in missing_ids:
            errors[child_id] = f"missing child ID: {child_id}"
        for child_id in duplicate_ids:
            if child_id in set(expected_ids):
                errors[child_id] = f"duplicate child ID: {child_id}"

        by_child = {child.local_id: child for child in children}
        prefixes: dict[str, str] = {}
        for item in response.items:
            if item.child_id not in by_child or item.child_id in duplicate_ids:
                continue
            prefix = item.prefix.strip()
            try:
                self._validate_prefix(
                    prefix=prefix,
                    child=by_child[item.child_id],
                    parent=resolved_parents[item.child_id],
                    document=document,
                )
            except Exception as exc:  # noqa: BLE001
                errors[item.child_id] = self._error_detail(exc)
            else:
                prefixes[item.child_id] = prefix
        return prefixes, errors

    def _validate_prefix(
        self,
        *,
        prefix: str,
        child: ChunkDraft,
        parent: ChunkDraft,
        document: DocumentContext,
    ) -> None:
        if not prefix:
            raise ValueError(f"empty prefix for {child.local_id}")
        if len(prefix) > self.max_prefix_chars:
            raise ValueError(
                f"prefix length for {child.local_id} exceeds {self.max_prefix_chars} characters"
            )
        if self._sentence_count(prefix) > 2:
            raise ValueError(f"prefix has more than 2 sentences for {child.local_id}")
        if not re.search(r"[\u3400-\u9fff]", prefix):
            raise ValueError(f"prefix is not a chinese relation explanation for {child.local_id}")
        if not any(marker in prefix for marker in _RELATION_MARKERS):
            raise ValueError(f"prefix is not a contextual relation for {child.local_id}")
        normalized_child = self._normalize_copy_text(child.text)
        normalized_prefix = self._normalize_copy_text(prefix)
        if normalized_child and normalized_child in normalized_prefix:
            raise ValueError(f"prefix copies complete child text for {child.local_id}")
        if self._copies_most_child(normalized_child, normalized_prefix):
            raise ValueError(f"prefix copies most of child text for {child.local_id}")

        corpus = "\n".join(
            [
                document.title,
                document.source_abstract or "",
                *document.section_outline,
                *child.section_path,
                parent.text,
                child.text,
            ]
        )
        corpus_entities = set(self._english_entities(corpus))
        added_entities = sorted(
            entity
            for entity in self._english_entities(prefix)
            if entity not in corpus_entities and entity not in _COMMON_ENGLISH_TERMS
        )
        if added_entities:
            raise ValueError(
                f"prefix adds inconsistent English entity for {child.local_id}: {added_entities}"
            )

        evidence_source = f"{parent.text}\n{child.text}"
        corpus_numbers = set(self._numbers(evidence_source))
        prefix_numbers = set(self._numbers(prefix))
        added_numbers = sorted(prefix_numbers - corpus_numbers)
        if added_numbers:
            raise ValueError(
                f"prefix adds unsupported numeric value for {child.local_id}: {added_numbers}"
            )
        unnecessary_numbers = sorted(
            number
            for number in prefix_numbers
            if not self._numeric_reference_is_necessary(
                number=number,
                prefix=prefix,
                evidence_source=evidence_source,
            )
        )
        if unnecessary_numbers:
            raise ValueError(
                f"prefix includes unnecessary numeric value for {child.local_id}: "
                f"{unnecessary_numbers}"
            )

    def _apply_prefix(self, child: ChunkDraft, prefix: str) -> ContextualizedChunk:
        contextualized_at = self.clock()
        if contextualized_at.tzinfo is None or contextualized_at.utcoffset() is None:
            raise ValueError("contextualized_at must be timezone-aware")
        contextualized = ContextualizedChunk.model_validate(
            {
                **child.model_dump(mode="python"),
                "contextual_prefix": prefix,
                "embedding_text": f"{prefix}\n\n{child.text}",
                "contextualization_model": self.contextualization_model,
                "contextualization_version": self.contextualization_version,
                "contextualization_prompt_version": self.prompt_version,
                "contextualized_at": contextualized_at,
            }
        )
        contextualized.metadata["contextual_prefix"] = {
            "text": prefix,
            "generated": True,
            "citation_eligible": False,
            "source_spans": [],
            "source_block_ids": [],
            "prompt_version": self.prompt_version,
        }
        return contextualized

    @staticmethod
    def _validate_inputs(
        children: list[ChunkDraft],
        parents: Mapping[str, ChunkDraft],
    ) -> dict[str, ChunkDraft]:
        errors: dict[str, str] = {}
        resolved: dict[str, ChunkDraft] = {}
        counts = Counter(child.local_id for child in children)
        for child in children:
            if counts[child.local_id] > 1:
                errors[child.local_id] = "duplicate input child ID"
                continue
            if child.chunk_role != "child":
                errors[child.local_id] = "input chunk_role must be child"
                continue
            if child.block_type not in RETRIEVABLE_BLOCK_TYPES:
                errors[child.local_id] = f"unsupported retrievable block type: {child.block_type}"
                continue
            if not child.parent_local_id or child.parent_local_id not in parents:
                errors[child.local_id] = "child parent cannot be resolved"
                continue
            parent = parents[child.parent_local_id]
            if parent.chunk_role != "parent":
                errors[child.local_id] = "resolved parent chunk_role must be parent"
                continue
            resolved[child.local_id] = parent
        if errors:
            raise ContextualizationFailed(errors)
        return resolved

    @staticmethod
    def _sentence_count(text: str) -> int:
        protected = re.sub(
            r"\b(?:e\.g\.|i\.e\.|et al\.|(?:[A-Za-z]\.){2,})",
            lambda match: match.group(0).replace(".", "<DOT>"),
            text,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"(?<!\d)\.(?!\d)", "。", protected)
        pieces = [piece.strip() for piece in re.split(r"[。！？!?]+", normalized) if piece.strip()]
        return max(1, len(pieces))

    @staticmethod
    def _normalize_copy_text(text: str) -> str:
        return re.sub(r"[^\w\u3400-\u9fff]+", "", text, flags=re.UNICODE).casefold()

    @staticmethod
    def _copies_most_child(normalized_child: str, normalized_prefix: str) -> bool:
        if len(normalized_child) < 20:
            return False
        longest = SequenceMatcher(
            None,
            normalized_child,
            normalized_prefix,
            autojunk=False,
        ).find_longest_match().size
        return longest >= 20 and longest / len(normalized_child) >= 0.60

    @staticmethod
    def _english_entities(text: str) -> list[str]:
        entities = re.findall(r"\b(?:[A-Za-z]\.){2,}", text)
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b", text)
        entities.extend(
            word
            for word in words
            if word[0].isupper()
            or any(character.isupper() for character in word[1:])
            or any(character.isdigit() for character in word)
            or "-" in word
            or "_" in word
        )
        return entities

    @staticmethod
    def _numbers(text: str) -> list[str]:
        return re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])", text)

    @classmethod
    def _numeric_reference_is_necessary(
        cls,
        *,
        number: str,
        prefix: str,
        evidence_source: str,
    ) -> bool:
        table_label = re.compile(
            rf"(?<![A-Za-z0-9])(?:表|table)\s*{re.escape(number)}(?![\d.])",
            re.IGNORECASE,
        )
        if table_label.search(prefix) and table_label.search(evidence_source):
            return True
        return any(
            cls._terms_are_near(prefix, marker, number)
            and cls._terms_are_near(evidence_source, marker, number)
            for marker in _NAMED_METRIC_MARKERS
        )

    @staticmethod
    def _terms_are_near(text: str, marker: str, number: str, max_distance: int = 24) -> bool:
        normalized = text.casefold()
        escaped_marker = re.escape(marker.casefold())
        marker_pattern = (
            rf"(?<![a-z0-9]){escaped_marker}(?![a-z0-9])"
            if marker.isascii()
            else escaped_marker
        )
        marker_positions = [
            match.start() for match in re.finditer(marker_pattern, normalized)
        ]
        number_positions = [
            match.start() for match in re.finditer(re.escape(number.casefold()), normalized)
        ]
        return any(
            abs(marker_position - number_position) <= max_distance
            for marker_position in marker_positions
            for number_position in number_positions
        )

    @staticmethod
    def _is_capacity_error(exc: Exception) -> bool:
        detail = str(exc).casefold()
        return any(
            marker in detail
            for marker in (
                "out of memory",
                "oom",
                "capacity",
                "context length",
                "context window",
                "insufficient memory",
                "requires more memory",
            )
        )

    @staticmethod
    def _error_detail(exc: Exception) -> str:
        detail = str(exc).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__

    @staticmethod
    def _errors_detail(errors: Mapping[str, str]) -> str:
        return "; ".join(f"{child_id}: {detail}" for child_id, detail in errors.items())
