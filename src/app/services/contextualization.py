from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.services.ai import ContextualizationOllamaClient
from app.services.semantic_chunking import ChunkDraft


RETRIEVABLE_BLOCK_TYPES = frozenset(
    {"narrative", "table", "figure", "formula", "caption", "appendix"}
)
_COMMON_ENGLISH_TERMS = frozenset(
    {"AI", "API", "CPU", "GPU", "JSON", "LLM", "ML", "NLP", "OCR", "PDF", "RAG", "SOTA"}
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
_METRIC_MARKERS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "bleu",
    "rouge",
    "准确率",
    "精确率",
    "召回率",
    "指标",
    "得分",
    "性能",
    "误差",
    "损失",
    "百分比",
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


class ContextualizationFailed(RuntimeError):
    def __init__(self, errors: Mapping[str, str]) -> None:
        self.errors = dict(errors)
        self.failed_child_ids = set(self.errors)
        details = "; ".join(f"{child_id}: {detail}" for child_id, detail in self.errors.items())
        super().__init__(f"contextualization failed for {sorted(self.failed_child_ids)}: {details}")


Checkpoint = Callable[[list[ChunkDraft]], None]


class ContextualizationService:
    """Generate and validate contextual prefixes without weakening source evidence."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
        max_prefix_chars: int = 240,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = get_settings()
        self.client = client or ContextualizationOllamaClient()
        self.batch_size = settings.contextualization_batch_size if batch_size is None else batch_size
        self.max_retries = (
            settings.contextualization_max_retries if max_retries is None else max_retries
        )
        self.max_prefix_chars = max_prefix_chars
        self.sleep = sleep
        self.prompt_version = self.client.prompt_version
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_prefix_chars <= 0:
            raise ValueError("max_prefix_chars must be positive")

    def contextualize(
        self,
        *,
        document: DocumentContext,
        children: Sequence[ChunkDraft],
        parents: Mapping[str, ChunkDraft],
        checkpoint: Checkpoint | None = None,
    ) -> list[ChunkDraft]:
        ordered_children = list(children)
        resolved_parents = self._validate_inputs(ordered_children, parents)
        if not ordered_children:
            return []

        successes: dict[str, ChunkDraft] = {}
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
    ) -> tuple[dict[str, ChunkDraft], dict[str, str]]:
        try:
            result = self._attempt(
                document=document,
                children=children,
                resolved_parents=resolved_parents,
                checkpoint=checkpoint,
            )
            return result, {}
        except Exception as exc:  # noqa: BLE001
            if self._is_capacity_error(exc):
                return self._shrink_for_capacity(
                    document=document,
                    children=children,
                    resolved_parents=resolved_parents,
                    checkpoint=checkpoint,
                    error=exc,
                )
            last_error = exc

        if self.max_retries >= 1:
            self.sleep(2.0)
            try:
                result = self._attempt(
                    document=document,
                    children=children,
                    resolved_parents=resolved_parents,
                    checkpoint=checkpoint,
                    correction=self._error_detail(last_error),
                )
                return result, {}
            except Exception as exc:  # noqa: BLE001
                if self._is_capacity_error(exc):
                    return self._shrink_for_capacity(
                        document=document,
                        children=children,
                        resolved_parents=resolved_parents,
                        checkpoint=checkpoint,
                        error=exc,
                    )
                last_error = exc

        if self.max_retries >= 2 and len(children) > 1:
            self.sleep(8.0)
            midpoint = len(children) // 2
            successes: dict[str, ChunkDraft] = {}
            errors: dict[str, str] = {}
            for half in (children[:midpoint], children[midpoint:]):
                try:
                    result = self._attempt(
                        document=document,
                        children=half,
                        resolved_parents=resolved_parents,
                        checkpoint=checkpoint,
                        correction=self._error_detail(last_error),
                    )
                    successes.update(result)
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

        detail = self._error_detail(last_error)
        return {}, {child.local_id: detail for child in children}

    def _shrink_for_capacity(
        self,
        *,
        document: DocumentContext,
        children: list[ChunkDraft],
        resolved_parents: Mapping[str, ChunkDraft],
        checkpoint: Checkpoint | None,
        error: Exception,
    ) -> tuple[dict[str, ChunkDraft], dict[str, str]]:
        if len(children) == 1:
            return {}, {children[0].local_id: self._error_detail(error)}
        midpoint = len(children) // 2
        successes: dict[str, ChunkDraft] = {}
        errors: dict[str, str] = {}
        for half in (children[:midpoint], children[midpoint:]):
            try:
                result = self._attempt(
                    document=document,
                    children=half,
                    resolved_parents=resolved_parents,
                    checkpoint=checkpoint,
                    correction=self._error_detail(error),
                )
                successes.update(result)
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
    ) -> dict[str, ChunkDraft]:
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
        by_id = self._validate_response(response, document, children, resolved_parents)
        contextualized = [self._apply_prefix(child, by_id[child.local_id]) for child in children]
        if checkpoint is not None:
            checkpoint(contextualized)
        return {item.local_id: item for item in contextualized}

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
    ) -> dict[str, str]:
        expected_ids = [child.local_id for child in children]
        actual_ids = [item.child_id for item in response.items]
        duplicate_ids = sorted(
            child_id for child_id, count in Counter(actual_ids).items() if count > 1
        )
        missing_ids = sorted(set(expected_ids) - set(actual_ids))
        extra_ids = sorted(set(actual_ids) - set(expected_ids))
        problems: list[str] = []
        if duplicate_ids:
            problems.append(f"duplicate child IDs: {duplicate_ids}")
        if missing_ids:
            problems.append(f"missing child IDs: {missing_ids}")
        if extra_ids:
            problems.append(f"extra child IDs: {extra_ids}")
        if problems:
            raise ValueError("; ".join(problems))

        by_child = {child.local_id: child for child in children}
        prefixes: dict[str, str] = {}
        for item in response.items:
            prefix = item.prefix.strip()
            self._validate_prefix(
                prefix=prefix,
                child=by_child[item.child_id],
                parent=resolved_parents[item.child_id],
                document=document,
            )
            prefixes[item.child_id] = prefix
        return prefixes

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
        if normalized_child and normalized_child in self._normalize_copy_text(prefix):
            raise ValueError(f"prefix copies complete child text for {child.local_id}")

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

        corpus_numbers = set(self._numbers(corpus))
        prefix_numbers = set(self._numbers(prefix))
        added_numbers = sorted(prefix_numbers - corpus_numbers)
        if added_numbers:
            raise ValueError(
                f"prefix adds unsupported numeric value for {child.local_id}: {added_numbers}"
            )
        metric_context = f"{parent.text}\n{child.text}".casefold()
        numbers_are_necessary = child.block_type in {"table", "figure", "formula"} or any(
            marker in metric_context for marker in _METRIC_MARKERS
        )
        if prefix_numbers and not numbers_are_necessary:
            raise ValueError(
                f"prefix includes unnecessary numeric value for {child.local_id}: "
                f"{sorted(prefix_numbers)}"
            )

    def _apply_prefix(self, child: ChunkDraft, prefix: str) -> ChunkDraft:
        contextualized = child.model_copy(deep=True)
        contextualized.embedding_text = f"{prefix}\n\n{child.text}"
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
        normalized = re.sub(r"(?<!\d)\.(?!\d)", "。", text)
        pieces = [piece.strip() for piece in re.split(r"[。！？!?]+", normalized) if piece.strip()]
        return max(1, len(pieces))

    @staticmethod
    def _normalize_copy_text(text: str) -> str:
        return re.sub(r"[^\w\u3400-\u9fff]+", "", text, flags=re.UNICODE).casefold()

    @staticmethod
    def _english_entities(text: str) -> list[str]:
        return re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b", text)

    @staticmethod
    def _numbers(text: str) -> list[str]:
        return re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?(?![A-Za-z])", text)

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
