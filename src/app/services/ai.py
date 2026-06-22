from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import httpx
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ExtractedEntity(BaseModel):#提取概念
    name: str
    entity_type: str = "concept"
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)


class ExtractedClaim(BaseModel):#知识三元组，主谓宾，
    subject: str
    predicate: str
    object_text: str
    expected_head: str = ""
    evidence_excerpt: str = ""
    confidence: float = 0.5
    source_chunk_ordinals: list[int] = Field(default_factory=list)
    source_sentence_refs: list[str] = Field(default_factory=list)
    relation_type: str = "fact"
    verification_errors: list[str] = Field(default_factory=list)
    growth_decision: str = "keep"


class GeneratedTriple(BaseModel):
    subject: str
    predicate: str
    object_text: str
    expected_head: str = ""
    evidence_excerpt: str = ""
    confidence: float = 0.5
    source_chunk_ordinals: list[int] = Field(default_factory=list)
    source_sentence_refs: list[str] = Field(default_factory=list)
    relation_type: str = "fact"


class DocumentAnalysisPayload(BaseModel):#页面解析结果
    title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    triples: list[GeneratedTriple] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class DocumentPagePayload(BaseModel):#解析页面
    page_label: str
    page_summary: str = ""
    page_markdown: str = ""
    sections: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class HeadAnalysisPayload(BaseModel):#核心实体分析报告
    head_entity: str
    summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    triples: list[GeneratedTriple] = Field(default_factory=list)
    related_entities: list[ExtractedEntity] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class TripleVerificationPayload(BaseModel):#三元组验证结果
    verdict: str = "local-only"
    accepted_indexes: list[int] = Field(default_factory=list)
    rejected_indexes: list[int] = Field(default_factory=list)
    errors_by_index: dict[str, list[str]] = Field(default_factory=dict)
    coverage_notes: list[str] = Field(default_factory=list)


class GrowthDecision(BaseModel):#单挑决策
    name: str
    item_type: str = "entity"
    decision: str = "keep"
    reason: str = ""


class GrowthDecisionPayload(BaseModel):#多条决策打包
    decisions: list[GrowthDecision] = Field(default_factory=list)


class DocumentExtraction(BaseModel):#文档最终抽取结果
    title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class QueryAnswerPayload(BaseModel):
    answer_markdown: str
    citations: list[int] = Field(default_factory=list)
    risk_level: str = "normal"


class VerificationPayload(BaseModel):
    verdict: str
    notes: str
    flagged_claim_indexes: list[int] = Field(default_factory=list)


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    score: float
    page_label: str | None
    excerpt: str# 分块内容摘要


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = settings.ollama_request_timeout

    def generate_structured(
        self,
        schema: type[SchemaT],
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> SchemaT:
        model_name = model or settings.ollama_generation_model
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": schema.model_json_schema(),# 关键：强制大模型按JSON Schema输出
        }
        data = self._post_chat(payload)
        content = self._message_content(data)
        try:
            return self._parse_structured_content(schema, content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structured schema response was invalid; retrying with JSON mode: %s", exc)
            retry_payload = self._json_mode_payload(
                schema=schema,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            retry_data = self._post_chat(retry_payload)
            return self._parse_structured_content(schema, self._message_content(retry_data))

    def generate_structured_with_images(
        self,
        schema: type[SchemaT],
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes | str | Path],
        model: str | None = None,
    ) -> SchemaT:
        model_name = model or settings.ollama_vision_model or settings.ollama_generation_model
        encoded_images = [self._encode_image(image) for image in images]
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": encoded_images,
                },
            ],
            "stream": False,
            "format": schema.model_json_schema(),
        }
        data = self._post_chat(payload)
        content = self._message_content(data)
        try:
            return self._parse_structured_content(schema, content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structured vision response was invalid; retrying with JSON mode: %s", exc)
            retry_payload = self._json_mode_payload(
                schema=schema,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=encoded_images,
            )
            retry_data = self._post_chat(retry_payload)
            return self._parse_structured_content(schema, self._message_content(retry_data))

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": settings.ollama_embedding_model, "input": text},
                )
                response.raise_for_status()
                vectors.append(response.json()["embeddings"][0])
        return vectors

    @staticmethod
    def _encode_image(image: bytes | str | Path) -> str:
        if isinstance(image, bytes):
            data = image
        else:
            path = Path(image)
            if path.exists():
                data = path.read_bytes()
            else:
                return str(image)
        return base64.b64encode(data).decode("utf-8")

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if settings.ollama_keep_alive is not None:
            payload = {**payload, "keep_alive": settings.ollama_keep_alive}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _message_content(data: dict[str, Any]) -> Any:
        content = (data.get("message") or {}).get("content", "")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        return content

    @staticmethod
    def _json_mode_payload(
        *,
        schema: type[SchemaT],
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        content = "\n\n".join(
            [
                user_prompt,
                "Return only one valid JSON object. Do not include markdown, code fences, or explanatory text.",
                "Use this compact JSON shape. Empty arrays are allowed when evidence is missing:",
                json.dumps(OllamaClient._compact_schema_shape(schema), ensure_ascii=False),
            ]
        )
        user_message: dict[str, Any] = {"role": "user", "content": content}
        if images:
            user_message["images"] = images
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "stream": False,
            "format": "json",
        }

    @classmethod
    def _compact_schema_shape(cls, annotation: Any, depth: int = 0) -> Any:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is list:
            item_type = args[0] if args else Any
            return [cls._compact_schema_shape(item_type, depth + 1)]
        if origin is dict:
            return {}
        if origin is not None and args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            return cls._compact_schema_shape(non_none_args[0], depth) if non_none_args else None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if depth >= 2:
                return {}
            return {
                name: cls._compact_schema_shape(field.annotation, depth + 1)
                for name, field in annotation.model_fields.items()
            }
        if annotation is str:
            return ""
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False
        return None

    @classmethod
    def _parse_structured_content(cls, schema: type[SchemaT], content: Any) -> SchemaT:
        if isinstance(content, dict):
            return schema.model_validate(content)
        if isinstance(content, list):
            return schema.model_validate(content)

        text = str(content or "").strip()
        if not text:
            raise ValueError(f"Ollama returned empty content for {schema.__name__}.")

        try:
            return schema.model_validate_json(text)
        except Exception:  # noqa: BLE001
            extracted = cls._extract_first_json_value(text)
            return schema.model_validate(extracted)

    @staticmethod
    def _extract_first_json_value(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise ValueError("No valid JSON object found in Ollama response.")


class ExternalVerifier:
    def __init__(self) -> None:
        self.enabled = settings.external_api_enabled and bool(settings.external_api_key)
        self.base_url = settings.external_api_base_url.rstrip("/")
        self.model = settings.external_api_model
        self.timeout = settings.external_api_timeout

    def verify_claims(self, summary: str, claims: list[ExtractedClaim | dict[str, Any]]) -> VerificationPayload:
        if not self.enabled:
            return VerificationPayload(verdict="skipped", notes="External verification disabled.", flagged_claim_indexes=[])

        schema = VerificationPayload.model_json_schema()
        prompt = {
            "summary": summary,
            "claims": [claim.model_dump() if isinstance(claim, BaseModel) else claim for claim in claims],
            "task": "Review whether the claims are supported by the summary and flag contradictions or uncertainty.",
        }
        headers = {"Authorization": f"Bearer {settings.external_api_key}"}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You verify research claims. Return strict JSON that matches the provided schema.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verification_payload", "schema": schema},
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        return VerificationPayload.model_validate_json(content)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:#向量相似度计算
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return numerator / (norm_a * norm_b)


def safe_model_call(func, fallback):
    try:
        return func()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model call failed, falling back: %s", exc)
        return fallback
