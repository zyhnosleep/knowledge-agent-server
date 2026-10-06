"""Decisions contain data, never executable tool names or user-defined scope."""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _decision_json_schema(schema: dict[str, Any]) -> None:
    properties = schema["properties"]
    title = schema.get("title", "AdaptiveDecision")
    retrieval = {name: properties[name] for name in ("action", "reason", "query", "document_id", "limit")}
    retrieval["action"] = {"const": "retrieve", "type": "string"}
    retrieval["query"] = {"type": "string", "minLength": 1, "maxLength": 1200}
    retrieval["limit"] = {"type": "integer", "minimum": 1, "maximum": 15}
    retrieval["document_id"] = {"type": "string", "minLength": 1, "maxLength": 128}
    terminal = {"action": {"enum": ["answer", "finish", "abstain"], "type": "string"},
                "reason": properties["reason"]}
    schema.clear()
    schema.update(title=title, oneOf=[
        {"type": "object", "properties": retrieval, "required": ["action", "query"], "additionalProperties": False},
        {"type": "object", "properties": terminal, "required": ["action"], "additionalProperties": False},
    ])


class AdaptiveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra=_decision_json_schema)
    strict_json_only: ClassVar[bool] = True
    json_retry_example: ClassVar[dict[str, Any]] = {"action": "abstain", "reason": "Insufficient evidence"}
    action: Literal["retrieve", "answer", "finish", "abstain"]
    reason: str = Field(default="", max_length=240)
    query: str | None = Field(default=None, min_length=1, max_length=1200)
    document_id: str | None = Field(default=None, min_length=1, max_length=128)
    limit: int | None = Field(default=None, ge=1, le=15)

    @field_validator("query", "document_id", mode="before")
    @classmethod
    def trim_strings(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def restrict_parameters(self):
        if self.action == "retrieve":
            if not self.query:
                raise ValueError("retrieve requires a query")
            if self.limit is None:
                self.limit = 5
        elif self.model_fields_set & {"query", "document_id", "limit"}:
            raise ValueError("terminal decisions cannot contain retrieval parameters")
        return self


class AdaptiveObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    conversation_summary: str
    evidence: list[dict[str, Any]]
    coverage_status: str
    coverage_missing_tables: list[str] = Field(default_factory=list)
    truncated: bool
    candidate_answer: str | None
    last_tool_error: str | None
    budget: dict[str, Any]
