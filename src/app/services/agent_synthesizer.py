from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.ai import OllamaClient

logger = logging.getLogger(__name__)


class LocalSynthesisPayload(BaseModel):
    answer_markdown: str
    cited_indexes: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class AgentSynthesizer:
    """Evidence synthesis using Ollama, an external API, or safe fallback.

    Allowed synthesis providers are ``auto``, ``external_api``, and ``local``.
    When the provider is ``auto``, the external API is only used when
    ``EXTERNAL_API_ENABLED=true`` and an API key is configured.

    External synthesis prompts the model to synthesize from evidence
    (not concatenate paragraphs) and to cite only from provided citations.
    """

    VALID_PROVIDERS = {"auto", "external_api", "local"}

    # Bounds for evidence-pack prompt section (Phase 3)
    MAX_EVIDENCE_PACK_ITEMS = 10
    MAX_EXCERPT_CHARS = 300

    def __init__(self, ollama_client: OllamaClient | None = None) -> None:
        self._settings = get_settings()
        self._ollama = ollama_client or OllamaClient()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str = "",
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize a final answer from RAG evidence.

        Optionally accepts ``evidence_pack`` (a dict with ``status`` and
        ``items`` keys) from the retrieve step.  The local fallback path
        ignores it; the external path includes evidence item excerpts in
        the prompt when available.

        Returns a dict with at least:
        ``answer_markdown``, ``cited_indexes``, ``warnings``,
        ``confidence``, ``provider``, and ``model``.
        """
        provider = self._resolve_provider()

        if provider == "local":
            return self._local_synthesize(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
            )

        # external_api path
        return self._external_synthesize(
            query=query,
            route=route,
            conversation_summary=conversation_summary,
            rag_answer=rag_answer,
            citations=citations,
            evidence_pack=evidence_pack,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> str:
        """Determine which provider to use based on settings."""
        configured = self._settings.agent_synthesis_provider
        if configured not in self.VALID_PROVIDERS:
            logger.warning(
                "Unknown AGENT_SYNTHESIS_PROVIDER=%r, falling back to 'auto'",
                configured,
            )
            configured = "auto"

        if configured == "local":
            return "local"

        if configured == "external_api":
            return "external_api"

        # "auto": use external API only if enabled and key is set
        if self._settings.external_api_enabled and bool(self._settings.external_api_key):
            return "external_api"
        return "local"

    def _local_fallback(
        self, rag_answer: str, citations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Deterministic local fallback: keep the RAG answer as-is."""
        return {
            "answer_markdown": rag_answer,
            "cited_indexes": list(range(len(citations))),
            "warnings": [],
            "confidence": 1.0,
            "provider": "local",
            "model": "local-fallback",
        }

    def _local_synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize a cited answer with the configured local Ollama model."""
        if not citations and not (evidence_pack or {}).get("items"):
            return self._local_fallback(rag_answer, citations)

        evidence_parts: list[str] = []
        for index, citation in enumerate(citations):
            excerpt = str(citation.get("excerpt") or "").strip()
            title = citation.get("page_title") or citation.get("document_id") or "untitled"
            if excerpt:
                evidence_parts.append(f"[{index}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) or "(no citation excerpts)"
        evidence_pack_text = self._format_evidence_pack_section(evidence_pack)

        system_prompt = (
            "You are an evidence-grounded knowledge-base assistant. Answer the user's "
            "question in the user's language by synthesizing the supplied evidence. "
            "Do not merely repeat the RAG draft. Do not add facts that are absent from "
            "the evidence. Cite supported statements with the supplied zero-based "
            "citation indexes, such as [0]. If evidence is insufficient, say exactly "
            "what cannot be established."
        )
        user_prompt = (
            f"User question: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG draft:\n{rag_answer}\n\n"
            f"Citation excerpts:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + "Return a concise final answer grounded only in this evidence."
        )
        model = self._settings.ollama_generation_model

        try:
            parsed = self._ollama.generate_structured(
                LocalSynthesisPayload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                think=False,
                options={"num_predict": 768},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local Ollama synthesis failed: %s", exc)
            result = self._local_fallback(rag_answer, citations)
            result["warnings"].append(
                f"Local Ollama synthesis failed; using evidence fallback: {exc}"
            )
            return result

        answer_markdown = parsed.answer_markdown.strip()
        if not answer_markdown:
            result = self._local_fallback(rag_answer, citations)
            result["warnings"].extend(parsed.warnings)
            result["warnings"].append(
                "Local Ollama synthesis returned an empty answer; using evidence fallback."
            )
            return result

        max_index = len(citations)
        cited_indexes = list(
            dict.fromkeys(
                index
                for index in parsed.cited_indexes
                if isinstance(index, int) and 0 <= index < max_index
            )
        )
        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": cited_indexes,
            "warnings": [str(warning) for warning in parsed.warnings],
            "confidence": max(0.0, min(1.0, float(parsed.confidence))),
            "provider": "local",
            "model": model,
        }

    @staticmethod
    def _format_evidence_pack_section(
        evidence_pack: dict[str, Any] | None,
    ) -> str:
        """Build a compact, bounded evidence-pack prompt section.

        Each included line preserves: item index, document_id, evidence_kind,
        source_stage, support_hint, and excerpt text.

        The section is explicitly capped:
        - At most ``MAX_EVIDENCE_PACK_ITEMS`` items.
        - Each excerpt truncated to ``MAX_EXCERPT_CHARS`` characters.

        Returns an empty string when *evidence_pack* is missing or empty.
        """
        if not evidence_pack or not evidence_pack.get("items"):
            return ""

        items: list[dict[str, Any]] = evidence_pack["items"]
        max_items = AgentSynthesizer.MAX_EVIDENCE_PACK_ITEMS
        max_chars = AgentSynthesizer.MAX_EXCERPT_CHARS
        emitted = 0
        total = len(items)

        lines: list[str] = []
        for item in items:
            if emitted >= max_items:
                break
            excerpt = str(item.get("excerpt", ""))
            if not excerpt:
                continue
            # Truncate long excerpts
            if len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars] + "…"
            idx = item.get("index", "?")
            doc_id = item.get("document_id", "?")
            kind = item.get("evidence_kind", "")
            stage = item.get("source_stage", "")
            hint = item.get("support_hint", "")
            lines.append(
                f"  [{idx}] doc={doc_id} kind={kind} stage={stage} "
                f"hint={hint}: {excerpt}"
            )
            emitted += 1

        if not lines:
            return ""

        header = f"Evidence pack items ({emitted} of {total}):"
        suffix = ""
        if emitted < total:
            suffix = f"\n  … ({total - emitted} more items omitted)"
        return header + "\n" + "\n".join(lines) + suffix

    def _external_synthesize(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the external API for evidence synthesis.

        On failure, returns a fallback result with a warning — never crashes.
        """
        if not self._settings.external_api_enabled or not bool(self._settings.external_api_key):
            result = self._local_fallback(rag_answer, citations)
            result["warnings"].append(
                "External synthesis requested but API is not configured; "
                "using local fallback."
            )
            return result

        # Build evidence context
        evidence_parts: list[str] = []
        for i, c in enumerate(citations):
            excerpt = c.get("excerpt", "")
            title = c.get("page_title") or c.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{i}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"

        # Include bounded evidence pack section when available
        evidence_pack_text = self._format_evidence_pack_section(evidence_pack)

        system_prompt = (
            "You are an evidence synthesis assistant. Your task is to synthesize "
            "a final answer from the provided RAG answer and evidence excerpts. "
            "You MUST synthesize — do NOT simply concatenate paragraphs. "
            "Only cite sources that are present in the provided evidence. "
            "Return a JSON object with keys: answer_markdown (string), "
            "cited_indexes (array of integers, only valid indexes from the evidence), "
            "warnings (array of strings), confidence (number 0.0-1.0)."
        )

        user_prompt = (
            f"Original query: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + "Synthesize a final answer from the above evidence. "
            "Return only a JSON object."
        )

        headers = {
            "Authorization": f"Bearer {self._settings.external_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._settings.external_api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            with httpx.Client(timeout=self._settings.external_api_timeout) as client:
                response = client.post(
                    f"{self._settings.external_api_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("External synthesis API call failed: %s", exc)
            result = self._local_fallback(rag_answer, citations)
            result["provider"] = "external_api"
            result["warnings"].append(
                f"External synthesis API call failed, using local fallback: {exc}"
            )
            return result

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, json.JSONDecodeError, IndexError) as exc:
            logger.warning("Failed to parse external synthesis response: %s", exc)
            result = self._local_fallback(rag_answer, citations)
            result["provider"] = "external_api"
            result["warnings"].append(
                f"Failed to parse external synthesis response, using local fallback: {exc}"
            )
            return result

        # Sanitize cited_indexes
        max_idx = len(citations)
        raw_indexes = parsed.get("cited_indexes", [])
        if isinstance(raw_indexes, list):
            sanitized = [i for i in raw_indexes if isinstance(i, int) and 0 <= i < max_idx]
        else:
            sanitized = []

        answer_markdown = parsed.get("answer_markdown", rag_answer)
        answer_markdown = str(answer_markdown or "").strip()
        warnings = [str(w) for w in (parsed.get("warnings") or [])]
        if not answer_markdown:
            if rag_answer.strip():
                answer_markdown = rag_answer
                warnings.append(
                    "External synthesis returned an empty answer; using RAG fallback."
                )
            else:
                answer_markdown = (
                    "The current knowledge base did not return enough relevant "
                    "evidence to answer this question reliably."
                )
                warnings.append(
                    "External synthesis returned an empty answer and no RAG "
                    "fallback was available."
                )

        # ---- bounded coverage retry (Phase 3) ----
        if answer_markdown and self._should_retry_for_coverage(
            route=route,
            evidence_pack=evidence_pack,
            citations=citations,
            answer_text=answer_markdown,
        ):
            retry_result = self._retry_with_anchors(
                query=query,
                route=route,
                conversation_summary=conversation_summary,
                rag_answer=rag_answer,
                citations=citations,
                evidence_pack=evidence_pack,
                first_answer=answer_markdown,
                headers=headers,
            )
            if retry_result is not None:
                return retry_result
            # Retry failed — fall through to original result
            warnings.append(
                "Evidence coverage retry did not produce a usable result; "
                "using first synthesis."
            )

        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": sanitized,
            "warnings": warnings,
            "confidence": float(parsed.get("confidence", 1.0)),
            "provider": "external_api",
            "model": self._settings.external_api_model,
        }

    # ------------------------------------------------------------------
    # coverage retry helpers (Phase 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_evidence_anchors(
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
    ) -> list[str]:
        """Extract high-signal anchor terms from evidence using generic rules.

        Anchors come from two sources:
        1. Evidence pack item excerpts (numeric patterns, abbreviations).
        2. Citation excerpts (numeric patterns, abbreviations).

        Rules are generic — no hard-coded domain terms.
        Returns at most 10 deduplicated anchors.
        """
        excerpts: list[str] = []

        # Collect excerpts from evidence pack
        if evidence_pack and evidence_pack.get("items"):
            for item in evidence_pack["items"]:
                ex = item.get("excerpt", "")
                if ex:
                    excerpts.append(str(ex))

        # Collect excerpts from citations
        for c in citations:
            ex = c.get("excerpt", "")
            if ex:
                excerpts.append(str(ex))

        if not excerpts:
            return []

        all_text = " ".join(excerpts)
        anchors: list[str] = []

        # Rule 1: numeric patterns with optional units
        numeric_pattern = re.compile(
            r"(?:^|[\s(])"
            r"[+-]?\d+\.?\d*"
            r"(?:\s*(?:kcal/mol|kJ/mol|kcal|kJ|nm|Å|eV|kcal·mol|kJ·mol|%|°C|K))?"
            r"(?:$|[\s,.);])",
        )
        for m in numeric_pattern.finditer(all_text.lower()):
            val = m.group().strip(" ,.();")
            if len(val) >= 2:
                anchors.append(val)

        # Rule 2: uppercase abbreviations (2–5 uppercase letters, whole-word)
        abbrev_pattern = re.compile(r"\b[A-Z]{2,5}\b")
        for m in abbrev_pattern.finditer(all_text):
            val = m.group()
            # Skip very common noise abbreviations
            if val in ("THE", "AND", "FOR", "ARE", "WAS", "HAS", "THIS", "THAT", "WITH", "FROM"):
                continue
            anchors.append(val)

        # Deduplicate preserving order, case-insensitive
        seen: set[str] = set()
        unique: list[str] = []
        for a in anchors:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique[:10]

    @staticmethod
    def _should_retry_for_coverage(
        *,
        route: str,
        evidence_pack: dict[str, Any] | None,
        citations: list[dict[str, Any]],
        answer_text: str,
    ) -> bool:
        """Decide whether a coverage retry is warranted.

        Returns True only when ALL of:
        - Route is evidence-heavy (``evidence_required``, ``table_or_metric``,
          ``multi_source_compare``).
        - Evidence pack is present and non-empty.
        - At least some extracted anchors are missing from the answer text.
        - Retry has not already been attempted (caller enforces this).
        """
        # Only retry on evidence-heavy routes
        if route not in ("evidence_required", "table_or_metric", "multi_source_compare"):
            return False

        # Must have a non-empty evidence pack
        if not evidence_pack or not evidence_pack.get("items"):
            return False

        anchors = AgentSynthesizer._extract_evidence_anchors(
            evidence_pack, citations
        )
        if not anchors:
            return False

        # Check coverage: count how many anchors appear in answer (case-insensitive)
        answer_lower = answer_text.lower()
        covered = sum(1 for a in anchors if a.lower() in answer_lower)

        # Retry if fewer than half of the anchors are covered
        return covered < max(1, len(anchors) // 2)

    def _retry_with_anchors(
        self,
        *,
        query: str,
        route: str,
        conversation_summary: str,
        rag_answer: str,
        citations: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None,
        first_answer: str,
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        """Retry external synthesis once with explicit missing-anchor hints.

        Calls the external API a second time with an augmented prompt that
        lists the missing evidence anchors and instructs the model to
        include them.  Returns ``None`` on any failure so the caller can
        fall back to the first synthesis result.
        """
        anchors = self._extract_evidence_anchors(evidence_pack, citations)
        if not anchors:
            return None

        answer_lower = first_answer.lower()
        missing = [a for a in anchors if a.lower() not in answer_lower]
        if not missing:
            return None  # all covered — no need to retry

        # Rebuild evidence context (same as first call)
        evidence_parts: list[str] = []
        for i, c in enumerate(citations):
            excerpt = c.get("excerpt", "")
            title = c.get("page_title") or c.get("document_id", "")
            if excerpt:
                evidence_parts.append(f"[{i}] {title}: {excerpt}")
        evidence_text = "\n\n".join(evidence_parts) if evidence_parts else "(no evidence)"

        evidence_pack_text = self._format_evidence_pack_section(evidence_pack)

        missing_list = ", ".join(missing[:8])

        system_prompt = (
            "You are an evidence synthesis assistant. Your task is to synthesize "
            "a final answer from the provided RAG answer and evidence excerpts. "
            "You MUST synthesize — do NOT simply concatenate paragraphs. "
            "IMPORTANT: The first answer omitted key evidence anchors. "
            "Your revised answer MUST explicitly include these evidence terms: "
            f"{missing_list}. "
            "Only cite sources that are present in the provided evidence. "
            "Return a JSON object with keys: answer_markdown (string), "
            "cited_indexes (array of integers, only valid indexes from the evidence), "
            "warnings (array of strings), confidence (number 0.0-1.0)."
        )

        user_prompt = (
            f"Original query: {query}\n"
            f"Route type: {route}\n"
            f"Conversation context: {conversation_summary or '(none)'}\n\n"
            f"RAG answer: {rag_answer}\n\n"
            f"Evidence excerpts with citation indexes:\n{evidence_text}\n\n"
            + (f"{evidence_pack_text}\n\n" if evidence_pack_text else "")
            + f"The first synthesis omitted these evidence anchors: {missing_list}\n"
            "Please produce a revised synthesis that explicitly includes "
            "these evidence terms where they are supported by the evidence. "
            "Return only a JSON object."
        )

        payload: dict[str, Any] = {
            "model": self._settings.external_api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            with httpx.Client(timeout=self._settings.external_api_timeout) as client:
                response = client.post(
                    f"{self._settings.external_api_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("Coverage retry API call failed: %s", exc)
            return None

        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, json.JSONDecodeError, IndexError) as exc:
            logger.warning("Failed to parse coverage retry response: %s", exc)
            return None

        # Sanitize cited_indexes
        max_idx = len(citations)
        raw_indexes = parsed.get("cited_indexes", [])
        if isinstance(raw_indexes, list):
            sanitized = [i for i in raw_indexes if isinstance(i, int) and 0 <= i < max_idx]
        else:
            sanitized = []

        answer_markdown = parsed.get("answer_markdown", "")
        answer_markdown = str(answer_markdown or "").strip()
        warnings = [str(w) for w in (parsed.get("warnings") or [])]
        warnings.append(
            "Coverage retry performed — first answer omitted evidence anchors"
        )

        if not answer_markdown:
            return None  # fall back to first answer

        return {
            "answer_markdown": answer_markdown,
            "cited_indexes": sanitized,
            "warnings": warnings,
            "confidence": float(parsed.get("confidence", 1.0)),
            "provider": "external_api",
            "model": self._settings.external_api_model,
        }
