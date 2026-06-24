from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.records import Document
from app.services.filesystem import strip_upload_prefix


PROFILE_VERSION = "paper-profile-v1"

PROFILE_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*\b")
TABLE_LABEL_RE = re.compile(r"\bTable\s*\d+\b", re.IGNORECASE)


@dataclass
class PaperProfile:
    document_id: str
    title: str
    one_sentence: str
    routing_summary: str
    profile_version: str = PROFILE_VERSION
    source_slug: str | None = None
    aliases: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "profile_version": self.profile_version,
            "document_id": self.document_id,
            "title": self.title,
            "source_slug": self.source_slug,
            "one_sentence": self.one_sentence,
            "routing_summary": self.routing_summary,
            "aliases": self.aliases,
            "key_terms": self.key_terms,
        }


def ensure_paper_profile(document: Document) -> dict:
    metadata = dict(document.metadata_json or {})
    existing = metadata.get("paper_profile")
    if isinstance(existing, dict) and existing.get("profile_version") == PROFILE_VERSION and existing.get("routing_summary"):
        return existing
    profile = build_paper_profile(document).to_dict()
    metadata["paper_profile"] = profile
    document.metadata_json = metadata
    return profile


def build_paper_profile(document: Document) -> PaperProfile:
    metadata = document.metadata_json or {}
    title = strip_upload_prefix(document.title or document.file_name or "Untitled document")
    clean_title = _clean_title(title)
    source_slug = str(metadata.get("source_slug") or metadata.get("page_slug") or "").strip() or None
    intelligence = metadata.get("document_intelligence") if isinstance(metadata.get("document_intelligence"), dict) else {}
    tables = intelligence.get("tables") if isinstance(intelligence, dict) else []
    figures = intelligence.get("figures") if isinstance(intelligence, dict) else []
    title_text = "\n".join(part for part in (clean_title, str(document.file_name or "")) if part)
    text = "\n".join(
        part
        for part in (
            title_text,
            (document.raw_text or "")[:5000],
            _join_table_text(tables),
            _join_figure_text(figures),
        )
        if part
    )
    aliases = _ordered_unique([clean_title, *_title_aliases(title)])
    table_terms = [match.group(0).replace(" ", " ") for match in TABLE_LABEL_RE.finditer(_join_table_text(tables))]
    figure_terms = [match.group(0).replace(" ", " ") for match in re.finditer(r"\b(?:Figure|Fig\.)\s*\d+\b", _join_figure_text(figures), re.IGNORECASE)]
    key_terms = _ordered_unique([*aliases, *table_terms, *figure_terms, *_keyword_terms(text)])[:48]
    one_sentence = _first_sentence(document.raw_text or "") or f"{clean_title} is a research paper about {', '.join(aliases[:3]) or clean_title}."
    routing_summary = " ".join(
        part
        for part in (
            one_sentence,
            f"Source slug: {source_slug}." if source_slug else "",
            f"Aliases: {', '.join(aliases[:10])}." if aliases else "",
            f"Key terms: {', '.join(key_terms[:16])}." if key_terms else "",
            f"Important tables: {', '.join(_ordered_unique(table_terms)[:8])}." if table_terms else "",
            f"Important figures: {', '.join(_ordered_unique(figure_terms)[:8])}." if figure_terms else "",
        )
        if part
    )
    return PaperProfile(
        document_id=document.id,
        title=clean_title,
        source_slug=source_slug,
        one_sentence=one_sentence[:500],
        routing_summary=routing_summary[:1600],
        aliases=aliases[:16],
        key_terms=key_terms,
    )


def paper_profile_text(document: Document) -> str:
    profile = ensure_paper_profile(document)
    parts = [
        str(profile.get("title") or document.title or ""),
        str(profile.get("one_sentence") or ""),
        str(profile.get("routing_summary") or ""),
        " ".join(str(item) for item in profile.get("aliases") or []),
        " ".join(str(item) for item in profile.get("key_terms") or []),
    ]
    return "\n".join(part for part in parts if part)


def _clean_title(value: str) -> str:
    text = strip_upload_prefix(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or value


def alias_in_text(alias: str, text: str) -> bool:
    alias = str(alias or "").strip()
    if not alias or not text:
        return False
    escaped = re.escape(alias)
    return bool(re.search(rf"(?<![A-Za-z0-9_/\-]){escaped}(?![A-Za-z0-9_/\-])", text, re.IGNORECASE))


def _title_aliases(title: str) -> list[str]:
    aliases: list[str] = []
    compact = re.sub(r"[^A-Za-z0-9]+", "", title)
    if 3 <= len(compact) <= 24:
        aliases.append(compact)
    for match in PROFILE_TERM_RE.finditer(title):
        token = match.group(0)
        uppercase_count = sum(1 for char in token if char.isupper())
        looks_like_model_name = (
            any(char.isdigit() for char in token)
            or uppercase_count >= 2
            or "-" in token
            or "/" in token
            or token.isupper()
        )
        if len(token) >= 3 and looks_like_model_name:
            aliases.append(token)
    return aliases


def _keyword_terms(text: str) -> list[str]:
    stopwords = {
        "with",
        "from",
        "that",
        "this",
        "using",
        "table",
        "figure",
        "results",
        "protein",
        "force",
        "field",
        "study",
        "paper",
    }
    terms: list[str] = []
    for match in PROFILE_TERM_RE.finditer(text):
        token = match.group(0)
        normalized = token.lower()
        if len(token) < 4 or normalized in stopwords:
            continue
        if token[:1].islower() and "-" not in token and "/" not in token:
            continue
        terms.append(token)
    return terms


def _join_table_text(tables: object) -> str:
    if not isinstance(tables, list):
        return ""
    parts: list[str] = []
    for table in tables[:12]:
        if isinstance(table, dict):
            parts.append(str(table.get("markdown") or ""))
        else:
            parts.append(str(table))
    return "\n".join(parts)


def _join_figure_text(figures: object) -> str:
    if not isinstance(figures, list):
        return ""
    parts: list[str] = []
    for figure in figures[:12]:
        if isinstance(figure, dict):
            parts.append(str(figure.get("note") or figure.get("caption") or ""))
        else:
            parts.append(str(figure))
    return "\n".join(parts)


def _first_sentence(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    match = re.search(r"(.{80,500}?[.!?])\s", normalized)
    if match:
        return match.group(1).strip()
    return normalized[:300].strip()


def _ordered_unique(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = re.sub(r"\s+", " ", text).lower()
        if not text or key in seen:
            continue
        ordered.append(text)
        seen.add(key)
    return ordered
