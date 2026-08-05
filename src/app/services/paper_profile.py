from __future__ import annotations

"""
Paper Profile：为每篇文档生成“画像”，用于检索路由和证据召回。

文档画像（PaperProfile）包含：
- title：清洗后的标题
- source_slug：文档在知识库中的稳定路径标识
- one_sentence：一句话摘要
- routing_summary：给路由/检索模型看的浓缩文本
- aliases：标题别名、模型名、缩写等
- key_terms：从全文提取的关键术语、数值+单位、表格/图标签等

主要入口：
- ensure_paper_profile(document): 确保文档已有画像，没有则生成并写入 metadata。
- paper_profile_data(document): 返回画像字典。
- paper_profile_retrieval_terms(document): 返回用于检索扩展的术语列表。
- source_fields_for_document(document): 返回 source chunk 的 page_slug / page_title。
"""

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.models.records import Document, DocumentChunk
from app.services.filesystem import slugify, strip_upload_prefix


# 画像版本号，用于缓存失效。
PROFILE_VERSION = "paper-profile-v1"

# 从文本中提取候选术语：字母数字开头，可包含 -_/ 连接符。
PROFILE_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*\b")
# 数值+单位模式：如 50%、500 K、1.2 kcal/mol、100 fs 等。
PROFILE_NUMERIC_UNIT_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:%|kT|K|kcal(?:\s*/\s*mol|\s+mol)?|fs|ps|ns|us|A)(?![\w/])",
    re.IGNORECASE,
)
# 表格标签：Table 1、Table 2 等。
TABLE_LABEL_RE = re.compile(r"\bTable\s*\d+\b", re.IGNORECASE)
# 预定义的领域短语模式，用于提升检索路由命中率。
PROFILE_PHRASE_PATTERNS = (
    re.compile(r"\bamino-acid specific\b", re.IGNORECASE),
    re.compile(r"\bC6\s+(?:coefficients?|dispersion|parameters?)\b", re.IGNORECASE),
    re.compile(r"\bcharge transfer\b", re.IGNORECASE),
    re.compile(r"\bcovalent relaxation\b", re.IGNORECASE),
    re.compile(r"\bexplicit hydrogen\b", re.IGNORECASE),
    re.compile(r"\bexpanded ensembles?\b", re.IGNORECASE),
    re.compile(r"\bhelical propensity\b", re.IGNORECASE),
    re.compile(r"\bhydration free energy\b", re.IGNORECASE),
    re.compile(r"\bLennard-Jones\b", re.IGNORECASE),
    re.compile(r"\bmolten globule\b", re.IGNORECASE),
    re.compile(r"\bMonte Carlo\b", re.IGNORECASE),
    re.compile(r"\bneutral state\b", re.IGNORECASE),
    re.compile(r"\bsalt bridge\b", re.IGNORECASE),
    re.compile(r"\bsteric clashes?\b", re.IGNORECASE),
    re.compile(r"\bvan der Waals\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+organic liquids\b", re.IGNORECASE),
)
PROFILE_CHI_RE = re.compile(r"(?:\\chi|\u03c7|chi)\s*_?\s*\{?\s*([12])\s*\}?", re.IGNORECASE)

@dataclass
class PaperProfile:
    """单篇文档的画像。"""

    document_id: str
    title: str
    one_sentence: str
    routing_summary: str
    profile_version: str = PROFILE_VERSION
    source_text_sha256: str = ""
    source_slug: str | None = None
    aliases: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """序列化为字典，便于存入 document.metadata_json。"""
        return {
            "profile_version": self.profile_version,
            "source_text_sha256": self.source_text_sha256,
            "document_id": self.document_id,
            "title": self.title,
            "source_slug": self.source_slug,
            "one_sentence": self.one_sentence,
            "routing_summary": self.routing_summary,
            "aliases": self.aliases,
            "key_terms": self.key_terms,
        }


def ensure_paper_profile(document: Document) -> dict:
    """确保文档已有 paper_profile；如果没有或版本过期，则重新生成并写入 metadata。"""
    metadata = dict(document.metadata_json or {})
    _ensure_source_identity(document, metadata)
    document.metadata_json = metadata
    existing = metadata.get("paper_profile")
    if _profile_is_current(existing, document):
        return existing
    profile = paper_profile_data(document)
    metadata["paper_profile"] = profile
    document.metadata_json = metadata
    return profile


def prepare_canonical_profile_source(
    document: Document,
    children: Iterable[DocumentChunk],
) -> str:
    """Install the canonical Child text and discard stale profile inputs."""
    ordered = sorted(children, key=lambda item: (item.ordinal, item.id))
    document.raw_text = "\n\n".join(
        chunk.text.strip()
        for chunk in ordered
        if chunk.block_type != "reference" and chunk.text.strip()
    )
    metadata = dict(document.metadata_json or {})
    metadata.pop("paper_profile", None)
    metadata.pop("document_intelligence", None)
    document.metadata_json = metadata
    return document.raw_text


def paper_profile_data(document: Document) -> dict:
    """返回文档画像字典（优先使用缓存）。"""
    metadata = dict(document.metadata_json or {})
    _ensure_source_identity(document, metadata)
    existing = metadata.get("paper_profile")
    if _profile_is_current(existing, document):
        document.metadata_json = metadata
        return existing
    return build_paper_profile(document).to_dict()


def paper_profile_retrieval_terms(document: Document) -> list[str]:
    """生成用于检索扩展的术语列表。

    来源包括：aliases、key_terms、从原文提取的关键词。去重后最多返回 96 个。
    """
    metadata = dict(document.metadata_json or {})
    _ensure_source_identity(document, metadata)
    document.metadata_json = metadata
    profile = paper_profile_data(document)
    existing_terms = profile.get("key_terms") if isinstance(profile.get("key_terms"), list) else []
    aliases = profile.get("aliases") if isinstance(profile.get("aliases"), list) else []
    source_text = _profile_source_text(document)
    return _ordered_unique(
        [
            *[str(term) for term in aliases],
            *_keyword_terms(source_text),
            *[str(term) for term in existing_terms],
        ]
    )[:96]


def ensure_source_identity(document: Document, preferred_title: str | None = None) -> dict:
    """确保文档的 source_slug 和 source_title 已写入 metadata。"""
    metadata = dict(document.metadata_json or {})
    _ensure_source_identity(document, metadata, preferred_title=preferred_title)
    document.metadata_json = metadata
    return metadata


def source_fields_for_document(document: Document) -> dict[str, str]:
    """返回文档 source summary chunk 的 page_slug / page_title / page_kind。"""
    metadata = dict(document.metadata_json or {})
    _ensure_source_identity(document, metadata)
    profile = metadata.get("paper_profile") if isinstance(metadata.get("paper_profile"), dict) else {}
    source_slug = str(metadata.get("source_slug") or profile.get("source_slug") or "").strip()
    source_title = str(metadata.get("source_title") or profile.get("title") or document.title or document.file_name or source_slug).strip()
    return {
        "page_slug": source_slug or source_slug_for_title(document.title or document.file_name or document.id),
        "page_title": strip_upload_prefix(source_title),
        "page_kind": "source_summary",
    }


def source_slug_for_title(title: str | None) -> str:
    """根据标题生成稳定的 source slug，格式为 sources/<slugified-title>。"""
    cleaned = strip_upload_prefix(str(title or "")).strip()
    return f"sources/{slugify(cleaned or 'untitled-document')}"


def build_paper_profile(document: Document) -> PaperProfile:
    """从零构建文档画像。

    构建步骤：
    1. 清洗标题，确定 source_slug；
    2. 拼接标题、正文、表格、图注文本；
    3. 提取 aliases（标题缩写、模型名等）；
    4. 提取 table_terms / figure_terms；
    5. 用 _keyword_terms 从全文提取关键术语；
    6. 生成一句话摘要和 routing_summary；
    7. 组装 PaperProfile。
    """
    metadata = document.metadata_json or {}
    title = strip_upload_prefix(document.title or document.file_name or "Untitled document")
    clean_title = _clean_title(title)
    source_slug = (
        str(metadata.get("source_slug") or metadata.get("page_slug") or "").strip()
        or source_slug_for_title(clean_title)
    )
    intelligence = metadata.get("document_intelligence") if isinstance(metadata.get("document_intelligence"), dict) else {}
    tables = intelligence.get("tables") if isinstance(intelligence, dict) else []
    figures = intelligence.get("figures") if isinstance(intelligence, dict) else []
    title_text = "\n".join(part for part in (clean_title, str(document.file_name or "")) if part)
    text = _profile_source_text(document, title_text=title_text, tables=tables, figures=figures)
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
        source_text_sha256=_profile_source_text_sha256(document),
        one_sentence=one_sentence[:500],
        routing_summary=routing_summary[:1600],
        aliases=aliases[:16],
        key_terms=key_terms,
    )


def _profile_source_text_sha256(document: Document) -> str:
    return hashlib.sha256((document.raw_text or "").encode("utf-8")).hexdigest()


def _profile_is_current(existing: object, document: Document) -> bool:
    return (
        isinstance(existing, dict)
        and existing.get("profile_version") == PROFILE_VERSION
        and bool(existing.get("routing_summary"))
        and existing.get("source_text_sha256")
        == _profile_source_text_sha256(document)
    )


def paper_profile_text(document: Document) -> str:
    """把画像拼接成一段文本，用于 embedding 或向量化。"""
    profile = paper_profile_data(document)
    parts = [
        str(profile.get("title") or document.title or ""),
        str(profile.get("one_sentence") or ""),
        str(profile.get("routing_summary") or ""),
        " ".join(str(item) for item in profile.get("aliases") or []),
        " ".join(str(item) for item in profile.get("key_terms") or []),
    ]
    return "\n".join(part for part in parts if part)


def _clean_title(value: str) -> str:
    """清洗标题：去除上传前缀、把 _/- 替换为空格、压缩空白。"""
    text = strip_upload_prefix(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or value


def alias_in_text(alias: str, text: str) -> bool:
    """判断别名是否作为独立词出现在文本中。"""
    alias = str(alias or "").strip()
    if not alias or not text:
        return False
    escaped = re.escape(alias)
    return bool(re.search(rf"(?<![A-Za-z0-9_\-]){escaped}(?![A-Za-z0-9_\-])", text, re.IGNORECASE))


def _title_aliases(title: str) -> list[str]:
    """从标题中提取别名：紧凑形式、模型名、缩写等。"""
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
    """从全文中提取关键术语并打分排序。

    提取规则：
    - 过滤常见停用词（with、from、table、paper 等）；
    - 优先保留：含数字、含分隔符、全大写缩写、大小写混合、数值+单位、领域术语；
    - 过滤看起来像文件路径、原子类型序列的噪音；
    - 重复出现加分，按分数降序、首次出现位置升序返回。
    """
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
        "page",
        "supporting",
        "information",
        "department",
        "university",
        "author",
    }
    domain_terms = {
        "alphal",
        "backbone",
        "barrier",
        "boltzmann",
        "chargetransfer",
        "covalentrelaxation",
        "cmap",
        "drude",
        "expandedensemble",
        "expandedensembles",
        "explicithydrogen",
        "fret",
        "helical",
        "helicalpropensity",
        "hydrationfreeenergy",
        "idp",
        "fitting",
        "lennardjones",
        "moltenglobule",
        "montecarlo",
        "neutralstate",
        "polarizability",
        "population",
        "propensity",
        "protocol",
        "protocols",
        "saltbridge",
        "restraint",
        "restraints",
        "rotamer",
        "rotamers",
        "sidechain",
        "side-chain",
        "steric",
        "stericclash",
        "stericclashes",
        "tetrapeptide",
        "torsion",
        "torsions",
        "vanderwaals",
        "vdw",
    }
    candidates: dict[str, tuple[str, float, int]] = {}

    def add_candidate(token: str, position: int) -> None:
        token = re.sub(r"\s+", " ", token).strip()
        normalized = token.lower()
        normalized_key = normalized.replace("\u03c7", "chi")
        compact = re.sub(r"[^a-z0-9]+", "", normalized_key)
        if len(compact) < 3 or normalized in stopwords:
            return
        if compact.startswith(("author", "department", "university")):
            return
        if _looks_like_artifact_path(token):
            return
        if _looks_like_atom_type_series(token):
            return
        has_digit = any(char.isdigit() for char in token)
        has_separator = any(char in token for char in "-_/")
        uppercase_count = sum(1 for char in token if char.isupper())
        is_acronym = uppercase_count >= 2 and token.upper() == token
        is_mixed_case = bool(re.search(r"[a-z][A-Z]", token)) or (
            uppercase_count >= 2 and any(char.islower() for char in token)
        )
        is_numeric_unit = bool(PROFILE_NUMERIC_UNIT_RE.fullmatch(token))
        is_domain_term = normalized in domain_terms or compact in domain_terms
        if not (has_digit or has_separator or is_acronym or is_mixed_case or is_numeric_unit or is_domain_term):
            return
        score = 1.0
        if has_digit and not is_numeric_unit:
            score += 5.0
        if has_separator and not is_numeric_unit:
            score += 4.0
        if is_acronym:
            score += 4.0
        if is_mixed_case:
            score += 7.0
        if is_numeric_unit:
            score += 2.0
        if is_domain_term:
            score += 3.0
        existing = candidates.get(normalized)
        if existing is None:
            candidates[normalized] = (token, score, position)
        else:
            first_token, existing_score, first_position = existing
            candidates[normalized] = (first_token, existing_score + 1.0, first_position)

    for match in PROFILE_NUMERIC_UNIT_RE.finditer(text):
        add_candidate(match.group(0), match.start())
    for match in PROFILE_PHRASE_PATTERNS:
        for phrase_match in match.finditer(text):
            add_candidate(phrase_match.group(0), phrase_match.start())
    for match in PROFILE_CHI_RE.finditer(text):
        add_candidate(f"\u03c7{match.group(1)}", match.start())
    for match in PROFILE_TERM_RE.finditer(text):
        add_candidate(match.group(0), match.start())
    ranked = sorted(candidates.values(), key=lambda item: (-item[1], item[2], item[0].lower()))
    return [term for term, _score, _position in ranked]


def _looks_like_atom_type_series(token: str) -> bool:
    """判断 token 是否像原子类型序列（如 C-N-CA-CB），这类噪音不应作为关键术语。"""
    parts = [part for part in re.split(r"[-_/]", token) if part]
    if len(parts) < 3:
        return False
    short_parts = sum(1 for part in parts if len(part) <= 3 and re.fullmatch(r"[A-Za-z0-9]+", part))
    return short_parts / len(parts) >= 0.75


def _looks_like_artifact_path(token: str) -> bool:
    """判断 token 是否像图片/附件路径或哈希文件名，这类噪音应过滤。"""
    lowered = token.lower()
    if lowered.startswith(("images/", "image/", "figures/", "figure/")):
        return True
    parts = [part for part in re.split(r"[/\\]", token) if part]
    if len(parts) >= 2 and re.fullmatch(r"[a-f0-9]{24,}", parts[-1], re.IGNORECASE):
        return True
    return len(token) > 48 and ("/" in token or "\\" in token)


def _profile_source_text(
    document: Document,
    title_text: str | None = None,
    tables: object | None = None,
    figures: object | None = None,
) -> str:
    """拼接用于画像分析的源文本：标题 + 正文 + 表格文本 + 图注文本。"""
    metadata = document.metadata_json or {}
    if tables is None or figures is None:
        intelligence = metadata.get("document_intelligence") if isinstance(metadata.get("document_intelligence"), dict) else {}
        tables = intelligence.get("tables") if isinstance(intelligence, dict) else []
        figures = intelligence.get("figures") if isinstance(intelligence, dict) else []
    if title_text is None:
        title = strip_upload_prefix(document.title or document.file_name or "Untitled document")
        title_text = "\n".join(part for part in (_clean_title(title), str(document.file_name or "")) if part)
    return "\n".join(
        part
        for part in (
            title_text,
            document.raw_text or "",
            _join_table_text(tables),
            _join_figure_text(figures),
        )
        if part
    )


def _join_table_text(tables: object) -> str:
    """从 intelligence tables 中提取 markdown 文本，最多取前 12 个表格。"""
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
    """从 intelligence figures 中提取 note/caption 文本，最多取前 12 个图。"""
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
    """从正文中提取第一个完整句子（80-500 字符），用于生成一句话摘要。"""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    match = re.search(r"(.{80,500}?[.!?])\s", normalized)
    if match:
        return match.group(1).strip()
    return normalized[:300].strip()


def _ensure_source_identity(document: Document, metadata: dict, preferred_title: str | None = None) -> None:
    """确保 metadata 中已有 source_slug 和 source_title。"""
    title = strip_upload_prefix(preferred_title or document.title or document.file_name or "Untitled document")
    metadata.setdefault("source_slug", source_slug_for_title(title))
    metadata.setdefault("source_title", _clean_title(title))


def _ordered_unique(values: list[str]) -> list[str]:
    """去重并保持首次出现顺序（按规范化小写 key 判断重复）。"""
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
