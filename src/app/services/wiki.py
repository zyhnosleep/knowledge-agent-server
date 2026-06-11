from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app.models.records import Claim, Document, Entity, Project, WikiPage
from app.services.ai import DocumentExtraction
from app.services.filesystem import project_paths, slugify


class WikiRenderer:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.paths = project_paths(project.slug)

    def render_document_summary(#生成文档摘要页
        self,
        document: Document,
        extraction: DocumentExtraction,# 之前提取的结构化数据
        claims: list[Claim] | None = None,# 验证通过的知识声明
        *,
        entity_links: dict[str, str] | None = None, # 实体→页面链接映射
        metadata: dict | None = None,
    ) -> tuple[str, str]:# 返回(页面slug, 完整Markdown内容)
        title = extraction.title or document.title
        slug = f"sources/{slugify(title)}"
        claims = claims or []
        entity_links = entity_links or {}
        verified_claims = [claim for claim in claims if claim.verification_status == "verified"]
        metadata = metadata or {
            "source_count": 1,
            "verified_claim_count": len(verified_claims),
            "key_terms": extraction.keywords[:12],
        }
        by_head: dict[str, list[Claim]] = defaultdict(list)
        for claim in verified_claims:
            by_head[(claim.metadata_json or {}).get("expected_head") or claim.subject].append(claim)
        key_fact_lines = [f"- {fact}" for fact in extraction.key_facts] or ["- None"]
        claim_lines: list[str] = []
        if by_head:
            for head, head_claims in sorted(by_head.items(), key=lambda item: item[0].lower()):
                claim_lines.append(f"### {self._wikilink(head, entity_links.get(head))}")
                claim_lines.extend(self._format_claim_line(claim, entity_links) for claim in head_claims)
                claim_lines.append("")
        else:
            claim_lines = ["- None"]
        evidence_lines = [
            f"- {self._wikilink(claim.subject, entity_links.get(claim.subject))} {claim.predicate}: {(claim.metadata_json or {}).get('evidence_excerpt', '').strip()}"
            for claim in verified_claims
            if (claim.metadata_json or {}).get("evidence_excerpt")
        ] or ["- None"]
        related_pages = [
            f"- {self._wikilink(name, slug)}"
            for name, slug in sorted(entity_links.items(), key=lambda item: item[0].lower())
        ] or ["- None"]
        intelligence_sections = self._render_document_intelligence_sections(document)
        # Keywords: prefer extraction.keywords, fall back to frontmatter key_terms.
        keywords_list = list(extraction.keywords) if extraction.keywords else []
        if not keywords_list:
            key_terms = metadata.get("key_terms", [])
            if isinstance(key_terms, list) and key_terms:
                keywords_list = [str(term) for term in key_terms[:12]]
        frontmatter = self._render_frontmatter(
            title=title,
            kind="source_summary",
            metadata=metadata,
        )
        content = "\n".join(
            [
                frontmatter,
                f"# {title}",
                "",
                "## Summary",
                extraction.summary.strip() or "No summary generated.",
                "",
                "## Key Facts",
                *key_fact_lines,
                "",
                "## Verified Triples By Head",
                *claim_lines,
                "",
                "## Evidence Notes",
                *evidence_lines,
                *intelligence_sections,
                "",
                "## Related Pages",
                *related_pages,
                "",
                "## Keywords",
                ", ".join(keywords_list) if keywords_list else "None",
                "",
                "## Concepts",
                "\n".join(f"- {concept}" for concept in extraction.concepts) or "- None",
            ]
        )
        return slug, content

    def _render_document_intelligence_sections(self, document: Document) -> list[str]:
        intelligence = (document.metadata_json or {}).get("document_intelligence", {})
        if not intelligence:
            return []
        sections: list[str] = ["", "## Page Structure"]

        # Pre-index tables and figures by page for fallback summaries.
        tables_by_page: dict[str, list[str]] = {}
        for table in (intelligence.get("tables") or []):
            label = str(table.get("page_label", "?"))
            markdown = str(table.get("markdown") or "").strip()
            if markdown:
                tables_by_page.setdefault(label, []).append(markdown)
        figures_by_page: dict[str, list[str]] = {}
        for figure in (intelligence.get("figures") or []):
            label = str(figure.get("page_label", "?"))
            note = str(figure.get("note") or "").strip()
            if note:
                figures_by_page.setdefault(label, []).append(note)

        page_outputs = intelligence.get("page_outputs", [])
        if isinstance(page_outputs, list) and page_outputs:
            for page in page_outputs[:8]:
                page_label = str(page.get("page_label", "?"))
                summary = str(page.get("page_summary") or "").strip()
                quality = str(page.get("text_quality") or "unknown")
                # Generate fallback summary from tables/figures on the same page.
                if not summary:
                    fallback_parts: list[str] = []
                    for tbl in tables_by_page.get(page_label, [])[:1]:
                        fallback_parts.append(f"Contains table: {tbl[:120]}")
                    for fig in figures_by_page.get(page_label, [])[:1]:
                        fallback_parts.append(f"Figure: {fig[:120]}")
                    page_text = str(page.get("page_markdown") or "").strip()
                    if not fallback_parts and page_text:
                        fallback_parts.append(page_text[:200])
                    summary = "; ".join(fallback_parts) if fallback_parts else "No summary available."
                sections.append(f"- Page {page_label} [{quality}]: {summary}")
        else:
            sections.append("- No structured page summaries available.")

        tables = intelligence.get("tables", [])
        if isinstance(tables, list) and tables:
            sections.extend(["", "## Tables"])
            for table in tables[:4]:
                page_label = table.get("page_label", "?")
                markdown = str(table.get("markdown") or "").strip()
                sections.append(f"### Page {page_label}")
                sections.append(markdown or "No table markdown captured.")

        formulas = intelligence.get("formulas", [])
        if isinstance(formulas, list) and formulas:
            sections.extend(["", "## Formulas"])
            for formula in formulas[:6]:
                page_label = formula.get("page_label", "?")
                text = str(formula.get("text") or "").strip()
                sections.append(f"- Page {page_label}: {text}")

        figures = intelligence.get("figures", [])
        if isinstance(figures, list) and figures:
            sections.extend(["", "## Figure Notes"])
            for figure in figures[:6]:
                page_label = figure.get("page_label", "?")
                note = str(figure.get("note") or "").strip()
                sections.append(f"- Page {page_label}: {note}")

        return sections

    def render_entity_pages(#生成实体详情页
        self,
        entities: list[Entity],
        claims: list[Claim],
        *,
        source_page_slug: str = "",
        source_page_title: str = "Source Page",
        entity_links: dict[str, str] | None = None,
        metadata_by_name: dict[str, dict] | None = None,
    ) -> list[tuple[str, str, str]]:
        by_entity: dict[str, list[Claim]] = defaultdict(list)
        inbound_by_entity: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            if claim.verification_status != "verified":
                continue
            by_entity[claim.subject].append(claim)
            inbound_by_entity[claim.object_text].append(claim)

        pages: list[tuple[str, str, str]] = []
        entity_links = entity_links or {}
        metadata_by_name = metadata_by_name or {}
        for entity in entities:
            slug = f"entities/{slugify(entity.name)}"
            entity_links.setdefault(entity.name, slug)
            claim_lines = [
                self._format_claim_line(claim, entity_links)
                for claim in by_entity.get(entity.name, [])
            ] or ["- No claims yet."]
            inbound_lines = [
                self._format_claim_line(claim, entity_links)
                for claim in inbound_by_entity.get(entity.name, [])
            ] or ["- None"]
            related_targets = sorted(
                {
                    *(
                        claim.object_text
                        for claim in by_entity.get(entity.name, [])
                        if (claim.metadata_json or {}).get("tail_growth_decision") == "grow"
                    ),
                    *(
                        claim.subject
                        for claim in inbound_by_entity.get(entity.name, [])
                        if (claim.metadata_json or {}).get("head_growth_decision") == "grow"
                    ),
                }
            )
            related_lines = [
                f"- {self._wikilink(name, entity_links.get(name, f'entities/{slugify(name)}'))}"
                for name in related_targets
                if name != entity.name
            ] or ["- None"]
            page_metadata = metadata_by_name.get(entity.name, {})
            frontmatter = self._render_frontmatter(
                title=entity.name,
                kind="entity",
                metadata=page_metadata,
            )
            content = "\n".join(
                [
                    frontmatter,
                    f"# {entity.name}",
                    "",
                    f"Type: `{entity.entity_type}`",
                    "",
                    "## Summary",
                    entity.summary or "No summary available.",
                    "",
                    "## Outbound Claims",
                    *claim_lines,
                    "",
                    "## Inbound References",
                    *inbound_lines,
                    "",
                    "## Related Pages",
                    *related_lines,
                    "",
                    "## Source Backlinks",
                    f"- {self._wikilink(source_page_title, source_page_slug)}",
                ]
            )
            pages.append((entity.name, slug, content))
        return pages

    def _format_claim_line(self, claim: Claim, entity_links: dict[str, str] | None = None) -> str:
        entity_links = entity_links or {}
        metadata = claim.metadata_json or {}
        evidence = metadata.get("evidence_excerpt", "")
        suffix = f"; evidence: {evidence[:180]}" if evidence else ""
        subject = self._wikilink(claim.subject, entity_links.get(claim.subject))
        object_text = claim.object_text
        if metadata.get("tail_growth_decision") == "grow" or claim.object_text in entity_links:
            object_text = self._wikilink(claim.object_text, entity_links.get(claim.object_text, f"entities/{slugify(claim.object_text)}"))
        return (
            f"- {subject} {claim.predicate} {object_text} "
            f"(confidence={claim.confidence:.2f}, verification={claim.verification_status}{suffix})"
        )

    @staticmethod
    def _wikilink(label: str, slug: str | None = None) -> str:
        if not slug:
            return label
        return f"[[{slug}|{label}]]"

    def _render_frontmatter(self, *, title: str, kind: str, metadata: dict) -> str:
        lines = [
            "---",
            f'title: "{self._yaml_escape(title)}"',
            f'kind: "{self._yaml_escape(kind)}"',
            f"source_count: {int(metadata.get('source_count', 0) or 0)}",
            f"verified_claim_count: {int(metadata.get('verified_claim_count', 0) or 0)}",
        ]
        growth_decision = metadata.get("growth_decision")
        if growth_decision:
            lines.append(f'growth_decision: "{self._yaml_escape(str(growth_decision))}"')
        key_terms = metadata.get("key_terms", [])
        if isinstance(key_terms, list) and key_terms:
            lines.append("key_terms:")
            for term in key_terms[:12]:
                lines.append(f'  - "{self._yaml_escape(str(term))}"')
        lines.extend(["---", ""])
        return "\n".join(lines)

    @staticmethod
    def _yaml_escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def write_page(self, slug: str, content: str) -> Path:
        path = self.paths["wiki_root"] / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def render_index(self, pages: list[WikiPage]) -> Path:
        lines = ["# Index", ""]
        for page in sorted(pages, key=lambda item: item.title.lower()):
            # Build links from the logical slug instead of persisted path strings.
            # This keeps index generation stable across Windows/Linux path formats.
            relative = Path(f"{page.slug}.md")
            metadata = page.metadata_json or {}
            summary = (metadata.get("summary") or page.markdown_content[:220]).replace("\n", " ").strip()
            key_terms = metadata.get("key_terms", [])
            if not isinstance(key_terms, list):
                key_terms = []
            key_terms_text = ", ".join(str(term) for term in key_terms[:8]) or "none"
            source_count = metadata.get("source_count", len(page.source_document_ids or []))
            verified_count = metadata.get("verified_claim_count", 0)
            lines.append(
                f"- [{page.title}]({relative.as_posix()}) | type={page.kind} | "
                f"sources={source_count} | verified_claims={verified_count} | "
                f"key_terms={key_terms_text} | summary={summary[:260]}"
            )
        path = self.paths["wiki_root"] / "index.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def render_log(self, document: Document, extraction: DocumentExtraction) -> Path:
        path = self.paths["wiki_root"] / "log.md"
        entry = "\n".join(
            [
                f"## {document.title}",
                "",
                f"- SHA256: `{document.sha256}`",
                f"- Summary length: {len(extraction.summary)}",
                f"- Key facts preserved: {len(extraction.key_facts)}",
                f"- Claims extracted: {len(extraction.claims)}",
                "",
            ]
        )
        previous = path.read_text(encoding="utf-8") if path.exists() else "# Log\n\n"
        path.write_text(previous + entry, encoding="utf-8")
        return path
