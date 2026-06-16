from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.records import Document, PageKind, Project, WikiPage
from app.services.filesystem import InvalidStoragePathError, safe_project_slug
from app.services.table_extraction import extract_structured_tables, structure_table_markdown, table_metric_values

settings = get_settings()


def build_ingest_quality_report(document: Document) -> dict:
    metadata = document.metadata_json or {}
    intelligence = metadata.get("document_intelligence") or {}
    tables = [table for table in intelligence.get("tables") or [] if isinstance(table, dict)]
    structured_tables = intelligence.get("structured_tables") or extract_structured_tables(tables)
    table_summaries: list[dict] = []
    warnings: list[str] = []

    for table in structured_tables:
        if not isinstance(table, dict):
            continue
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        headers = table.get("headers") if isinstance(table.get("headers"), list) else []
        flags = table.get("quality_flags") if isinstance(table.get("quality_flags"), list) else []
        metrics = table_metric_values(str(table.get("markdown") or ""))
        summary = {
            "label": table.get("label"),
            "page_label": table.get("page_label"),
            "caption": table.get("caption"),
            "row_count": len(rows),
            "column_count": len(headers),
            "headers": headers,
            "quality_flags": flags,
            "metric_values": metrics,
        }
        table_summaries.append(summary)
        if flags:
            warnings.append(f"{table.get('label') or 'table'} has quality flags: {', '.join(str(flag) for flag in flags)}")
        if table.get("label") and not rows:
            warnings.append(f"{table.get('label')} has no structured rows")

    if not table_summaries and str(metadata.get("parser_mode") or "").startswith("pdf"):
        warnings.append("No tables were captured for this PDF document")

    page_outputs = intelligence.get("page_outputs") if isinstance(intelligence.get("page_outputs"), list) else []
    report = {
        "parser_mode": metadata.get("parser_mode"),
        "page_count": metadata.get("pages"),
        "page_output_count": len(page_outputs),
        "table_count": len(tables),
        "structured_table_count": len(table_summaries),
        "formula_count": len(intelligence.get("formulas") or []),
        "figure_count": len(intelligence.get("figures") or []),
        "tables": table_summaries,
        "warnings": warnings,
    }
    return report


def lint_project_wiki(db: Session, project_slug: str, *, limit: int = 50, offset: int = 0) -> dict:
    project = db.scalar(select(Project).where(Project.slug == project_slug))
    if project is None:
        raise ValueError(f"Project '{project_slug}' not found")
    try:
        project_path_slug = safe_project_slug(project.slug)
    except InvalidStoragePathError as exc:
        raise ValueError(f"Project '{project.slug}' has an unsafe slug") from exc

    pages = db.scalars(select(WikiPage).where(WikiPage.project_id == project.id)).all()
    issues: list[dict] = []
    index_path = settings.wiki_dir / project_path_slug / "index.md"
    if not index_path.exists():
        issues.append(_issue("missing_index", "high", "wiki/index.md does not exist", {}))
    else:
        index_text = index_path.read_text(encoding="utf-8")
        for page in pages:
            if page.kind == PageKind.query_answer.value:
                continue
            if f"{page.slug}.md" not in index_text:
                issues.append(_issue("missing_index_entry", "medium", f"{page.slug} is missing from index.md", {"page_slug": page.slug}))

    page_by_slug = {page.slug: page for page in pages}
    for page in pages:
        issues.extend(_lint_page_links(page, page_by_slug))
        if page.kind == PageKind.source_summary.value:
            issues.extend(_lint_source_page_tables(page))
        if page.kind == PageKind.query_answer.value:
            issues.extend(_lint_query_answer_citations(page))

    safe_limit = max(1, min(limit, 200))
    safe_offset = max(0, offset)
    returned_issues = issues[safe_offset : safe_offset + safe_limit]
    return {
        "project_slug": project.slug,
        "page_count": len(pages),
        "issue_count": len(issues),
        "limit": safe_limit,
        "offset": safe_offset,
        "returned_issue_count": len(returned_issues),
        "issues": returned_issues,
    }


def _lint_source_page_tables(page: WikiPage) -> list[dict]:
    issues: list[dict] = []
    body = page.markdown_content or ""
    if "## Tables" not in body:
        return issues
    table_blocks = _extract_table_blocks_from_page(body)
    if not table_blocks:
        issues.append(_issue("tables_section_empty", "medium", f"{page.slug} has a Tables section but no parseable markdown tables", {"page_slug": page.slug}))
        return issues
    for block in table_blocks:
        table = structure_table_markdown(block)
        if table.quality_flags:
            issues.append(
                _issue(
                    "table_quality_flags",
                    "medium",
                    f"{page.slug} {table.label or 'table'} has quality flags: {', '.join(table.quality_flags)}",
                    {"page_slug": page.slug, "table_label": table.label, "quality_flags": table.quality_flags},
                )
            )
        if table.label and not table.rows:
            issues.append(_issue("table_without_rows", "high", f"{page.slug} {table.label} has no structured rows", {"page_slug": page.slug, "table_label": table.label}))
    return issues


def _lint_query_answer_citations(page: WikiPage) -> list[dict]:
    text = page.markdown_content or ""
    issues: list[dict] = []
    unresolved = sorted(set(re.findall(r"\[[^\]\d][^\]]*\](?!\()", text)))
    if unresolved:
        issues.append(_issue("unresolved_citation_labels", "medium", f"{page.slug} contains nonnumeric citation labels", {"page_slug": page.slug, "labels": unresolved[:8]}))
    return issues


def _lint_page_links(page: WikiPage, page_by_slug: dict[str, WikiPage]) -> list[dict]:
    issues: list[dict] = []
    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", page.markdown_content or ""):
        if "://" in link or link.startswith("#"):
            continue
        target = link[:-3] if link.endswith(".md") else link
        target = str(Path(page.slug).parent.joinpath(target)).replace("\\", "/") if not target.startswith(("sources/", "entities/", "queries/")) else target
        target = re.sub(r"\.md$", "", target)
        if target not in page_by_slug and target not in {"index", "log"}:
            issues.append(_issue("broken_wiki_link", "medium", f"{page.slug} links to missing page {link}", {"page_slug": page.slug, "target": link}))
    return issues


def _extract_table_blocks_from_page(markdown: str) -> list[str]:
    blocks: list[str] = []
    in_tables = False
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## Tables"):
            in_tables = True
            continue
        if in_tables and line.startswith("## "):
            break
        if not in_tables:
            continue
        if line.startswith("### Page") and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [block.strip() for block in blocks if "|" in block]


def _issue(kind: str, severity: str, detail: str, payload: dict) -> dict:
    return {
        "kind": kind,
        "severity": severity,
        "detail": detail,
        "payload": payload,
    }
