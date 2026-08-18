#!/usr/bin/env python3
"""批量下载力场/分子模拟文献 PDF 到 internal-research raw 目录。

来源：
  1. PMC OA 子集（E-utilities esearch 按期刊矩阵 + 关键词，oa.fcgi 拿 PDF）——正式发表版，优先
  2. arXiv API（physics.chem-ph / q-bio.BM / cond-mat.soft / cs.LG × 力场关键词）——含已发表预印本

命名对齐现有格式：<uuid32>-<slug>.pdf；输出 corpus-manifest.json（标题/期刊/年份/来源 ID）。
幂等：已存在文件与 manifest 记录跳过，可断点续跑。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib import parse, request

UA = "knowledge-agent-corpus-builder/1.0 (mailto:ka-corpus@example.com)"
RAW_DIR = Path.home() / "knowledge-agent-dev/runtime/data/raw/internal-research"
MANIFEST = RAW_DIR / "corpus-manifest.json"

# ---------- arXiv ----------
ARXIV_QUERIES = [
    'cat:physics.chem-ph AND (ti:"force field" OR ti:"molecular dynamics" OR ti:"water model")',
    'cat:q-bio.BM AND (ti:"force field" OR ti:"molecular dynamics" OR ti:"water model" OR ti:"coarse-grained")',
    'cat:cond-mat.soft AND (ti:"force field" OR ti:"molecular dynamics" OR ti:"coarse-grained")',
    'cat:cs.LG AND (ti:"force field" OR ti:"neural network potential")',
    'cat:physics.chem-ph AND (ti:"CHARMM" OR ti:"AMBER" OR ti:"OPLS" OR ti:"Martini" OR ti:"GROMACS")',
    'cat:physics.chem-ph AND (ti:"reactive force field" OR ti:"ReaxFF" OR ti:"machine-learned potential")',
]

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}


def http_get(url: str, timeout: int = 40) -> bytes:
    req = request.Request(url, headers={"User-Agent": UA})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def arxiv_search(query: str, max_results: int) -> list[dict]:
    url = (
        "http://export.arxiv.org/api/query?search_query=" + parse.quote(query)
        + f"&start=0&max_results={max_results}&sortBy=relevance"
    )
    try:
        text = http_get(url).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  arXiv query failed: {exc}", file=sys.stderr)
        return []
    entries = []
    for entry in ET.fromstring(text).findall("a:entry", ARXIV_NS):
        title = re.sub(r"\s+", " ", entry.findtext("a:title", "") or "").strip()
        aid = (entry.findtext("a:id", "") or "").rsplit("/abs/", 1)[-1]
        jref = (entry.findtext("ar:journal_ref", "") or "").strip()
        doi = (entry.findtext("ar:doi", "") or "").strip()
        published = (entry.findtext("a:published", "") or "")[:10]
        entries.append({"id": aid, "title": title, "journal_ref": jref, "doi": doi, "published": published})
    return entries


# ---------- PMC ----------
PMC_JOURNALS = [
    "J Chem Theory Comput", "J Chem Phys", "J Phys Chem B", "J Phys Chem A",
    "Phys Chem Chem Phys", "Nat Commun", "Proc Natl Acad Sci U S A", "eLife",
    "Wiley Interdiscip Rev Comput Mol Sci", "J Comput Chem", "J Mol Model",
    "J Chem Inf Model", "PLoS Comput Biol", "Biophys J", "Proteins",
    "Chem Rev", "Chem Soc Rev", "J Am Chem Soc", "ACS Omega", "npj Comput Mater",
    "Digital Discovery", "J Mol Graph Model",
]

PMC_QUERY = (
    '("force field"[tiab] OR "molecular dynamics"[tiab] OR "CHARMM"[tiab] '
    'OR "AMBER"[tiab] OR "OPLS"[tiab] OR "water model"[tiab] OR "coarse-grained"[tiab] '
    'OR "neural network potential"[tiab]) AND ('
    + " OR ".join(f'"{j}"[ta]' for j in PMC_JOURNALS) + ")"
)


def pmc_search(retmax: int) -> list[str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json"
        f"&retmax={retmax}&term={parse.quote(PMC_QUERY)}"
    )
    data = json.loads(http_get(url).decode("utf-8", "replace"))
    return data.get("esearchresult", {}).get("idlist", [])


def pmc_oa_pdf(pmcid: str) -> str | None:
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}&format=json"
    try:
        data = json.loads(http_get(url).decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None
    for rec in data.get("records", []):
        href = ((rec.get("link") or {}).get("href") or "").strip()
        if href:
            if href.startswith("ftp://"):
                href = "https://" + href[len("ftp://"):]
            return href
    return None


def pmc_metadata(pmcid: str) -> dict:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&retmode=json"
        f"&id={pmcid}"
    )
    try:
        data = json.loads(http_get(url).decode("utf-8", "replace"))
        rec = data.get("result", {}).get(pmcid, {})
        return {
            "title": rec.get("title", ""),
            "journal": (rec.get("fulljournalname") or rec.get("source") or ""),
            "published": str(rec.get("pubdate", ""))[:10],
            "doi": (rec.get("elocationid", "") or "").replace("doi:", ""),
        }
    except Exception:  # noqa: BLE001
        return {"title": "", "journal": "", "published": "", "doi": ""}


# ---------- 通用 ----------
def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def slug_of(title: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", title.lower())
    slug = "-".join(words[:5])
    return slug[:40] or "untitled"


def download(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            data = http_get(url, timeout=90)
            if len(data) < 2000:
                print(f"  too small ({len(data)}B): {url}", file=sys.stderr)
                return False
            dest.write_bytes(data)
            return True
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"  FAIL {url}: {exc}", file=sys.stderr)
                return False
            time.sleep(5)


def load_seen() -> tuple[set[str], list[dict]]:
    seen: set[str] = set()
    entries: list[dict] = []
    if MANIFEST.is_file():
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for e in data.get("entries", []):
            seen.add(norm_title(e.get("title", "")))
            entries.append(e)
    return seen, entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-limit", type=int, default=40)
    parser.add_argument("--pmc-limit", type=int, default=250)
    parser.add_argument("--max-downloads", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=3.0)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen_titles, entries = load_seen()
    downloads = 0
    skipped_dup = 0

    def maybe_download(src: str, pdf_url: str, meta: dict) -> None:
        nonlocal downloads, skipped_dup
        title = meta.get("title", "")
        key = norm_title(title)
        if not key or key in seen_titles:
            skipped_dup += 1
            return
        if downloads >= args.max_downloads:
            return
        filename = f"{uuid.uuid4().hex}-{slug_of(title)}.pdf"
        dest = RAW_DIR / filename
        if args.dry_run:
            print(f"  [dry] {src} {title[:70]}")
            seen_titles.add(key)
            downloads += 1
            entries.append({"filename": filename, "source": src, "title": title, **{k: v for k, v in meta.items() if k != "title"}})
            return
        print(f"  [{downloads + 1}] {src} {title[:70]}")
        if download(pdf_url, dest):
            seen_titles.add(key)
            downloads += 1
            entries.append({"filename": filename, "source": src, "title": title, **{k: v for k, v in meta.items() if k != "title"}})
        else:
            dest.unlink(missing_ok=True)

    # 1) PMC OA（正式版优先）
    print("== PMC OA ==")
    pmcids = pmc_search(args.pmc_limit)
    print(f"esearch found {len(pmcids)} PMCIDs")
    for pmcid in pmcids:
        if downloads >= args.max_downloads:
            break
        pdf_url = pmc_oa_pdf(pmcid)
        time.sleep(args.sleep)
        if not pdf_url:
            continue
        meta = pmc_metadata(pmcid)
        time.sleep(args.sleep)
        meta["pmcid"] = pmcid
        maybe_download("pmc", pdf_url, meta)

    # 2) arXiv
    print("== arXiv ==")
    for query in ARXIV_QUERIES:
        if downloads >= args.max_downloads:
            break
        for entry in arxiv_search(query, args.arxiv_limit):
            if downloads >= args.max_downloads:
                break
            time.sleep(args.sleep)
            meta = {
                "title": entry["title"],
                "journal": entry["journal_ref"],
                "published": entry["published"],
                "doi": entry["doi"],
                "arxiv_id": entry["id"],
            }
            maybe_download("arxiv", f"https://arxiv.org/pdf/{entry['id']}", meta)

    MANIFEST.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "total": len(entries),
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsummary: downloaded={downloads} skipped_dups={skipped_dup} total_in_manifest={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
