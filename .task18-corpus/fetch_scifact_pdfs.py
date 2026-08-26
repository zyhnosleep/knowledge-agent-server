#!/usr/bin/env python3
"""下载 SciFact 评测语料 PDF（PMC OA）到 raw/scifact-bench/。

来源：BEIR scifact qrels（test+train 引用的全部唯一 PMID）+ 随机 100 篇非引用干扰项。
流程：PMID → efetch 拿 PMCID → oa.fcgi 拿 PDF 链接 → 下载。
命名：<pmid>-<uuid8>.pdf（评测时按文件名映射回 PMID）。
输出：scifact-bench-manifest.json（pmid/pmcid/filename/status）。
"""
from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path
from urllib import parse, request

UA = "knowledge-agent-scifact-bench/1.0 (mailto:ka-corpus@example.com)"
BASE = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/beir-scifact/scifact"
RAW_DIR = Path.home() / "knowledge-agent-dev/runtime/data/raw/scifact-bench"
MANIFEST = RAW_DIR / "scifact-bench-manifest.json"
DISTRACTOR_COUNT = 100
SLEEP = 1.0  # NCBI 限速 3 req/s


def http_get(url: str, timeout: int = 60) -> bytes:
    req = request.Request(url, headers={"User-Agent": UA})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def collect_pmids() -> tuple[set[str], set[str]]:
    """返回 (qrels 引用的 PMID 集, BEIR corpus 全部 PMID 集)。"""
    cited: set[str] = set()
    for name in ("test.tsv", "train.tsv"):
        for line in (BASE / "qrels" / name).read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                cited.add(parts[1])
    all_ids: set[str] = set()
    for line in (BASE / "corpus.jsonl").read_text().splitlines():
        try:
            all_ids.add(json.loads(line)["_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return cited, all_ids


def pmid_to_pmcid(pmid: str) -> str | None:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc"
        f"&id={pmid}&rettype=pmcids&retmode=json"
    )
    try:
        data = json.loads(http_get(url).decode("utf-8", "replace"))
        pmcid = (data.get("pmcids") or {}).get("pmcid")
        return pmcid or None
    except Exception:  # noqa: BLE001
        return None


def pmcid_to_oa_pdf(pmcid: str) -> str | None:
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


def download(url: str, dest: Path) -> bool:
    for _ in range(3):
        try:
            data = http_get(url, timeout=120)
            if len(data) < 2000:
                return False
            dest.write_bytes(data)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(4)
    return False


def main() -> int:
    cited, all_ids = collect_pmids()
    distractors = random.Random(42).sample(sorted(all_ids - cited), DISTRACTOR_COUNT)
    targets = sorted(cited | set(distractors))
    print(f"cited={len(cited)} distractor={len(distractors)} total={len(targets)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    if MANIFEST.is_file():
        entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["entries"]

    for i, pmid in enumerate(targets, 1):
        if pmid in entries and entries[pmid].get("status") in ("downloaded", "no-oa"):
            continue
        print(f"[{i}/{len(targets)}] PMID {pmid}", flush=True)
        pmcid = pmid_to_pmcid(pmid)
        time.sleep(SLEEP)
        if not pmcid:
            entries[pmid] = {"pmid": pmid, "pmcid": None, "filename": None, "status": "no-pmcid"}
            continue
        pdf_url = pmcid_to_oa_pdf(pmcid)
        time.sleep(SLEEP)
        if not pdf_url:
            entries[pmid] = {"pmid": pmid, "pmcid": pmcid, "filename": None, "status": "no-oa"}
            continue
        filename = f"{pmid}-{uuid.uuid4().hex[:8]}.pdf"
        if download(pdf_url, RAW_DIR / filename):
            entries[pmid] = {"pmid": pmid, "pmcid": pmcid, "filename": filename, "status": "downloaded"}
            print(f"  ok: {filename}")
        else:
            entries[pmid] = {"pmid": pmid, "pmcid": pmcid, "filename": None, "status": "failed"}

    MANIFEST.write_text(
        json.dumps(
            {"total_targets": len(targets), "cited": len(cited), "distractors": len(distractors),
             "entries": entries},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    statuses = {}
    for e in entries.values():
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    print("summary:", statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
