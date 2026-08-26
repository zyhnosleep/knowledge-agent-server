#!/usr/bin/env python3
"""BEIR SciFact corpus → .md 文件（qrels 引用全集 + 随机干扰项）。

输出目录：~/knowledge-agent-dev/runtime/data/raw/scifact-bench/
命名：<pmid>-<uuid8>.md（评测按文件名前缀映射回 PMID）。
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

BASE = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/beir-scifact/scifact"
RAW_DIR = Path.home() / "knowledge-agent-dev/runtime/data/raw/scifact-bench"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distractors", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="0 = 全部（引用 + 干扰）")
    args = parser.parse_args()

    cited: set[str] = set()
    for name in ("test.tsv", "train.tsv"):
        for line in (BASE / "qrels" / name).read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                cited.add(parts[1])

    docs: dict[str, dict] = {}
    for line in (BASE / "corpus.jsonl").read_text().splitlines():
        d = json.loads(line)
        docs[d["_id"]] = d
    distractors = random.Random(42).sample(sorted(set(docs) - cited), args.distractors)
    targets = sorted(cited | set(distractors))
    if args.limit > 0:
        targets = targets[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    made, skipped = 0, 0
    for pmid in targets:
        doc = docs[pmid]
        dest = RAW_DIR / f"{pmid}-{uuid.uuid4().hex[:8]}.md"
        if dest.exists():
            skipped += 1
            continue
        text = (doc.get("text") or "").strip()
        title = (doc.get("title") or "").strip()
        content = f"# {title}\n\n{text}\n" if title else f"{text}\n"
        dest.write_text(content, encoding="utf-8")
        made += 1
    print(f"cited={len(cited)} distractors={len(distractors)} targets={len(targets)} made={made} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
