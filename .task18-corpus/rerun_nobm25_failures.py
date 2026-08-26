#!/usr/bin/env python3
"""重跑 nobm25 验证的 2 条措辞漂移失败题，确认重跑通过（基线同法）。"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

CASES = Path("/home/<user>/knowledge-agent-dev/runtime/task15/internal-research-overlap30-full-answer-cases.json")
API = "http://127.0.0.1:8002/api/agent/query"

payload = json.loads(CASES.read_text(encoding="utf-8"))
cases = payload.get("cases", payload)
for cid in ("charmm36_overview", "ff99sb_ildn_mechanism"):
    case = next(c for c in cases if c["id"] == cid)
    body = {
        "project_slug": case["project_slug"],
        "query": case["question"],
        "session_id": "rerun-nobm25-%s-%d" % (cid, int(time.time())),
        "constraints": {"timeout_seconds": 240},
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    ans = str(result.get("final_answer") or "")
    terms = case.get("answer_required_terms", [])
    missing = [t for t in terms if t.casefold() not in ans.casefold()]
    print(
        "%s: status=%s citations=%d missing=%s ms=%d"
        % (cid, result.get("status"), len(result.get("citations") or []), missing, int((time.monotonic() - started) * 1000)),
        flush=True,
    )
