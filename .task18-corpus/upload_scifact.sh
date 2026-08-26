#!/usr/bin/env bash
# SciFact 语料幂等批量导入：test-cited(284) + distractor(300) ≈ 584 篇 → sci-fact-bench 项目
# 用法：nohup bash upload_scifact.sh > /tmp/scifact_upload.log 2>&1 &
set -uo pipefail

HOST="http://127.0.0.1:8002"
RAW=~/knowledge-agent-dev/runtime/data/raw/scifact-bench
SLUG=sci-fact-bench
LIST=/tmp/scifact_upload_list.txt
DONE=/tmp/scifact_done.txt
LOG=/tmp/scifact_uploaded.log
FAIL=/tmp/scifact_upload_fail.log

# 1) 生成待导入文件列表
python3 - <<'PY'
import json
from pathlib import Path
BASE = Path.home() / "knowledge-agent-dev/runtime/sci-fact-bench/beir-scifact/scifact"
test_cited, all_cited = set(), set()
for fname, s in (("test.tsv", test_cited), ("train.tsv", all_cited)):
    for line in (BASE / "qrels" / fname).read_text().splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 2:
            s.add(p[1])
raw = Path.home() / "knowledge-agent-dev/runtime/data/raw/scifact-bench"
files = [p.name for p in sorted(raw.glob("*.md"))
         if p.name.split("-")[0] in test_cited or p.name.split("-")[0] not in all_cited]
Path("/tmp/scifact_upload_list.txt").write_text("\n".join(files) + "\n")
print(f"[plan] test_cited={len(test_cited)} all_cited={len(all_cited)} upload_files={len(files)}")
PY

# 2) 已有 file_name 集合（翻页拉全）
: > "$DONE"
OFF=0
while :; do
  PAGE=$(curl -s -m 20 "$HOST/api/documents?project_slug=$SLUG&limit=500&offset=$OFF")
  N=$(echo "$PAGE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d))")
  echo "$PAGE" | python3 -c "import json,sys; d=json.load(sys.stdin); [print(x['file_name']) for x in d]" >> "$DONE"
  if [ "$N" -lt 500 ]; then break; fi
  OFF=$((OFF + N))
done
echo "[plan] already_in_system=$(wc -l < "$DONE")"

# 3) 循环上传（跳过已存在 / 已成功）
CNT=0; OK=0; FAILCNT=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if grep -qxF "$f" "$DONE" 2>/dev/null || grep -qxF "$f" "$LOG" 2>/dev/null; then
    continue
  fi
  BODY=$(curl -s -m 180 -F "file=@$RAW/$f" "$HOST/api/ingest/upload?project_slug=$SLUG")
  CODE=$(echo "$BODY" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('status','ERR'))" 2>/dev/null || echo ERR)
  case "$CODE" in
    queued|pending|completed|processing|ready|parsing|embedding|indexing|contextualizing|canonicalizing|chunking)
      echo "$f" >> "$LOG"; OK=$((OK+1)) ;;
    *)
      echo "$f\t$CODE\t$BODY" >> "$FAIL"; FAILCNT=$((FAILCNT+1)) ;;
  esac
  CNT=$((CNT+1))
  if [ $((CNT % 20)) -eq 0 ]; then
    echo "[progress] attempted=$CNT ok=$OK fail=$FAILCNT $(date '+%H:%M:%S')"
  fi
done < "$LIST"
echo "[done] attempted=$CNT ok=$OK fail=$FAILCNT"
