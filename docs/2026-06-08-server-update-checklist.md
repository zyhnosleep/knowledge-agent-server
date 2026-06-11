# 服务器更新后命令版 Checklist

这份清单面向当前无 Docker 服务器版，假设项目目录为：

```sh
~/llm_wiki_server
```

说明：

- 下面命令按 POSIX `sh` 兼容方式编写
- 第 0 步“上传覆盖文件”需要你先在本地完成
- 从第 1 步开始，可以直接在服务器终端复制执行

## 0. 先在本地上传覆盖这些文件

必须覆盖：

```text
src/app/services/ai.py
src/app/services/pipeline.py
src/app/services/wiki.py
src/app/services/search.py
```

可选覆盖：

```text
README.md
docs/deploy-no-docker.md
docs/obsidian-workflow.md
docs/2026-06-08-server-update-checklist.md
```

## 1. 进入项目并激活虚拟环境

```sh
cd ~/llm_wiki_server
. .venv/bin/activate
pwd
python --version
```

## 2. 确认新文件已经覆盖到位

```sh
ls -l src/app/services/ai.py
ls -l src/app/services/pipeline.py
ls -l src/app/services/wiki.py
ls -l src/app/services/search.py
```

如果你也上传了文档：

```sh
ls -l docs/obsidian-workflow.md
ls -l docs/2026-06-08-server-update-checklist.md
```

## 3. 先做基础代码检查

```sh
python -m compileall src
```

如果服务器装了测试依赖，再执行：

```sh
pytest -q
```

## 4. 重启 API 和 worker

```sh
pkill -f "uvicorn app.main:app" || true
pkill -f "python -m app.workers.runner" || true
./scripts/start_api.sh
./scripts/start_worker.sh
./scripts/status.sh
```

## 5. 查看最近日志

```sh
tail -n 80 logs/api.log
tail -n 80 logs/worker.log
```

如果想持续观察：

```sh
tail -f logs/api.log logs/worker.log
```

## 6. 确认服务已恢复

```sh
curl http://127.0.0.1:8000/api/health
```

也可以检查文档页：

```sh
curl -I http://127.0.0.1:8000/docs
curl -I http://127.0.0.1:8000/
```

## 7. 准备一个新的测试文档，避免被去重

如果项目里已经有示例文件，可以直接在服务器上复制出一个“新文件名 + 新时间戳”的版本：

```sh
mkdir -p tmp
cp docs/medical_test_case.md "tmp/medical_test_case_$(date +%Y%m%d_%H%M%S).md"
ls -l tmp
```

如果你没有这个示例文件，也可以手工生成一个最小测试文档：

```sh
cat > "tmp/followup_test_$(date +%Y%m%d_%H%M%S).md" <<'EOF'
# 随访测试文档

患者诊断为高血压。

医生建议患者在3个月后进行复查，并继续当前治疗方案。

当前用药为二甲双胍 500mg，每日两次。
EOF

ls -l tmp
```

## 8. 上传测试文档到本机 API

先找出刚才最新生成的测试文件：

```sh
LATEST_FILE="$(ls -t tmp/*.md | head -n 1)"
echo "$LATEST_FILE"
```

上传：

```sh
curl -X POST "http://127.0.0.1:8000/api/ingest/upload?project_slug=internal-research&project_name=Internal%20Research" \
  -F "file=@${LATEST_FILE}"
```

## 9. 检查 documents 和 runs

```sh
curl http://127.0.0.1:8000/api/documents
curl http://127.0.0.1:8000/api/runs
```

如果 worker 正在处理，可以多看几次：

```sh
watch -n 2 'curl -s http://127.0.0.1:8000/api/runs'
```

如果服务器没有 `watch`，就手工重复：

```sh
curl -s http://127.0.0.1:8000/api/runs
sleep 2
curl -s http://127.0.0.1:8000/api/runs
```

## 10. 检查 wiki 输出是否生成

```sh
find data/raw/internal-research -maxdepth 2 -type f
find data/wiki/internal-research -maxdepth 3 -type f
```

重点看：

```sh
cat data/wiki/internal-research/index.md
cat data/wiki/internal-research/log.md
```

如果有 source/entity 页面，再逐个看：

```sh
find data/wiki/internal-research/sources -maxdepth 2 -type f -name "*.md" -print
find data/wiki/internal-research/entities -maxdepth 2 -type f -name "*.md" -print
```

查看最新 source 页：

```sh
LATEST_SOURCE="$(find data/wiki/internal-research/sources -type f -name "*.md" | tail -n 1)"
echo "$LATEST_SOURCE"
cat "$LATEST_SOURCE"
```

## 11. 发起一次 query，验证 wiki-first

生成请求文件：

```sh
cat > tmp/query.json <<'EOF'
{
  "project_slug": "internal-research",
  "question": "医生建议患者多久后进行复查？",
  "save_answer": true
}
EOF
```

发送 query：

```sh
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  --data @tmp/query.json
```

## 12. 检查 query 是否写回 wiki

```sh
find data/wiki/internal-research/queries -maxdepth 1 -type f -name "*.md"
cat data/wiki/internal-research/index.md
cat data/wiki/internal-research/log.md
```

查看最新 query 页：

```sh
LATEST_QUERY="$(find data/wiki/internal-research/queries -type f -name "*.md" | tail -n 1)"
echo "$LATEST_QUERY"
cat "$LATEST_QUERY"
```

## 13. 检查 review 队列

```sh
curl http://127.0.0.1:8000/api/reviews
```

## 14. 如果你要同步到本地 Obsidian

服务器端确认这个目录已经是最新结果：

```sh
find data/wiki/internal-research -maxdepth 3 -type f
```

你后续本地同步的重点目录是：

```text
data/wiki/internal-research/
```

## 15. 出现异常时优先排查

```sh
./scripts/status.sh
tail -n 120 logs/api.log
tail -n 120 logs/worker.log
curl http://127.0.0.1:8000/api/runs
curl http://127.0.0.1:8000/api/documents
curl http://127.0.0.1:8000/api/reviews
```

## 16. 本次更新的通过标准

执行完上面步骤后，你应该能看到这些结果：

- API 和 worker 都正常运行
- 新文档成功 ingest，而不是被去重跳过
- `data/wiki/internal-research/sources/*.md` 出现新版 source 页
- source 页里能看到更清晰的 key facts / verified triples / evidence notes
- `entities/*.md` 生成更合理
- query 返回优先命中 wiki citation
- `wiki/queries/*.md` 成功写回
