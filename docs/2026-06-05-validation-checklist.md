# 2026-06-05 联调验证清单

这份清单用于完成今天的服务器联调任务，目标是验证一条完整链路：

1. 上传 1 个 `txt` 或 `md` 文档
2. 确认 `worker` 消费队列
3. 检查 `data/raw/` 和 `data/wiki/`
4. 检查项目目录下的 `index.md` 和 `log.md`
5. 发起 1 次问答，确认返回 `citations`
6. 确认 `save_answer=true` 会写入 `wiki/queries/`
7. 重复上传同一文档，确认去重生效

建议先使用内容较短、事实清晰的 `md` 或 `txt` 文件，不要一开始就使用 `pdf` 或 `docx`。

## 1. 开始前检查

先确认服务状态：

```sh
cd ~/llm_wiki_server
./scripts/status.sh
```

建议同时打开日志观察窗口：

```sh
cd ~/llm_wiki_server
tail -f logs/api.log logs/worker.log
```

开始前的通过标准：

- `api` 进程在运行
- `worker` 进程在运行
- `redis` 在运行
- `ollama` 在运行

如果这里已经异常，先不要上传文件，优先修复基础服务。

## 2. 上传 1 个测试文档

建议使用 API 方式上传，便于观察返回结果。

```sh
curl -X POST "http://127.0.0.1:8000/api/ingest/upload?project_slug=internal-research&project_name=Internal%20Research" \
  -F "file=@/path/to/your-test.md"
```

说明：

- 把 `/path/to/your-test.md` 替换成服务器上的真实文件路径
- 第一次上传时，返回里的 `status` 很可能是 `queued`

这一阶段的通过标准：

- 接口返回 `document_id`
- 接口返回 `run_id`
- 没有 4xx 或 5xx 错误

## 3. 确认 worker 消费队列

上传完成后，检查文档状态和运行状态：

```sh
curl http://127.0.0.1:8000/api/documents
curl http://127.0.0.1:8000/api/runs
```

通过标准：

- `documents` 中能看到新上传的文件
- 该文档状态最终从 `pending` 或 `processing` 变为 `ready`
- `runs` 中最近一条任务状态最终变为 `completed`
- `logs/worker.log` 没有持续报错

如果长时间没有处理，优先检查：

- `logs/worker.log`
- `logs/api.log`
- Redis 是否正常
- Ollama 是否正常

## 4. 检查 raw 和 wiki 目录

检查原始文件和 Wiki 投影是否生成：

```sh
find data/raw/internal-research -maxdepth 2 -type f
find data/wiki/internal-research -maxdepth 3 -type f
```

预期至少看到：

- `data/raw/internal-research/` 下有带随机前缀的原始文件
- `data/wiki/internal-research/index.md`
- `data/wiki/internal-research/log.md`
- `data/wiki/internal-research/sources/...md`

如果抽取到了实体，还可能出现：

- `data/wiki/internal-research/entities/...md`

通过标准：

- 原始文件已写入 `data/raw/internal-research/`
- Wiki 页面已写入 `data/wiki/internal-research/`

## 5. 检查 index.md 和 log.md

直接查看这两个文件：

```sh
cat data/wiki/internal-research/index.md
cat data/wiki/internal-research/log.md
```

通过标准：

- `index.md` 中至少有 1 条页面链接
- `log.md` 中包含本次文档的记录
- `log.md` 中能看到 `SHA256`
- `log.md` 中能看到 `Summary length`
- `log.md` 中能看到 `Claims extracted`

如果 `index.md` 或 `log.md` 缺失，优先回看：

- `logs/worker.log`
- 文档状态是否真的已经变为 `ready`

## 6. 发起 1 次问答，确认 citations

当前查询逻辑已经改成更接近 `llm_wiki` 的 `wiki-first` 方式：

- 优先检索 `WikiPage`
- 以 wiki 页面正文作为主要回答上下文
- 必要时再补 source chunk 证据
- `save_answer=true` 会将回答写入 `wiki/queries/`

为避免终端内嵌中文 JSON 出现解析问题，建议先写请求文件再发送：

```sh
cat > /tmp/query.json <<'EOF'
{"project_slug":"internal-research","question":"医生建议患者多久后进行复查？","save_answer":true}
EOF
```

然后发送：

```sh
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/query.json
```

通过标准：

- 返回中有 `answer_markdown`
- 返回中有 `citations`
- `citations` 是非空数组
- 返回中有 `verification_status`
- `citations` 中至少有 1 条能对应到 wiki 页或 source chunk

当前配置下，`verification_status` 常见值通常是：

- `local-only`

如果返回答案但内容仍然明显偏离文档，优先检查：

- `data/wiki/internal-research/sources/...md` 是否已经正确写入复查时间
- 提问是否与文档中的表达接近
- 返回的 `citations` 是否命中了正确 wiki 页面

## 7. 检查 query 是否写回 wiki

因为当前请求里 `save_answer=true`，回答应当写入 `wiki/queries/`，并更新 `index.md`、`log.md`。

执行：

```sh
find data/wiki/internal-research/queries -maxdepth 1 -type f
cat data/wiki/internal-research/index.md
cat data/wiki/internal-research/log.md
```

通过标准：

- `data/wiki/internal-research/queries/` 下出现新的 `.md` 文件
- `index.md` 中新增了 query 页面链接
- `log.md` 中新增了 `query | ...` 记录

## 8. 重复上传同一文档，确认去重

再次上传完全相同的文件：

```sh
curl -X POST "http://127.0.0.1:8000/api/ingest/upload?project_slug=internal-research&project_name=Internal%20Research" \
  -F "file=@/path/to/your-test.md"
```

这次预期会比第一次更快返回。

再检查运行记录：

```sh
curl http://127.0.0.1:8000/api/runs
```

通过标准：

- 最新一次返回很快完成
- 最近一条运行记录里有重复跳过的迹象
- 不应生成一条新的独立文档内容

当前代码里，重复文档命中去重时，任务备注通常应为：

```text
Duplicate document skipped.
```

注意：

- 只有“完全相同的文件内容”才会命中去重
- 哪怕只改了一个空格，`SHA256` 都会变化，去重也不会命中

## 9. 今日建议

- 第一轮只用 1 个小型 `md` 或 `txt`
- 问答先测事实题，再测总结题
- 出现问题时，先看日志再重复操作，避免把问题放大
- 如果首轮链路跑通，再考虑换更真实的文档继续试

## 10. 今日完成判定

今天这轮联调可以视为完成，当且仅当以下 7 项都成立：

1. 成功上传 1 个 `txt` 或 `md`
2. `worker` 成功消费任务
3. `data/raw/` 和 `data/wiki/` 生成了对应内容
4. `index.md` 和 `log.md` 正常生成
5. 问答返回了非空 `citations`
6. `save_answer=true` 成功写入 `wiki/queries/`
7. 重复上传命中了去重逻辑
