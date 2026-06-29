# MinerU/RAG 下一阶段计划

## 当前进度

- PDF 入口已经接入 MinerU 优先路径：`MINERU_ENABLED=true` 时先调用 `mineru -p <pdf> -o <cache_dir> -b pipeline`，成功后读取 `content_list_v2.json`、`content_list.json` 和 Markdown，再映射到 `ParsedDocument`、`ParsedChunk`、表格、公式、图注和页面级 metadata。
- MinerU 失败、超时、未安装或输出不可用时，会自动回退到现有 PDF 文档智能链路；最后仍可退到文本层解析。
- 项目 extra 已经按要求使用 `mineru[all]>=3.4,<3.5`。公网验证时，PyPI 可见最新版本为 `mineru 3.4.0`，本机 `D:\Miniconda3` 已安装并验证为 `mineru, version 3.4.0`。
- 真实 CLI smoke 已通过：`Knowledge graph.pdf` 第 1 页使用 `pipeline` 后端、`MINERU_MODEL_SOURCE=modelscope` 解析成功，产出 `content_list_v2.json`、Markdown、layout/span/origin PDF 等文件。
- 应用侧 `_parse_pdf_with_mineru()` 已用真实 CLI 路径验证通过，输出 `parser_mode=pdf_mineru`、`backend=pipeline`、`page_outputs=1`、`chunks=11`。
- loop runner 已扩展为可选真实 MinerU parser smoke：默认不跑真实 MinerU，显式传 `--run-mineru-smoke` 时才调用 `scripts/mineru_parser_smoke.py`，并把 summary 写入 manifest。
- loop runner 已扩展为可选真实服务 ingest gate：默认不上传文件，显式传 `--run-service-ingest-smoke` 时才调用 `/api/ingest/upload`、轮询 `/api/runs`，再抓 `/api/documents/{id}/quality` 和 `/api/wiki/lint` 写入 manifest。
- 本地隔离端到端 smoke 已通过：RAG-only ingest 使用真实 MinerU 解析 `Knowledge graph.pdf` 第 1 页，文档 `ready`、run `completed`、`parser_mode=pdf_mineru`、11 个 chunks；在无 Ollama 条件下 query fallback 返回 2 条 citations。

## OCR 和 Qwen 判断

- 当前主 OCR/版面解析方向由 MinerU pipeline 承担，不把本地 Qwen 当基础 OCR 引擎。
- 代码里仍保留 `pytesseract + PIL` 的 OCR fallback，只在 `OCR_FALLBACK_ENABLED=true` 且文本层质量低时作为 fail-safe 使用；默认关闭。
- 本地 Qwen 更适合做 MinerU 之后的语义处理：页面摘要、表格解释、事实抽取、答案生成、证据验证和低质量页的多模态补强。
- 如果后续要尝试 Qwen-VL 或本地视觉模型参与 OCR，应先用 MinerU 输出做基线对照，按页比较文字、表格、公式、图注和 citation 命中率。

## 何时借鉴开源项目

- 阶段 1：先完成 MinerU 输出接入与真实 PDF 验收，再借鉴 RAGFlow 的复杂文档解析、chunk 组织、表格/公式保真思路。
- 阶段 2：当 citation 和页面证据链路打通后，借鉴 kotaemon 的 PDF 引用预览、证据定位和 source-first QA 体验。
- 阶段 3：当检索 API 稳定后，借鉴 R2R 的 API-first 检索服务、混合检索、文档管理和评估接口。
- 阶段 4：当需要团队/企业知识入口时，再借鉴 Onyx 的连接器、后台同步、权限和企业搜索入口。
- Dify、FastGPT、MaxKB 可在工作流编排和中文知识库管理上局部参考；Open WebUI、AnythingLLM 更适合作为外部 UI 或轻量个人知识库参考，不建议整体替换当前主线。

## 下一步执行顺序

1. 本地 loop 门槛：持续运行 quick/query profile；需要真实 MinerU 时显式加 `--run-mineru-smoke`，确认 parser smoke summary 为 `status=passed`。
2. 服务器环境验证：在目标服务器执行 `pip install -e ".[mineru]"`，确认 `mineru --version`、`MINERU_MODEL_SOURCE=modelscope`、GPU/CPU 路径可用。
3. 真实服务 ingest：先用独立 project，例如 `mineru-rag-smoke`，通过 `/api/ingest/upload` 摄入代表性 PDF，检查 `/api/documents/{id}/quality` 和 `/api/wiki/lint`，确认没有 fallback 到非 MinerU。
4. 检索与问答验收：先跑剩余 6 个失败 case 的 targeted query eval，再跑 full30；每个失败归因到 parse/chunk/retrieval/citation/generation/benchmark。
5. 开源借鉴落地：真实验收问题暴露后，再按模块吸收 RAGFlow/kotaemon/R2R/Onyx 的具体做法，避免提前引入整套平台。
6. agent+前端门槛：真实 ingest 可复跑、query eval 可复跑、剩余失败有明确归因、full30 不低于当前 24/30，才进入 agent 和前端页面搭建。
7. 运行与资源策略：单卡服务器采用 MinerU 和 Ollama 顺序执行；重 PDF 解析前把 `OLLAMA_KEEP_ALIVE=0` 或暂停 Ollama，解析后再进入 Qwen 抽取和问答阶段。

## 本轮验证结果

- 安装：`D:\Miniconda3\python.exe -m pip install -e ".[mineru]"` 成功；`pip check` 通过。
- 版本：`D:\Miniconda3\Scripts\mineru.exe --version` -> `mineru, version 3.4.0`。
- CLI 参数：MinerU 3.4 仍支持 `-p`、`-o`、`-b pipeline`、`--start`、`--end`；默认后端偏向 `hybrid-engine`，项目显式传 `pipeline` 保持稳定。
- 真实 smoke：`Knowledge graph.pdf` 第 1 页解析成功，输出目录 `tmp/mineru-smoke-kg-page0/Knowledge graph/auto/`。
- Parser smoke：`_parse_pdf_with_mineru()` 真实调用成功，未再出现 Windows GBK 解码异常。
- 测试：`D:\Miniconda3\python.exe -m pytest tests/test_parser_document_intelligence.py tests/test_loop_runner.py -q` -> `22 passed`。
- 回归入口：`D:\Miniconda3\python.exe scripts/run_mineru_rag_loop.py --profile quick --python D:\Miniconda3\python.exe` -> `overall_status=passed`，manifest 为 `tmp/loop_runs/mineru_rag_20260626_160840_398629/manifest.json`。
- 新增 loop smoke：`D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --profile quick --python D:\Miniconda3\python.exe --run-mineru-smoke --mineru-smoke-pdf "Knowledge graph.pdf" --mineru-smoke-start 0 --mineru-smoke-end 0 --mineru-bin D:\Miniconda3\Scripts\mineru.exe --mineru-model-source modelscope --timeout 900` -> `overall_status=passed`，manifest 为 `tmp/loop_runs/mineru_rag_20260626_162941_579993/manifest.json`。
- 本地 RAG-only ingest smoke：隔离目录 `tmp/e2e_mineru_ingest_20260627_001505/`，输出 `ingest_summary.json` 和 `query_smoke_summary.json`；Ollama 当前不可用，所以该 smoke 只证明 ingest/retrieval/citation 最小链路，不证明最终回答质量。
- 新增真实服务 ingest gate：公网远端执行 `scripts/run_mineru_rag_loop.py --skip-pytest --run-service-ingest-smoke ...` 通过，manifest 为 `tmp/service_ingest_real_20260627_0052/manifest.json`；证据为 `duplicate_skipped=false`、`parser_mode=pdf_mineru`、16 页、`page_output_count=16`、10 张表、6 张图、`content_list_v2.json` 存在。`/api/wiki/lint` 可达但返回 1 个 `missing_index`，原因是该远端 smoke 当前为 RAG-only ingest、`SAC_KG_ENABLED=false`，未生成 wiki index。
- 新增 query attribution gate：`--run-query-eval` 现在会同时生成 `query_attribution.json`，记录 expected term 命中/缺失、citation source、source hint 是否匹配和 likely failure stage。公网远端 target6 为 `6/6 passed`；显式 full30 query gate 为 `25/30 passed`，满足 `min_query_passed=24`、`max_query_failed=6`、`require_failure_attribution=true`，剩余 5 条均为 answer-stage 缺词，citation/source 已命中。

## 剩余失败归因

当前 full30 已到 `25/30 passed`，剩余 5 个失败的下一轮定位如下：

- `charmm36m_overview`：answer 缺 `NMR`，citation/source 已命中。
- `ff99sb_ildn_mechanism`：answer 缺 `0.5 kcal/mol`，citation/source 已命中。
- `opls4_overview`：answer 缺 `van der Waals`、`GLH`，citation/source 已命中。
- `opls4_mechanism`：answer 缺 `GLH`，citation/source 已命中。
- `opls5_mechanism`：answer 缺 `FXA`、`-2.4`，citation/source 已命中。

## 可执行入口

```powershell
D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --profile quick --python D:\Miniconda3\python.exe
```

带真实 MinerU parser smoke：

```powershell
D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --profile quick --python D:\Miniconda3\python.exe --run-mineru-smoke --mineru-smoke-pdf "Knowledge graph.pdf" --mineru-smoke-start 0 --mineru-smoke-end 0 --mineru-bin D:\Miniconda3\Scripts\mineru.exe --mineru-model-source modelscope --timeout 900
```

需要真实服务 query eval 时显式开启：

```powershell
D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --profile query --python D:\Miniconda3\python.exe --run-query-eval --base-url http://127.0.0.1:8000 --timeout 120 --limit 5
```

该命令会同时生成 `query_attribution.json`，用于查看每个 case 的 expected term 命中、citation source、source hint match 和 likely failure stage。

作为进入 agent/frontend 前的 full30 gate，可显式加阈值：

```powershell
D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --profile query --python D:\Miniconda3\python.exe --run-query-eval --base-url http://127.0.0.1:8000 --timeout 120 --min-query-passed 24 --max-query-failed 6 --require-failure-attribution
```

需要真实服务 ingest gate 时显式开启：

```powershell
D:\Miniconda3\python.exe scripts\run_mineru_rag_loop.py --skip-pytest --run-service-ingest-smoke --service-ingest-pdf "Knowledge graph.pdf" --service-ingest-project-slug mineru-rag-smoke --service-ingest-project-name "MinerU RAG Smoke" --service-ingest-timeout 1800 --base-url http://127.0.0.1:8000
```

默认输出写入 `tmp/loop_runs/`。真实 MinerU CLI 只在显式传入 `--run-mineru-smoke` 时调用；`/api/query` 只在显式传入 `--run-query-eval` 时访问；真实服务上传只在显式传入 `--run-service-ingest-smoke` 时执行。service ingest gate 默认会拒绝服务端的 duplicate-skip 结果，避免旧文档让 smoke 假通过；只有显式传 `--allow-service-ingest-duplicate-skip` 才允许复用已 ready 的同 SHA 文档。
