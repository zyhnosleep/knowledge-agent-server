# PDF 问答质量提升计划（Round 2 修复 + MinerU 集成）

## Summary

当前 PDF 主链已经能通过 `PyMuPDF 渲染 + Ollama Vision + 文本层融合` 生成可用 wiki，但表格、公式、多栏阅读顺序和 citation 精度仍有上限。本轮目标分两部分推进：

- Round 2：继续修复 wiki 内容完整性、检索精度、citation excerpt、page_label 和数值反幻觉。
- MinerU：把 MinerU 作为可选 PDF 结构化解析引擎，优先产出页面级 Markdown、表格、公式和图注，再交给现有 SAC-KG/wiki/query 流水线消费。

服务器硬件为 RTX 3090 24GB + 90GB RAM，满足本地 MinerU pipeline 运行条件。部署策略采用 **GPU sequential mode**：MinerU 和 Ollama 共享同一块 GPU，但通过 CLI 子进程顺序执行，避免同时抢显存。

## Key Changes

- `parser.py` 增加 MinerU 优先分支：`MINERU_ENABLED=true` 时先调用 `mineru` CLI，成功则使用 MinerU 输出；失败或未安装时自动回退到现有 Ollama Vision，再回退到 pypdf 文本层。
- MinerU 接入使用 CLI 而不是内部 Python API：命令形态为 `mineru -p <pdf> -o <cache_dir> -b pipeline`，避免绑定不稳定的内部模块路径。
- 输出解析优先读取 `content_list_v2.json`，再 fallback 到旧 `content_list.json`；解析结果统一映射到现有 `ParsedDocument`、`ParsedChunk` 和 `metadata.document_intelligence`。
- 表格、公式、图注被写入现有 `document_intelligence.tables/formulas/figures`，后续 wiki 渲染、检索、问答接口不需要改协议。
- 新增配置项：`MINERU_ENABLED`、`MINERU_BIN`、`MINERU_BACKEND`、`MINERU_MODEL_SOURCE`、`MINERU_OUTPUT_DIR`、`MINERU_TIMEOUT`、`MINERU_EXTRA_ARGS`。
- 依赖改为可选安装：`pip install -e ".[mineru]"`，其中 `mineru` extra 使用 `mineru[pipeline]`，不是旧的 `magic-pdf[full]`。

## Server Deployment Notes

推荐在服务器虚拟环境中安装：

```sh
cd ~/llm_wiki_server
. .venv/bin/activate
pip install -U pip
pip install -e ".[mineru]"
```

`.env` 建议启用：

```dotenv
MINERU_ENABLED=true
MINERU_BIN=mineru
MINERU_BACKEND=pipeline
MINERU_MODEL_SOURCE=modelscope
MINERU_OUTPUT_DIR=./data/cache/mineru
MINERU_TIMEOUT=3600
MINERU_EXTRA_ARGS=
```

GPU 使用建议：

- 不让 MinerU 和 Ollama 同时处理大任务。
- 大批量 PDF 解析前可通过 `nvidia-smi` 确认 Ollama 是否仍占用显存。
- 如果 Ollama 长时间保留 Qwen 模型，可在 `.env` 中设置 `OLLAMA_KEEP_ALIVE=5m`；大批量 PDF 解析前可临时改成 `0` 或停掉 Ollama。
- 第一版先使用 MinerU `pipeline` 后端；如果后续需要 vLLM/VLM 后端，再通过 `MINERU_EXTRA_ARGS` 或专门配置扩展。

## Test Plan

- 单元测试：用模拟 MinerU `content_list_v2.json` 验证 title/text/table/equation/image 能正确转成 `ParsedDocument`、chunks 和 `document_intelligence`。
- 回退测试：MinerU CLI 不存在、退出码非 0、输出缺失、输出为空时，`_parse_pdf_with_mineru()` 返回 `None`，由现有 PDF 链路继续处理。
- 回归测试：`MINERU_ENABLED=false` 时现有 `python -m compileall src tests` 和 `pytest -q` 必须通过。
- 服务器验收：启用 MinerU 后重新 ingest SAC-KG PDF，确认 `sources/*.md` 中 Tables/Figures/Formulas 更完整，查询 Table/Figure/ablation/metrics 时 citation 能命中 wiki page。
- 性能验收：用 `nvidia-smi` 观察 MinerU 运行期间 GPU 占用，确认任务结束后显存释放，再进入 Ollama 抽取阶段。

## Assumptions

- 第一版不改数据库 schema，不新增 API，不改变 upload/query 的外部协议。
- MinerU 默认关闭，未安装 MinerU 的环境行为不变。
- `data/cache/mineru` 属于运行缓存，不进入 Git。
- 3090 24GB 显存足够单独运行 MinerU pipeline 或 Ollama Qwen，但两者不应并发跑重任务。
- PDF 解析质量更新后，需要对旧 PDF 重新 ingest，旧 wiki/source page 不会自动重写。
