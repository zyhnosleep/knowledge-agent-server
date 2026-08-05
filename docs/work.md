# Knowledge Agent 工作记录

## 2026-07-30 Task 15：`search.py` 恢复、PDF 表格一致性与本地最终复核

- 已确认用户提到的 `search.py` 误改真实存在：该文件曾被部分旧实现覆盖。现已恢复完整 Parent（不再按 1600/2400 字符截断）、10,000 个真实 tokenizer token 的回答预算、最多 6 个唯一 Parent、表格前后 Child 的独立 citation，以及 staged `parse_version_map` 对 paper profile、SQL、vector store 和 `_finalize_contexts()` 的全链路透传。另修复遗留的多余 `@classmethod`，避免 `_is_high_risk()` 实例调用报错；无 shadow map 时仍兼容旧 vector-store 测试替身。
- Narrative overlap 去重已改为精确 suffix/prefix 匹配，并用当前固定 tokenizer 的真实 token counter 限定重叠不超过 50 token；不再使用按空格分隔的 `\S+` 近似。最终 prompt 渲染不会通过 `citation.excerpt` 把已删除 overlap 重新拼回，但 citation excerpt 本身保持原始 Child 文本不变。英文和中文无空格文本均有回归覆盖。
- MinerU HTML 表格现在统一经过 `CanonicalArtifactStore._canonical_table_from_html()`：第一逻辑行始终作为 header，adapter、单表质量门、`TableValidator`、跨页质量门和跨页 validator 共用同一规则。mixed `<td>/<th>`、全 `<td>` 跨页表以及 `rowspan/colspan` 不再因不同模块的 header 规则而互相判不一致。
- 当 MinerU 同时提供 HTML 和 Markdown 时，HTML cell/span grid 是权威结构；canonical `source_markdown` 由 caption 与 canonical grid 重新生成，MinerU 原始 Markdown 保存在 `metadata.parser_source_markdown` 供审计。这样 span 覆盖格不会再因 HTML 留空、Markdown 重复展开而触发 `source_markdown_mismatch`。HTML 原文、cell span 和 SourceSpan 均继续保留。
- 结构化 Child 的两项完整性问题已修复：Figure Parent 仍保留附近正文，但 Parent 超限时 Figure Child 只对图源描述无损分窗，不再重复超长 nearby narrative；超长表格行的脚注在能完整放入时进入直接嵌入 Child，放不下时生成独立、可引用、有 SourceSpan 的有界 footnote Child，长脚注可继续无损分窗。所有 Child 仍满足 `embedding_text == text`。
- PDF 文本层逐页审计新增低覆盖率诊断：`coverage_ratio < 0.90` 的页面写入 `text_layer_low_coverage_page_indices` 并产生页级 warning；该阈值只用于告警，不替代“任意连续缺失达到 80 个规范化字符即恢复”的主规则。
- 两轮独立审查的结果已全部处理：无 Critical；审查提出的最终 prompt overlap、HTML span、mixed header、跨页 header、Figure nearby、表格脚注、中文 tokenizer overlap 和低页面覆盖率问题均已有回归测试。
- 最新本地验证：四个核心模块 `220 passed`；Task 15 扩大回归 `702 passed`；完整测试 `1778 passed`、0 failed；`python -m compileall -q src/app scripts` 通过；`git diff --check` 无 whitespace error，仅有工作区已有 LF/CRLF 提示。首次全量运行因测试进程 `PATH` 找不到 Conda Node 出现 1 个环境错误，加入既有 `/home/zyh/miniconda3/bin` 后完整重跑通过，未为此修改仓库。
- 本轮修复尚未同步服务器，也未启动新 rebuild、未修改 active pointer、未执行 cleanup。2026-07-30 本地门禁完成后，只读内网 SSH 在握手阶段被 `192.168.31.20:22` 直接关闭；既有公网入口又因 DNS 无法解析而失败，因此已停在同步前，没有重试轰炸、没有终止未知 SSH 进程。SSH 恢复后的下一步仅向 `/home/zhangyh/knowledge-agent-dev` 精确同步生产/config 变更，先对文档 `3f783e8e-b28f-4d5d-9425-bbf948ea224a`（`114110_1_5.0197592.pdf`）执行 GPU0 staged canary；测试环境保持旧版本只读，GPU1 不使用，canary 不激活。确认 pages 26-29 补齐、TABLE III/IV 无视觉 LLM repair 即通过且状态为 `ready_to_activate` 后，才使用新 config hash/suffix 运行全部 17 篇。旧 chunks、vectors、ParseVersions、MinerU 输出、artifacts 和源文件全部保留。

## 2026-07-29 Task 15：Tokenizer fidelity 与批量原子激活（本地完成）

- Task 1-5 已在 `codex/internal-pilot` 本地实现完成。生产 ingestion 现在强制使用本地固定的 `Qwen/Qwen3-Embedding-4B` tokenizer（revision `5cf2132abc99cad020ac570b19d031efec650f2b`），tokenizer 内容哈希和完整 ingestion 配置哈希进入 ParseVersion identity；运行时缺少或不匹配时直接诊断失败，不允许 UTF-8 byte fallback。
- PDF/Canonical fidelity 已改为逐页连续缺失检测与无损恢复；所有 narrative、table、formula、figure Child 都必须通过真实 tokenizer 上限校验，`token_count` 必须等于重新分词结果，且所有 Child 继续保持直接嵌入 `embedding_text == text`。`paper_profile` 依据当前 canonical Child 文本指纹刷新，不调用 LLM。
- Parent/table 回答上下文不再按 1600/2400 字符截断。Top-10 检索排名保持不变，回答阶段最多扩展 6 个唯一 Parent，并在 10,000 个真实 tokenizer token 内按完整单元装配；放不下完整 Parent 时退回命中的 Child，不截断证据。表格邻行仍保持独立 citation/source span。
- 全量重建改为 batch-atomic：每篇文档只运行到 `index`，新 ParseVersion 停在 `ready_to_activate`，旧 active pointer 持续提供服务。验收通过 `parse_version_map` 对 SQL、SQLite vector、PGVector、`QueryService`、`RAGAdapter` 和 acceptance 全链路做只读 shadow 路由；全部文档同时通过后，`activate_batch()` 才在一个事务中锁定并复验全部版本，统一更新 active pointer、状态、`raw_text`、source identity、`paper_profile`、checkpoint、run progress 和 document status。任一验证或 flush 失败会整体回滚。
- Shadow routing 与激活后 routing 已共用同一画像源规则：按稳定顺序拼接非 reference Child，移除旧 `paper_profile` 和旧 `document_intelligence` 后重新生成画像，避免验收前后选文语料漂移。active-index acceptance 即使没有 shadow map，也会加载并校验当前 ingestion 配置、source/version identity 和 Child token 上限。
- Cleanup 防护已收紧，但本阶段不会执行 cleanup：必须绑定成功激活记录且 active pointer 与 version map 完全一致；只处理该激活批次内明确为 `superseded` 的 ParseVersion 和该批次的 legacy 数据；`ready_to_activate`、failed、queued、running 版本以及未参与批次的 legacy active 数据全部保留。旧 chunk、vector、ParseVersion、artifact、MinerU 输出和源文件当前均不删除。
- 两轮独立审查共发现并修复 7 个 Important 问题：partial shadow map 指标范围、shadow 路由使用旧文本/画像、激活信任 checkpoint 而未直接复验 source/token、cleanup 误删 staged、无 shadow map 的 active acceptance 门禁恒失败、shadow/active 画像源不一致、partial cleanup 误伤未映射 legacy 文档。对应回归测试均已加入。
- 最终本地验证：Task 5 聚焦回归 `224 passed`；完整测试 `1760 passed, 3 skipped`；`py_compile` 通过；`git diff --check` 无 whitespace error，仅有 Windows 工作区既有 LF/CRLF 提示。尚未向服务器同步本轮新代码，也未修改开发/测试数据库或服务。

### 剩余 Task 7：开发环境 staged 对比与验收

- 只同步新代码和配置到 `/home/zhangyh/knowledge-agent-dev`；测试环境 `/home/zhangyh/knowledge-agent-test` 保持旧版本、只读且不重启、不迁移、不写入。所有开发重建/推理命令必须设置 `CUDA_VISIBLE_DEVICES=0`，不得使用 GPU1。
- 先下载并校验固定 tokenizer snapshot，生成 tokenizer preflight；冻结开发/测试文档清单与 source SHA，并从旧测试版本保存同一组 30 个 byte-matched case 的只读 baseline。
- 在开发环境为 17 篇文档并行数据版本做 side-by-side rebuild，但全部只到 `ready_to_activate`。在所有文档 source fidelity、structured limit、config identity、source/version identity、embedding、vector、artifact 等严格指标均为 `1.0` 前，不切换任何 active pointer。
- 使用 staged `parse_version_map` 先跑不生成答案的 30 题 shadow retrieval：要求 Recall@5=`1.0`、Recall@10=`1.0`、source identity validity=`1.0`，且每个相同 case 不得低于旧测试 baseline；失败即停止，不进入答案测试。
- 检索通过后再跑 staged shadow full30 answer：要求至少 `29/30`，总分不得低于旧测试 baseline，并单独记录 P50/P95 latency。所有硬门禁通过后才执行一次批量原子激活；继续保留所有旧版本与 artifact，不执行 cleanup。最后复核测试环境 HEAD、服务、文档/chunk 数量和代表性 source SHA 完全不变。

## 2026-07-29 Task 15：V4 全量重建完成

- 服务器自动升级 NVIDIA 驱动后曾出现内核模块 `580.159.03` 与 NVML `580.173.02` 不一致；经用户确认重启后，内核模块与用户态库均为 `580.173.02`。重启同时暴露 pgvector 容器 `llm-wiki-pgvector-rehearsal` 的 restart policy 为 `no`，已仅恢复该既有容器及原服务，未修改数据库配置或数据。
- 全量续跑发现两个真实阻断并以测试驱动修复：其一，repair 生成模型默认驻留 5 分钟，与下一篇 MinerU 解析模型叠加导致 GPU0 OOM；重建脚本现在每篇文档结束后查询开发 Ollama `/api/ps` 并对实际驻留模型发送 `keep_alive=0`。其二，摘要检测器能识别标题/作者后的 `Abstract`，但抽取器只检查候选首行，导致 `abstract_missing` 错误依赖视觉修复；现在抽取器扫描候选任意行，PDF audit 仅在摘要为空时从前两页 pypdf 文本层确定性补齐文档级摘要，MinerU blocks、tables 和主解析器身份保持不变。聚焦回归 `158 passed`。
- 修复后的 `ff99sb-ildn` 定向报告 `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-ff99sb-ildn-v4.json` 与 `114110_1_5.0197592` 定向报告 `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-114110-v4.json` 均为 1/1 成功、0 失败、全部严格指标 `1.0`、`ready_for_acceptance=true`。每篇完成后 GPU0/GPU1 均回落到 15 MiB，GPU1 全程无计算进程。
- 正式全量报告 `/home/zhangyh/knowledge-agent-dev/runtime/task15/canonical-rebuild-report.json` 已通过：17/17 文档成功、0 失败，3029 个 Child 全部完成原文直嵌入和 pgvector 索引；`contextualization_eligible_children=0`、`contextualized_children=0`，parse、contextual prefix 空集、plain embedding、embedding、pgvector、table、source span、artifact 指标全部为 `1.0`，`ready_for_acceptance=true`。
- 数据库逐行复核：3029/3029 Child 满足 `embedding_text == text`、无任何 contextual 字段且具有 2560 维 JSON embedding；1011/1011 Parent 均未嵌入。17/17 active version 均为 `canonical-v4-*`，17/17 开发源文件 SHA-256 匹配。旧 legacy/V1/V2/V3 共 6 个 inactive parse version、旧 chunk/vector、21 个历史/active artifact 目录和 MinerU 产物继续保留，未执行任何删除参数。
- 测试环境复核仍为 HEAD `e40425c5e559e3cb5af04f2e390f20c6e6de8ef3`、15 文档、1628 chunks、15/15 源文件 SHA-256 匹配；服务状态恢复到冻结快照（test API/Ollama active、test worker inactive）。下一步只生成公平重叠题集审计，之后才进行旧测试版与新开发版的同题检索对比。

## 2026-07-28 Task 15：选择性结构化上下文化调整

- 已停止 `ff14sb` 的旧 `canonical-v1` canary。该次运行在约 465 个 Child 上执行全量 LLM contextualization，超过 120 次模型调用、耗时约 50 分钟，期间出现 32K prompt 截断和重试；停止后确认无 `rebuild_canonical_index.py` 进程，开发推理始终只使用 GPU0。
- 用户最终确认：`narrative`、`caption`、`appendix`、`table` 直接使用 `child.text` 做 embedding，不调用 LLM；仅 `figure`、`formula` 生成 1–2 句 contextual prefix，并以 `contextual_prefix + "\n\n" + child.text` 做 embedding。表格仍保留独立 Child、Markdown 内容、页码和来源信息；Parent 保持不嵌入。
- 已停止仍在运行的 `ff14sb` V2 canary，并在服务器确认无 `rebuild_canonical_index.py` 残留进程。新策略使用独立 `canonical-v3-<sha>` parse version，保留并且不覆盖旧 `canonical-v1`、中断的 `canonical-v2` checkpoint、chunk、vector、artifact、MinerU 输出和源文件。测试环境继续作为旧版本只读基线。
- 完整性门禁拆分为两项：结构化 Child 的 `contextual_prefix_completeness=1.0`，普通 Child 的 `plain_embedding_completeness=1.0`；所有 Child 仍必须满足 embedding、pgvector、source span 和 artifact 完整性。
- 本地已完成 V3 策略的测试先行修改；聚焦回归 `180 passed`，证明表格 `embedding_text == text` 且无 contextual 字段，只有 figure/formula 进入 LLM。开发环境同步与新的 `ff14sb` V3 canary 尚未开始。
- `ff14sb` V3 dry-run 选择 `canonical-v3-45f41d8c39a2` 且零源文件错误；正式 canary 在 GPU0 上运行，GPU1 无计算进程。表格退出 LLM 后，contextualization 候选从 295 个降为 8 个（5 figure、3 formula）。
- V3 canary 三次均在 contextualize 严格门禁停止，未进入 embed/index/activate，旧 active 数据未替换。根因证据显示 `qwen3.5:9b` 对 figure 响应不稳定遵循 JSON schema，会交替输出 `relations/relation_description`、`items/relation_summary`、顶层数组、`relation_explanation` 或 `{child_id: prefix}`；部分前缀引用论文名 `ff14sb` 时又与当前数字标识校验冲突。
- 用户最终批准彻底移除 ingestion-time LLM contextualization。V4 将 `narrative/table/figure/formula/caption/appendix` 全部设为直接 Child embedding，要求 `embedding_text == text` 且无 contextual 字段；contextual eligible/contextualized 固定为 0。新策略使用 `canonical-v4-<sha>`，不复用或覆盖 V1/V2/V3 checkpoint。本地聚焦回归 `194 passed`，等待全量测试和开发 V4 canary。
- `ff14sb` V4 canary 已在开发环境 GPU0 成功完成并激活 `canonical-v4-45f41d8c39a2`；报告位于 `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-ff14sb-v4.json`。本次 1/1 文档成功、0 失败，共生成 395 个可检索 Child（100 narrative、287 table、5 figure、3 formula）；`contextualization_eligible_children=0`、`contextualized_children=0`、`plain_embedding_children=395`，全部 Child 均满足 `embedding_text == text`、无 contextual 字段且具有有效向量。`contextual_prefix_completeness`、`plain_embedding_completeness`、embedding、pgvector、table、source span 和 artifact 指标均为 `1.0`，`ready_for_acceptance=true`。MinerU 仍为主解析器，50/50 页覆盖；仅第 7、8、9、15、16、45、46、48、50 页以 `pypdf_text_layer` 补齐，原因均为 `mineru_page_missing`。运行期间 GPU1 保持 22 MiB 且无计算进程，结束后无重建残留进程。
- `opls5` V4 表格 canary 已在开发环境 GPU0 成功完成并激活 `canonical-v4-c0948ba8a49a`；报告位于 `/home/zhangyh/knowledge-agent-dev/runtime/task15/canary-opls5-v4.json`。本次 1/1 文档成功、0 失败，共 156 个 Child（89 narrative、57 table、7 figure、3 formula），156/156 均满足 `embedding_text == text`、无 contextual 字段且向量维度有效。10/10 张 canonical 表格状态均为 `accepted_mineru`，全部具有非空 `normalized_markdown` 和有效 source span，且无 `validation_failed`；所有严格完整性指标均为 `1.0`，`ready_for_acceptance=true`。旧 `canonical-v1-c0948ba8a49a` 失败版本继续保留且未激活；测试环境复核仍为 15 文档/1628 chunks。运行期间仅 GPU0 有计算进程，GPU1 保持 22 MiB，完成后无重建残留进程。
- 全量 V4 dry-run 报告 `/home/zhangyh/knowledge-agent-dev/runtime/task15/canonical-rebuild-dry-run.json` 已通过：17/17 源文件存在、0 失败，计划版本全部以 `canonical-v4-` 开头。随后启动正式全量重建；按用户 2026-07-28 当日暂停要求，以 SIGINT 安全中断在 `ff99sb-ildn`（文档 `1b4c28ed-2954-4b06-b858-ac544e819c0e`）的 `repair` 阶段。当前 `ff14sb` 与 `opls5` 两个 V4 版本保持 active；`ff99sb-ildn` 的 parse 已完成，repair checkpoint 状态为 running/attempt 1，明日等待 lease 过期后使用同一命令和 `--resume` 续跑。中断后确认无 `rebuild_canonical_index.py` 残留进程，未开始 retrieval/full30/citation 对比，也未删除或覆盖任何旧数据。

## 2026-07-28 Canonical Contextual RAG：Task 14

- 新增 `scripts/rebuild_canonical_index.py`：支持 `--dry-run`、`--resume`、`--document-id`、`--report`、`--delete-old-after-acceptance` 和 `--confirm-delete-old-data`。dry-run 只检查原始文件与计划版本，不创建 parse version、目录或数据库记录；已有 checkpoint 未显式传入 `--resume` 时拒绝继续。
- 重建按文档执行完整八阶段 canonical pipeline，并生成全有或全无的 readiness 报告；任一文档失败或解析、contextual prefix、embedding、表格、source span、artifact、pgvector 指标不完整时返回非零，不允许进入验收。
- 旧数据清理必须同时提供两个删除保护参数，并读取 `ready_for_acceptance=true` 且 `failed_documents=0` 的既有重建报告；只删除 inactive/legacy chunk、对应向量、旧 parse version/artifact 和 MinerU 临时输出，保留原始上传文件、active canonical artifact、active chunk/prefix/embedding/vector。
- 新增 `scripts/evaluate_canonical_retrieval.py`：记录 parse/contextualization/embedding/pgvector/table/source-location 完整度、Recall@5/10、答案与引用结果、P50/P95 检索延迟及失败归因。检索断言只读取 citation `excerpt`，不使用 `context_text`，并拒绝 contextual prefix 泄漏、References 命中、错误 source identity 和无效 bbox/text locator。
- 新增 `docs/query_acceptance/canonical_ingestion_v1.json`，覆盖 narrative、中文问英文论文、长表、跨页表、Figure、Formula、Appendix、References 排除、contextual prefix 引用排除和 bbox 定位十类严格案例。
- 提交：`ce881cd Add canonical rebuild acceptance tooling`。计划指定回归 `19 passed`；canonical parsing/artifact/quality/retrieval/stage 扩展回归 `440 passed, 3 skipped`；`git diff --check` 通过。
- 按用户要求在 Task 14 完成后暂停。服务器未连接，未执行真实历史重建、真实 full30/canonical acceptance 或旧数据删除；Task 15 保持未开始。

## 2026-07-27 Canonical Contextual RAG：Task 12

- 新增四个只读接口：当前解析状态、当前 canonical Markdown、当前 canonical.md 下载，以及按 Child chunk 查询来源位置；四个接口都强制校验 `project_slug`，只暴露文档的 active parse version。
- Markdown 与下载接口同时校验数据库 checkpoint、manifest 身份、输入指纹和 Markdown SHA-256；artifact 路径继续拒绝目录越界、符号链接和 Windows reparse point。
- JSON Markdown 视图限制为 10 MiB；下载不受该限制，先复制到受控临时快照并校验，再只流式发送这份快照。同一 inode 原地改写、路径替换、客户端断连、取消或发送异常都不会返回未经验证的字节，源句柄和快照句柄都会关闭。
- source location 使用严格类型和长度限制，只返回原始 Child 的 `source_spans`；支持 PDF 多页 bbox、DOCX 段落、HTML DOM path、文本行范围，并保留真实表格定位。`contextual_prefix`、`embedding_text`、内部路径和内部 metadata 不进入 API。
- 状态接口返回版本、解析器、八阶段进度、质量状态、定向修复页、manifest warning 数量和下载可用性；不提供历史版本入口。
- 提交：`a65c1fa`、`2f57432`、`678d5b7`。最终规格复核：`SPEC COMPLIANT`；API/认证聚焦回归 `76 passed`；主全量回归 `1632 passed, 3 skipped`。
- 下一步：Task 13 在现有运行列表和 source drawer 中加入当前解析进度、Markdown 查看/下载，以及基于 `source_spans` 的 PDF/Markdown 引用高亮；不展示历史解析版本或 contextual prefix。

## 2026-07-27 Canonical Contextual RAG：Task 13

- 在现有运行看板的 source drawer 中加入 active parse 状态栏，显示版本、MinerU 解析器版本、八阶段进度、质量状态、修复页和警告数量。
- 新增“查看解析 Markdown”和“下载 canonical.md”工具按钮；Markdown 以 DOM `textContent`/文本节点渲染，不使用动态 `innerHTML`，长文在独立视口中滚动。
- 回答中的 citation marker 和来源列表现在可以打开对应文档，并调用 `/citations/{chunk_id}/location`；前端按当前后端的 0 起始 citation 编号定位，越界时兼容旧的 1 起始标记。
- PDF 引用会导航到来源页，并对当前页的一个或多个 `normalized_bbox` 绘制不拦截鼠标的覆盖层；文本来源按 `line_start`/`line_end` 高亮。Figure/AI observation 明确显示“图像分析（非原文文本）”，不冒充 source text。
- UI 只提供当前 active parse，不显示历史解析版本，也不显示 `contextual_prefix` 或 `embedding_text`。用户确认本项目不做移动端，本任务最终只以 1440×900 桌面视口为浏览器门禁。
- 提交：`bd965b6 Add canonical parse and citation UI`。静态前端与 canonical API 聚焦回归 `108 passed`；Playwright 桌面验证通过，Markdown 可滚动、下载为 `canonical.md`、bbox 覆盖为 1、无横向溢出、无控制台错误。
- 下一步：Task 14 增加全量重建、断点续跑、旧数据确认删除和 canonical retrieval acceptance 工具。

## 当前目标

- 在同一台双 GPU 服务器上隔离开发与测试环境。
- 开发环境固定 GPU 0，测试环境固定 GPU 1。
- 两套环境统一使用 `qwen3.5:9b` 和 `qwen3-embedding:4b`。
- 测试环境保留现有文章、账号、会话和公网入口。
- 开发环境使用空数据库且只通过 SSH 转发访问。

## 运行约束

- 生成上下文固定 32K，并发容量为 1。
- Flash Attention 和 q8_0 KV Cache 开启。
- 模型闲置 5 分钟后释放 GPU。
- DeepSeek 等外部 API 默认关闭。
- 引用编号、页码和来源链接由后端验证。
- 开发提交只有人工验收后才能发布到测试环境。

详细设计和执行步骤见：

- `docs/superpowers/specs/2026-07-17-knowledge-agent-environment-isolation-design.md`
- `docs/superpowers/plans/2026-07-17-knowledge-agent-environment-isolation.md`

## 2026-07-18 隔离部署记录

部署代码基线：`c009875`（`codex/internal-pilot`）。

GitHub 仓库已重命名为 `zyhnosleep/knowledge-agent-server`；本地和两套服务器代码目录的 `origin` 均已更新为新地址。

### 开发环境

- 目录：`/home/zhangyh/knowledge-agent-dev`
- API：`127.0.0.1:8002`，仅通过 SSH 转发访问
- 数据库：`knowledge_agent_dev`，空开发数据集
- Redis：`redis://127.0.0.1:6379/1`
- GPU/Ollama：GPU 0，`127.0.0.1:11435`
- 服务：`knowledge-agent-dev-api`、`knowledge-agent-dev-worker`、`knowledge-agent-dev-ollama`
- 真实验收：上传在 0.116 秒返回排队状态，worker 完成解析；JSON embedding 与 pgvector 均为 2560 维；问答模型为 `qwen3.5:9b` 且引用有效。
- 队列验收：DB 1 排队、运行中、失败任务均为 0，注册 worker 为 1。

### 测试环境

- 目录：`/home/zhangyh/knowledge-agent-test`
- API：`127.0.0.1:8001`
- 数据库：`knowledge_agent_test`
- Redis：`redis://127.0.0.1:6379/2`
- GPU/Ollama：GPU 1，`127.0.0.1:11436`
- 服务：`knowledge-agent-test-api`、`knowledge-agent-test-worker`、`knowledge-agent-test-ollama`
- 临时公网：`https://planets-intranet-dog-therapy.trycloudflare.com`，由原有 Caddy Basic Auth 网关转发到新测试 API
- 迁移结果：15 个文档、1778 个 chunk；4B 重建成功 15/15，失败 0；JSON embedding 与 pgvector 1778/1778 有效，维度 2560。
- 检索基线：旧 8B `recall@5=1.0`、引用有效率 100%、P95 846 ms。
- 新模型验收：4B `recall@5=1.0`、引用有效率 100%、P95 1098 ms，验收通过。
- 公网验收：未授权 401、授权页面 200、异步上传完成、Agent 回答与引用有效、PDF Range 返回 206。
- 新上传文件写入 `/home/zhangyh/knowledge-agent-test/runtime/data/raw/...`，不写入迁移备份目录。
- 队列验收：DB 2 排队、运行中、失败任务均为 0，注册 worker 为 1。
- 空闲释放：停止模型请求约 5 分钟后，开发与测试 `/api/ps` 均为空；GPU 1 回落到 22 MiB，GPU 0 仅保留既有 ComfyUI 的 256 MiB。

### 公共运行组件与备份

- Redis 服务：`knowledge-agent-redis.service`
- Redis 版本：服务器现有 Redis 7.4，仅监听 `127.0.0.1:6379`，AOF 位于 `/home/zhangyh/knowledge-agent-runtime/redis`
- 共享依赖运行时：`/home/zhangyh/knowledge-agent-runtime/.venv`
- 共享 Ollama 模型实体目录：`/home/zhangyh/knowledge-agent-models`，不再依赖旧目录软链接
- 迁移前备份：`/home/zhangyh/knowledge-agent-backups/20260718T015755Z`
- 重建报告：`/home/zhangyh/knowledge-agent-test/runtime/reindex-report.json`
- 重建校验：`/home/zhangyh/knowledge-agent-test/runtime/reindex-verify.json`
- 固定题集与报告：备份目录下 `acceptance/knowledge_agent_retrieval_v1.json`、`baseline-8b.json`、`new-4b.json`

### 旧版本清理

2026-07-18 经用户确认完成以下清理：

- 删除三个旧 API/Ollama systemd 服务及其 unit 文件。
- 删除旧代码与运行目录。
- 删除旧应用数据库；PostgreSQL 仅保留 `knowledge_agent_dev` 和 `knowledge_agent_test`。
- 公网 Caddy 配置迁移到 `/home/zhangyh/knowledge-agent-test/runtime/tunnel`，原 Cloudflare Quick Tunnel 进程保持运行，因此临时网址不变。
- 删除服务器主目录下剩余的 6 个 `llm_wiki*` 旧仓库与历史备份目录，释放约 11.56 GB；检查确认无活动服务、进程或软链接引用。
- 将约 38 GB 模型实体原地迁移到 `knowledge-agent-models`，删除旧 `llm_runtime` 缓存、临时文件和日志，额外释放约 3.62 GB；开发、测试两端均通过 2560 维 embedding 与 9B 生成实测。
- 删除 0 字节的失败备份 `20260718T014131Z`，仅保留已验证可恢复的 `20260718T015755Z`。

本次未授权删除的 14B/27B/8B 模型和 108 MB 迁移前备份继续保留。若需要回滚数据，应从 `/home/zhangyh/knowledge-agent-backups/20260718T015755Z` 恢复到新建数据库，不再依赖已删除的旧服务或旧目录。

## 2026-07-20 至 2026-07-27 Canonical Contextual RAG 升级进展

### 已确认的目标架构

- 学术 PDF 以 MinerU 为主解析器；Document Intelligence 只做质量门控后的定向修复，不作为默认重复解析链路。
- 解析结果先形成 canonical document，再执行 section-aware semantic Parent/Child 切分。
- 仅 Child 进入普通检索；Parent 用于命中后的上下文展开。表格、公式、图片等结构块保持独立语义边界。
- 每个 Child 在嵌入前必须生成 1–2 句 LLM contextual prefix；实际嵌入文本为 `contextual_prefix + "\n\n" + child.text`，但 prefix 不进入引用原文。
- References 默认排除普通 RAG；Appendix 可以检索。表格损坏阻断 readiness，图片/公式分析允许带 warning 继续。
- 新版本必须完整经过 parse、repair、canonicalize、semantic_split、contextualize、embed、index、activate 八个阶段；激活前旧 active 版本持续对外可用。

### Task 7：语义 Parent/Child 切分

- 已完成 section-aware semantic chunking，替换固定字符切块作为 canonical 主路径。
- Parent 按章节和结构组织，Child 按语义边界切分；跨页、表格、公式、图片等边界保持可追溯。
- chunk ID 包含 parse version、结构标识、来源 block、角色、序号和文本指纹，避免版本间 ID 冲突。
- 最终门禁：规格通过、代码质量通过；Task 7 完成点全量测试 `1427 passed, 3 skipped`。

### Task 8：强制 contextual prefix

- 已实现批量 contextualization、纠错重试、拆批/OOM 降级、精确失败 ID 和 durable checkpoint。
- prefix、模型、版本、prompt 版本和时间成为一等字段；原始 `text` 与 source spans 保持不变。
- 对表格 Child 使用同一“解释该块与全文关系”的原则，但上下文锚点来自表题、表头、行组、章节和 Parent，不复制表格正文。
- 最终门禁：规格通过、代码质量通过；Task 8 完成点全量测试 `1538 passed, 3 skipped`。

### Task 9：可恢复八阶段流水线

- 已实现八个具名队列：`ingest.parse`、`ingest.repair`、`ingest.canonicalize`、`ingest.semantic_split`、`ingest.contextualize`、`ingest.embed`、`ingest.index`、`ingest.activate`。
- checkpoint 持久化、stage lease、确定性 job ID、RQ 重复任务恢复、失败重试和进程恢复已落地。
- lease 默认覆盖 `queue_job_timeout + 300s`；超长异常会截断、清洗并记录 SHA-256，避免失败记录本身再次失败。
- Task 10 完成前的中间版本不会覆盖旧 active 文档标题、元数据、状态或检索可见性；artifact 路径拒绝越界、符号链接和 Windows reparse point。
- 最终门禁：`SPEC COMPLIANT`、质量 `APPROVED`；全量测试 `1576 passed, 3 skipped`。

### Task 10：版本化向量与原子激活

- 提交：`3f12f8b Index contextual embeddings by parse version`。
- `ChunkVector` 和 `VectorHit` 已携带 `parse_version`；pgvector、SQLite-vec 和 JSON fallback 均按 `Document.active_parse_version` 过滤。
- 仅具有完整 contextual prefix 的 Child 调用 embedding；Parent 仍写入版本化 `DocumentChunk` 供后续展开，但 embedding 为 `null`。
- index 阶段校验 Child 数量、向量数量、唯一 local ID 和配置维度；SQLite-vec 兼容旧元表并补充 `parse_version` 列。
- activate 阶段要求 retrievable、contextualized、embedded、backend-indexed、valid-source-span 和 artifact Child 数量非零且完全一致，随后在同一数据库事务内切换 active pointer；失败保持旧 active 版本。
- 已修复无 document scope 时 inactive SQLite 近邻占满候选窗口导致 active 结果漏召回的问题。
- 当前验证：Task 10 聚焦及检索回归 `349 passed`；此前实现代理全量测试 `1583 passed, 3 skipped`。真实 Redis registry 竞争和真实 PostgreSQL/pgvector 并发激活仍需在部署验收阶段补集成测试。

### 当前状态与下一步

- 当前分支：`codex/internal-pilot`，Task 11 HEAD 为 `bc7a18b`。
- 用户现有的 canonical 相关未提交改动、设计文档删除项及 `agent_synthesizer.py` 改动保持原样，未纳入 Task 7–11 提交。

### Task 11：Child 命中与 Parent 证据展开

- 主提交：`2712e8d Expand contextual child hits to source parents`；审查修复提交：`2d8f49f`、`922c66b`、`bc7a18b`。
- canonical 主路径严格只检索 active parse version 的 Child，并排除 `block_type='reference'`；legacy 空字段兼容仅限 legacy 分支。
- 命中 Child 后批量加载 Parent 和必要邻居，citation excerpt 始终等于原始 `child.text`；`contextual_prefix` 和 `embedding_text` 不进入引用或 source evidence。
- overview/comparison 可在独立预算内加入前后邻居；表格 prompt 使用 caption/header/命中 row-group；Figure/Formula 返回有据的 typed provenance ID。
- typed ID 由 semantic chunking 写入派生 source span metadata，不再用 block ID、asset path 或 relationship ID 冒充；metadata-only provenance 不会绕过 Task 10 的 source locator 激活门。
- source candidates 先排序取 Top-K，再一次性批量加载 Parent/neighbor，N+1 回归从 32 次 SELECT 降到不超过 4 次。
- draft token budget 按最终 `"\n\n".join([index + rendered context])` 完整表示计算，包含 section 分隔符和 excerpt 合成后的真实成本。
- 最终门禁：`SPEC COMPLIANT`、质量 `APPROVED`；实现代理全量测试 `1602 passed, 3 skipped`，主代理聚焦回归 `288 passed`。

### 下一步

- Task 12：增加只读 canonical parse、Markdown 下载与 citation source-location API，只暴露 active parse version，并复用 artifact 路径边界校验。

## 2026-07-31 Task 16：canonical 表格证据组装审计与本地修复

- 根因已收敛到答案上下文层，不是 MinerU、pypdf、semantic chunk、embedding 或 active/staged 路由：staged 的 Recall@5/10、citation 和 source location 已通过，剩余失败集中在表格 Child 证据没有完整进入答案 prompt。
- 保持解析器、Chunk 边界、tokenizer、embedding、active 指针、旧数据和 test 环境不变；canonical 表格仍按 `document_id + parse_version + table_id` 组装同表 Child，每个 Child 保留独立 citation、原文 excerpt、source span 和 parse version。
- `src/app/services/search.py` 已改为由严格 retrieval-token budget 控制 prompt 大小，不再使用普通 Parent 的 1600 字符窗口或 canonical table 的 2400 字符窗口；普通 Parent 最多完整展开 6 个，后续回退到原 Child excerpt，表格 Child 不受该 Parent 上限影响。
- 新增普通文本 suffix/prefix overlap 去重（含无空格中文），只改变 prompt 文本，不改变 citation excerpt；shadow table 仍只读指定 staged parse version，不修改 active pointer。
- 本地验证：`tests/test_canonical_retrieval.py` 35 passed；`tests/test_vector_retrieval.py` 22 passed；`python -m py_compile src/app/services/search.py` 通过；相关 `test_query_service.py` 完整长上下文回归 2 passed。
- `test_query_service.py` 仍有 4 个此前存在的旧确定性表格/默认 5 条上下文上限失败，本轮未扩大范围处理；未同步服务器、未运行 rebuild、未激活 staged、未删除旧数据。

### 下一步

- 在开发服务器恢复后，仅将已验证的 `search.py` 同步到 `/home/zhangyh/knowledge-agent-dev`，所有运行命令显式使用 `CUDA_VISIBLE_DEVICES=0`。
- 先跑 staged retrieval-only 30 题；Recall、citation、source location 或 parse-version isolation 任何回退都停止，不进入 full-answer 评测。通过后再跑 staged full-answer，对比旧 test baseline；staged 保持 inactive。

## 2026-07-31 Task 16 follow-up: canonical table evidence regression fixed

- Root causes confirmed in `src/app/services/search.py`: ordinary same-page evidence was allowed to grow to the global limit instead of the historical five-item per-page cap; table citation ranking used raw retrieval score and could prefer unrelated simulation-condition tables; generic table answers consumed the eight-row budget from the first table and hid later requested facets.
- Fixes are limited to query-time answer/retrieval behavior. Explicit table labels receive a strong relevance priority, requested row/facet terms contribute deterministic ranking, ordinary non-table per-page deduplication remains capped at five, and generic table rows are selected round-robin so each requested table contributes evidence before the remaining row budget is used.
- Local verification: `tests/test_query_service.py` = `195 passed`; combined table/canonical/vector/agent/evaluation regression = `380 passed`; `python -m py_compile src/app/services/search.py` passed.
- Canonical table assembly remains version-scoped by `document_id + parse_version + table_id`; full table facts remain lossless and every Child keeps its own citation/source spans. No parser, MinerU, chunk boundary, tokenizer, embedding, active pointer, test data, or server test environment was changed.
- Current state: development server sync and staged retrieval/full-answer evaluation are still pending. The staged version remains inactive; old data is retained; any server command must explicitly set `CUDA_VISIBLE_DEVICES=0`.
- Full local suite after the regression fix: `1805 passed, 3 skipped` (187.18s). `git diff --check` reports no whitespace errors.
- Server preflight was attempted read-only with passwordless SSH: `llm-wiki-server` (`192.168.31.20:22`) timed out and `ffsampling-public` was refused. No files, services, databases, pointers, or GPU workloads were changed remotely.

## 2026-08-03 Task 16：Agent 表格引用桥接、首次合成精确性与开发同步

- 已完成本地 Agent 表格证据修复：`agent_executor.py` 会把完整 canonical table evidence 和 `table_facts.source_chunk_ids` 对应的原始 Child 加入 synthesis citations，并在合成后重新映射 citation marker；同编号但不同内容的 Table 会显示可读的 Table 标签与 caption/source legend。
- 已完成本地 Ollama synthesis provider：开发配置使用 `AGENT_SYNTHESIS_PROVIDER=ollama`，模型为 `qwen3.5:9b`；合成失败仍安全回退到 local evidence answer。
- 发现并补齐 LLM fallback：`search.py::_draft_answer()` 对可重试 Ollama 超时最多重试 2 次（间隔 5 秒），三次失败才返回带 `[系统提示：LLM 生成暂时失败，以下为原始检索证据，仅供参考]` 的降级答案。
- 根据 30 题首轮 Agent 评测逐题报告，citation pass rate 和 synthesis applied rate 均为 `1.0`，但首次 Ollama synthesis 遗漏了 QM/NMR、Table N、精确数字和行列名等证据锚点。已新增两层保护：
  - 首次 synthesis prompt 对 `table_or_metric`、`evidence_required`、`multi_source_compare` 明确要求原样保留精确数字/单位、技术缩写、数据集名、Table 标签、行名和列名；Canonical table facts 优先，禁止只写趋势而丢原值。
  - 首次结果缺少证据锚点时，Ollama 最多执行一次 coverage retry；重试仍失败则保留首次结果并记录 warning。
- 本地 TDD/回归：新增 fallback 重试、Ollama coverage retry、首次精确 prompt 测试；聚焦回归 `299 passed`，全量回归 `1832 passed, 3 skipped`，`py_compile` 通过，`git diff --check` 无空白错误。
- 开发环境已同步最终 `search.py` 与 `agent_synthesizer.py`，远程 SHA-256 分别为 `01cc4025525591cd58bea49e6cfd0185f15eac954b597b5cf5368a13ef3c53ca`、`ee94644580dd620c14254b633a8684beaa4b37f2c1402f7073e9cf966bbc559f`；API 已于 18:07 重启并监听 `127.0.0.1:8002`，API/Worker/Ollama 均 active。
- 开发 Ollama 已确认启用推理加速：`OLLAMA_FLASH_ATTENTION=1`、`OLLAMA_KV_CACHE_TYPE=q8_0`、`OLLAMA_NUM_PARALLEL=1`、`CUDA_VISIBLE_DEVICES=0`、`OLLAMA_KEEP_ALIVE=5m`；GPU1 全程未参与计算。
- 评测记录：旧版（coverage retry 前）完整 30 题报告保留在 `/home/zhangyh/knowledge-agent-dev/runtime/task15/task16-agent-full30-pre-ollama-coverage-retry-20260803.json`，结果为 `passed_cases=5/30`、`answer_pass_rate=0.1667`、`citation_pass_rate=1.0`、`synthesis_applied_rate=1.0`；第二轮已按用户要求在第 3 题前停止，部分日志保留在 `/home/zhangyh/knowledge-agent-dev/runtime/task15/task16-agent-full30-20260803-partial-coverage-retry-20260803.log`，未生成最终报告。
- 当前状态：最终 prompt 修复已同步但尚未重新跑完整 30 题评测；staged 保持 inactive，test 环境未同步/未重启，旧数据未删除。明日第一步是在现有 GPU0 加速配置下重新运行完整 Agent 30 题，并比较旧报告与新报告的 answer/citation/synthesis/latency 指标。

## 2026-08-04 Task 16：最终 Agent 30 题评测已完成（未通过）

- 只读预检确认开发 API（8002）、RQ Worker 和开发 Ollama 均为 user-level systemd `active`；测试 API/Ollama 仍隔离，staged 保持 inactive，旧数据未删除。
- 开发 Ollama 使用 `127.0.0.1:11435`、`qwen3.5:9b`，`CUDA_VISIBLE_DEVICES=0`、Flash Attention、`q8_0` KV cache、单并发；Worker 同样绑定 GPU0。启动时 GPU1 保持空闲。
- 已启动最终 prompt 版本的完整 30 题 Agent 评测，远端进程 PID 为 `1308723`；日志为 `/home/zhangyh/knowledge-agent-dev/runtime/task15/task16-agent-full30-20260804.log`，最终报告为 `/home/zhangyh/knowledge-agent-dev/runtime/task15/task16-agent-full30-20260804.json`。
- 最终报告已生成：`/home/zhangyh/knowledge-agent-dev/runtime/task15/task16-agent-full30-20260804.json`；评测进程已结束，30/30 题均已记录。
- 最终指标：`passed_cases=3/30`、`answer_pass_rate=0.1`、`citation_pass_rate=0.8667`、`synthesis_applied_rate=0.8667`、`p50_latency_ms=147404`、`p95_latency_ms=300100`，`strict_pass=false`。与旧版 `5/30`、`0.1667`、`1.0`、`1.0` 相比，本轮整体回退，不能进入激活或生产更新。
- 状态分布：`completed=23`、`timeout=3`、`error=4`；3 个完全通过题为 `charmm36idpsff_overview`、`charmm36m_table_metrics`、`ff99sb_disp_table_metrics`。
- 失败分组：4 个 `error`（`charmm36_overview`、`charmm36_table_metrics`、`ff14sb_table_metrics`、`ff99sb_ildn_table_parameters`）同时缺答案和 citation；3 个 `timeout`（`charmm36idpsff_table_metrics`、`oplsaa_overview`、`oplsaa_table_metrics`）；另有 20 个已完成合成但遗漏一个或多个问题要求的精确术语/数字。总计 `27` 个 answer 失败、`4` 个 citation 失败、`7` 个 agent status 失败。
- 评测中观察到两组 RAG draft 三次超时后 fallback；fallback 保护避免了进程中断，但没有解决答案覆盖率问题。下一步应先定位 timeout 与 synthesis 遗漏的共同根因，再重新评测；暂不修改 active/staged、测试环境或旧数据。
- 评测期间不并发启动第二个评测、不切换 active/staged、不修改测试环境或清理旧数据。
