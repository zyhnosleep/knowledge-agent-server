# Agent 质量运营与实验平台：第一阶段交付

日期：2026-09-21  
状态：产品与技术设计完成；DeepSeek 与远程 embedding API 已完成真实 smoke test，
开发 MaaS endpoint 当前选用 `qwen3.7-text-embedding`（1024 维）；尚未运行全量文献基线。
目标：在现有 Knowledge Agent 上建立反馈、案例、评测、实验与发布判断闭环。

## 1. 产品定义

主要用户是负责 AI 应用质量的产品负责人，协作用户是算法/后端工程师与人工评测者。科研用户在现有问答界面提供反馈。

核心任务：产品负责人收到“回答不可靠”的反馈后，能够找到对应回答与证据、确认失败原因、建立回归用例，并比较改进版本是否值得发布。

需求假设尚未经过用户访谈验证：反馈缺少上下文、复现困难、版本比较不一致是优先痛点。首批试用需验证这些假设，不将其写成已完成调研。

第一版成功条件：至少一个真实失败案例经过反馈、审核、回归用例、实验对比与人工发布判断的全流程；所有结论可追溯到固定版本的数据和结果。目标值与实际结果分别存储。

## 2. 当前能力审计

审计代码：codex/internal-pilot，提交 aeb2a99。此处为静态审计，不证明服务器健康或测试通过。

| 能力 | 已核实证据 | 实施决定 |
|---|---|---|
| Agent 请求、会话、引用、trace | src/app/api/agent_routes.py | 复用现有接口与用户归属校验 |
| trace 的请求、答案、模型、耗时和 token 字段 | src/app/models/records.py 的 AgentTraceRun | 关联反馈；扩展逐调用 usage，先验证现有 token 是否为实际计量 |
| 质量报告 API | src/app/api/quality_routes.py | 保留，新增产品质量视图，不把入库成功率当问答正确率 |
| 聊天、文档库、运行页面 | src/app/static/index.html | 保留样式与信息结构，追加反馈入口和质量页面 |
| Agent 评测脚本 | scripts/evaluate_agent_full.py | 复用评分逻辑前核实其语义；增加认证、版本与跨轮用例支持 |
| 文本生成和 embedding | search.py、pipeline.py、agent_synthesizer.py 直接依赖 OllamaClient；已新增 DeepSeek/Qwen API 传输适配 | 后续继续把调用审计与 provider 接口下沉，DeepSeek 负责文本生成 |
| 反馈数据模型与业务接口 | 相关 models/schemas/api 中未发现 | 新增 |

历史 23/30、30/30、SciFact 指标只作历史材料，不作为 DeepSeek 版本基线。ui-references 目录中的图片是设计参考，不作为当前界面运行验收证据。

## 3. 第一版范围与优先级

P0：DeepSeek 文本生成适配、可靠调用记录、回答反馈、案例审核、版本化评测集、两版本对照、发布判断、访问隔离。

P1：质量概览、CSV/JSON 导出、案例优先级与负责人、手工任务成功标注。

后续：Langfuse/Promptfoo 集成、线上随机 A/B 分流、自动聚类、自动优化 Prompt、多应用统一 SDK。第一版的离线对照不称作线上 A/B 实验。

发布判断只输出通过、不通过、待补证据；部署仍由现有流程执行。

## 4. DeepSeek 接入与运行边界

目标运行方式为服务端调用 DeepSeek API。服务部署地点与运行时 API 密钥后续配置。浏览器只访问应用后端。

三个独立能力接口：TextGenerator、Embedder、DocumentParser。DeepSeek 只承担已验证支持的文本生成能力；不假设其提供当前项目需要的 embedding 或视觉解析接口。

embedding：开发 MaaS endpoint 未暴露原计划的 Qwen/Qwen3-Embedding-4B，已验证可用的完整模型为
`qwen3.7-text-embedding`，API 返回 1024 维；因此开发环境不复用旧的 2560 维向量，必须按新的
provider/model/dimensions 配置创建新索引并评测后激活，不能混用新旧向量。若后续部署真正的
Qwen3-Embedding-4B（2560 维），同样必须生成新的 ingestion identity。PDF 解析确定使用 MinerU；
扫描件的 OCR/视觉能力随 MinerU 后端配置验证。

配置契约（API 适配已实现；真实凭据与服务商 endpoint 仍由部署环境提供）：

| 配置 | 用途 |
|---|---|
| GENERATION_PROVIDER=deepseek | 选择文本生成实现 |
| DEEPSEEK_API_KEY | 服务端环境变量/密钥管理；不写前端、日志或实验配置 |
| DEEPSEEK_BASE_URL | 实施时按官方文档核对 HTTPS endpoint |
| DEEPSEEK_MODEL | 明确模型 ID，不依赖“最新模型”别名假设 |
| GENERATION_TIMEOUT_SECONDS | 单次调用超时，受整个请求剩余时间约束 |
| GENERATION_MAX_RETRIES | 传输重试初始最多 1 次，与答案质量重试共用请求预算 |
| GENERATION_MAX_OUTPUT_TOKENS | 输出上限 |
| GENERATION_CONCURRENCY | 应用级并发上限，初始 1 |
| EXPERIMENT_MAX_COST_CNY | 实验估算费用上限；缺价格配置时禁用付费批跑 |
| EMBEDDING_PROVIDER=openai-compatible | 选择远程嵌入实现 |
| EMBEDDING_API_BASE_URL / EMBEDDING_API_KEY | 远程 embedding API 地址与服务端密钥 |
| EMBEDDING_API_MODEL=qwen3.7-text-embedding | 固定模型标识；按所选服务商实际 ID 配置 |
| EMBEDDING_API_TIMEOUT | 嵌入调用超时 |
| EMBEDDING_DIMENSIONS=1024 | 向量索引维度；与现有索引不一致时必须新建索引 |

实施前核对官方模型 ID、结构化输出、流式协议、usage 字段与价格。不预填未验证价格。现有 Ollama 的 format=json_schema、模型卸载、GPU 租约不能直接应用到 DeepSeek。

失败处理：401/403 配置失败不自动重试；429/可恢复 5xx 在总时限和预算内退避重试；网络超时保留失败与用量未知状态；响应格式错误单独记录。等待、限流、超时、取消均向用户返回可解释状态。取消请求立即停止后续调用，不能承诺服务商端已停止计费。

调用审计字段：request_id、trace_id、attempt_id、provider、model、operation、prompt_hash、开始/结束时间、status、error_code、usage、usage_source、pricing_version、estimated_cost。只保存必要证据；用户上传内容和历史会话中被选用的片段会发送给 API 服务商，界面和部署说明同步改掉“全本地、不出内网”的表述。

费用规则：按官方返回的输入/输出及缓存分类 token 和版本化单价估算，避免缓存 token 重复计费；没有 usage 时记 unknown，不记 0。估算费用与服务商账单不等价。重试逐次计量；预算采用调用前最坏情况预留与结束后结算。

## 5. 页面草图与状态

```text
现有回答
  回答正文 [引用1] [引用2]
  [有帮助] [无帮助]  [补充说明]
  无帮助 → 引用不支持 / 事实错误 / 信息不全 / 太慢 / 其他
  提交状态：保存中 → 已保存（可修改）；失败时保留输入并重试

质量概览
  项目 | 时间范围 | 应用版本 | 作答模式
  请求数  完成率  超时率  P95耗时  估算费用/未知用量数
  反馈覆盖率  负反馈率  已审核断言支持率（显示样本数）
  高频失败原因 → 对应案例列表

失败案例
  状态/原因/优先级筛选 | 分页
  问题摘要 | 用户原因 | 人工归因 | 状态 | 更新时间
  案例详情：原回答与引用 | 执行阶段 | 调用耗时 | 人工审核记录
  [标记待修复] [提交验证] [加入评测集] [关闭并说明理由]

评测与实验
  数据集版本 | 基线配置 | 候选配置 | 用例数 | 费用预估
  [启动实验] [取消]    完成/失败/未执行计数
  同题比较：答案A | 答案B | 原文依据 | 自动检查 | 人工评分
  结果：改善/退化/持平/不可比 | 质量门槛 | 发布判断与理由
```

空列表说明如何产生数据；无权限不展示其他用户内容；模型未配置显示配置缺失；实验中断可查看已完成结果。未知指标显示“暂无数据/待审核”，不能显示 0%。界面沿用现有样式，第一版不迁移前端框架。

## 6. 数据与接口契约

所有记录带 project_id、应用标识和访问范围。第一版普通用户仅能对自己的回答反馈；质量管理者仅查看被授权项目。权限角色与现有认证连接，不能因为登录成功就开放全部 trace。

| 对象 | 核心字段/约束 |
|---|---|
| AnswerFeedback | trace_id、author_id、rating、reason、note、timestamps；同用户同回答一条当前反馈，修改保留审计 |
| QualityCase | feedback_id、trace_id、分类、优先级、负责人、状态、审核说明；自动创建需幂等 |
| DatasetVersion | dataset_id、version、content_hash、split、创建人；发布后不可修改 |
| EvaluationCase | 场景、问题/跨轮脚本、证据快照、期望行为、评分规则、来源案例；审核后入集 |
| Experiment | 数据集版本、基线/候选配置快照、代码版本、语料/parse_version、预算、状态、时间 |
| ExperimentResult | experiment_id、case_id、variant、attempt、答案/引用快照、状态、耗时、人工/自动评分；唯一键防重跑覆盖 |
| ModelCall | trace_id、attempt_id、usage、错误类别、模型/提示词版本、费用与计量来源 |
| AuditEvent | 操作者、对象、动作、前后状态摘要、时间 |

反馈原因属于用户主观描述，人工归因属于审核结果，二者分开。trace 默认可能被定期清理，纳入案例/评测时保存必要快照和来源哈希；删除原数据时按删除策略处理快照，不能仅靠悬空 trace 外键。

拟新增接口前缀 /api/quality-ops：

- PUT /feedback/{trace_id}：幂等提交/修改反馈。
- GET /cases、GET /cases/{id}、PATCH /cases/{id}：分页筛选和状态变更。
- POST /datasets、POST /datasets/{id}/versions：建立审核后数据集快照。
- POST /experiments：明确启动付费对照实验，支持幂等键与预算预留。
- GET /experiments/{id}、POST /experiments/{id}/cancel：查询/取消后台任务。
- GET /metrics：按项目、窗口、版本、模式返回指标及分母。

案例状态：new → triaged → in_progress → ready_for_validation → resolved；验证失败回 in_progress；无效反馈以 dismissed 关闭并写理由。实验状态：queued → running → completed / failed / cancelled；部分结果不能冒充完整通过。

## 7. 指标与实验方法

线上请求与离线实验使用独立流量标签，不互相污染。

| 指标 | 固定口径 |
|---|---|
| 负反馈率 | 窗口内收到反馈的回答中，当前反馈为负的占比；同时显示样本量 |
| 反馈覆盖率 | 收到有效反馈的回答数 / 可反馈回答数 |
| 引用支持率 | 人工审核中，被证据支持的关键断言数 / 已审核关键断言数；展示不可判定数量 |
| 任务成功率 | 人工确认成功的任务数 / 有效任务数，不能用 HTTP 200 代替 |
| 超时率 | 超时请求数 / 已进入服务的请求数；取消另列 |
| P95 耗时 | 完成请求端到端耗时的 nearest-rank 95分位；同时显示超时数及最长等待 |
| 费用 | 已计量估算总费用、未知用量请求数；未知不当零 |

首批 40 条审核用例目标分布：事实 10、表格 8、跨文档 6、跨轮 6、证据不足 6、异常请求 4。30 条开发集、10 条验收集，按来源文献/问题族分组隔离；该规模只支持探索性结论。样本必须有来源与期望行为，不能凑题数。

对照固定同一语料快照、嵌入模型、检索配置、会话输入和评分规则。先比较 Prompt/作答策略这一变量；模型变化另开实验。跨轮用例保持组内 session，一组执行完再评分。记录温度等采样参数与执行时间；小样本变化不能称为统计显著。

初版发布门槛（暂定，基线后校准）：运行覆盖全部验收题；无新增严重事实/引用错误；人工任务成功率不低于基线；超时率不高于基线；P95不超过基线的1.2倍；估算费用不超预算。存在未审核、用量未知影响预算判断、语料版本不一致时，结论为待补证据。关键词检查只叫关键词覆盖，不叫正确率。

## 8. 开发任务与验收

| 顺序 | 任务 | 验收 |
|---|---|---|
| T0 | 基线与环境盘点 | 记录代码/索引版本、现有测试结果、API依赖；历史与新报告分开 |
| T1 | 文本生成provider与DeepSeek适配 | 模拟正常、401、429、5xx、超时、畸形JSON、缺失usage；生成与嵌入解耦 |
| T2 | 逐调用审计与预算 | 重试计数/计费正确，未知用量不归零，密钥不落库 |
| T3 | 反馈接口与聊天入口 | 保存/修改/刷新成功，重复提交幂等，越权与CSRF测试通过 |
| T4 | 案例管理 | 反馈到案例幂等；状态流转与审核记录完整 |
| T5 | 评测集版本与后台实验 | 数据集不可变，跨轮保留上下文，任务可取消，失败/未执行不丢失 |
| T6 | 指标与发布判断 | 固定小样本校验分母、P95、费用、未知值与门槛；逐题可追溯 |
| T7 | 联调与真实试用 | 一条真实负反馈完成全流程；产出两轮实验与人工复盘 |

预计第1–2天 T0/设计，第3–4天 T1/T2，第5–6天 T3/T4，第7–9天 T5，第10天 T6/联调，第3周 T7。API与嵌入/解析环境未就绪期间完成模拟契约测试；真实基线与试用顺延。

## 9. 本次完成与下一步

已完成：代码能力审计、PRD、页面草图、数据模型、接口契约、指标口径、DeepSeek迁移边界、任务与验收拆分。

尚未完成：provider实现、业务页面、数据库迁移、测试执行、在线DeepSeek调用、服务器部署、真实基线与用户验证。

下一步先执行 T0/T1；在隔离开发环境配置服务端密钥和嵌入/解析服务后，先用一条公开文献问题验证连接与usage，再在明确的费用上限内执行基线。个人作品演示使用可公开文献与可使用代码；当前实习数据不默认进入外部API。
