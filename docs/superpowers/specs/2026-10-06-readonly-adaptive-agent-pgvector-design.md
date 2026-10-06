# 复杂题只读自适应 Agent 与个人服务器 pgvector 部署设计

日期：2026-10-06。状态：**用户已确认书面设计及实施计划，开始实施**。

## 1. 用户目标与成功条件

用户已选择：普通问题保留原路径；复杂题增加只读自适应循环；先在个人服务器完整跑通，再同步到有发布授权的 GitHub 项目；实际向量后端为 pgvector，移除会误导部署的旧文件。

不把静态 ComplexPlan 改名当作自主 Agent。自主性的验收依据是：同一问题在不同检索 observation 下，模型选择不同的下一步查询或停止；后续查询参数来源于新观察，而不是预先固定。

成功必须同时满足：

1. 个人服务器上的数据库、模型、普通问答、视觉问答和复杂题动态链路都实测通过；HTTP 200 或 mocked 单测不能代替这一条件。
2. 普通问题不开启额外规划调用，上一阶段视觉证据、会话、预算及取消保护保持有效。
3. 工具不能修改知识源、索引或模型，不能跨项目、跨会话、跨冻结解析版本获取证据。
4. 生产检索实用 pgvector 和同一 2048 维多模态 embedding 空间，不以 SQLite 或 JSON 回退冒充成功。
5. 只有验收过、去除敏感内容的代码差异才能推送；不覆盖原工作树未提交修改。

本次不新增训练，不修改原冻结 50 题，不顺带删除文献、数据库、模型、评测结果或备份。

## 2. 已实测的现场，而非历史推断

SSH：`root@connect.bjb1.seetacloud.com:37974`，已认证成功。

该端口返回的 ED25519 指纹与本机此前信任的同域旧端口完全相同；本次用既有主机信任别名连接，保持 StrictHostKeyChecking=yes，没有关闭主机校验。Windows ssh-keyscan 的失败另涉及其不支持协商到的密钥交换算法，不代表服务器拒绝执行权限。

服务器项目：`/root/autodl-tmp/llm_wiki_multimodal_20260922`，当前不是 Git checkout。

| 项目 | 本轮核查结果 |
| --- | --- |
| GPU | 单张 RTX 4090 D，49140 MiB，核查时显存占用为 0 |
| 数据盘 | 150G，已用约 80G，可用约 71G |
| PID 1 | bash，不是 Systemd；旧 Systemd user 双环境教程不适用 |
| PostgreSQL | 16.15；16/main down；pg_isready 无响应；存在遗留 socket |
| PG 控制文件 | 集群状态 in production，最近 checkpoint 为 10 月 3 日；尚未恢复启动，不能据此断言数据库损坏或完整 |
| pgvector | vector.so 与 0.8.6 扩展文件存在；库内扩展版本待数据库启动后 SELECT 核实 |
| API / 模型 | 127.0.0.1:18002 与 :18080 均 connection refused |
| 运行配置 | postgresql+psycopg；pgvector；qwen3-vl-embedding:2b；2048 维 |
| 生成模型 | qwen3-vl:4b；视觉模型同名；LoRA 文件存在，不代表正在启用或模型已加载 |
| 语料文件 | corpus 下 7 个 PDF、parsed 下 7 个文档目录；业务行数及激活版本待 SELECT 核实 |
| API 环境 | 项目 .venv，Python 3.11.16，FastAPI 0.141.1、Pydantic 2.13.5、SQLAlchemy 2.0.54 |
| 模型环境 | /root/autodl-tmp/embed_env，Python 3.10.8、torch 2.7.1、transformers 5.6.0 |
| schema 迁移 | alembic.ini 指向 src/app/db/alembic；迁移源码实际存在，并非根目录缺 alembic 就表示缺迁移 |
| GitHub | 现有 origin 为 zyhnosleep/knowledge-agent-server，main HEAD 可读取；用户本轮已明确确认使用该仓库并有代码发布权限 |

本地修复源码：`D:\项目复现\tmp\knowledge_audit_20261003`。

本地上一阶段报告：`D:\项目复现\output\knowledge_harness_repair_20261005`。全套测试曾为 2265 passed / 16 failed / 4 skipped，相关 17 文件为 894 passed / 0 failed / 1 skipped；这些不是服务器真实模型验收。

本轮 SHA256 比较：agent_executor、ai、rag_adapter、search 在服务器与本地修复快照不同；服务器没有新增 execution_budget.py；vector_store 一致。这确认上一阶段修复尚未上线。新发布需覆盖完整变更清单而非只传新增 Agent 文件。

## 3. 方案选择与范围

已按用户选择采用方案 B：

- A：继续固定 RAG 工作流，迁移代价小，但没有 observation 驱动的补证能力。
- **B：复杂题才启用只读自适应循环，复用现有 harness、模型客户端与检索服务。** 范围可控，能检验真实动态决策。
- C：所有问题自由 ReAct。增加普通查询延迟及越界风险，当前不采用。

不引入新的 Agent 框架，不开放 shell、任意 SQL、文件写入、任意网络或动态工具注册。外部网络开关也不能增加动态 Agent 的工具权限。

“只读”限定模型可调用的业务工具：不能写知识源、解析版本、向量或模型。应用原有的会话轮次、审计 trace 及明确的图片缓存属于基础设施记录，仍允许由受信应用代码写入，不声称整个请求零写入。

## 4. 入口、循环与预算

普通问答入口维持现有普通文本/视觉链路。`/api/agent/query` 与其 SSE 入口在服务端开关启用且 PolicyRouter 路由为 complex_multi_hop 时使用动态路径。multi_source_compare 不自动等同复杂题；跨文档多跳问题必须通过明确的复杂模式识别测试。

第一版服务端配置：AGENT_ADAPTIVE_ENABLED 默认 false；个人服务器验收时显式 true。AGENT_ADAPTIVE_MAX_DECISIONS 默认 3；AGENT_ADAPTIVE_MAX_SUPPLEMENT_RETRIEVALS 默认 2。这是上限，不承诺每次都用满。

循环顺序：

1. 校验当前会话与请求作用域，完成路由，冻结合法文档与 parse_version_map。
2. 初次检索，建立 PreparedEvidence 和可信证据身份集合。
3. 将问题、会话摘要、证据 observation、缺口、上次工具结果和剩余预算交给决策模型。
4. 严格解析动作，受信执行器校验参数后执行只读检索或基于当前证据作答。
5. 若允许且仍有缺口，用新 observation 再决策；有答案时也只能在预算内补证，不能无限改写。
6. 通过现有引用与结构校验结束；无充分证据则明确拒答/说明缺口。结构校验不宣称事实正确。

允许的决策为 retrieve、answer、finish、abstain。finish 只接受已有候选答案，不能携带自造答案、证据或引用；初次没有候选答案时不能 finish。abstain 返回简短、明确的证据不足说明。模型不给出或保存隐藏思考链，只保存动作和简短审计理由。

所有模型调用，包括 planner、格式修复、视觉回答和综合，都进入现有 ExecutionBudget。模型调用数、工具调用数、step、token、deadline、cancel 共用一套账本，格式修复共用既有单次额度，不另建重试池。候选答案生成后不为了“循环”无条件再调用模型。

保留 AgentConstraints 默认值和客户端显式限制；不因为启用自适应自动放大 max_steps、timeout 或 token。执行器在决定补证之前保留可行的 answer/verify/finalize 步骤额度；如果无法完成安全尾段，就提前停止并标记预算终止，不把截断结果标成 completed。真实复杂题验收可以显式配置较宽预算，但固定路径与动态路径必须使用同一预算。

异常策略：未知动作、额外字段、越权参数不执行；JSON 格式错误最多使用一次共享格式修复；仍不合法则停止并显示 decision_invalid。已验证的现有候选答案可以在有效预算内完成校验返回，但不隐式改走另一模型或另一索引。重复查询且没有新证据时停止，记录 no_progress；服务失败和预算终止与证据不足分别标记。

## 5. 组件与证据接口

新增小模块 `services/adaptive_agent.py` 管理状态与循环，`services/agent_decider.py` 管理预算内结构化模型决策；动作 schema 放入专门 schema 模块，避免继续扩大现有大执行器。AgentExecutor 只承担入口选择、现有会话/trace 接线与最终响应整合。

RequestScope 由执行器创建，包含 project、session、请求 document 限制、冻结解析版本和 embedding 身份；不由模型填写或覆盖。检索 query 长度限制为 1–1200 字符，limit 为 1–15。可选聚焦 document 只能缩小当前合法文档集合，不能扩大请求 document 范围。工具名不接受任意字符串映射到 Python 对象。

所有补证复用同一 RequestScope 和 parse_version_map；新激活版本不会改变正在处理的轮次。新增证据沿用 QueryService 的 canonical 身份重建与去重，并保持表格 facts、coverage、图片和引用为同一集合。禁止只追加 EvidencePack 文本而遗漏 PreparedEvidence 里的真实像素上下文。

决策 observation 只读，限制最多 15 个证据条目、每条摘录最多 450 字符、摘录总量最多 6000 字符；明确标记裁剪，不把未展示误称为缺失。内容包含文档/版本/图表身份、可见摘录、coverage、工具错误和预算。论文文本是非可信数据，即使包含“忽略规则/执行指令”也不能改变系统提示或权限。

视觉生成仍通过真实图片回答；planner 的文字观察不能代替读图，不能凭 caption 确认像素事实。图片路径必须由合法 canonical 资源解析，限定缓存/解析目录，校验 resolve 后的实际路径与符号链接；不能使用论文或模型输出提供的文件路径。

trace 与 SSE 记录 execution_mode、decision_index、动作参数摘要、工具结果、新增证据身份、预算及终止原因；不输出私钥、环境凭据、隐藏思考链或完整公司文档。普通路径不伪造 adaptive 标记。

## 6. pgvector 个人服务器部署

生产 profile 明确指定 PostgreSQL + pgvector。启动前核对 driver、扩展、vector 列类型/维度、embedding provider/model/revision/processor 身份以及激活 manifest。维度相同不代表模型空间相同；旧 2560 维索引与其备份不混入新查询。

生产 readiness 不允许配置缺失后静默选 SQLite，也不以 JSON fallback 宣称 pgvector 可用。pgvector 不可用、向量身份不一致或激活状态不一致时给出明确错误并阻断上线。SQLite 的测试 fixture 与兼容实现不是生产入口，第一版不为删它们而破坏大量无关测试；默认生产说明、环境模板和依赖分组必须消除误导。

2048 维 vector 当前没有证据表明存在 ANN 索引，不承诺 HNSW 加速；先用现有精确余弦检索核验排序和实际执行后端。不要未经验证为超过普通 vector ANN 维度上限的列建索引，也不要为加索引改变维度或混用 halfvec。

数据库恢复顺序：先核查 PGDATA 实际路径、占用和备份空间；对停止的集群保存可恢复的完整副本后，用 PostgreSQL 正常启动执行 WAL recovery。不得 initdb、删 postmaster.pid、pg_resetwal 或覆盖原库。启动失败先读诊断，不自动重建。启动成功后确认库内 schema、扩展、业务数据、active 指针与向量身份，再做新的逻辑备份；10 月 3 日的业务状态不能仅靠 9 月 22 日备份推断。

迁移先 introspect 当前 schema 与 Alembic revision，严禁无条件重跑 pgvector_2048_up.sql 或 down.sql。若需要新幂等迁移，必须先在备份恢复的隔离库测试；已有生产索引和解析数据保留。

启动入口收敛为一套容器可用的项目控制脚本：preflight/start/status/stop。按 PG → 本地模型 → API 顺序，精确校验 PID、命令参数、工作目录和端口归属，再做健康检查；不能只看 PID 存活，不用模糊 pkill，不杀其他任务，端口被非本项目占用时停止并报告。

API 与模型保持 loopback，数据库无公网暴露。当前 AUTH_ENABLED=false 仅能用于 loopback + SSH 转发；公网发布属于另行的认证与网络配置，不在本次默认范围。图片根目录收紧到 runtime/data/parsed 与 runtime/data/cache，若确需其他缓存目录则逐项加入已验证白名单，不默认放宽到 /root/autodl-tmp。

API 和模型环境保持分离：不要将项目 transformers<5 依赖直接装进已运行 Qwen3-VL 的 transformers 5.6.0 模型环境。记录实测版本后建立各自可重建的依赖清单，不能为通过单测盲目降级模型库。config.py 的重复 Ollama 字段作同值去重，保留 aliases 和已有调用契约。

## 7. 精确清理范围与保留项

以下是**待实施前再次核对的清理清单**。先保存哈希、引用清单和可恢复副本，再从活跃发布树移除；归档不进入 GitHub。不是按文件名批量删目录。

| 目标（服务器项目相对路径） | 处置与理由 |
| --- | --- |
| deploy/internal-pilot.md | 移除旧双 GPU/Systemd 部署入口，改由 deploy/standalone.md 说明当前 pgvector 拓扑 |
| docs/deploy-no-docker.md | 移除旧双环境/2560 维教程，README 与 runbook 改链到唯一当前部署说明 |
| .env.development.example、.env.test.example | 当前发布树移除，改成单实例 pgvector 模板；同步改造仍断言旧拓扑的部署及配置测试，保留认证/隔离测试覆盖 |
| deploy/systemd/knowledge-agent-dev-api.service、knowledge-agent-test-api.service | 移除当前无法使用的双环境 API units |
| deploy/systemd/knowledge-agent-dev-ollama.service、knowledge-agent-test-ollama.service | 移除错误的 GPU 0/1 与旧文本模型部署 units |
| deploy/systemd/knowledge-agent-dev-worker.service、knowledge-agent-test-worker.service | 移除旧双环境 units；保留实际入库 worker 功能及其当前启动说明 |
| deploy/systemd/knowledge-agent-dev-feishu-bot.service、knowledge-agent-test-feishu-bot.service | 移除公司部署专用的服务模板；本轮不顺带拆除集成业务模块 |
| deploy/systemd/knowledge-agent-redis.service、knowledge-agent-test-tunnel.service | 移除不适用容器的 service 模板；Redis 是否必需由当前入库配置与真实入库验收决定，不直接删其依赖 |
| deploy/cloudflared/config.yml.example | 在确认无活跃引用后移除旧公网隧道示例；不影响用户既有实际隧道或凭据 |
| scripts/launch_multimodal_pilot.py | 移除硬编码旧 8B 模型、/generate 旁路及实验路径的启动器，替换为统一项目控制入口 |
| scripts/serve_qwen_multimodal.py | 确认无新入口/测评引用后移出当前发布树；保留在离线历史归档，而非与当前服务并列 |
| README.md、docs/project-runbook.md、deploy/multimodal-pilot.env.example | 重写当前环境、模型协议与后端说明，不整份删除使用说明 |

不直接删除 `multimodal_rag.py`、serve_multimodal_app.py、run_multimodal_pilot.py、verify_multimodal_pgvector.py：当前验收脚本/测试仍引用旧旁路。先迁移必要的检查到普通/Agent 正式接口，停止把旧旁路当主入口；只有引用迁移完且历史评测可追溯后才能从活跃发布树移出，保留独立的 pgvector 验证。

保留：SQLite → PostgreSQL 的一次性迁移工具（标注只用于迁移，不是推荐运行后端）、pgvector 迁移和验证工具、schema 历史、原始文献、parse/cache、数据库和 SQLite 历史备份、2560 维回退表、模型及 LoRA、冻结题集及测评产物、上一阶段修复报告与原始源码备份。

服务器 `.env`、`.env.bak*`、`.env.sqlite-20260922.bak` 是私有配置/恢复资料，不发布，不为清理误导而不可恢复删除。旧 probe 与 audit 文件逐项核对引用和复现价值，不能把名称含 probe/test 的文件统一当垃圾。

## 8. 验收矩阵与发布门槛

实施采用 TDD。先有可复现失败，再修实现，不能靠删测试通过。新增覆盖：结构化决策、非法工具/参数/注入、动态补证改变行为、无进展、跨项目/会话/版本隔离、像素与引用对应、共享预算/格式修复、取消/超时、SSE/JSON 一致，以及普通路径零额外 planner。

服务器真实验收分四级：

1. 基础设施：PG recovery 后业务/schema/vector 身份核对；实际 PGVectorStore 查询与手算余弦距离、合法过滤后排序一致；服务 readiness 不接受静默回退。
2. 普通链路：文本首轮、跨轮查询、换题、表格数值、视觉像素问题、无证据拒答；trace 证明用了预期模型、图片和 pgvector。若项目支持上传入库，则补充独立小文档的 parse→canonical→embed→index→activate→query，不修改原冻结语料。
3. 动态链路：新增约 20 题，与原 50 题分开，覆盖同文档多跳、图表交叉、跨文档矛盾、缺证据、跨轮与注入。固定 base 模型和相同索引/解析版本/预算，比较固定工作流与动态路径，避免把 LoRA 差异混入编排差异。
4. 迁移可用性：脱离 SSH 后服务存活、精确 stop/start 不误停其他服务、重复 start 不重复占端口、重启恢复入口和错误日志可定位。

新题集保存问题、合法文档/图表身份、参考关键事实和不可回答条件。主指标为关键事实正确率、引用对应性、适当拒答；辅指标为补证成功、越权拦截、调用数、token、延迟、终止原因。结构校验 passed 不是正确率，自建 20 题成绩不是 SPIQA 论文成绩。未运行的单元格保持空，不补造数据。

硬门槛：全部新增安全/预算/证据身份测试通过；普通已确认用例不退化；真实动态补证至少展示多个 observation 导致动作改变的案例；真实链路无索引空间混用或静默 JSON/SQLite 回退。答错题报告而不掩盖，系统跑通与质量提升分别结论。若动态收益不足，可部署保留关闭开关的实现并如实说明，不声称已提升准确率。

全套原有 16 个失败逐项按根因处理：缺文件的部署断言随当前迁移修复；pytest 异步依赖与 FastAPI introspection 按环境兼容处理；历史 db=None 探针保留明确边界。必须给出本次实际 pass/fail/skip 和失败原因，不能沿用旧数字冒充本次结果。

## 9. GitHub 安全同步与回退

现有原工作树 `D:\LLM_wiki\.worktrees\internal-pilot` 很脏；不覆盖、reset、全量 add 或提交其所有用户修改。完整源码以服务器实况为部署基底，叠加本地 10 月 3 日快照已审查的变更及本次改造；GitHub 集成在单独工作副本/分支进行，对每个新增、修改、删除文件做来源和 diff 审查。

用户本轮已明确确认目标为 zyhnosleep/knowledge-agent-server，并确认具有代码发布权限。服务器端完整验收先于任何 push；默认只推独立变更分支，不直接覆盖 main、不 force push、不自动合并 PR。若需新分支名称，实施计划中明确后确认。

发布白名单只包括授权代码、测试、迁移、脱敏模板、当前部署文档和必要依赖清单。排除私有 .env、SSH 私钥、公钥授权配置、密码/token、runtime、论文/题集原文、数据库 dump、模型权重、公司业务数据、压缩包、egg-info、pycache 及机器特定日志。使用 git diff/staged inventory 与敏感模式扫描复核，扫描不代替文件来源审查。

服务端发布前保留当前完整源码/config 副本、最新 DB 恢复资料、版本清单与 SHA256；原数据库保持独立，不用代码发布覆盖 runtime。回退先关 AGENT_ADAPTIVE_ENABLED 恢复固定路径；必要时按本次完整发布备份恢复代码/config。上一阶段部分测试没有修改前备份，不能宣称其报告可以完整一键回滚。

快照不是 Git 仓库，本文只保存为本地设计，尚未 commit 或 push。正式集成时可以提交已批准设计，但不能为满足文档提交要求去污染原脏工作树。

## 10. 当前状态与后续门禁

- [x] 确认用户方向与成功条件。
- [x] SSH 指纹比对和认证；服务器、源码、后端及启动拓扑只读核查。
- [x] 设计草案、可恢复清理范围、真实验收与发布安全边界。
- [x] 用户确认 GitHub 目标及代码发布授权。
- [x] 用户审阅并确认本书面设计（本轮回复“继续”）。
- [x] 依据已批准设计编写实施计划，由用户确认本对话内执行方式（回复“执行”）。
- [ ] 实施本地 TDD、环境迁移、精确清理和服务端真实验收。
- [ ] 验收后审查发布差异、同步已授权 GitHub 分支并报告证据。

设计确认前不新增产品代码、不安装产品依赖、不恢复/变更数据库、不删除文件、不部署或推送。
