# 科研文献问答运行手册

此页对应当前单实例多模态部署，命令见 [部署说明](../deploy/standalone.md)，配置使用 standalone.env.example。带日期的工作记录/历史演示稿不是当前部署步骤。

## 数据链路

上传注册 → parse → repair → canonicalize → semantic_split → contextualize → embed → index → activate。

解析版本保存配置/产物身份和阶段检查点，质量不合格不能激活。旧版本/索引/源文档保留；禁止为了 readiness 修改 status 跳过质量门。无 Redis 时同步执行；使用队列时另行运行串行 worker，API 成功不等于入库完成。

文字块/图块共用 Qwen3-VL-Embedding-2B 2048 维空间。固定本地 tokenizer 仅计数，不是第二个检索 embedding。更换 encoder/processor/loader 要新 shadow 重建核验，不给历史向量补当前身份。PostgreSQL + pgvector 先使用精确余弦，不宣称已有 ANN。

## 普通查询

/api/query 为原 RAG；/api/agent/query 是普通/复杂题的有界编排入口，stream 为 SSE。

首轮查询可 draft 直通，跨轮查询进入带会话上下文的综合路径；换题不继承不相关的视觉意图。上下文化只改检索 query，不改原始证据。显式预算即使与 schema 默认相等也优先于服务端默认。

检索包含 pgvector、词法和结构补证。strict profile 遇到无效向量/库/空间/版本错误必须阻断，不静默 JSON/SQLite 回退；正常空命中可说明缺证据，backend 如实记录。

表格值来自 canonical facts，coverage 按完整文档/版本/表格身份计算。视觉作答送受限目录真实像素，caption 不能代替读图。引用由可信 DB 身份重建，同名图表与补证编号不得串文档。

## 复杂题只读自适应 Agent

只在 complex_multi_hop + AGENT_ADAPTIVE_ENABLED=true 启用，普通题零额外 planner，multi_source_compare 不自动等于复杂题。

初次检索 → observation（证据/coverage/预算）→ 模型选择 retrieve / answer / finish / abstain → 受信校验 → 工具结果/新 observation → 收尾。

最多 3 次决策/2 次补证，不保证用满。实际查询由新 observation 决定；静态 ComplexPlan 仍称固定工作流。只记录动作/简短理由，不保存隐藏思考链。

模型不能填写 project/session/parse map、路径、SQL、shell 或任意工具；document focus 只能缩小冻结范围。补证保留初始合法 attachments，重验 canonical chunk/图表身份，整体合并 PreparedEvidence（像素/facts/coverage/引用），不是拼文本。

planner/answer/格式修复共用 ExecutionBudget、模型租约与单次格式修复池，为 verify/finalize 留额度。无进展、非法决策、服务失败、预算停止和证据不足分别标 stop_reason。SSE 关闭/断连取消同一预算，释放队列/租约。

## 验证与测评

静态/动态共用收尾，一份 agent 轮次/trace；JSON/SSE 同样报告 execution_mode、决策/停止、实际 backend、预算和结构验证。结构规则检查答案形状/引用，不证明事实正确，completed 也不是正确率。

事实、引用、适当拒答用逐题 bool/None 判分，未运行/无参考留空。新 20 题编排配对与原 50 题/SPIQA 论文测评分开，不能混称论文成绩。

## 运维边界

scripts/project_ctl.py 统一 PG→模型→API，精确验证 PID/start_time/cwd/exe/argv/socket inode。API/模型 loopback，图片只读 parsed/cache；stop 不停数据库/陌生进程，Redis/worker 不归 controller 管理。

API 与模型 Python 环境分离，不向 GPU 环境安装 API transformers<5。私有 .env、runtime、文献/题集、权重、dump、凭据、备份不进入 GitHub。本次不启动飞书/公网隧道、不训练。

代码单测、服务器跑通、质量提升是独立结论，只使用各自实际报告，不沿用历史成绩。
