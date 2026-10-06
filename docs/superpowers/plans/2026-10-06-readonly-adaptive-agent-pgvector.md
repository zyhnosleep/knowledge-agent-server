# 只读自适应 Agent 与 pgvector 个人部署 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 用户已选择在本对话执行：主智能体直接实施，最后一次独立全分支审查；不另开会话或并行派实现者。

**Goal:** 在个人服务器跑通 pgvector 文本/视觉 RAG 与复杂题只读自适应循环，清理误导部署入口，验收后将授权代码同步 GitHub。

**Architecture:** 保留普通问题的现有工作流，只将 complex_multi_hop 接入 observation 驱动的受限状态机。复用 PreparedEvidence、canonical 校验、ExecutionBudget 和模型租约；生产部署单独使用严格 pgvector profile 与容器可用的统一启动入口。数据库和模型与代码发布分离，先备份、再实测、最后推独立分支。

**Tech Stack:** Python 3.11+ API、FastAPI、Pydantic 2、SQLAlchemy 2、psycopg 3、PostgreSQL 16 + pgvector 0.8.6、现有 Ollama 协议 Qwen3-VL 模型服务、pytest；模型 Python 3.10.8 环境保持独立。

**Spec:** [已确认设计](../specs/2026-10-06-readonly-adaptive-agent-pgvector-design.md)。用户已回复“执行”，本计划已批准并开始实施。

## Global Constraints

- 普通问题零额外 planner；只有 complex_multi_hop + AGENT_ADAPTIVE_ENABLED=true 启用循环。
- AGENT_ADAPTIVE_ENABLED 默认 false；MAX_DECISIONS 默认 3；MAX_SUPPLEMENT_RETRIEVALS 默认 2。
- 检索 query 1–1200 字符，limit 1–15；observation 最多 15 条、每条摘录 450 字符、摘录总量 6000 字符，标记裁剪。
- 只允许 retrieve、answer、finish、abstain；禁止 shell、任意 SQL、写知识源、动态工具注册及任意外网。
- 补证使用同一项目/会话/document 约束、冻结版本和 embedding 身份，不能扩大范围；视觉回答仍送真实像素。
- planner、answer、格式修复共用 ExecutionBudget 和单次格式修复额度；不得自动扩大用户预算或把跳过校验标为通过。
- 生产 PostgreSQL + pgvector + qwen3-vl-embedding:2b + 2048 维；不以 SQLite/JSON 回退冒充 pgvector。
- 禁止 initdb、pg_resetwal、删 postmaster.pid、覆盖原数据库、盲跑升降维 SQL；不得混入旧 2560 维空间。
- API/模型监听 loopback；关闭 auth 时不得公网暴露；图片白名单限定解析/缓存实际目录。
- 不在原脏工作树或其 .venv 安装/覆盖；API/model 依赖分离，不降级现有模型 transformers 5.6.0。
- 不训练、不修改原冻结 50 题，不删除语料、模型、历史索引、备份或既有评测。
- 服务器验收先于 GitHub push；用户已授权 zyhnosleep/knowledge-agent-server；不推 main、不 force push、不自动合并。

## Review Focus

1. 客户端显式提交与 schema 默认值相等的预算：仍必须尊重显式值，不能被服务器配置替换（Task 6）。
2. 补证期间新增文档、空版本映射或激活指针切换：只能访问开始时冻结的合法集合（Task 4）。
3. 新证据合并后引用重新编号、同名图表跨论文：像素、facts 与最终引用身份必须一致（Task 4、6）。
4. PID 文件过期/复用、端口被非本项目占用、并发 start：不能误停其他服务或报告假成功（Task 7）。
5. 本地模型返回自然语言/半截 JSON 或同维度不同 embedding 身份：明确拒绝决策/部署，不绕过校验（Task 2、3）。

## 工作位置、文件边界与阶段

实施工作副本：`D:\项目复现\tmp\knowledge_agent_delivery_20261006`。通过 using-git-worktrees 在原 repository 创建隔离 worktree，若原生工具不支持此项目则使用 Git worktree fallback；预先核查目标不存在，存在时先确认身份并复用，不覆盖。基线 origin/main 在设计核查时为 `77d04bab2111c311848fe7b67b6f865111647207`，执行时重新核对最新 ref；分支名 `feat/readonly-adaptive-pgvector`，若同名已有其他工作则报告冲突，不 reset 或覆盖。

服务器工作项目：`/root/autodl-tmp/llm_wiki_multimodal_20260922`。私有验收/备份目录：`/root/autodl-tmp/knowledge_agent_delivery_20261006`；先核对绝对路径及剩余空间。报告目录：`D:\项目复现\output\knowledge_adaptive_delivery_20261006`。所有报告/评测/备份默认不发布。

主要新文件职责：

- `src/app/schemas/adaptive_agent.py`：严格动作与有限 observation 数据模型。
- `src/app/services/agent_decider.py`：受预算与租约保护的结构化模型决策。
- `src/app/services/adaptive_agent.py`：不可扩权的 RequestScope、状态机与 terminal outcome。
- `src/app/services/runtime_contract.py`：生产数据库、embedding/激活身份核验；只读检查，不自动修库。
- `scripts/project_ctl.py`：容器内 preflight/start/status/stop、PID/端口识别和就绪检查。
- `scripts/verify_standalone.py`：普通/视觉/Agent 接口及实际 pgvector 端到端验收。
- `scripts/evaluate_adaptive_agent.py`：固定工作流与动态链路配对实验，输出未填造的指标。
- `deploy/standalone.env.example`、`deploy/standalone.md`：唯一当前部署配置与说明。
- `requirements-model.lock.txt`：记录实测模型环境依赖；不与 API 的 pyproject 混装。

现有修改集中于 config、search/RAGAdapter、AgentExecutor/agent_routes、vector_store、model readiness 和必要启动服务协议；不重构无关前端或整套入库系统。

本计划共 9 个任务：先完整源码/保护资料，再本地 TDD，最后服务器恢复验收与安全同步。Task 1–7 的提交只保留本地，Task 9 才 push。

命令约定：以下本地 `python` 均指工作副本 `.venv/Scripts/python.exe`，PowerShell 使用 `& .\.venv\Scripts\python.exe`，不能用当前 PATH 上的 MSYS Python；Task 1 建环境前指定 `D:\LLM_wiki\.worktrees\internal-pilot\.venv\Scripts\python.exe`。服务器 API 测试使用现有项目 `.venv/bin/python`、cwd/PYTHONPATH 显式指向 staging；模型进程仅使用 embed_env/bin/python。JUnit 与日志路径统一使用报告目录的绝对路径。

### Task 1: 完整发布工作副本、来源清单与可恢复备份

**Files:** 全部操作限隔离 worktree；恢复 Git 基线的 pyproject.toml、README.md、alembic.ini、`src/app/db/alembic/` 等包装。叠加服务器真实 `src/app/` 与当前 `scripts/serve_local_models.py` 的必要差异，再按上一阶段 change-ledger 的 current_sha256 精确叠加 15 个改动文件（visual_evidence.py 未改）。记录 `output/knowledge_adaptive_delivery_20261006/provenance.json`、`baseline.xml`、`rollback-inventory.json`，不发布。

**Interfaces:** 产出完整工作副本、原始/修复/新代码来源与 SHA256、测试失败基线，以及服务器源码/config 和停止 PG 集群的备份路径；不产生新业务接口。

- [ ] 核查 worktree、ref、AGENTS/CONTEXT/ADR，建立隔离分支；读原项目只用于来源/解释器，不写其文件。
- [ ] 审核服务器每个待复制应用文件的 diff；机械复制只允许授权源码，不递归复制 .env、runtime、文献、权重、egg-info/pycache。校验所有修复 ledger 哈希，确认工作副本导入的 app.__file__ 位于该副本。
- [ ] 核查服务器 PGDATA 为 `/var/lib/postgresql/16/main`、集群确实停止、无外置 tablespace/数据路径遗漏；统计空间后保存冷备份及源码/config 副本，权限 0700/0600。若发现外置 tablespace，把它纳入同一可恢复快照再继续，不能只复制符号链接。不要删除遗留 PID/socket。
- [ ] 以原 `.venv/Scripts/python.exe` 只作为解释器在副本执行 `-m pytest -p no:cacheprovider --junitxml=D:/项目复现/output/knowledge_adaptive_delivery_20261006/baseline.xml`，设置 PYTHONPATH 为副本 src、PYTHONDONTWRITEBYTECODE=1、不读取原工作树 .env。记录实际结果，核对上一阶段 16 类失败，不将基线失败当新改造成功。
- [ ] 创建副本自己的 `.venv`；从 API pyproject 安装依赖，dev 加 `pytest-asyncio>=0.23,<2`，后续不改原解释器依赖。记录 Windows/API 与 Linux/API 的版本，模型环境仅只读盘点。本地提交审核过的源码基底与设计/计划，使用逐文件 git add，不全量 add。

### Task 2: 严格 pgvector 运行契约与部署配置

**Files:** Create `services/runtime_contract.py`、`tests/test_runtime_contract.py`、`deploy/standalone.env.example`；Modify `core/config.py`、`services/vector_store.py`、`services/search.py`、`services/model_readiness.py`、`scripts/serve_local_models.py`、`pyproject.toml`。

**Interfaces:** `EmbeddingIdentity` frozen dataclass：provider/model/revision/processor_hash 为 str，dimensions 为 int。`RuntimeContractError(reason: str)` 不含凭据。`check_pgvector_contract(db: Session, settings: Settings) -> dict[str, Any]` 核对 driver/extension/vector 列/激活 manifest。`check_embedding_contract(db: Session, identity: EmbeddingIdentity, parse_version_map: Mapping[str,str]) -> None`；模型服务新增只读 `GET /api/embedding_identity` 提供实际 loader/权重/processor 身份，不因健康检查运行推理。`VECTOR_STORE_STRICT` 默认 false，standalone 明确 true；严格模式失败不回退，空命中仍允许正常词法补充且元数据如实标记。

- [ ] 写 `test_strict_profile_rejects_sqlite_without_json_fallback`、`test_same_dimensions_different_model_revision_blocks_ready`、`test_missing_identity_is_unverified_not_backfilled`：断言错误原因、JSON 检索未调用、readiness 非 ready。普通空检索不伪造配置错误。

```python
assert json_fallback.call_count == 0
assert readiness["status"] != "ready"
assert configured_identity.dimensions == runtime_identity.dimensions == 2048
assert configured_identity.revision != runtime_identity.revision
```

- [ ] 执行 `python -m pytest tests/test_runtime_contract.py -q` 确认 RED，保留日志。
- [ ] 实现上述只读核验及实际后端 trace；同值去重 config.py 的重复 Ollama 字段，保持 alias；新生产模板明确 PostgreSQL/pgvector/2048/loopback 和 base 别名。SQLite 可选依赖移入测试/兼容 extras，保留测试功能，不卸载用户环境。
- [ ] 运行该测试及现有 `test_vector_retrieval.py`、模型就绪/配置测试，确认无静默回退或别名契约回归。若旧向量缺可靠 revision 身份，阻断 ready；不得用当前权重指纹直接给历史向量补身份。修复须在隔离重建/校验后才激活，并保留旧索引；需要超出既有迁移能力时报告新设计需求。
- [ ] 提交该独立变更及测试；依赖锁采用实测版本，不将 API transformers<5 装进模型环境。

### Task 3: 结构化动作与预算内决策模型

**Files:** Create `schemas/adaptive_agent.py`、`services/agent_decider.py`、`tests/test_agent_decider.py`；Modify `core/config.py`。

**Interfaces:** `AdaptiveDecision` strict Pydantic：action 为 retrieve/answer/finish/abstain；reason 最多 240 字符；retrieve 才允许 query/document_id/limit，query 1–1200、limit 1–15，其余动作禁止携带答案或工具参数，extra=forbid。`AdaptiveObservation` 包含 question、conversation_summary、evidence、coverage_status、truncated、candidate_answer、last_tool_error 和 budget；evidence 为具有 canonical 身份的有限 dict 列表。`build_observation(*, question: str, conversation_summary: str, prepared: PreparedEvidence, candidate: QueryResponse|None, budget: ExecutionBudget, last_tool_error: str|None=None) -> AdaptiveObservation` 负责按指定阈值裁剪。`AgentDecider(client: OllamaClient, runtime: ModelRuntime)`；`decide(observation: AdaptiveObservation, target: InferenceTarget) -> AdaptiveDecision`。

- [ ] 写失败测试，至少断言：`action="shell"`、伪造 answer、额外 project_slug、bool limit、query 长度 1201 均拒绝；450/6000/15 的观察裁剪被标记；论文里要求执行 shell 不改变 system prompt 或工具白名单。

```python
with pytest.raises(ValidationError):
    AdaptiveDecision(action="shell", query="whoami")
assert len(observation.evidence) <= 15
assert max(len(item["excerpt"]) for item in observation.evidence) <= 450
assert sum(len(item["excerpt"]) for item in observation.evidence) <= 6000
assert observation.truncated is True
```

- [ ] `python -m pytest tests/test_agent_decider.py -q` 得到 RED。
- [ ] 实现严格 schema 与提示词：非可信 observation 单独 JSON 编码；调用现有 OllamaClient.generate_structured，think=false，num_predict=512、num_ctx=target.context_length。使用 runtime.acquire(target.profile) 与当前 ExecutionBudget；不自建重试/HTTP 通路，不接受自由文本为动作。
- [ ] 测试自然语言/截断 JSON 最多一次共享格式修复，失败后 decision_invalid；取消/租约等待计入 deadline；实际模型请求和未知 usage 的保守记账有断言。现有 budget/model_runtime 测试同时 GREEN。
- [ ] 提交配置、schema、decider 与对应测试。

### Task 4: 冻结合法集合和真实像素补证

**Files:** Modify `services/search.py`、`services/rag_adapter.py`；Create `tests/test_adaptive_evidence.py`。RequestScope 在 Task 5 文件中定义，本任务接口不依赖它。

**Interfaces:** 保留 `prepare_evidence(..., parse_version_map: dict[str,str] | None=None)`；None 表示本次首次冻结，显式映射（包括空映射）表示不得补入新文档。新增 `QueryService.merge_prepared_snapshots(base: PreparedEvidence, extra: PreparedEvidence) -> PreparedEvidence` 和 RAGAdapter 对应薄封装；验证相同项目、冻结版本与请求范围，再按 canonical 身份合并完整 contexts/pack/表格 coverage，而非只拼 excerpt。

- [ ] 写 `test_supplement_cannot_add_newly_created_document`、`test_empty_frozen_map_does_not_refresh_active_versions`、`test_focus_document_cannot_expand_request_scope`；指针由 v5→v6 变化后仍只读 v5。

```python
assert supplement.parse_version_map == initial.parse_version_map
assert "new_document" not in supplement.parse_version_map
assert {item.parse_version for item in supplement.pack.items} == {"v5"}
assert empty_frozen.pack.items == []
```

- [ ] 写图表合并测试：同名图 2 分属两篇论文、表格引用重新编号、伪造跨版本 chunk；断言合法图片仍随最终 answer 请求发送，像素/facts/citation 身份一致，非法证据没有进入任何通道。
- [ ] 运行新测试确认 RED，然后实现严格区分 None 与空映射、全检索/邻居/图表展开只用冻结文档集合及上述合并方法。图片路径必须 resolve 后落在已核验根目录，符号链接越界拒绝。
- [ ] 运行 `test_adaptive_evidence.py`、`test_harness_repair.py`、表格/visual evidence 相关原测试，确认 GREEN；这些关键测试纳入每次后续回归。
- [ ] 提交冻结/合并变更和测试，不为新增补证绕过原 canonical 身份契约。

### Task 5: 只读自适应状态机

**Files:** Create `services/adaptive_agent.py`、`tests/test_adaptive_agent.py`。

**Interfaces:** `RequestScope` frozen dataclass(project_slug: str, session_id: str, document_id: str|None, parse_version_map: Mapping[str,str], embedding_identity: EmbeddingIdentity)；构造时复制并用 MappingProxyType 固定映射。`AdaptiveOutcome` dataclass(prepared: PreparedEvidence, answer: QueryResponse|None, stop_reason: str, decisions: int, supplement_retrievals: int)。`AdaptiveAgent(decider: AgentDecider, retrieve: Callable[[str,str|None,int],PreparedEvidence], answer: Callable[[PreparedEvidence],QueryResponse], emit_step: Callable[[AgentStep],None])`；`run(*, question: str, conversation_summary: str, scope: RequestScope, initial: PreparedEvidence, target: InferenceTarget, budget: ExecutionBudget, max_decisions: int=3, max_supplements: int=2) -> AdaptiveOutcome`。受信 callbacks 注入固定 scope，模型不能指定 db/文件/工具对象。

- [ ] 用确定性假 decider 写 RED：同一问题初次只有图 2 时选择检索表 3；初次已有两份证据时直接 answer；断言第二次实际 query 取自新的 observation 而非原问题固定复制。

```python
assert sparse_case.retrieval_queries == ["论文 A 表 3 最优方法指标"]
assert complete_case.retrieval_queries == []
assert sparse_case.outcome.supplement_retrievals == 1
assert complete_case.outcome.supplement_retrievals == 0
assert sparse_case.outcome.decisions <= 3
```

- [ ] 写 no_progress、超限、无候选 finish、无证据 abstain、拒绝越权 document 的测试。对已有候选答案，finish 不能自造引用；工具失败与缺证据分别终止。
- [ ] 实现 observation→decision→validate→tool→new observation 状态机，使用 Task 4 合并接口；最多 3 决策/2 补证，保留尾段步骤/工具预算。候选生成后无缺口不无条件再决策；缺预算不调用模型、不伪造 completed。
- [ ] 测试在默认 max_steps=8 下所有 trace 步骤不超限，低 token/45s deadline/主动 cancel 都停止，共享格式重试最多一次；新证据必须真正增加合法身份才能认为 progress。
- [ ] `python -m pytest tests/test_adaptive_agent.py tests/test_adaptive_evidence.py tests/test_agent_decider.py tests/test_execution_budget.py -q` GREEN 后提交。

### Task 6: JSON/SSE 接线、会话及显式预算修复

**Files:** Modify `services/agent_executor.py`、`api/agent_routes.py`；Create `tests/test_adaptive_integration.py`；扩展现有 `tests/test_agent_executor.py`、`tests/test_agent_streaming.py`、`tests/test_agent_routes.py`。

**Interfaces:** 在已有 route 之后且常规检索/生成之前分流，不重复 touch_session/add_turn。新增私有 `_execute_adaptive(request: AgentQueryRequest, budget: ExecutionBudget, *, route: AgentRouteDecision, inference_target: InferenceTarget, session_id: str, steps: list[AgentStep], usage: AgentUsage) -> AgentQueryResponse`；复用现有 trace/finalize/verify/helpers，把公共收尾提取为一个方法，静态与动态各调用一次。接口仍为原 AgentQueryResponse，不新增客户端可扩权字段。

- [ ] RED：普通查询 decider=0 次、复杂但开关关闭=0 次、complex+true 动态；text 和 image 的跨轮查询均正确使用当前 session，换题不错误沿用视觉意图。
- [ ] RED：客户端显式 timeout_seconds=45/max_steps=8 而服务器配置更大，仍保持 45/8；字段省略才取服务端默认。改 `_apply_server_constraint_defaults` 按 Pydantic model_fields_set，而非值是否等于默认值判断。

```python
explicit = AgentQueryRequest(project_slug="p", query="q",
    constraints={"timeout_seconds": 45, "max_steps": 8})
resolved = _apply_server_constraint_defaults(explicit)
assert resolved.constraints.timeout_seconds == 45
assert resolved.constraints.max_steps == 8
assert plain_query_decider.call_count == 0
```

- [ ] 接入 Task 5；统一 JSON/SSE 的 metadata：execution_mode、decision_index、实际 backend、预算、终止原因。terminal 只写一次会话 agent 轮次及 trace，结构校验 passed/failed/skipped 如实；后续不能把旧 candidate 引用到新版本。
- [ ] RED→GREEN：SSE 中途断连和 socket 卡住取消，队列/租约释放；结构校验 skip 不等于通过；图片与引用重新编号正确。trace 不包含隐藏思考、秘密配置或完整 observation 原文。
- [ ] 运行新增和原 Agent、harness、budget 相关文件，然后全套 pytest 输出 JUnit；实际失败逐项修复（缺异步插件、FastAPI 路由 introspection、db=None 探针按真实接口边界），不删测试掩盖。提交通过的接线及兼容修复。

### Task 7: 统一启动与精确清理

**Files:** Create `scripts/project_ctl.py`、`deploy/standalone.md`、`tests/test_project_ctl.py`、`tests/test_standalone_deployment.py`；Modify README/runbook/pyproject、原部署与上下文化配置测试；删除设计第 7 节列明且已消除引用的旧模板/启动器。保留必要迁移、worker、独立 pgvector 验证。

**Interfaces:** `ProjectController(root: Path)`；`preflight() -> dict[str,Any]`、`start() -> dict[str,Any]`、`status() -> dict[str,Any]`、`stop() -> dict[str,Any]`。CLI `python scripts/project_ctl.py preflight|start|status|stop`，失败 exit 1，结果为脱敏 JSON。PG 使用原集群管理命令；controller 默认 stop 仅停止自己启动并验证身份的 model/API/可选 worker，不停止数据库或未知既有进程。

- [ ] RED：过期 PID 被其他程序复用、陌生进程占端口、两个 start 并发、健康探测返回旧进程、图片根符号链接越界；断言绝不 kill 陌生进程，也不假报 ready。

```python
assert foreign_pid not in kill_calls
assert foreign_port_result["status"] != "ready"
assert simultaneous_start_model_launch_count == 1
assert controller.status()["api"]["owned"] is False
```

- [ ] 实现 root+argv+解释器+监听 inode 校验、跨进程启动锁、PG→model→API 顺序、脱离 SSH 的 subprocess session、失败日志及健康 deadline。不用 shell=True 或模糊 pkill，不依赖 Systemd/ss；只读状态不执行服务变更。
- [ ] 模型配置由私有 env 提供绝对模型/adapter 目录；默认统一 VL embedding 和 base alias；根目录仅 parsed/cache 已验证路径。模板无凭据/公司路径，控制器禁止 AUTH_ENABLED=false 的非 loopback 配置。
- [ ] 按设计清单确认每个目标解析路径位于工作副本/服务器项目，保存副本+哈希后移出活跃树；逐项更新 README/runbook 与测试的旧链接/环境断言。旧旁路仍被历史脚本引用者保留或迁移检查后再归档，不能整模块盲删。用 rg 检查活跃入口无旧 Systemd/GPU1/8B/2560 部署指示，schema 历史和备份中的历史记录不要求删除。
- [ ] 启动/部署测试及全套回归 GREEN 后提交 controller、文档、依赖和逐项删除。保留本地 cleanup-ledger.json（原路径/哈希/归档路径/理由），Git 不包含备份或数据。

### Task 8: 服务器真实恢复、发布及配对评估

**Files:** Create `scripts/verify_standalone.py`、`scripts/evaluate_adaptive_agent.py`、`tests/test_standalone_acceptance.py`、`tests/test_adaptive_evaluation.py`；题集/推理/指标仅在私有 `runtime/acceptance/20261006/`，最终报告在本地 output。生产 .env 只在服务器备份后配置，不提交。

**Interfaces:** 验收 CLI `python scripts/verify_standalone.py --base-url http://127.0.0.1:18002 --project multimodal-pilot --output runtime/acceptance/20261006/smoke`；比较 CLI `python scripts/evaluate_adaptive_agent.py --base-url http://127.0.0.1:18002 --cases runtime/acceptance/20261006/complex20.jsonl --output runtime/acceptance/20261006/paired`；`_aggregate(records: list[dict[str,Any]]) -> dict[str,Any]` 返回 fact_accuracy/citation_accuracy/abstention_accuracy 及各自 judged_count、总 run/error/ungraded 数。判分字段 fact_correct/citation_correct/abstention_appropriate 为 bool 或 None，不从 verification_status 推导。Case 字段固定 id/question/document_scope/reference_facts/required_evidence/unanswerable；20 题分组各 4：同文档多跳、图表核对、跨文档/矛盾、缺证据/澄清、跨轮/注入。

- [ ] RED→GREEN 测试验收脚本的失败 exit、后端证据、空指标、两臂预算一致以及结构 passed 不当作 fact accuracy。配对实验的模式切换仅由本机运维控制服务开关，不新增公开可写切换 API；两臂同 base、相同身份/索引/版本/预算。

```python
assert _aggregate([])["fact_accuracy"] is None
assert _aggregate([{"verification_status": "passed"}])["fact_accuracy"] is None
assert static_record["constraints"] == adaptive_record["constraints"]
assert failed_backend_smoke_exit_code == 1
```

- [ ] 重新只读核查 SSH 指纹、实例、PG 状态和备份哈希；若停机状态变化重新备份。正常 `pg_ctlcluster 16 main start`，核查恢复日志及 SELECT 业务数据/schema/Alembic/vector extension/2048 列/激活 map；失败停止诊断，不 reset。生成新的 pg_dump 并验证可读取内容表，原库不被覆盖。
- [ ] 发布审核后的源码到独立 staging，通过服务器 .venv 在 staging 跑同套测试；切换代码前保存旧源码。逐文件/按清单部署，不用 --delete 同步项目根，不复制或覆盖 runtime/model/data/私有配置。控制器 preflight 必须通过身份检查后才启动 API；新路径稳定前开关保持 false。
- [ ] 真实 `/api/query` 和 `/api/agent/query`/stream 测文本首轮、跨轮、换题、表格、像素视觉、拒答；PgVectorStore 距离与手算 cosine 对齐且过滤排序一致。新增隔离测试项目上传一个人工小文档，经 parse→canonical→embed→index→activate→query 跑通；不修改原七篇/50 题。Redis/worker 若为真实入库所必需，按项目真实队列启用，不凭猜测删依赖。
- [ ] 构造独立 20 题，人工核验参考关键事实；固定路径/动态路径使用相同约束（max_steps=12、max_tool_calls=6、budget_tokens=60000、timeout_seconds=120），单 GPU 首版顺序评测，不让并发混入延迟比较。记录关键事实正确率、引用对应、适当拒答及补证/调用/token/延迟/停止原因；尚未跑或无参考的指标保持空并说明。至少展示多个 observation 改变动作的真实案例。
- [ ] 脱离当前 SSH 后从新连接复查健康；重复 start/精确 stop/start 与回退开关可用。汇总全部实际测试、已知失败、错题与清理账本。安全/身份/普通回归失败不得 push；若只有动态收益不足，关闭默认开关并如实报告，不宣称精度提升。
- [ ] 提交验收工具及其测试（不提交私有题集/推理文本/语料），服务器最终源码哈希与待发布提交核对。

### Task 9: 最终审查与验收后 GitHub 同步

**Files:** 仅该 worktree 的授权代码/测试/脱敏模板/文档/迁移/依赖清单；报告与原工作树不入提交。

**Interfaces:** 产出 GitHub 独立分支 `feat/readonly-adaptive-pgvector`、确切 commit SHA、服务器发布哈希、验收报告和可恢复清理记录。

- [ ] 按 executing-plans / requesting-code-review 进行一次全分支独立审查，范围覆盖 spec、安全、像素、冻结集合、预算、服务控制和发布差异。必要修复先写 RED 再 GREEN；收到意见按 receiving-code-review 技能核实，不盲从。所有重要问题闭环后重新核对真实端到端验收与代码哈希。
- [ ] `git diff --check`、逐文件 diff/source 复核、`git ls-files` 白名单检查及敏感模式扫描：排除 .env、SSH/private key、凭据、runtime、PDF/文献原文、dump、权重、压缩包、egg-info/pycache、公司数据。扫描发现疑似项只显示文件/行位置，不打印秘密；扫描不能替代来源审查。
- [ ] 重新运行全套与关键相关测试，记录实际 pass/fail/skip；核对服务器已部署的是这些提交文件。只以确认的证据宣称完成，不使用 10 月 5 日数字替代本次结果。
- [ ] push 前重新读取远端 main/目标分支 refs；同名远端分支非本任务时不覆盖，main 漂移则在隔离分支处理兼容并重跑必要验收，不 force push。仅执行 `git push -u origin feat/readonly-adaptive-pgvector`，不自动合并 main 或创建发布 tag。
- [ ] 报告服务器当前状态、实际测评结果、GitHub 分支/commit、删除的精确对象及归档可恢复位置；原工作树修改保持不变。

## 执行检查点

- [x] 用户已确认设计与 GitHub 发布权限。
- [x] 已保存并自审实施计划；执行方式沿用本对话 Native。
- [x] 用户审阅确认本计划（回复“执行”）。
- [ ] Tasks 1–7：保护资料与本地实现/回归。
- [ ] Task 8：服务器全链路与配对评估。
- [ ] Task 9：独立审查、最终核验、GitHub 同步。

本计划已批准；逐任务进度以隔离工作副本的执行 ledger 为准。
