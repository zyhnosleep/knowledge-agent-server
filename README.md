# Knowledge Agent

科研文献的文本/图表证据问答：保留原始文件、canonical 解析版本、页码和来源；结构引用验证不等于事实正确率。

## 当前部署

单实例 Linux/容器：PostgreSQL 16 + pgvector、2048 维 Qwen3-VL-Embedding-2B；文本和图片共用同一向量空间。生成/视觉模型为 Qwen3-VL-4B-Instruct，默认 base，LoRA 可选。API 127.0.0.1:18002，模型服务 127.0.0.1:18080；不开公网入口。

API 与 GPU 模型使用独立 Python 环境。当前部署配置只使用 [standalone 模板](deploy/standalone.env.example)，步骤和恢复边界见 [部署说明](deploy/standalone.md)。模板路径需要按实际安装修改，实际 .env、模型、数据和凭据不能提交。

## 运行与验证

```bash
.venv/bin/python scripts/project_ctl.py preflight
.venv/bin/python scripts/project_ctl.py start
.venv/bin/python scripts/project_ctl.py status
.venv/bin/python scripts/project_ctl.py stop
```

controller 依次检查 PG、模型、API；只管理有精确进程/监听归属记录的模型和 API，stop 不停止数据库。旧索引缺 revision/processor 身份会被阻断：必须新建受控 shadow 版本、质量核验后激活，禁止补写旧向量身份。

## 工作流与 Agent

当前采用固定工作流，自适应实验暂停，部署时保持 `AGENT_ADAPTIVE_ENABLED=false`；普通题和复杂题均不调用自主 planner。跨论文比较对已识别的论文分别检索，保留两侧证据；回答提示区分来源，显式图号优先定位请求的图，长表格追问可恢复上一轮检索目标。模型仍可能读错曲线或混淆方法，不能把结构校验通过当作事实正确。

自适应代码仅保留供后续研究：只有 complex_multi_hop + AGENT_ADAPTIVE_ENABLED=true 才使用 observation 驱动的只读循环，最多 3 次决策、2 次补证，共享 step/tool/token/deadline/cancel 预算；本轮修复与验收不启用它。

版本、项目、会话和图片作用域由受信执行器冻结，模型不能扩权或执行 shell/任意 SQL/写知识源。默认开关关闭，服务器真实验收后再决定是否启用。它是受限 Agent + harness，不是把固定工作流更名为自主 Agent。

## 开发

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -p no:cacheprovider -q
```

SQLite 仅为测试/历史迁移兼容，不是生产替代后端。不要把 API 的 transformers<5 安装到独立模型环境。完整运行链路见 [运行手册](docs/project-runbook.md)。

单测通过不代表服务器已部署；真实文本、表格、像素、跨轮、拒答、pgvector 排序和 Agent JSON/SSE 必须另行验收。私有语料、题集、评测输出和备份不发布。
