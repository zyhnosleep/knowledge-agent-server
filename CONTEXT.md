# Knowledge Agent Server

科学文档知识库问答服务：解析论文 PDF，经语义分块/向量检索/确定性表格组装，用本地 Ollama 模型组织答案；单轮查询追求低延迟直通，多轮查询保证历史语境下的综合。

## Language

### 对话

**会话（session）**:
一次多轮问答的载体，以 session_id 标识；对话记录按轮次落库并随会话增长触发轮次压缩。
_Avoid_: chat, thread, conversation

**轮次（turn）**:
会话中一次用户查询及紧随的 agent 产出（检索、生成、验证）。
_Avoid_: round, step, message

**首轮查询（first-turn query）**:
会话第 1 轮的查询，无前文可引用；唯一允许 draft 直通的轮次。
_Avoid_: 单轮问题, initial question

**跨轮查询（cross-turn query）**:
第 2 轮起引用前文信息的查询；路由上强制走综合路径，不做词表检测。
_Avoid_: 追问, follow-up, 多轮问题

### 答案路径

**draft 直通（direct draft）**:
无历史注入的 LLM 组织答案路径；保留给首轮查询。
_Avoid_: 直答, direct answer, rag-direct

**综合路径（synthesize）**:
带会话历史摘要与检索证据的 LLM 组织答案路径；跨轮查询的必经路径。
_Avoid_: 综合生成, synthesis fallback

**上下文化（contextualization）**:
把短引用查询（≤30 字符）包装为"上一轮问题+当前追问"后再检索；仅作用于跨轮查询的检索阶段。
_Avoid_: 历史注入, context packing

### 记忆

**轮次压缩（turn compaction）**:
会话超出容量上限（200 轮）时删除最旧轮次的保险丝机制；压缩后轮次索引单调递增、不复用。
_Avoid_: 摘要压缩, history summarization

**轮次索引（turn index）**:
轮次在会话内的排序键；单调递增、压缩删除后不复用，保证指代解析可见完整历史。
_Avoid_: id, sequence number
