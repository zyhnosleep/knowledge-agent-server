"""
handler.py —— 机器人事件处理入口（过滤 → @ 检查 → 分派）
==========================================================

``BotHandler.handle`` 是长连接收到消息事件后的统一入口：
1. 租户/群白名单过滤（``EventPolicy.allow``）。
2. @ 机器人检查（按 bot_open_id）。
3. 按消息类型分派：文本 → 问答（T3）、文件 → 入库（T2）、
   文件+文字 → 先入库后问答（T4）。当前（T1）仅回固定"收到，处理中"回执。
4. 不支持的类型的消息（图片等）静默忽略。
"""

from __future__ import annotations

import threading

from app.services.feishu_bot.events import EventPolicy, MessageEvent


class BotHandler:
    """机器人事件处理入口。

    参数：
    - ``client``：飞书 API 客户端（发消息用，鸭子类型：send_text(chat_id, text)）。
    - ``policy``：过滤策略。
    - ``bot_open_id``：机器人自身 open_id（@ 判断基准）。
    """

    def __init__(
        self,
        client,
        policy: EventPolicy,
        bot_open_id: str,
        *,
        ingestor=None,
        querier=None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._bot_open_id = bot_open_id
        self._ingestor = ingestor
        self._querier = querier
        # 每群一把锁：文件入库持有锁期间，同群的文本问答等待入库完成
        # （混合场景"先入库后回答"，答案可引用刚入库的文档）
        self._chat_locks: dict[str, threading.Lock] = {}

    def handle(self, event: MessageEvent) -> None:
        """处理一条消息事件；过滤不通过或未 @ 机器人则静默。"""
        if not self._policy.allow(event):
            return
        if not self._policy.mentions_bot(event, self._bot_open_id):
            return
        self._dispatch(event)

    def _dispatch(self, event: MessageEvent) -> None:
        """按消息类型分派：文件 → 入库（T2）；文本 → 问答（T3）。"""
        if event.is_file:
            self._file_scenario(event)
        elif event.is_text:
            self._text_scenario(event)

    def _file_scenario(self, event: MessageEvent) -> None:
        """文件入库场景：占位回执 → 入库（持锁）→ 结果回执（失败给明确原因）。"""
        self._client.send_text(event.chat_id, "正在入库…")
        if self._ingestor is None:
            self._client.send_text(event.chat_id, "入库功能未启用")
            return
        with self._chat_lock(event.chat_id):
            result = self._ingestor.ingest_file(event)
        if result.ok:
            self._client.send_text(event.chat_id, f"已入库 1 篇：{result.file_name}")
        else:
            name = result.file_name or "文件"
            self._client.send_text(event.chat_id, f"{name} 入库失败：{result.error}")

    def _text_scenario(self, event: MessageEvent) -> None:
        """文本问答场景：空问题给提示；否则占位回执 → 等入库完成 → 答案（带引用）。"""
        question = event.clean_text()
        if not question:
            self._client.send_text(
                event.chat_id,
                "请 @ 我并说明你的问题，例如：@知识助手 这篇论文的结论是什么？",
            )
            return
        self._client.send_text(event.chat_id, "处理中…")
        if self._querier is None:
            self._client.send_text(event.chat_id, "问答功能未启用")
            return
        # 持同群锁：若文件正在入库则等待完成，答案才能命中刚入库的文档
        with self._chat_lock(event.chat_id):
            result = self._querier.answer(question)
        if result.ok:
            text = result.answer or ""
            if result.citations:
                text += "\n\n参考：" + "、".join(result.citations)
            self._client.send_text(event.chat_id, text)
        else:
            self._client.send_text(event.chat_id, f"问答失败：{result.error}")

    def _chat_lock(self, chat_id: str) -> threading.Lock:
        """获取（或创建）指定群的锁。"""
        lock = self._chat_locks.get(chat_id)
        if lock is None:
            lock = threading.Lock()
            self._chat_locks[chat_id] = lock
        return lock
