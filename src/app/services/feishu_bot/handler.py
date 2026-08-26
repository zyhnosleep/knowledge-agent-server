"""
handler.py —— 机器人事件处理入口（过滤 → @ 检查 → 分派）
==========================================================

``BotHandler.handle`` 是长连接收到消息事件后的统一入口：
1. 租户/群白名单过滤（``EventPolicy.allow``）。
2. @ 机器人检查（按 bot_open_id）。
3. 按消息类型分派：文件 → 入库（T2）、文本 → 问答（T3）。
   飞书单条消息要么文本要么文件（无混合类型）→ 混合场景（T4）由
   "相邻两条消息 + 每群 pending 计数"实现：文本提问等待同群所有
   进行中的入库完成，答案才能引用刚入库的文档。
4. 不支持的类型的消息（图片等）静默忽略。
"""

from __future__ import annotations

import logging
import threading

from app.services.feishu_bot.client import FeishuClientProtocol
from app.services.feishu_bot.events import EventPolicy, MessageEvent
from app.services.feishu_bot.inbox import InboxIngestor
from app.services.feishu_bot.query import Answerer, truncate_answer

logger = logging.getLogger(__name__)


class BotHandler:
    """机器人事件处理入口。

    参数：
    - ``client``：飞书 API 客户端（发消息用，FeishuClientProtocol）。
    - ``policy``：过滤策略。
    - ``bot_open_id``：机器人自身 open_id（@ 判断基准）。
    - ``ingestor``/``querier``：可选，缺省时对场景回"功能未启用"提示。
    """

    def __init__(
        self,
        client: FeishuClientProtocol,
        policy: EventPolicy,
        bot_open_id: str,
        *,
        ingestor: InboxIngestor | None = None,
        querier: Answerer | None = None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._bot_open_id = bot_open_id
        self._ingestor = ingestor
        self._querier = querier
        # 每群进行中的入库文件数：文本提问等待 pending 归零（T4"先入库后回答"，
        # 多文件同传时答案等全部入库完成，不只等第一个）
        self._pending: dict[str, int] = {}
        self._cond = threading.Condition()

    def handle(self, event: MessageEvent) -> None:
        """处理一条消息事件；过滤不通过或未 @ 机器人则静默。"""
        if not self._policy.allow(event):
            logger.info(
                "feishu bot: event dropped by policy (chat_id=%s)", event.chat_id
            )
            return
        if not self._policy.mentions_bot(event, self._bot_open_id):
            logger.info(
                "feishu bot: event dropped (bot not mentioned, chat_id=%s)", event.chat_id
            )
            return
        self._dispatch(event)

    def _dispatch(self, event: MessageEvent) -> None:
        """按消息类型分派：文件 → 入库（T2）；文本 → 问答（T3）。"""
        if event.is_file:
            self._file_scenario(event)
        elif event.is_text:
            self._text_scenario(event)

    def _file_scenario(self, event: MessageEvent) -> None:
        """文件入库场景：占位回执 → 入库（计 pending）→ 结果回执（失败给明确原因）。"""
        self._client.send_text(event.chat_id, "正在入库…")
        if self._ingestor is None:
            self._client.send_text(event.chat_id, "入库功能未启用")
            return
        with self._cond:
            self._pending[event.chat_id] = self._pending.get(event.chat_id, 0) + 1
        try:
            result = self._ingestor.ingest_file(event)
        finally:
            with self._cond:
                self._pending[event.chat_id] -= 1
                self._cond.notify_all()
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
        self._client.send_text(event.chat_id, f"处理中：{question[:20]}")
        if self._querier is None:
            self._client.send_text(event.chat_id, "问答功能未启用")
            return
        # 等同群进行中的入库全部完成（pending 归零），答案才能命中刚入库的文档
        with self._cond:
            while self._pending.get(event.chat_id, 0) > 0:
                self._cond.wait()
        result = self._querier.answer(question)
        if result.ok:
            text = result.answer or ""
            if result.citations:
                text += "\n\n参考：" + "、".join(result.citations)
            # 引用拼接后再截断一次：保证最终消息不超飞书长度限制
            self._client.send_text(event.chat_id, truncate_answer(text))
        else:
            self._client.send_text(event.chat_id, f"问答失败：{result.error}")
