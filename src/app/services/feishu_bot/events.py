"""
events.py —— im.message.receive_v1 事件解析与过滤
=================================================

职责：
- 把飞书推送的原始事件 dict 解析为 ``MessageEvent``（类型安全、字段齐全）。
- ``EventPolicy`` 过滤策略：租户限定 + 可选群白名单 + @ 提及判断。
  策略遵循"配置存在才收紧"：白名单为空 = 租户内不限群。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class MessageEvent:
    """解析后的群消息事件（im.message.receive_v1）。"""

    event_id: str
    tenant_key: str
    app_id: str
    chat_id: str
    message_id: str
    message_type: str
    sender_open_id: str
    # 会话类型：group（群聊）/ p2p（单聊）
    chat_type: str = "group"
    # 被 @ 的 open_id 列表（含机器人与他人）
    mentions: list[str] = field(default_factory=list)
    # 与 mentions 对应的 @ 标记字符串（如 "@_user_1"，用于从文本中剥离）
    mention_keys: list[str] = field(default_factory=list)
    # 文本消息的原文（未去除 @ 提及标记）
    text: str | None = None
    # 文件消息的资源信息（message_type == "file" 时非空）
    file_key: str | None = None
    file_name: str | None = None

    def clean_text(self) -> str:
        """去除文本中的 @ 提及标记（如 ``@_user_1``）后的干净提问文本。"""
        if not self.text:
            return ""
        cleaned = self.text
        # mentions 的 key 形如 @_user_1，直接按 key 字符串替换
        for mention_key in self.mention_keys:
            cleaned = cleaned.replace(mention_key, "")
        return cleaned.strip()

    @property
    def is_file(self) -> bool:
        return self.message_type == "file" and bool(self.file_key)

    @property
    def is_text(self) -> bool:
        return self.message_type == "text"


def parse_message_event(raw: dict) -> MessageEvent:
    """解析飞书推送的原始事件 dict；结构不符抛 ValueError。"""
    header = raw.get("header") or {}
    event = raw.get("event") or {}
    sender = event.get("sender") or {}
    sender_id = sender.get("sender_id") or {}
    message = event.get("message") or {}

    content_raw = message.get("content") or ""
    if isinstance(content_raw, str):
        try:
            content = json.loads(content_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"message content is not valid JSON: {content_raw!r}") from exc
    else:
        content = content_raw

    mentions = []
    mention_keys = []
    for mention in message.get("mentions") or []:
        mention = mention or {}
        mention_id = mention.get("id") or {}
        if mention_id.get("open_id"):
            mentions.append(mention_id["open_id"])
            if mention.get("key"):
                mention_keys.append(mention["key"])

    parsed = MessageEvent(
        event_id=header.get("event_id", ""),
        tenant_key=header.get("tenant_key", ""),
        app_id=header.get("app_id", ""),
        chat_id=message.get("chat_id", ""),
        message_id=message.get("message_id", ""),
        message_type=message.get("message_type", ""),
        sender_open_id=sender_id.get("open_id", ""),
        chat_type=message.get("chat_type", "group"),
        mentions=mentions,
        mention_keys=mention_keys,
    )
    if parsed.message_type == "text":
        parsed.text = content.get("text") if isinstance(content, dict) else None
    elif parsed.message_type == "file":
        if isinstance(content, dict):
            parsed.file_key = content.get("file_key")
            parsed.file_name = content.get("file_name")
    return parsed


class EventPolicy:
    """事件过滤策略：租户限定 + 可选群白名单 + @ 提及判断。

    - ``allowed_tenant``：非空时仅放行该租户的事件。
    - ``allowed_chat_ids``：非空时仅放行这些群的会话。
    """

    def __init__(self, allowed_tenant: str | None, allowed_chat_ids: list[str]) -> None:
        self._allowed_tenant = allowed_tenant
        self._allowed_chat_ids = list(allowed_chat_ids)

    def allow(self, event: MessageEvent) -> bool:
        """租户与群白名单双通过才算放行。"""
        if self._allowed_tenant and event.tenant_key != self._allowed_tenant:
            return False
        if self._allowed_chat_ids and event.chat_id not in self._allowed_chat_ids:
            return False
        return True

    def mentions_bot(self, event: MessageEvent, bot_open_id: str) -> bool:
        """事件中是否 @ 了机器人（按 open_id 判断）。

        私聊（p2p）单聊没有 @ 提及，且单聊里的消息必然是发给机器人的 → 直接放行。
        """
        if event.chat_type == "p2p":
            return True
        return bot_open_id in event.mentions
