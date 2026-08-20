"""
entry.py —— 飞书机器人进程入口
===============================

独立常驻进程（systemd user service 拉起），与 API 服务分离：

- 启动：校验配置 → 拉取机器人 open_id → 组装过滤器/处理器 → 长连接长驻。
- 任一启动失败（配置缺失/机器人信息获取失败）以退出码 1 结束，
  交给 systemd ``Restart=on-failure`` 拉起；网络恢复后自然起来。
- 事件处理：解析失败丢弃并告警；解析成功后放线程池执行
  （处理是同步 HTTP 调用，不阻塞 ws 收帧）。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.services.feishu_bot.client import FeishuClient
from app.services.feishu_bot.events import EventPolicy, parse_message_event
from app.services.feishu_bot.handler import BotHandler
from app.services.feishu_bot.inbox import InboxIngestor, ServerApi
from app.services.feishu_bot.long_conn import LongConnClient
from app.services.feishu_bot.query import Answerer

logger = logging.getLogger(__name__)

ClientFactory = Callable[[str, str], FeishuClient]


def _split_chat_ids(raw: str) -> list[str]:
    """把逗号分隔的群白名单配置解析为列表（去空项）。"""
    return [part.strip() for part in raw.split(",") if part.strip()]


async def _on_event(handler: BotHandler, data: dict) -> None:
    """ws 事件回调：解析 → 线程池处理（不阻塞收帧）。

    处理中的任何异常（如发消息失败）只记日志不冒泡——否则会断掉
    健康的长连接并丢后续事件（飞书侧 3 秒未 ack 才会重推，无必要代价）。
    """
    try:
        event = parse_message_event(data)
    except ValueError:
        logger.warning("feishu bot: unparseable event dropped: %r", str(data)[:200])
        return
    try:
        await asyncio.to_thread(handler.handle, event)
    except Exception:
        logger.exception("feishu bot: event handling failed (event_id=%s)", event.event_id)


async def run_bot(
    settings: Settings | None = None,
    client_factory: ClientFactory = FeishuClient,
) -> None:
    """机器人主流程（可注入 settings/client_factory 便于测试）。"""
    settings = settings or get_settings()
    if not settings.feishu_bot_enabled:
        logger.info("feishu bot disabled (FEISHU_BOT_ENABLED=false); exiting")
        return
    if not settings.feishu_app_id or not settings.feishu_app_secret:
        logger.error("feishu bot: FEISHU_APP_ID/FEISHU_APP_SECRET missing; exiting")
        raise SystemExit(1)

    client = client_factory(settings.feishu_app_id, settings.feishu_app_secret)
    try:
        bot_open_id = client.get_bot_open_id()
    except Exception as exc:
        logger.error("feishu bot: cannot resolve bot open_id: %s; exiting", exc)
        raise SystemExit(1) from exc

    policy = EventPolicy(
        allowed_tenant=settings.feishu_allowed_tenant,
        allowed_chat_ids=_split_chat_ids(settings.feishu_bot_allowed_chat_ids),
    )
    # 本地 API 复用：文件入库 + 问答都经 dev API（不直接触碰数据库）
    api = ServerApi(settings.feishu_bot_api_base_url)
    ingestor = InboxIngestor(
        feishu=client,
        api=api,
        inbox_project=settings.feishu_bot_inbox_project,
        inbox_project_name="飞书收件箱",
    )
    querier = Answerer(api, project_slug=settings.feishu_bot_inbox_project)
    handler = BotHandler(
        client=client,
        policy=policy,
        bot_open_id=bot_open_id,
        ingestor=ingestor,
        querier=querier,
    )
    conn = LongConnClient(client)
    logger.info("feishu bot starting (bot_open_id=%s)", bot_open_id)
    await conn.run(lambda data: _on_event(handler, data))


def main() -> None:
    """命令行入口：python -m app.services.feishu_bot.entry"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
