"""飞书机器人进程入口单测：配置缺失/机器人信息获取失败 → 退出（systemd 拉起重试）。"""

import asyncio

import httpx
import pytest

from app.core.config import Settings
from app.services.feishu_bot.client import FeishuClient
from app.services.feishu_bot.entry import run_bot


def _settings(**overrides) -> Settings:
    # Settings 未开 populate_by_name → kwargs 必须用 env alias 名
    base = {
        "FEISHU_BOT_ENABLED": True,
        "FEISHU_APP_ID": "cli_x",
        "FEISHU_APP_SECRET": "secret",
        "FEISHU_ALLOWED_TENANT": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_run_bot_returns_when_disabled():
    settings = _settings(**{"FEISHU_BOT_ENABLED": False})
    asyncio.run(run_bot(settings=settings))  # 应正常返回，不启动任何东西


def test_run_bot_exits_without_credentials():
    settings = _settings(**{"FEISHU_APP_ID": None, "FEISHU_APP_SECRET": None})
    with pytest.raises(SystemExit):
        asyncio.run(run_bot(settings=settings))


def _bot_info_fails_client(app_id: str, app_secret: str) -> FeishuClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 10003, "msg": "bot disabled"})

    return FeishuClient(app_id, app_secret, transport=httpx.MockTransport(handler))


def test_run_bot_exits_when_bot_info_unavailable():
    settings = _settings()
    with pytest.raises(SystemExit):
        asyncio.run(run_bot(settings=settings, client_factory=_bot_info_fails_client))
