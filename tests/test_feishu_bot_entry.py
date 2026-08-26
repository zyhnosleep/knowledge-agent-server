"""飞书机器人进程入口单测：配置缺失/机器人信息获取失败 → 退出（systemd 拉起重试）。"""

import threading

import httpx
import pytest

from app.core.config import Settings
from app.services.feishu_bot.client import FeishuClient
from app.services.feishu_bot.entry import _on_event, run_bot


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
    run_bot(settings=settings)  # 应正常返回，不启动任何东西


def test_run_bot_exits_without_credentials():
    settings = _settings(**{"FEISHU_APP_ID": None, "FEISHU_APP_SECRET": None})
    with pytest.raises(SystemExit):
        run_bot(settings=settings)


def _bot_info_fails_client(app_id: str, app_secret: str) -> FeishuClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 10003, "msg": "bot disabled"})

    return FeishuClient(app_id, app_secret, transport=httpx.MockTransport(handler))


def test_run_bot_exits_when_bot_info_unavailable():
    settings = _settings()
    with pytest.raises(SystemExit):
        run_bot(settings=settings, client_factory=_bot_info_fails_client)


def _event_payload() -> dict:
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt_1",
            "event_type": "im.message.receive_v1",
            "create_time": "0",
            "token": "tok",
            "app_id": "cli_app",
            "tenant_key": "tenant_a",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_user"}, "sender_type": "user", "tenant_key": "tenant_a"},
            "message": {
                "message_id": "om_1",
                "create_time": "0",
                "chat_id": "oc_group",
                "chat_type": "group",
                "message_type": "text",
                "content": '{"text":"@_user_1 hi"}',
                "mentions": [{"key": "@_user_1", "id": {"open_id": "ou_bot"}, "name": "x", "tenant_key": "tenant_a"}],
            },
        },
    }


class _BoomHandler:
    """handle 必炸：模拟发消息失败等下游异常。"""

    def __init__(self, called=None) -> None:
        self.called = called

    def handle(self, event) -> None:
        if self.called is not None:
            self.called.set()
        raise RuntimeError("send failed")


def test_on_event_swallows_handler_failures():
    """handler.handle 抛异常（如 send_text 失败）不冒泡断链：记日志继续。"""
    called = threading.Event()
    _on_event(_BoomHandler(called), _event_payload())  # 不应抛异常
    assert called.wait(timeout=1)


def test_on_event_drops_unparseable_event():
    """解析失败的帧被丢弃（不炸循环），也不调 handler。"""
    handled = []

    class Recorder(_BoomHandler):
        def handle(self, event) -> None:
            handled.append(event)

    _on_event(Recorder(), {"type": "EVENT", "data": {"garbage": 1}})
    assert handled == []
