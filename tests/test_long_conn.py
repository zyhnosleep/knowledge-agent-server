"""Tests for the official-SDK Feishu long-connection adapter."""

import json

from app.services.feishu_bot.long_conn import LongConnClient


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
            "sender": {
                "sender_id": {"open_id": "ou_user"},
                "sender_type": "user",
                "tenant_key": "tenant_a",
            },
            "message": {
                "message_id": "om_1",
                "create_time": "0",
                "chat_id": "oc_group",
                "chat_type": "group",
                "message_type": "text",
                "content": '{"text":"@_user_1 hi"}',
                "mentions": [
                    {
                        "key": "@_user_1",
                        "id": {"open_id": "ou_bot"},
                        "name": "bot",
                        "tenant_key": "tenant_a",
                    }
                ],
            },
        },
    }


def test_official_sdk_dispatches_message_event_as_raw_dict():
    created = {}

    class FakeSdkClient:
        def __init__(self, app_id, app_secret, **kwargs):
            created.update(
                app_id=app_id,
                app_secret=app_secret,
                event_handler=kwargs["event_handler"],
            )

        def start(self):
            created["event_handler"]._do_without_validation(
                json.dumps(_event_payload()).encode()
            )

    events = []
    LongConnClient("cli_x", "secret", client_factory=FakeSdkClient).run(events.append)

    assert created["app_id"] == "cli_x"
    assert created["app_secret"] == "secret"
    assert events == [_event_payload()]


def test_sdk_client_starts_even_without_receiving_an_event():
    started = []

    class FakeSdkClient:
        def __init__(self, app_id, app_secret, **kwargs):
            pass

        def start(self):
            started.append(True)

    LongConnClient("cli_x", "secret", client_factory=FakeSdkClient).run(lambda _: None)
    assert started == [True]
