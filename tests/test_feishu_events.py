"""事件解析（im.message.receive_v1）与过滤策略（租户/群白名单/@ 提及）单测。"""

import pytest

from app.services.feishu_bot.events import (
    EventPolicy,
    MessageEvent,
    parse_message_event,
)


def _text_event(**overrides) -> dict:
    base = {
        "schema": "2.0",
        "header": {
            "event_id": "evt_1",
            "event_type": "im.message.receive_v1",
            "create_time": "1700000000000",
            "token": "tok",
            "app_id": "cli_app",
            "tenant_key": "tenant_a",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": "ou_user", "union_id": "u", "user_id": "uid"},
                "sender_type": "user",
                "tenant_key": "tenant_a",
            },
            "message": {
                "message_id": "om_1",
                "root_id": "",
                "parent_id": "",
                "create_time": "1700000000000",
                "chat_id": "oc_group",
                "chat_type": "group",
                "message_type": "text",
                "content": '{"text":"@_user_1 你好"}',
                "mentions": [
                    {
                        "key": "@_user_1",
                        "id": {"open_id": "ou_bot"},
                        "name": "知识助手",
                        "tenant_key": "tenant_a",
                    }
                ],
            },
        },
    }
    return _deep_merge(base, overrides)


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def test_parse_text_event_extracts_fields():
    event = parse_message_event(_text_event())
    assert isinstance(event, MessageEvent)
    assert event.event_id == "evt_1"
    assert event.tenant_key == "tenant_a"
    assert event.app_id == "cli_app"
    assert event.chat_id == "oc_group"
    assert event.message_id == "om_1"
    assert event.message_type == "text"
    assert event.sender_open_id == "ou_user"
    assert event.mentions == ["ou_bot"]
    assert event.text == "@_user_1 你好"
    assert event.clean_text() == "你好"


def test_parse_text_event_without_mentions():
    event = parse_message_event(_text_event(event={"message": {"mentions": []}}))
    assert event.mentions == []
    assert event.text == "@_user_1 你好"
    assert event.clean_text() == "@_user_1 你好"


def test_parse_file_event_extracts_file_key_and_name():
    event = parse_message_event(
        _text_event(
            event={
                "message": {
                    "message_type": "file",
                    "content": '{"file_key":"file_abc","file_name":"报告.pdf","file_size":12345}',
                }
            }
        )
    )
    assert event.message_type == "file"
    assert event.file_key == "file_abc"
    assert event.file_name == "报告.pdf"


def test_parse_bad_content_json_raises():
    with pytest.raises(ValueError):
        parse_message_event(_text_event(event={"message": {"content": "not json"}}))


def test_parse_event_id_preserved_from_header():
    event = parse_message_event(_text_event(header={"event_id": "evt_unique"}))
    assert event.event_id == "evt_unique"


def test_policy_allows_matching_tenant():
    policy = EventPolicy(allowed_tenant="tenant_a", allowed_chat_ids=[])
    event = parse_message_event(_text_event())
    assert policy.allow(event)


def test_policy_rejects_foreign_tenant():
    policy = EventPolicy(allowed_tenant="tenant_a", allowed_chat_ids=[])
    event = parse_message_event(_text_event(header={"tenant_key": "tenant_b"}))
    assert not policy.allow(event)


def test_policy_tenant_unset_does_not_filter():
    policy = EventPolicy(allowed_tenant=None, allowed_chat_ids=[])
    event = parse_message_event(_text_event(header={"tenant_key": "any"}))
    assert policy.allow(event)


def test_policy_chat_whitelist_limits_groups():
    policy = EventPolicy(allowed_tenant=None, allowed_chat_ids=["oc_ok"])
    # oc_group 不在白名单 → 拒绝
    assert not policy.allow(parse_message_event(_text_event()))
    assert not policy.allow(
        parse_message_event(_text_event(event={"message": {"chat_id": "oc_other"}}))
    )
    # 白名单内的群放行
    assert policy.allow(
        parse_message_event(_text_event(event={"message": {"chat_id": "oc_ok"}}))
    )


def test_policy_empty_whitelist_allows_all_groups():
    policy = EventPolicy(allowed_tenant=None, allowed_chat_ids=[])
    event = parse_message_event(_text_event())
    assert policy.allow(event)


def test_policy_mentions_bot():
    policy = EventPolicy(allowed_tenant=None, allowed_chat_ids=[])
    event = parse_message_event(_text_event())
    assert policy.mentions_bot(event, bot_open_id="ou_bot")


def test_policy_mentions_other_user_is_not_bot():
    policy = EventPolicy(allowed_tenant=None, allowed_chat_ids=[])
    event = parse_message_event(_text_event())
    assert not policy.mentions_bot(event, bot_open_id="ou_other")


def test_policy_no_mentions_is_not_bot():
    policy = EventPolicy(allowed_tenant=None, allowed_chat_ids=[])
    event = parse_message_event(_text_event(event={"message": {"mentions": []}}))
    assert not policy.mentions_bot(event, bot_open_id="ou_bot")
