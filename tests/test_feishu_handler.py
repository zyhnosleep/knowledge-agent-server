"""BotHandler 单测：过滤 + @ 检查 + 分派（T1 回执 / T2 文件入库 / T3 问答 / T4 混合）。"""

import threading
import time

from app.services.feishu_bot.events import EventPolicy, parse_message_event
from app.services.feishu_bot.handler import BotHandler
from app.services.feishu_bot.inbox import IngestResult
from app.services.feishu_bot.query import QueryResult


class _FakeClient:
    """记录 send_text 调用的替身（不触网）。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send_text(self, chat_id: str, text: str) -> None:
        self.sent.append((chat_id, text))


class _FakeIngestor:
    """InboxIngestor 替身：结果可控。"""

    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    def ingest_file(self, event) -> "object":
        self.calls += 1
        return self.result


class _FakeQuerier:
    """Answerer 替身：结果可控。"""

    def __init__(self, result) -> None:
        self.result = result
        self.questions: list[str] = []

    def answer(self, question: str) -> "object":
        self.questions.append(question)
        return self.result


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
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _deep_merge(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _handler(client=None, **kwargs):
    return BotHandler(
        client=client or _FakeClient(),
        policy=EventPolicy(allowed_tenant="tenant_a", allowed_chat_ids=[]),
        bot_open_id="ou_bot",
        **kwargs,
    )


def test_handler_acks_when_mentioned_and_allowed():
    client = _FakeClient()
    _handler(client).handle(parse_message_event(_text_event()))
    # T3 起文本场景：占位回执回显问题（"处理中：<问题>"）；无问答器时补"未启用"提示
    assert client.sent == [("oc_group", "处理中：你好"), ("oc_group", "问答功能未启用")]


def test_handler_silent_when_tenant_not_allowed():
    client = _FakeClient()
    event = parse_message_event(
        _text_event(header={"tenant_key": "tenant_b"})
    )
    _handler(client).handle(event)
    assert client.sent == []


def test_handler_silent_when_chat_not_in_whitelist():
    client = _FakeClient()
    handler = BotHandler(
        client=client,
        policy=EventPolicy(allowed_tenant="tenant_a", allowed_chat_ids=["oc_ok"]),
        bot_open_id="ou_bot",
    )
    handler.handle(parse_message_event(_text_event()))
    assert client.sent == []


def test_handler_silent_when_bot_not_mentioned():
    client = _FakeClient()
    event = parse_message_event(_text_event(event={"message": {"mentions": []}}))
    _handler(client).handle(event)
    assert client.sent == []


def test_handler_ignores_unsupported_message_types():
    client = _FakeClient()
    event = parse_message_event(
        _text_event(event={"message": {"message_type": "image", "content": "{}"}})
    )
    _handler(client).handle(event)
    assert client.sent == []


def _file_event() -> dict:
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt_file",
            "event_type": "im.message.receive_v1",
            "create_time": "1700000000000",
            "token": "tok",
            "app_id": "cli_app",
            "tenant_key": "tenant_a",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_user"}, "sender_type": "user", "tenant_key": "tenant_a"},
            "message": {
                "message_id": "om_f",
                "create_time": "1700000000000",
                "chat_id": "oc_group",
                "chat_type": "group",
                "message_type": "file",
                "content": '{"file_key":"file_abc","file_name":"报告.pdf","file_size":123}',
                "mentions": [{"key": "@_user_1", "id": {"open_id": "ou_bot"}, "name": "助手", "tenant_key": "tenant_a"}],
            },
        },
    }


def test_handler_file_ingestion_sends_placeholder_then_result():
    client = _FakeClient()
    ingestor = _FakeIngestor(
        IngestResult(ok=True, document_id="doc_1", file_name="报告.pdf")
    )
    _handler(client, ingestor=ingestor).handle(parse_message_event(_file_event()))
    assert client.sent == [
        ("oc_group", "正在入库…"),
        ("oc_group", "已入库 1 篇：报告.pdf"),
    ]
    assert ingestor.calls == 1


def test_handler_file_ingestion_failure_reports_reason():
    client = _FakeClient()
    ingestor = _FakeIngestor(
        IngestResult(ok=False, file_name="报告.pdf", error="入库失败（parse_failed）")
    )
    _handler(client, ingestor=ingestor).handle(parse_message_event(_file_event()))
    assert client.sent == [
        ("oc_group", "正在入库…"),
        ("oc_group", "报告.pdf 入库失败：入库失败（parse_failed）"),
    ]


def test_handler_text_without_querier_keeps_ack_receipt():
    client = _FakeClient()
    _handler(client).handle(parse_message_event(_text_event()))
    assert client.sent == [("oc_group", "处理中：你好"), ("oc_group", "问答功能未启用")]


def test_handler_mention_only_asks_for_question():
    client = _FakeClient()
    event = parse_message_event(
        _text_event(event={"message": {"content": '{"text":"@_user_1 "}'}})
    )
    _handler(client).handle(event)
    assert client.sent == [("oc_group", "请 @ 我并说明你的问题，例如：@知识助手 这篇论文的结论是什么？")]


def test_handler_text_question_sends_processing_then_answer_with_citations():
    client = _FakeClient()
    querier = _FakeQuerier(QueryResult(ok=True, answer="结论是……", citations=["报告.pdf"]))
    _handler(client, querier=querier).handle(parse_message_event(_text_event()))
    assert client.sent == [
        ("oc_group", "处理中：你好"),
        ("oc_group", "结论是……\n\n参考：报告.pdf"),
    ]
    assert querier.questions == ["你好"]


def test_handler_final_text_truncated_after_citations_appended():
    """答案+引用拼接后再截断：总长度不超飞书限制。"""
    client = _FakeClient()
    querier = _FakeQuerier(
        QueryResult(ok=True, answer="长" * 3990, citations=["报告.pdf", "附录.docx"])
    )
    _handler(client, querier=querier).handle(parse_message_event(_text_event()))
    text = client.sent[-1][1]
    assert len(text) <= 4000
    assert "截断" in text


def test_handler_text_question_failure_reports_reason():
    client = _FakeClient()
    querier = _FakeQuerier(QueryResult(ok=False, error="问答超时"))
    _handler(client, querier=querier).handle(parse_message_event(_text_event()))
    assert client.sent == [
        ("oc_group", "处理中：你好"),
        ("oc_group", "问答失败：问答超时"),
    ]


def test_handler_mixed_file_then_question_answers_after_ingestion():
    """文件+提问先后到达：问答必须等入库完成（答案能引用刚入库的文档）。"""
    client = _FakeClient()
    order: list[str] = []
    release_ingest = threading.Event()

    class SlowIngestor(_FakeIngestor):
        def ingest_file(self, event):
            order.append("ingest-start")
            release_ingest.wait(3)
            order.append("ingest-end")
            return IngestResult(ok=True, document_id="doc_1", file_name="报告.pdf")

    class OrderedQuerier(_FakeQuerier):
        def answer(self, question):
            order.append("answer")
            return QueryResult(ok=True, answer="基于刚入库文档的答案", citations=["报告.pdf"])

    handler = _handler(
        client,
        ingestor=SlowIngestor(IngestResult(ok=True)),
        querier=OrderedQuerier(QueryResult(ok=True)),
    )

    file_thread = threading.Thread(target=handler.handle, args=(parse_message_event(_file_event()),))
    text_thread = threading.Thread(target=handler.handle, args=(parse_message_event(_text_event()),))
    file_thread.start()
    time.sleep(0.1)  # 确保文件事件先拿到同 chat 的锁
    text_thread.start()
    time.sleep(0.1)
    release_ingest.set()  # 放行入库
    file_thread.join(3)
    text_thread.join(3)

    assert not file_thread.is_alive() and not text_thread.is_alive()
    # 问答严格发生在入库完成之后
    assert order.index("answer") > order.index("ingest-end")
    # 消息序列：入库占位 → 问答占位 → 已入库 → 答案（含引用，命中刚入库文档）
    assert client.sent[:2] == [
        ("oc_group", "正在入库…"),
        ("oc_group", "处理中：你好"),
    ]
    assert client.sent[-2:] == [
        ("oc_group", "已入库 1 篇：报告.pdf"),
        ("oc_group", "基于刚入库文档的答案\n\n参考：报告.pdf"),
    ]


def test_handler_question_waits_for_all_inflight_files():
    """多文件同时入库：提问必须等全部入库完成（不只第一个）。"""
    client = _FakeClient()
    order: list[str] = []
    release = threading.Event()
    ingest_calls = 0

    class MultiSlowIngestor(_FakeIngestor):
        def ingest_file(self, event):
            nonlocal ingest_calls
            ingest_calls += 1
            order.append(f"ingest-start-{ingest_calls}")
            release.wait(3)
            order.append(f"ingest-end-{ingest_calls}")
            return IngestResult(ok=True, document_id=f"doc_{ingest_calls}", file_name=f"f{ingest_calls}.pdf")

    class OrderedQuerier(_FakeQuerier):
        def answer(self, question):
            order.append("answer")
            return QueryResult(ok=True, answer="答案")

    handler = _handler(
        client,
        ingestor=MultiSlowIngestor(IngestResult(ok=True)),
        querier=OrderedQuerier(QueryResult(ok=True)),
    )

    file_threads = [
        threading.Thread(target=handler.handle, args=(parse_message_event(_file_event()),))
        for _ in range(2)
    ]
    text_thread = threading.Thread(target=handler.handle, args=(parse_message_event(_text_event()),))
    file_threads[0].start()
    time.sleep(0.05)
    file_threads[1].start()
    time.sleep(0.05)
    text_thread.start()
    time.sleep(0.1)
    release.set()
    for t in file_threads:
        t.join(3)
    text_thread.join(3)

    assert not any(t.is_alive() for t in file_threads) and not text_thread.is_alive()
    # 问答严格发生在第二个文件也入库完成之后
    assert order.index("answer") > order.index("ingest-end-2")


def test_handler_question_alone_does_not_wait_for_previous_chat():
    """同 chat 没有进行中的入库时，提问不应被阻塞。"""
    client = _FakeClient()
    querier = _FakeQuerier(QueryResult(ok=True, answer="快答案"))
    handler = _handler(client, querier=querier)
    started = time.monotonic()
    handler.handle(parse_message_event(_text_event()))
    assert time.monotonic() - started < 0.5
    assert querier.questions == ["你好"]
