"""LongConnClient 单测：本地 websockets 服务器模拟飞书长连接帧协议。

覆盖：PING→PONG、EVENT→PONG ack + 分发、断线重连。
"""

import asyncio
import json

import pytest
import websockets

from app.services.feishu_bot.long_conn import LongConnClient


class _FakeApi:
    """替身：只提供 fetch_ws_url（鸭子类型 FeishuClient 的长连接所需部分）。"""

    def __init__(self, ws_url: str) -> None:
        self.ws_url = ws_url

    def fetch_ws_url(self, region: str = "cn") -> str:
        return self.ws_url


async def _start_server(handler) -> tuple:
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_ping_gets_pong_and_event_dispatched():
    events: list[dict] = []

    async def server_handler(websocket):
        # PING → 客户端应回 PONG（同 client_msg_id）
        await websocket.send(json.dumps({"type": "PING", "client_msg_id": "p1"}))
        pong1 = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert pong1 == {"type": "PONG", "client_msg_id": "p1"}

        # EVENT → 客户端应回 PONG ack，并把 data 交给 on_event
        await websocket.send(
            json.dumps({"type": "EVENT", "client_msg_id": "e1", "data": {"k": "v"}})
        )
        pong2 = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert pong2 == {"type": "PONG", "client_msg_id": "e1"}
        await asyncio.sleep(0.1)  # 等客户端分发

    server, ws_url = await _start_server(server_handler)
    client = LongConnClient(_FakeApi(ws_url), reconnect_delay=60)
    task = asyncio.create_task(client.run(events.append))
    try:
        await asyncio.wait_for(task, timeout=3)
    except asyncio.TimeoutError:
        pass  # run 是长驻循环，靠 cancel 退出
    finally:
        task.cancel()
        server.close()
    assert events == [{"k": "v"}]


@pytest.mark.asyncio
async def test_reconnects_after_connection_dropped():
    connections = 0
    connected_twice = asyncio.Event()

    async def server_handler(websocket):
        nonlocal connections
        connections += 1
        if connections == 1:
            await websocket.close()  # 第一次连接立即断开
            return
        # 第二次连接：收 PING 并验证 PONG
        await websocket.send(json.dumps({"type": "PING", "client_msg_id": "p2"}))
        pong = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert pong == {"type": "PONG", "client_msg_id": "p2"}
        connected_twice.set()

    server, ws_url = await _start_server(server_handler)
    client = LongConnClient(_FakeApi(ws_url), reconnect_delay=0.05)
    task = asyncio.create_task(client.run(lambda ev: None))
    try:
        await asyncio.wait_for(connected_twice.wait(), timeout=5)
    finally:
        task.cancel()
        server.close()
    assert connections == 2


@pytest.mark.asyncio
async def test_event_ack_before_slow_handler():
    """耗时处理不应阻塞 ack：服务端先收到 PONG 再看到分发结果。"""
    order: list[str] = []

    async def server_handler(websocket):
        await websocket.send(
            json.dumps({"type": "EVENT", "client_msg_id": "e1", "data": {"slow": True}})
        )
        frame = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
        assert frame == {"type": "PONG", "client_msg_id": "e1"}
        order.append("ack")
        await asyncio.sleep(0.3)
        order.append("done")  # 此时慢 handler 应已跑完

    async def slow_handler(event):
        await asyncio.sleep(0.1)
        order.append("handled")

    server, ws_url = await _start_server(server_handler)
    client = LongConnClient(_FakeApi(ws_url), reconnect_delay=60)
    task = asyncio.create_task(client.run(slow_handler))
    try:
        await asyncio.wait_for(task, timeout=3)
    except asyncio.TimeoutError:
        pass
    finally:
        task.cancel()
        server.close()
    assert order.index("ack") < order.index("handled")


@pytest.mark.asyncio
async def test_reconnect_backoff_doubles_on_repeated_failure(monkeypatch):
    """连续断连时重连间隔指数退避（0.01 → 0.02 → 0.04），成功连接后重置。"""
    fetched = asyncio.Event()
    attempts = 0

    class FlakyApi(_FakeApi):
        def fetch_ws_url(self, region: str = "cn") -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 4:
                raise RuntimeError("endpoint down")
            fetched.set()
            return self.ws_url

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.services.feishu_bot.long_conn.sleep", fake_sleep)

    server, ws_url = await _start_server(lambda ws: ws.recv())  # 挂着等取消
    client = LongConnClient(FlakyApi(ws_url), reconnect_delay=0.01)
    task = asyncio.create_task(client.run(lambda ev: None))
    try:
        await asyncio.wait_for(fetched.wait(), timeout=5)
    finally:
        task.cancel()
        server.close()
    # 三次失败对应三次退避：0.01、0.02（翻倍）、0.04（再翻倍）
    assert sleeps == [0.01, 0.02, 0.04]
