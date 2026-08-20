"""
long_conn.py —— 飞书长连接（WebSocket）客户端
==============================================

流程：
1. 经 ``FeishuClient.fetch_ws_url`` 拿 ``ws_url``（POST /open-apis/ws/v1/endpoint）。
2. WebSocket 连接；帧协议为 JSON 文本帧：
   - 收 ``{"type":"PING","client_msg_id":...}`` → 回 PONG（同 client_msg_id）。
   - 收 ``{"type":"EVENT","client_msg_id":...,"data":{...}}`` → 先回 PONG ack
     （3 秒超时重推时限内必须应答），再把 ``data`` 交给 ``on_event`` 处理。
   - 其余帧类型忽略。
3. 断线/异常 → 退避重连（``reconnect_delay`` 秒后重试），进程内永续。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import websockets

logger = logging.getLogger(__name__)

OnEvent = Callable[[dict], Awaitable[None] | None]


class LongConnClient:
    """飞书长连接客户端。

    参数：
    - ``feishu_client``：飞书 API 客户端（需具备 ``fetch_ws_url(region=...)``）。
    - ``region``：端点区域（国内 cn）。
    - ``reconnect_delay``：断线后的重连间隔秒数。
    """

    def __init__(self, feishu_client, *, region: str = "cn", reconnect_delay: float = 5.0) -> None:
        self._api = feishu_client
        self._region = region
        self._reconnect_delay = reconnect_delay

    async def run(self, on_event: OnEvent) -> None:
        """长驻运行：连接 → 收帧分发；断线退避重连，直至被取消。"""
        while True:
            try:
                await self._connect_once(on_event)
            except Exception as exc:  # 断线/端点获取失败等，退避重连
                logger.warning(
                    "feishu long connection error: %r; reconnect in %.1fs",
                    exc,
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)

    async def _connect_once(self, on_event: OnEvent) -> None:
        ws_url = self._api.fetch_ws_url(region=self._region)
        async with websockets.connect(ws_url) as ws:
            logger.info("feishu long connection established: %s", ws_url)
            async for raw in ws:
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("feishu ws: dropped non-JSON frame: %r", raw[:200])
                    continue
                await self._handle_frame(ws, frame, on_event)

    async def _handle_frame(self, ws, frame: dict, on_event: OnEvent) -> None:
        frame_type = frame.get("type")
        client_msg_id = frame.get("client_msg_id")
        if frame_type == "PING":
            await ws.send(json.dumps({"type": "PONG", "client_msg_id": client_msg_id}))
            return
        if frame_type == "EVENT":
            # ack 必须先于处理：飞书 3 秒未 ack 会重推事件
            await ws.send(json.dumps({"type": "PONG", "client_msg_id": client_msg_id}))
            await on_event(frame.get("data") or {})
        # PONG / 未知帧：忽略
