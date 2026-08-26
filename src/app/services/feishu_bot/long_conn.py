"""Feishu long-connection adapter backed by the official SDK.

Feishu's WebSocket transport uses binary protobuf frames, including control
frames, heartbeats, fragmented payloads, and acknowledgements. Keeping that
wire protocol in the official SDK avoids a connection that handshakes
successfully but never dispatches message events.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol

import lark_oapi as lark
from lark_oapi.ws import Client as LarkWsClient

logger = logging.getLogger(__name__)

OnEvent = Callable[[dict], None]


class WsClientProtocol(Protocol):
    def start(self) -> None: ...


WsClientFactory = Callable[..., WsClientProtocol]


class LongConnClient:
    """Run the official Feishu long-connection client and expose raw events."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        client_factory: WsClientFactory = LarkWsClient,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._client_factory = client_factory

    def run(self, on_event: OnEvent) -> None:
        """Block forever while the SDK handles protobuf frames and reconnects."""

        def dispatch(event: Any) -> None:
            payload = json.loads(lark.JSON.marshal(event))
            on_event(payload)

        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(dispatch)
            .build()
        )
        client = self._client_factory(
            self._app_id,
            self._app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )
        logger.info("feishu official long-connection client starting")
        client.start()
