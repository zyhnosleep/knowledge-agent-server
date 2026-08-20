"""
inbox.py —— 文件入库链路（T2）：下载 → 上传 ingest → 轮询 ready → 回执
=======================================================================

``ServerApi``：本地 knowledge-agent API 客户端（upload / 文档状态查询），
机器人进程经 HTTP 调用 dev API，不直接触碰数据库。

``InboxIngestor``：一条文件消息的处理编排——下载失败/上传失败/解析失败/
轮询超时均返回带明确原因的 ``IngestResult``，不静默吞错。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.services.feishu_bot.events import MessageEvent

# 文档状态终态：ready 可检索；各 failed 变体为失败终态
TERMINAL_OK = {"ready"}
TERMINAL_FAIL = {
    "failed",
    "parse_failed",
    "table_repair_failed",
    "contextualization_failed",
    "embedding_failed",
    "activation_failed",
}


class ServerApiError(RuntimeError):
    """本地 API 调用失败（业务错误或响应结构不符）。"""


class ServerApi:
    """本地 knowledge-agent API 客户端（同步 httpx）。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _decode(self, response: httpx.Response) -> dict:
        """解析响应；FastAPI 错误形态（detail）或非 JSON 统一转 ServerApiError。"""
        try:
            payload = response.json()
        except ValueError as exc:
            raise ServerApiError(f"API error: HTTP {response.status_code}") from exc
        detail = payload.get("detail")
        if detail:
            raise ServerApiError(str(detail))
        return payload

    def upload(
        self,
        project_slug: str,
        project_name: str,
        filename: str,
        content: bytes,
    ) -> str:
        """上传文件触发摄取流水线（项目不存在会自动创建），返回 document_id。"""
        response = self._client.post(
            "/api/ingest/upload",
            params={"project_slug": project_slug, "project_name": project_name},
            files={"file": (filename, content)},
        )
        response.raise_for_status()
        payload = self._decode(response)
        document_id = payload.get("document_id")
        if not document_id:
            raise ServerApiError("upload response missing document_id")
        return document_id

    def document_status(self, document_id: str) -> str:
        """查询文档状态（ready / processing / failed / parse_failed ...）。"""
        response = self._client.get(f"/api/documents/{document_id}")
        response.raise_for_status()
        payload = self._decode(response)
        status = payload.get("status")
        if not status:
            raise ServerApiError("document response missing status")
        return status

    def query(self, project_slug: str, question: str) -> dict:
        """同步执行一次 Agent 问答（POST /api/agent/query），返回完整响应体。"""
        response = self._client.post(
            "/api/agent/query",
            json={"project_slug": project_slug, "query": question},
            timeout=120.0,
        )
        response.raise_for_status()
        return self._decode(response)


@dataclass
class IngestResult:
    """一条文件消息的入库结果。"""

    ok: bool
    document_id: str | None = None
    file_name: str | None = None
    error: str | None = None


class InboxIngestor:
    """文件入库编排：下载 → 上传 → 轮询 ready → 结果。"""

    def __init__(
        self,
        feishu,
        api: ServerApi,
        *,
        inbox_project: str,
        inbox_project_name: str,
        poll_interval: float = 2.0,
        poll_timeout: float = 120.0,
    ) -> None:
        self._feishu = feishu
        self._api = api
        self._project = inbox_project
        self._project_name = inbox_project_name
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout

    def ingest_file(self, event: MessageEvent) -> IngestResult:
        """处理一条文件消息；任一环节失败返回带原因的结果。"""
        if not event.file_key:
            return IngestResult(ok=False, file_name=event.file_name, error="消息中无文件资源")
        try:
            content = self._feishu.download_file(event.message_id, event.file_key)
        except Exception as exc:
            return IngestResult(ok=False, file_name=event.file_name, error=f"下载失败：{exc}")
        try:
            document_id = self._api.upload(
                self._project, self._project_name, event.file_name or "file", content
            )
        except Exception as exc:
            return IngestResult(ok=False, file_name=event.file_name, error=f"上传失败：{exc}")
        return self._poll(document_id, event.file_name)

    def _poll(self, document_id: str, file_name: str | None) -> IngestResult:
        """轮询文档状态至终态或超时。"""
        deadline = time.monotonic() + self._poll_timeout
        while True:
            try:
                status = self._api.document_status(document_id)
            except Exception as exc:
                return IngestResult(
                    ok=False,
                    document_id=document_id,
                    file_name=file_name,
                    error=f"状态查询失败：{exc}",
                )
            if status in TERMINAL_OK:
                return IngestResult(ok=True, document_id=document_id, file_name=file_name)
            if status in TERMINAL_FAIL:
                return IngestResult(
                    ok=False,
                    document_id=document_id,
                    file_name=file_name,
                    error=f"入库失败（{status}）",
                )
            if time.monotonic() >= deadline:
                return IngestResult(
                    ok=False,
                    document_id=document_id,
                    file_name=file_name,
                    error="入库超时",
                )
            time.sleep(self._poll_interval)
