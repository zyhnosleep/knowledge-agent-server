"""
ai.py —— AI 模型接入层：Ollama 客户端 / 外部验证器 / 数据结构与工具
==================================================================

职责：
- 封装 Ollama（本地 LLM）的多种调用形态，并提供独立的 DeepSeek
  OpenAI-compatible 文本生成客户端：
  - 普通对话补全（非流式 ``generate_chat`` 与流式 ``stream_chat``）。
  - 结构化输出 ``generate_structured`` / ``generate_structured_with_images``：
    通过 ``format=schema.model_json_schema()`` 强制模型按 JSON Schema
    输出，解析失败时自动退化为 ``format=json`` 的宽松 JSON 模式重试。
- 文本嵌入 ``embed``（本地 Ollama 或 Qwen3-Embedding-4B API）与模型卸载
  ``unload_loaded_models``。
- 提供一套用于文档分析 / 知识抽取的数据模型（Pydantic BaseModel），
  描述实体、三元组、页面解析、三重验证、增长决策、抽取结果等。
- 提供 ``ContextualizationOllamaClient``：把 Ollama 传输层"冻结"到
  contextualization（上下文增强）专用模型配置。
- 提供 ``ExternalVerifier``：对接外部 API（OpenAI 兼容 chat/completions
  + json_schema 响应格式）验证文档 claims 是否被摘要支持。
- 提供向量相似度计算、可重试错误判定与安全调用工具函数。

结构说明：
- 本模块同时充当"传输层"与"数据层"：Pydantic 模型既用于请求 Schema
  约束，也用于解析响应。所有模型均为纯数据类，无副作用。
"""

from __future__ import annotations

import base64
import json
import logging
import time
import math
import threading
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import httpx
from pydantic import BaseModel, Field, PrivateAttr

from app.core.config import get_settings
from app.services.execution_budget import BudgetExceeded, current_execution_budget

logger = logging.getLogger(__name__)
settings = get_settings()
# 泛型参数：绑定到 BaseModel 子类，用于结构化输出类型约束
SchemaT = TypeVar("SchemaT", bound=BaseModel)
# 哨兵对象：区分"未显式传入 keep_alive"与"显式传 None"
_DEFAULT_KEEP_ALIVE = object()


def request_timeout(configured: float) -> float:
    budget = current_execution_budget()
    return budget.http_timeout(configured) if budget else configured


@contextmanager
def _model_request(payload: dict[str, Any], timeout: float, *, ollama: bool):
    """Admit one actual attempt; failed/unknown attempts keep their reserve."""
    if ollama:
        payload = dict(payload)
        options = dict(payload.get('options') or {})
        options.setdefault('num_ctx', settings.ollama_generation_context_length)
        options.setdefault('num_predict', settings.generation_max_output_tokens)
        payload['options'] = options
    budget = current_execution_budget()
    reservation = None
    if budget is not None:
        options = dict(payload.get('options') or {})
        context_length = options.get('num_ctx', settings.ollama_generation_context_length)
        output_limit = options.get('num_predict', settings.generation_max_output_tokens) if ollama else payload.get('max_tokens', settings.generation_max_output_tokens)
        messages = list(payload.get('messages') or [])
        if isinstance(payload.get('format'), dict):
            messages.append({'role':'format', 'content':json.dumps(payload['format'], ensure_ascii=False)})
        if payload.get('response_format'):
            messages.append({'role':'format', 'content':json.dumps(payload['response_format'], ensure_ascii=False)})
        reservation = budget.reserve_model_call(messages=messages, model=str(payload.get('model', '')),
            max_output_tokens=output_limit, context_length=context_length)
        payload = dict(payload)
        if ollama:
            options['num_ctx'] = context_length
            options['num_predict'] = reservation.max_output_tokens
            payload['options'] = options
        else:
            payload['max_tokens'] = reservation.max_output_tokens
    try:
        yield payload, request_timeout(timeout), reservation
    finally:
        if reservation is not None and not reservation.settled:
            budget.finish_model_call(reservation, prompt_tokens=None, completion_tokens=None)


def _settle_model_response(reservation, data: dict[str, Any], *, ollama: bool):
    budget = current_execution_budget()
    if budget is not None and reservation is not None and not reservation.settled:
        usage = data if ollama else data.get('usage', {})
        usage = usage if isinstance(usage, dict) else {}
        budget.finish_model_call(reservation,
            prompt_tokens=usage.get('prompt_eval_count' if ollama else 'prompt_tokens'),
            completion_tokens=usage.get('eval_count' if ollama else 'completion_tokens'))
        budget.check_deadline()


def budgeted_completion_post(url: str, *, payload: dict[str, Any], timeout: float,
                             headers: dict[str, str]) -> dict[str, Any]:
    """Shared OpenAI-compatible HTTP boundary (including legacy verifier)."""
    with _model_request(payload, timeout, ollama=False) as (bounded, remaining, reservation):
        with httpx.Client(timeout=remaining) as client:
            data = _budgeted_json_post(client, url, payload=bounded, headers=headers)
        _settle_model_response(reservation, data, ollama=False)
        return data


@contextmanager
def _stream_budget_guard(response: httpx.Response, cancel_event: threading.Event | None):
    """Interrupt a blocked body read, not merely reject its next yielded line.

    HTTPX read timeouts are inactivity limits. Its native transport exposes
    the network stream: shutting down that request's socket wakes recv before
    closing the response. Custom transports must make close interrupt reads.
    """
    budget = current_execution_budget()
    if budget is None and cancel_event is None:
        yield
        return
    finished = threading.Event()

    def watch():
        while not finished.is_set():
            expired = cancel_event is not None and cancel_event.is_set()
            if budget is not None:
                try:
                    budget.check_deadline()
                except BudgetExceeded:
                    expired = True
            if expired:
                network_stream = response.extensions.get('network_stream')
                try:
                    native_socket = network_stream.get_extra_info('socket') if network_stream else None
                    if native_socket is not None:
                        native_socket.shutdown(socket.SHUT_RDWR)
                except (OSError, AttributeError):
                    pass  # Already closed; response.close is still required.
                try:
                    response.close()
                except Exception:
                    logger.debug('Interrupted stream close failed', exc_info=True)
                return
            delay = min(0.05, max(0.001, budget.deadline - budget.clock())) if budget else 0.05
            finished.wait(delay)

    watcher = threading.Thread(target=watch, name='model-stream-budget', daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join(timeout=0.2)


def _budgeted_json_post(client, url, *, payload, headers=None, timeout=None):
    """Guard nonstream JSON from connection completion through headers/body.

    HTTP core's trace extension exposes this request's native stream as soon
    as TCP/TLS connects, before response headers. No global socket scanning,
    background model-call thread or uncancellable response-body wait.
    """
    budget = current_execution_budget()
    kwargs = {'json': payload}
    if headers is not None:
        kwargs['headers'] = headers
    if timeout is not None:
        kwargs['timeout'] = timeout
    if budget is None:
        response = client.post(url, **kwargs)
        response.raise_for_status()
        return response.json()
    budget.check_deadline()
    # Every guarded request owns a fresh connection, including embedding batches;
    # an earlier pooled stream must not escape the pre-header trace hook.
    request_headers = httpx.Headers(headers or {})
    request_headers['Connection'] = 'close'
    kwargs['headers'] = request_headers
    class OwnedRequest:
        def __init__(self):
            self.extensions = {}
            self.response = None
        def close(self):
            if self.response is not None:
                self.response.close()
            client.close()
    owned = OwnedRequest()
    def trace(event, info):
        if event in ('connection.connect_tcp.complete', 'connection.connect_unix_socket.complete', 'connection.start_tls.complete'):
            stream = info.get('return_value')
            if stream is not None:
                owned.extensions['network_stream'] = stream
                # Cancellation may have arrived before connection completed.
                try:
                    budget.check_deadline()
                except BudgetExceeded:
                    native_socket = stream.get_extra_info('socket')
                    if native_socket is not None:
                        try:
                            native_socket.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    raise
    try:
        with _stream_budget_guard(owned, budget.cancel_event):
            with client.stream('POST', url, extensions={'trace': trace}, **kwargs) as response:
                owned.response = response
                budget.check_deadline()
                response.raise_for_status()
                response.read()
                budget.check_deadline()
                return response.json()
    except Exception:
        budget.check_deadline()  # An interrupted transport is a budget stop, not a retry.
        raise


class ExtractedEntity(BaseModel):
    """抽取出的概念/实体。"""

    name: str  # 实体名称
    entity_type: str = "concept"  # 实体类型（默认概念）
    summary: str = ""  # 实体摘要
    aliases: list[str] = Field(default_factory=list)  # 别名列表


class ExtractedClaim(BaseModel):
    """知识三元组（主谓宾结构）及验证信息。"""

    subject: str  # 主语
    predicate: str  # 谓语/关系
    object_text: str  # 宾语
    expected_head: str = ""  # 期望归属的实体（head）
    evidence_excerpt: str = ""  # 证据摘录
    confidence: float = 0.5  # 置信度
    source_chunk_ordinals: list[int] = Field(default_factory=list)  # 来源分块序号
    source_sentence_refs: list[str] = Field(default_factory=list)  # 来源句引用
    relation_type: str = "fact"  # 关系类型（默认事实）
    verification_errors: list[str] = Field(default_factory=list)  # 验证错误
    growth_decision: str = "keep"  # 增长决策（保留/丢弃等）


class GeneratedTriple(BaseModel):
    """模型生成的原始三元组（不含验证字段）。"""

    subject: str
    predicate: str
    object_text: str
    expected_head: str = ""
    evidence_excerpt: str = ""
    confidence: float = 0.5
    source_chunk_ordinals: list[int] = Field(default_factory=list)
    source_sentence_refs: list[str] = Field(default_factory=list)
    relation_type: str = "fact"


class DocumentAnalysisPayload(BaseModel):
    """整篇文档的解析结果负载。"""

    title: str  # 标题
    summary: str  # 摘要
    keywords: list[str] = Field(default_factory=list)  # 关键词
    key_facts: list[str] = Field(default_factory=list)  # 关键事实
    entities: list[ExtractedEntity] = Field(default_factory=list)  # 实体
    concepts: list[str] = Field(default_factory=list)  # 概念列表
    triples: list[GeneratedTriple] = Field(default_factory=list)  # 三元组
    coverage_notes: list[str] = Field(default_factory=list)  # 覆盖说明


class DocumentPagePayload(BaseModel):
    """单个页面（Page）的解析结果负载。"""

    page_label: str  # 页码标签
    page_summary: str = ""  # 页摘要
    page_markdown: str = ""  # 页 Markdown
    sections: list[str] = Field(default_factory=list)  # 章节
    tables: list[str] = Field(default_factory=list)  # 表格
    figures: list[str] = Field(default_factory=list)  # 图表
    formulas: list[str] = Field(default_factory=list)  # 公式
    key_facts: list[str] = Field(default_factory=list)  # 关键事实
    entities: list[ExtractedEntity] = Field(default_factory=list)  # 实体
    evidence_spans: list[str] = Field(default_factory=list)  # 证据片段
    coverage_notes: list[str] = Field(default_factory=list)  # 覆盖说明
    # 私有字段：解析来源（默认 document_intelligence）
    _analysis_source: str = PrivateAttr(default="document_intelligence")

    @property
    def analysis_source(self) -> str:
        """只读属性：该页解析结果的分析来源标识。"""
        return self._analysis_source


class HeadAnalysisPayload(BaseModel):
    """核心实体（head entity）分析报告。"""

    head_entity: str  # 核心实体名
    summary: str = ""  # 摘要
    key_facts: list[str] = Field(default_factory=list)  # 关键事实
    triples: list[GeneratedTriple] = Field(default_factory=list)  # 三元组
    related_entities: list[ExtractedEntity] = Field(default_factory=list)  # 相关实体
    related_concepts: list[str] = Field(default_factory=list)  # 相关概念
    coverage_notes: list[str] = Field(default_factory=list)  # 覆盖说明


class TripleVerificationPayload(BaseModel):
    """三元组批量验证结果。"""

    verdict: str = "local-only"  # 总判定（默认 local-only）
    accepted_indexes: list[int] = Field(default_factory=list)  # 通过的下标
    rejected_indexes: list[int] = Field(default_factory=list)  # 拒绝的下标
    errors_by_index: dict[str, list[str]] = Field(default_factory=dict)  # 各下标错误
    coverage_notes: list[str] = Field(default_factory=list)  # 覆盖说明


class GrowthDecision(BaseModel):
    """单条增长决策。"""

    name: str  # 决策对象名
    item_type: str = "entity"  # 类型（实体/概念等）
    decision: str = "keep"  # 决策结果（keep 等）
    reason: str = ""  # 决策理由


class GrowthDecisionPayload(BaseModel):
    """多条增长决策的打包负载。"""

    decisions: list[GrowthDecision] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    """文档最终抽取结果（供入库/检索）。"""

    title: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)  # 知识三元组（带验证）
    key_facts: list[str] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)


class QueryAnswerPayload(BaseModel):
    """问答系统答案负载。"""

    answer_markdown: str  # Markdown 答案正文
    citations: list[int] = Field(default_factory=list)  # 引用下标
    risk_level: str = "normal"  # 风险等级


class VerificationPayload(BaseModel):
    """外部验证结果负载。"""

    verdict: str  # 判定
    notes: str  # 说明
    flagged_claim_indexes: list[int] = Field(default_factory=list)  # 被标记的下标


@dataclass
class SearchHit:
    """检索命中项。"""

    chunk_id: str
    document_id: str
    score: float
    page_label: str | None
    excerpt: str  # 分块内容摘要


class OllamaClient:
    """Ollama HTTP 客户端封装。

    通过 ``/api/chat`` 与 ``/api/embed`` 端点与本地 Ollama 服务通信。
    结构化输出依赖 Ollama 的 ``format`` 参数（JSON Schema 或 ``"json"``）。
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        """初始化客户端：生成/嵌入两个 base_url 与超时配置。

        默认读取全局配置 ``settings.ollama_generation_base_url`` 与
        ``settings.ollama_embedding_base_url``；尾随斜杠被剥掉，便于拼接。
        """
        self.base_url = (base_url or settings.ollama_generation_base_url).rstrip("/")
        self.embedding_base_url = (
            embedding_base_url or settings.ollama_embedding_base_url
        ).rstrip("/")
        self.timeout = settings.ollama_request_timeout

    def generate_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        context_length: int,
        max_output_tokens: int = 768,
    ) -> dict[str, Any]:
        """Generate one non-streaming Markdown response with bounded context.

        生成一次非流式的 Markdown 响应，并限制上下文长度。

        - ``stream=False``：一次返回完整结果。
        - ``think=False``：关闭推理模式（部分模型支持）。
        - ``options.num_ctx``：模型上下文窗口大小。
        - ``options.num_predict``：最大输出 token 数。
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": context_length,
                "num_predict": max_output_tokens,
            },
        }
        data = self._post_chat(payload)
        return self._chat_result(data)

    def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        context_length: int,
        max_output_tokens: int = 768,
        cancel_event: threading.Event | None = None,
    ):
        """Yield Ollama NDJSON chat chunks until completion or cancellation.

        流式生成：逐行 yield Ollama 的 NDJSON 聊天分块，直到完成或取消。

        - 使用 ``client.stream`` 保持连接并逐行读取。
        - ``cancel_event`` 被置位时提前终止（用于客户端取消/超时）。
        - 每行解析为 JSON，标准化为 ``_chat_result``，并附加 ``done``
          标志标识流是否结束。
        """
        payload = self._with_keep_alive(
            {
                "model": model,
                "messages": messages,
                "stream": True,
                "think": False,
                "options": {
                    "num_ctx": context_length,
                    "num_predict": max_output_tokens,
                },
            }
        )
        with _model_request(payload, self.timeout, ollama=True) as (bounded, remaining, reservation):
            with httpx.Client(timeout=remaining) as client:
                with client.stream('POST', f'{self.base_url}/api/chat', json=bounded) as response, _stream_budget_guard(response, cancel_event):
                    response.raise_for_status()
                    lines = iter(response.iter_lines())
                    while True:
                        budget = current_execution_budget()
                        if budget:
                            budget.check_deadline()
                        if cancel_event is not None and cancel_event.is_set():
                            if budget:
                                raise BudgetExceeded('cancelled')
                            break
                        try:
                            line = next(lines)
                        except StopIteration:
                            if budget:
                                budget.check_deadline()
                            if cancel_event is not None and cancel_event.is_set() and budget:
                                raise BudgetExceeded('cancelled')
                            break
                        except (httpx.TransportError, OSError):
                            if budget:
                                budget.check_deadline()
                            if cancel_event is not None and cancel_event.is_set():
                                if budget:
                                    raise BudgetExceeded('cancelled')
                                break
                            raise
                        if budget:
                            budget.check_deadline()
                        if cancel_event is not None and cancel_event.is_set():
                            if budget:
                                raise BudgetExceeded('cancelled')
                            break
                        if not line:
                            continue
                        data = json.loads(line)
                        result = self._chat_result(data)
                        result['done'] = bool(data.get('done'))
                        if result['done']:
                            _settle_model_response(reservation, data, ollama=True)
                        yield result
                        if result['done']:
                            break

    def generate_structured(
        self,
        schema: type[SchemaT],
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        think: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> SchemaT:
        """按 JSON Schema 生成结构化输出；解析失败时退化重试为 JSON 模式。

        参数：
        - ``schema``：目标 Pydantic 模型，决定输出结构。
        - ``system_prompt`` / ``user_prompt``：提示词。
        - ``model``：模型名，默认 ``settings.ollama_generation_model``。
        - ``think``：是否启用推理链，默认 ``False``（与 ``generate_chat``
          对齐；qwen3.5 推理模型开启思考链会显著拖慢生成）；``options``
          透传给 Ollama。

        实现要点：
        1. ``format=schema.model_json_schema()`` —— 关键：强制大模型按
           JSON Schema 输出（Ollama 的 guided 生成）。
        2. 首次解析失败（模型输出不合规）时，降级为 ``format=json``
           的宽松模式再试一次，并在日志中记录告警。
        """
        model_name = model or settings.ollama_generation_model
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": schema.model_json_schema(),  # 关键：强制大模型按JSON Schema输出
            # 默认关闭推理链（与 generate_chat 对齐）：qwen3.5 是推理模型，
            # 不传 think 时思考链默认开启，生成耗时实测慢约 6 倍。
            "think": False if think is None else think,
        }
        if options:
            payload["options"] = options
        data = self._post_chat(payload)
        content = self._message_content(data)
        try:
            return self._parse_structured_content(schema, content)
        except Exception as exc:  # noqa: BLE001
            budget = current_execution_budget()
            if isinstance(exc, BudgetExceeded) or (budget and not budget.consume_format_retry()):
                raise
            strict_json = bool(getattr(schema, "strict_json_only", False))
            # 模型原始输出随异常透传（调用方"自由文本接受"决策依据）：
            # ollama 对 qwen3.5 的 format=schema 是提示式软约束，模型高频
            # 输出自然语言回答而非 JSON，原始文本必须到达调用方才能被接受。
            try:
                if not strict_json:
                    exc.add_note(f"raw_content={str(content)[:6000]}")
            except Exception:  # noqa: BLE001
                pass
            # Schema 约束输出失败：退回宽松 JSON 模式重试
            if strict_json:
                logger.warning("Invalid strict decision JSON; using shared format repair (%s)", schema.__name__)
            else:
                logger.warning(
                    "Structured schema response was invalid; retrying with JSON mode: %s (raw=%.500s)",
                    exc, str(content)[:500],
                )
            retry_payload = self._json_mode_payload(
                schema=schema,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                think=think,
                options=options,
            )
            retry_data = self._post_chat(retry_payload)
            retry_content = self._message_content(retry_data)
            try:
                return self._parse_structured_content(schema, retry_content)
            except Exception as retry_exc:  # noqa: BLE001
                try:
                    if not strict_json:
                        retry_exc.add_note(f"raw_content={str(retry_content)[:6000]}")
                except Exception:  # noqa: BLE001
                    pass
                # JSON 模式重试也失败：记录原始输出后原样抛出（调用方重试/兜底）
                if strict_json:
                    logger.warning("Strict decision format repair failed (%s)", schema.__name__)
                else:
                    logger.warning(
                        "JSON mode retry also failed for %s: %s (raw=%.500s)",
                        schema.__name__, retry_exc, str(retry_content)[:500],
                    )
                raise

    def generate_structured_with_images(
        self,
        schema: type[SchemaT],
        *,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes | str | Path],
        model: str | None = None,
    ) -> SchemaT:
        """带图片输入的结构化生成（视觉模型）。

        - 模型默认取 ``settings.ollama_vision_model``，缺省时退回
          ``settings.ollama_generation_model``。
        - 图片统一编码为 base64 字符串（``_encode_image``），放入 user
          message 的 ``images`` 字段。
        - 同样先 Schema 约束，失败则 JSON 模式重试。
        """
        model_name = model or settings.ollama_vision_model or settings.ollama_generation_model
        encoded_images = [self._encode_image(image) for image in images]
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": encoded_images,
                },
            ],
            "stream": False,
            "format": schema.model_json_schema(),
        }
        data = self._post_chat(payload)
        content = self._message_content(data)
        try:
            return self._parse_structured_content(schema, content)
        except Exception as exc:  # noqa: BLE001
            budget = current_execution_budget()
            if isinstance(exc, BudgetExceeded) or (budget and not budget.consume_format_retry()):
                raise
            logger.warning("Structured vision response was invalid; retrying with JSON mode: %s", exc)
            retry_payload = self._json_mode_payload(
                schema=schema,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=encoded_images,
            )
            retry_data = self._post_chat(retry_payload)
            return self._parse_structured_content(schema, self._message_content(retry_data))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量计算文本嵌入向量。

        POST ``/api/embed``；返回 ``embeddings`` 数组。

        显存管理（与生成模型共卡的关键）：
        - ``keep_alive=0``：嵌入模型用完即卸载，绝不驻留与生成模型
          争抢显存（曾因 5m 驻留窗口挤掉生成模型导致 schema 输出崩坏）。
        - ``options.num_ctx=16384``：嵌入模型不需要 32K 上下文；16384
          覆盖现存全部 chunk（实测 max 16859 字符的表格块，token 化
          最坏情况接近 1 字符/token，仍需 8K+），KV cache ~3.2G，
          与生成模型共存峰值 ~18G < 23G。若未来出现 >16K token 的
          chunk，应修切块上限而非继续放大 num_ctx（超长 chunk 的
          嵌入向量信息平均化，检索精度本就差）。注意此处不走
          ``_with_keep_alive``（全局配置会覆盖显式值），body 直接构造。
        """
        if not texts:
            return []
        provider = settings.active_embedding_provider
        if provider == "openai-compatible":
            return self._embed_openai_compatible(texts)
        if provider != "ollama":
            raise ValueError(
                "Unsupported EMBEDDING_PROVIDER: "
                f"{settings.embedding_provider!r}; expected 'ollama' or "
                "'openai-compatible'."
            )
        with httpx.Client(timeout=request_timeout(self.timeout)) as client:
            return _budgeted_json_post(client, f"{self.embedding_base_url}/api/embed", payload={
                    "model": settings.ollama_embedding_model,
                    "input": texts,
                    "keep_alive": "0",
                    "options": {"num_ctx": 16384},
                })["embeddings"]

    def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        """Call a hosted OpenAI-compatible embeddings endpoint.

        The endpoint is intentionally vendor-neutral.  ``EMBEDDING_API_BASE_URL``
        points at the API root (normally ending in ``/v1``); this method appends
        ``/embeddings``.  Response rows are sorted by their explicit ``index`` so
        batching cannot silently reorder document chunks.
        """
        base_url = (settings.embedding_api_base_url or "").strip().rstrip("/")
        api_key = settings.embedding_api_key
        if not base_url:
            raise ValueError(
                "EMBEDDING_API_BASE_URL is required when "
                "EMBEDDING_PROVIDER=openai-compatible."
            )
        if not api_key or api_key.strip().upper() in {"CHANGE_ME", "YOUR_API_KEY"}:
            raise ValueError(
                "EMBEDDING_API_KEY is required when "
                "EMBEDDING_PROVIDER=openai-compatible."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        embeddings: list[list[float]] = []
        batch_size = settings.embedding_api_batch_size
        with httpx.Client(timeout=request_timeout(settings.embedding_api_timeout)) as client:
            for offset in range(0, len(texts), batch_size):
                budget = current_execution_budget()
                if budget:
                    budget.check_deadline()
                batch = texts[offset : offset + batch_size]
                payload = _budgeted_json_post(client, f"{base_url}/embeddings", headers=headers,
                    payload={"model": settings.embedding_api_model, "input": batch},
                    timeout=budget.http_timeout(settings.embedding_api_timeout) if budget else None)
                rows = payload.get("data")
                if not isinstance(rows, list):
                    raise ValueError("Embedding API response is missing a data array.")
                if len(rows) != len(batch) or any(
                    not isinstance(row, dict) for row in rows
                ):
                    raise ValueError(
                        "Embedding API returned an invalid number or shape of embeddings."
                    )
                # OpenAI-compatible servers normally include an integer ``index``.
                # A few providers omit it while preserving response order, so
                # accept the all-omitted form but reject partial, duplicate, or
                # out-of-range indices within each batch.  The batch-local index
                # is intentional: concatenating validated batches preserves the
                # original document order without assuming a global index.
                indices = [row.get("index") for row in rows]
                if all(index is None for index in indices):
                    ordered = rows
                elif all(
                    isinstance(index, int) and not isinstance(index, bool)
                    for index in indices
                ):
                    expected_indices = set(range(len(batch)))
                    actual_indices = {
                        index for index in indices if isinstance(index, int)
                    }
                    if actual_indices != expected_indices:
                        raise ValueError(
                            "Embedding API returned invalid or duplicate data indices."
                        )
                    ordered = sorted(rows, key=lambda row: row["index"])
                else:
                    raise ValueError(
                        "Embedding API returned invalid or partial data indices."
                    )

                raw_embeddings = [row.get("embedding") for row in ordered]
                if any(not isinstance(item, list) for item in raw_embeddings):
                    raise ValueError(
                        "Embedding API returned an invalid number or shape of embeddings."
                    )
                dimensions = {len(item) for item in raw_embeddings}
                expected_dimensions = settings.active_embedding_dimensions
                if dimensions != {expected_dimensions}:
                    raise ValueError(
                        "Embedding API dimensions do not match the configured index: "
                        f"received {sorted(dimensions)}, expected "
                        f"{expected_dimensions}."
                    )
                for item in raw_embeddings:
                    vector: list[float] = []
                    for value in item:
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            raise ValueError(
                                "Embedding API returned a non-numeric vector value."
                            )
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise ValueError(
                                "Embedding API returned a non-finite vector value."
                            )
                        vector.append(numeric)
                    embeddings.append(vector)
        return embeddings

    def unload_loaded_models(self) -> list[str]:
        """卸载当前加载的全部模型（内存回收）。

        流程：
        1. 对生成与嵌入两个 base_url 去重后逐个查询 ``/api/ps``
           获取已加载模型列表。
        2. 对每个模型 POST ``/api/generate`` 且 ``keep_alive=0``，
           指示 Ollama 释放该模型内存。

        返回：被卸载的模型名列表。
        """
        unloaded: list[str] = []
        # dict.fromkeys 去重并保持顺序
        for base_url in dict.fromkeys((self.base_url, self.embedding_base_url)):
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{base_url}/api/ps")
                response.raise_for_status()
                models = response.json().get("models") or []
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    model = str(item.get("name") or item.get("model") or "").strip()
                    if not model:
                        continue
                    # keep_alive=0 表示立即卸载
                    release = client.post(
                        f"{base_url}/api/generate",
                        json={"model": model, "keep_alive": 0},
                    )
                    release.raise_for_status()
                    unloaded.append(model)
        return unloaded

    @classmethod
    def _chat_result(cls, data: dict[str, Any]) -> dict[str, Any]:
        """把 Ollama chat 响应标准化为紧凑结果字典（含 token 统计）。

        统计字段：prompt_eval_count / eval_count / 各阶段耗时
        （load/prompt_eval/eval/total duration），便于监控与日志。
        """
        return {
            "content": str(cls._message_content(data) or ""),
            "model": str(data.get("model") or ""),
            "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
            "eval_count": int(data.get("eval_count") or 0),
            "prompt_eval_duration": int(data.get("prompt_eval_duration") or 0),
            "eval_duration": int(data.get("eval_duration") or 0),
            "total_duration": int(data.get("total_duration") or 0),
            "load_duration": int(data.get("load_duration") or 0),
        }

    @staticmethod
    def _encode_image(image: bytes | str | Path) -> str:
        """把图片编码为 base64 字符串（供 Ollama ``images`` 字段）。

        - bytes：直接编码。
        - str / Path：若是存在的文件路径则读取文件内容编码；
          否则视为已是 base64 字符串，原样返回。
        """
        if isinstance(image, bytes):
            data = image
        else:
            path = Path(image)
            if path.exists():
                data = path.read_bytes()
            else:
                # 非文件路径：视为已是 base64 文本
                return str(image)
        return base64.b64encode(data).decode("utf-8")

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/chat（自动注入 keep_alive）并返回 JSON。"""
        with _model_request(payload, self.timeout, ollama=True) as (bounded, remaining, reservation):
            with httpx.Client(timeout=remaining) as client:
                data = _budgeted_json_post(client, f'{self.base_url}/api/chat', payload=self._with_keep_alive(bounded))
            _settle_model_response(reservation, data, ollama=True)
            return data

    @staticmethod
    def _with_keep_alive(payload: dict[str, Any]) -> dict[str, Any]:
        """按全局配置向请求注入 ``keep_alive``（模型驻留时间）。

        - 配置为 None：不注入。
        - 配置为字符串且内容为纯数字（可带 +/- 前缀）：转为 int。
        """
        if settings.ollama_keep_alive is None:
            return payload
        keep_alive: str | int = settings.ollama_keep_alive
        if isinstance(keep_alive, str):
            normalized = keep_alive.strip()
            if normalized.lstrip("+-").isdigit():
                keep_alive = int(normalized)
        return {**payload, "keep_alive": keep_alive}

    @staticmethod
    def _message_content(data: dict[str, Any]) -> Any:
        """提取响应中的 message.content。

        content 可能是字符串，也可能是多段内容列表（每段含 text 字段）；
        列表时拼接所有段的文本。
        """
        content = (data.get("message") or {}).get("content", "")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        return content

    @staticmethod
    def _json_mode_payload(
        *,
        schema: type[SchemaT],
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
        think: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构造"宽松 JSON 模式"的请求负载（Schema 约束失败后的重试）。

        在 user prompt 中追加：
        - "只返回一个合法 JSON 对象，不要 Markdown/代码围栏/解释文本。"
        - 期望的紧凑 JSON 形状示例（``_compact_schema_shape`` 生成），
          并允许证据缺失时空数组。

        使用 ``format="json"`` 而非完整 Schema，给模型更大的容错空间。
        """
        content = "\n\n".join(
            [
                user_prompt,
                "Return only one valid JSON object. Do not include markdown, code fences, or explanatory text.",
                "Use this compact JSON shape. Empty arrays are allowed when evidence is missing:",
                json.dumps(getattr(schema, "json_retry_example", None) or
                           OllamaClient._compact_schema_shape(schema), ensure_ascii=False),
            ]
        )
        user_message: dict[str, Any] = {"role": "user", "content": content}
        if images:
            user_message["images"] = images
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "stream": False,
            "format": "json",
            "think": False if think is None else think,
        }
        if options:
            payload["options"] = options
        return payload

    @classmethod
    def _compact_schema_shape(cls, annotation: Any, depth: int = 0) -> Any:
        """根据类型注解生成"紧凑 JSON 形状"示例值（供 prompt 展示）。

        映射规则（按类型）：
        - list[T] -> [示例(T)]；dict -> {}。
        - Optional[T]（含 None 联合）-> 展开为 T 的示例。
        - BaseModel 子类 -> 字段名到示例的字典；嵌套超过 2 层时返回 {}
          以避免 prompt 膨胀。
        - 基础类型：str -> ""、int -> 0、float -> 0.0、bool -> False。
        - 其他 -> None。
        """
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is list:
            item_type = args[0] if args else Any
            return [cls._compact_schema_shape(item_type, depth + 1)]
        if origin is dict:
            return {}
        # Optional[T] / Union[T, None]：取非 None 分支
        if origin is not None and args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            return cls._compact_schema_shape(non_none_args[0], depth) if non_none_args else None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if depth >= 2:
                return {}  # 深嵌套截断，避免 prompt 过长
            return {
                name: cls._compact_schema_shape(field.annotation, depth + 1)
                for name, field in annotation.model_fields.items()
            }
        if annotation is str:
            return ""
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False
        return None

    @classmethod
    def _parse_structured_content(cls, schema: type[SchemaT], content: Any) -> SchemaT:
        """把模型返回内容解析为指定 schema 的实例。

        依次尝试：
        1. content 已是 dict / list：直接 ``model_validate``。
        2. 空文本：抛 ValueError。
        3. 文本整体 JSON 解析（``model_validate_json``）。
        4. 失败则从文本中提取第一个 JSON 值（``_extract_first_json_value``，
           处理 Markdown 围栏/前后缀噪音）再验证。
        """
        if isinstance(content, dict):
            return schema.model_validate(content)
        if isinstance(content, list):
            return schema.model_validate(content)

        text = str(content or "").strip()
        if not text:
            raise ValueError(f"Ollama returned empty content for {schema.__name__}.")

        if getattr(schema, "strict_json_only", False):
            def unique_object(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate decision JSON key")
                    result[key] = value
                return result
            return schema.model_validate(json.loads(text, object_pairs_hook=unique_object))

        try:
            return schema.model_validate_json(text)
        except Exception:  # noqa: BLE001
            # 整体解析失败：尝试从文本中挖出第一个 JSON 值
            extracted = cls._extract_first_json_value(text)
            return schema.model_validate(extracted)

    @staticmethod
    def _extract_first_json_value(text: str) -> Any:
        """从文本中提取第一个合法的 JSON 值（含列表/对象）。

        处理步骤：
        1. 去除首尾空白；若是 Markdown 代码围栏（```...```）则剥掉围栏，
           并去掉可选的 ``json`` 语言标注。
        2. 逐字符扫描，找到 ``{`` 或 ``[`` 即尝试 ``raw_decode``；
           成功即返回解析出的值，失败继续向后扫描。

        抛 ValueError：文本中不存在任何合法 JSON 值。
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
        raise ValueError("No valid JSON object found in Ollama response.")


class DeepSeekAPIError(RuntimeError):
    """A classified DeepSeek transport or response error.

    ``status_code`` and ``retryable`` are retained for trace/audit callers so
    they can distinguish credential failures (401/403), throttling (429),
    transient server failures (5xx), and exhausted network retries without
    parsing provider-specific exception strings.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.error_code = error_code


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek chat client with bounded retries.

    The client is deliberately independent from ``OllamaClient``: generation
    credentials, timeout, retry budget, model identity, and usage accounting
    come from the DeepSeek-specific settings.  It never logs the API key and
    does not retry credential errors (401/403).
    """

    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.deepseek_api_key
        self.model = model or settings.deepseek_model
        self.timeout = (
            timeout if timeout is not None else settings.generation_timeout_seconds
        )
        self.max_retries = (
            max_retries if max_retries is not None else settings.generation_max_retries
        )
        self.retry_backoff_seconds = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else settings.generation_retry_backoff_seconds
        )

    def generate_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate one non-streaming response and preserve provider usage.

        Retries are bounded by ``GENERATION_MAX_RETRIES``.  Only 429/5xx and
        transport timeouts/connectivity errors are retried; malformed response
        payloads and authentication failures fail immediately.
        """
        if not self.base_url:
            raise DeepSeekAPIError("DEEPSEEK_BASE_URL is required.")
        if not self.api_key or self.api_key.strip().upper() in {"CHANGE_ME", "YOUR_API_KEY"}:
            raise DeepSeekAPIError(
                "DEEPSEEK_API_KEY is required.", error_code="missing_credentials"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_output_tokens or settings.generation_max_output_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: DeepSeekAPIError | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            try:
                with _model_request(payload, self.timeout, ollama=False) as (bounded, remaining, reservation):
                    with httpx.Client(timeout=remaining) as client:
                        response = client.post(
                        f"{self.base_url}/chat/completions",
                        json=bounded,
                        headers=headers,
                        )
                    if 200 <= response.status_code < 300:
                        _settle_model_response(reservation, response.json(), ollama=False)
            except BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001
                retryable = isinstance(
                    exc,
                    (
                        httpx.TimeoutException,
                        httpx.NetworkError,
                        ConnectionError,
                        TimeoutError,
                    ),
                )
                last_error = DeepSeekAPIError(
                    f"DeepSeek request failed: {type(exc).__name__}: {exc}",
                    retryable=retryable,
                    error_code="transport_error",
                )
                if retryable and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error from exc

            status_code = int(getattr(response, "status_code", 0) or 0)
            if not 200 <= status_code < 300:
                retryable = status_code in self._RETRYABLE_STATUS_CODES
                if retryable and attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                message = f"DeepSeek API returned HTTP {status_code}."
                try:
                    response.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    last_error = DeepSeekAPIError(
                        message,
                        status_code=status_code,
                        retryable=retryable,
                        error_code="http_error",
                    )
                    raise last_error from exc
                raise DeepSeekAPIError(
                    message,
                    status_code=status_code,
                    retryable=retryable,
                    error_code="http_error",
                )

            try:
                data = response.json()
                content = self._content_from_response(data)
            except Exception as exc:  # noqa: BLE001
                raise DeepSeekAPIError(
                    f"DeepSeek response format is invalid: {exc}",
                    status_code=status_code,
                    error_code="invalid_response",
                ) from exc
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
            return {
                "content": content,
                "model": str(data.get("model") or self.model),
                "provider": "deepseek",
                "usage": usage,
                "usage_source": "provider" if usage else "unknown",
                "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                "completion_tokens": usage.get("completion_tokens") if usage else None,
                "total_tokens": usage.get("total_tokens") if usage else None,
            }

        # The loop either returns or raises; retain a defensive guard for
        # static analyzers and future changes to retry conditions.
        raise last_error or DeepSeekAPIError("DeepSeek request failed.")

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = max(0.0, float(self.retry_backoff_seconds)) * (2**attempt)
        if delay:
            budget = current_execution_budget()
            time.sleep(budget.http_timeout(delay) if budget else delay)
        budget = current_execution_budget()
        if budget:
            budget.check_deadline()

    @staticmethod
    def _content_from_response(data: Any) -> str:
        if not isinstance(data, dict):
            raise ValueError("response must be an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices is missing or empty")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("first choice is invalid")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("message is missing")
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content or "")


class ContextualizationOllamaClient(OllamaClient):
    """Ollama transport frozen to the contextualization model configuration.

    把 Ollama 传输层"冻结"到 contextualization（上下文增强）专用的模型
    配置：base_url / model / timeout / prompt_version / keep_alive 均取自
    contextualization 相关配置（或显式传入覆盖）。
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        prompt_version: str | None = None,
        keep_alive: str | int | None | object = _DEFAULT_KEEP_ALIVE,
    ) -> None:
        """初始化：继承基类 base_url 解析，并锁定专用配置。

        - ``keep_alive`` 默认取 ``settings.ollama_keep_alive``；
          显式传 None 则表示不注入 keep_alive。
        """
        super().__init__(base_url=base_url or settings.contextualization_base_url)
        self.model = model or settings.contextualization_model
        self.timeout = timeout if timeout is not None else settings.contextualization_timeout
        self.prompt_version = prompt_version or settings.contextualization_prompt_version
        self.keep_alive = (
            settings.ollama_keep_alive
            if keep_alive is _DEFAULT_KEEP_ALIVE
            else keep_alive
        )

    def generate_contextualization(
        self,
        schema: type[SchemaT],
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        """生成上下文增强（contextualization）的结构化输出。

        使用本客户端锁定的模型配置；Schema 约束输出，关闭 think。
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "format": schema.model_json_schema(),
        }
        data = self._post_contextualization(payload)
        return self._parse_structured_content(schema, self._message_content(data))

    def _post_contextualization(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/chat（使用本客户端的 keep_alive 配置）。"""
        with _model_request(payload, self.timeout, ollama=True) as (bounded, remaining, reservation):
            with httpx.Client(timeout=remaining) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=self._with_configured_keep_alive(bounded),
                )
                response.raise_for_status()
                data = response.json()
            _settle_model_response(reservation, data, ollama=True)
            return data

    def _with_configured_keep_alive(self, payload: dict[str, Any]) -> dict[str, Any]:
        """按本客户端锁定的 keep_alive 注入请求（None 则不注入）。"""
        if self.keep_alive is None:
            return payload
        keep_alive = self.keep_alive
        if isinstance(keep_alive, str):
            normalized = keep_alive.strip()
            if normalized.lstrip("+-").isdigit():
                keep_alive = int(normalized)
        return {**payload, "keep_alive": keep_alive}


class ExternalVerifier:
    """外部 API 验证器：让外部模型校验文档 claims 是否被摘要支持。

    对接 OpenAI 兼容的 ``/chat/completions`` 端点，使用
    ``response_format={"type": "json_schema", ...}`` 要求结构化输出。
    未启用（``external_api_enabled`` 为假或缺少 key）时所有调用返回
    ``verdict="skipped"``。
    """

    def __init__(self) -> None:
        """初始化：根据配置决定是否启用外部验证。"""
        self.enabled = settings.external_api_enabled and bool(settings.external_api_key)
        self.base_url = settings.external_api_base_url.rstrip("/")
        self.model = settings.external_api_model
        self.timeout = settings.external_api_timeout

    def verify_claims(self, summary: str, claims: list[ExtractedClaim | dict[str, Any]]) -> VerificationPayload:
        """调用外部模型验证 claims 是否被 summary 支持。

        参数：
        - ``summary``：文档摘要（作为判定依据）。
        - ``claims``：待验证的三元组列表（ExtractedClaim 或原始 dict）。

        返回：
        - 未启用时：``VerificationPayload(verdict="skipped", ...)``。
        - 启用时：用 json_schema 响应格式请求外部模型，解析为
          ``VerificationPayload``（含被标记的 claim 下标）。
        """
        if not self.enabled:
            return VerificationPayload(verdict="skipped", notes="External verification disabled.", flagged_claim_indexes=[])

        schema = VerificationPayload.model_json_schema()
        prompt = {
            "summary": summary,
            "claims": [claim.model_dump() if isinstance(claim, BaseModel) else claim for claim in claims],
            "task": "Review whether the claims are supported by the summary and flag contradictions or uncertainty.",
        }
        headers = {"Authorization": f"Bearer {settings.external_api_key}"}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You verify research claims. Return strict JSON that matches the provided schema.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            # OpenAI 兼容的结构化输出：强制按 json_schema 返回
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verification_payload", "schema": schema},
            },
        }
        data = budgeted_completion_post(f'{self.base_url}/chat/completions',
            payload=payload, timeout=self.timeout, headers=headers)
        # 内容可能是字符串或多段列表，统一为字符串再解析
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
        return VerificationPayload.model_validate_json(content)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:  # 向量相似度计算
    """计算两个向量的余弦相似度（0 范数时返回 0.0）。"""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Embedding dimension mismatch: {len(vec_a)} != {len(vec_b)}")
    if not all(math.isfinite(value) for value in (*vec_a, *vec_b)):
        raise ValueError("Embedding contains non-finite values")
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return numerator / (norm_a * norm_b)


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient errors worth retrying.

    判断异常是否值得重试（瞬时性错误）。

    可重试类型：httpx 的超时/连接/读写/池超时；HTTP 状态码
    429/500/502/503/504；低层 ``ConnectionError`` / ``TimeoutError``。
    其余异常视为不可重试。
    """
    import httpx
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.ReadError):
        return True
    if isinstance(exc, httpx.WriteError):
        return True
    if isinstance(exc, httpx.PoolTimeout):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 服务端繁忙/临时故障类状态码可重试
        return exc.response.status_code in (429, 500, 502, 503, 504)
    # ConnectionError, TimeoutError from low-level networking
    # 低层网络错误（ConnectionError / TimeoutError）
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def safe_model_call(func, fallback):
    """安全调用模型函数：任何异常都退回 fallback 值。

    用途：模型调用失败不应使整个流程崩溃；记录警告并返回兜底值。
    """
    try:
        return func()
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model call failed, falling back: %s", exc)
        return fallback
