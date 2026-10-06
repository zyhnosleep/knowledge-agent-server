#!/usr/bin/env python
"""本地模型服务:一个进程托管三个 Qwen 模型,对外讲 Ollama 协议。

为什么要有这个文件
==================
项目的检索/生成链路全都通过 HTTP 跟 Ollama 说话——``/api/embed``、
``/api/chat``。客户端在 ``app/services/ai.py`` 的 ``OllamaClient`` 里,
已经带了 JSON schema 约束、失败重试、兜底降级。要加多模态能力,最稳的
做法是**照旧讲这个协议**,把 ``base_url`` 指向本进程,而不是去改客户端。
回退也只是改一个 URL 的事。

托管什么
========
=======================  ==========================  ======  =============================
端点                       模型                         维度    谁在用
=======================  ==========================  ======  =============================
``POST /api/embed``          Qwen3-VL-Embedding-2B       2048    文字块与图块**共用**
``POST /api/embed_image``    同上                         2048    图块(以及文字查询)
``POST /api/chat``           Qwen3-VL-4B-Instruct         —      生成 / 图问答(base ↔ LoRA 热切)
=======================  ==========================  ======  =============================

**为什么文字和图用同一个模型**:这个模型把两种模态放在同一个向量空间里。
共用之后,一个文字查询编出来的向量,可以同时跟"文字块"和"图块"算相似度——
一次检索天然排出两类结果,不需要把两个榜单独拼起来,也不需要人为定权重。

代价是放弃了专门的文字模型 ``Qwen3-Embedding-4B``。实测过(363 个块):
top-1 同文档率 89.5%(4B) vs 89.0%(VL),实质打平。留了
``LOCAL_MODEL_EMBED_BACKEND=text`` 可以切回 4B,以便回退。

模型常驻显存(约 13 GB / 48 GB),各自一把锁把前向串行化,避免并发
互相踩。语料本来就只有几十张图,串行完全够用。

两条安全约束(交接文档 §14)
===========================
1. 只监听回环地址,不对外。
2. ``/api/embed_image`` 与 ``/api/chat`` 里用**文件路径**引用的图,只能落在
   ``LOCAL_MODEL_IMAGE_ROOTS`` 列出的根目录内——也就是页面缓存与解析产物
   那一小片地方,不放大成任意路径读取。

``keep_alive`` 的语义
=====================
Ollama 用 ``keep_alive=0`` 卸载模型。本服务的三个模型是**故意常驻**的
(重载一次要几十秒,而实验要反复调用),所以:
- ``/api/ps`` 返回空的 ``models`` 列表 —— 没有可卸载的东西;
- ``/api/generate`` 收下 ``keep_alive=0`` 但不真的卸载。
两者都是如实的:没有模型被卸载,也没有模型被卸载后又被重新加载。

环境变量
========
``LOCAL_MODEL_HOST`` / ``LOCAL_MODEL_PORT``    监听地址与端口(默认 127.0.0.1:18080)
``LOCAL_MODEL_TEXT_EMBED``                     Qwen3-Embedding-4B 目录(仅回退时用)
``LOCAL_MODEL_IMAGE_EMBED``                    Qwen3-VL-Embedding-2B 目录
``LOCAL_MODEL_EMBED_BACKEND``                   ``vl``(默认,统一)或 ``text``(退回 4B)
``LOCAL_MODEL_CHAT``                           Qwen3-VL-4B-Instruct 目录
``LOCAL_MODEL_CHAT_ADAPTER``                   LoRA 适配器目录;**留空则不启用 ft**
``LOCAL_MODEL_CHAT_ALIAS``                     基座模型名(默认 ``qwen3-vl:4b``)
``LOCAL_MODEL_CHAT_FT_ALIAS``                  微调模型名(默认 ``<alias>-ft``)
``LOCAL_MODEL_IMAGE_ROOTS``                    允许读图的根目录,逗号分隔
``LOCAL_MODEL_PRELOAD``                        启动时预载哪些(``all``/``none``/逗号列表)

跑起来::

    /root/autodl-tmp/embed_env/bin/python scripts/serve_local_models.py
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import sys
import threading
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.services.model_asset_identity import fingerprint_embedding_assets

LOGGER = logging.getLogger("local_models")


def _embedding_assets(path: Path) -> dict[str, str]:
    return fingerprint_embedding_assets(path, loader_identity={
        "loader": "sentence_transformers.SentenceTransformer",
        "encode": {"normalize_embeddings": True, "convert_to_numpy": True, "trust_remote_code": True},
        "packages": {name: package_version(name) for name in
                     ("sentence-transformers", "transformers", "torch")},
    })

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

DEFAULT_TEXT_EMBED = "/root/autodl-tmp/model_cache/models/Qwen--Qwen3-Embedding-4B/snapshots/master"
DEFAULT_IMAGE_EMBED = "/root/autodl-tmp/model_cache/models/Qwen--Qwen3-VL-Embedding-2B"
DEFAULT_CHAT = "/root/autodl-tmp/model_cache/models/Qwen--Qwen3-VL-4B-Instruct"
DEFAULT_IMAGE_ROOTS = "/root/autodl-tmp"

TEXT_EMBED_PATH = Path(os.environ.get("LOCAL_MODEL_TEXT_EMBED") or DEFAULT_TEXT_EMBED)
IMAGE_EMBED_PATH = Path(os.environ.get("LOCAL_MODEL_IMAGE_EMBED") or DEFAULT_IMAGE_EMBED)
CHAT_PATH = Path(os.environ.get("LOCAL_MODEL_CHAT") or DEFAULT_CHAT)
CHAT_ADAPTER_RAW = os.environ.get("LOCAL_MODEL_CHAT_ADAPTER", "").strip()
CHAT_ADAPTER_PATH = Path(CHAT_ADAPTER_RAW) if CHAT_ADAPTER_RAW else None
CHAT_ALIAS = os.environ.get("LOCAL_MODEL_CHAT_ALIAS") or "qwen3-vl:4b"
CHAT_FT_ALIAS = os.environ.get("LOCAL_MODEL_CHAT_FT_ALIAS") or f"{CHAT_ALIAS}-ft"
HOST = os.environ.get("LOCAL_MODEL_HOST") or "127.0.0.1"
PORT = int(os.environ.get("LOCAL_MODEL_PORT") or "18080")
PRELOAD = os.environ.get("LOCAL_MODEL_PRELOAD") or "all"

#: 图像索引与图问答的 embedder 名字,出现在 /api/tags 里供客户端探测。
TEXT_EMBED_NAME = os.environ.get("LOCAL_MODEL_TEXT_EMBED_NAME") or "qwen3-embedding:4b"
IMAGE_EMBED_NAME = os.environ.get("LOCAL_MODEL_IMAGE_EMBED_NAME") or "qwen3-vl-embedding:2b"

#: 文字索引交给谁编。
#: - ``vl``(默认):也用 Qwen3-VL-Embedding-2B,文字块与图块落在同一空间;
#: - ``text``:退回 Qwen3-Embedding-4B 单独编文字(2560 维),用于回退对照。
EMBED_BACKEND = (os.environ.get("LOCAL_MODEL_EMBED_BACKEND") or "vl").strip().lower()


def _parse_roots(raw: str) -> list[Path]:
    """把逗号分隔的根目录串解析成已 resolve 的 Path 列表。"""
    roots: list[Path] = []
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        try:
            roots.append(Path(candidate).expanduser().resolve())
        except OSError:  # pragma: no cover - resolve 基本不会抛
            continue
    return roots


IMAGE_ROOTS = _parse_roots(os.environ.get("LOCAL_MODEL_IMAGE_ROOTS") or DEFAULT_IMAGE_ROOTS)

if not IMAGE_ROOTS:
    raise SystemExit("LOCAL_MODEL_IMAGE_ROOTS 解析后为空;拒绝在'任意路径可读'的状态下启动")


# --------------------------------------------------------------------------
# 图片的读取与校验
# --------------------------------------------------------------------------


def _decode_data_uri(raw: str) -> bytes:
    """解出 base64 图片字节,容忍 ``data:image/png;base64,`` 前缀。"""
    payload = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"图片 base64 解不开: {exc}") from exc


def _resolve_image_path(raw: str) -> Path:
    """把文件路径解析成真实路径,并确认它落在允许的根目录内。

    这是交接文档 §14 的那条约束:模型服务只允许打开页面缓存/解析产物里的
    图,不放宽成任意路径读取。越界一律 403,不做静默降级。
    """
    candidate = Path(raw).expanduser()
    # 先判越界,再查存在性——顺序不能反。反过来的话,调用方能从"不存在"
    # 和"越界"两种回答的差别里,探出根目录外某个文件在不在。
    # strict=False 不要求路径存在,但仍会把 ..、相对路径和已有的符号链接
    # 展开成绝对路径,所以"指向根目录外的符号链接"在这里也会被判越界。
    probed = candidate.resolve(strict=False)
    for root in IMAGE_ROOTS:
        if probed == root or root in probed.parents:
            break
    else:
        raise HTTPException(
            status_code=403,
            detail=(
                f"图片路径越界: {raw} 不在允许的根目录内 "
                f"({', '.join(str(root) for root in IMAGE_ROOTS)})"
            ),
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(status_code=404, detail=f"图片不存在: {raw}") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"不是文件: {raw}")
    return resolved


# 超过这个长度、且不含换行的字符串一律当 base64。为什么要靠长度分:
# base64 的字母表**含** ``/``,而 JPEG 的 base64 恰好以 ``/9j/`` 开头——
# 早先按"开头有没有斜杠"来判,结果每一张 JPEG 都被当成了文件路径。
# 真实的图片路径不会长到几百个字符,这个分界够用。
_BASE64_MIN_LEN = 512


def _load_image(raw: Any, *, source: str = "auto") -> Image.Image:
    """把一条图片输入变成 RGB 的 PIL 图像。

    ``source`` 说明这条输入到底是什么,调用方知道就别让服务猜:

    - ``"base64"``:Ollama 的 ``images`` 字段规定的就是裸 base64 / data URI。
    - ``"path"``  :服务器上的文件路径,受 ``IMAGE_ROOTS`` 限制。
    - ``"auto"``  :长且无换行 → base64,否则当路径。
    """
    if isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    elif isinstance(raw, Image.Image):
        return raw.convert("RGB")
    elif isinstance(raw, str):
        stripped = raw.strip()
        mode = source
        if mode == "auto":
            mode = (
                "base64"
                if len(stripped) >= _BASE64_MIN_LEN and "\n" not in stripped
                else "path"
            )
        if mode == "base64":
            data = _decode_data_uri(stripped)
        else:
            data = _resolve_image_path(stripped).read_bytes()
    else:
        raise HTTPException(status_code=400, detail=f"不认识的图片输入类型: {type(raw).__name__}")
    try:
        with Image.open(io.BytesIO(data)) as handle:
            return handle.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"图片解不开: {exc}") from exc


def _unit_vectors(vectors: Any) -> list[list[float]]:
    """把一批向量显式归一化成单位长度,再转成普通的 float 列表。

    为什么不直接信 ``normalize_embeddings=True``:底层跑的是 bfloat16,
    尾数只有 8 位,实测归一化后范数会落到 0.998 这种量级(误差约 0.2%)。
    pgvector 的余弦距离自己会除范数,``vector_store`` 写入前也会再归一化
    一次,所以不影响正确性——但让服务交出来的向量范数就精确等于 1,
    少一个日后要排查的"到底谁没归一化"的疑点。
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.size == 0:
        return []
    if array.ndim == 1:  # 单条向量
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.clip(norms, 1e-12, None)).tolist()


def _looks_like_path(raw: str) -> bool:
    """判断一个字符串该当成图片路径,还是当作文本/base64。

    认两种:绝对路径、或带图片后缀。其余全当文本,免得把一句正常的话
    误判成路径。长度超过 ``_BASE64_MIN_LEN`` 的一律不算路径——base64 的
    字母表里含 ``/``,光看开头有没有斜杠会把每张 JPEG 都认成路径。
    """
    stripped = raw.strip()
    if not stripped or "\n" in stripped:
        return False
    if len(stripped) >= _BASE64_MIN_LEN:
        return False
    if stripped.startswith("/"):
        return True
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    return Path(stripped).suffix.lower() in suffixes


# --------------------------------------------------------------------------
# 三个模型
# --------------------------------------------------------------------------


class TextEmbedder:
    """文本嵌入:Qwen3-Embedding-4B,sentence-transformers 结构,2560 维。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._model: Any = None
        self._lock = threading.Lock()
        self.dimensions: int | None = None
        self.asset_identity: dict[str, str] | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self) -> Any:
        """惰性加载(双检锁:热路径上先看已加载就不再抢锁)。"""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                LOGGER.info("加载文本嵌入模型: %s", self.path)
                started = time.monotonic()
                identity = _embedding_assets(self.path)
                model = SentenceTransformer(str(self.path), device="cuda", trust_remote_code=True)
                model.eval()
                if _embedding_assets(self.path) != identity:
                    raise RuntimeError("embedding_assets_changed_during_load")
                self.asset_identity = identity
                self._model = model
                self.dimensions = int(model.get_sentence_embedding_dimension())
                LOGGER.info(
                    "文本嵌入就绪: %d 维, 耗时 %.1fs", self.dimensions, time.monotonic() - started
                )
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量编码,返回 L2 归一化后的向量。"""
        if not texts:
            return []
        model = self._ensure()
        with self._lock, torch.inference_mode():
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return _unit_vectors(vectors)


class ImageEmbedder:
    """图像/文本嵌入:Qwen3-VL-Embedding-2B,2048 维,图与文字共享同一空间。

    所以同一个 ``embed`` 既能编图,也能编文字查询——检索时两边可比。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._model: Any = None
        self._lock = threading.Lock()
        self.dimensions: int | None = None
        self.asset_identity: dict[str, str] | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                LOGGER.info("加载图像嵌入模型: %s", self.path)
                started = time.monotonic()
                identity = _embedding_assets(self.path)
                model = SentenceTransformer(str(self.path), device="cuda", trust_remote_code=True)
                model.eval()
                if _embedding_assets(self.path) != identity:
                    raise RuntimeError("embedding_assets_changed_during_load")
                self.asset_identity = identity
                self._model = model
                self.dimensions = int(model.get_sentence_embedding_dimension())
                LOGGER.info(
                    "图像嵌入就绪: %d 维, 耗时 %.1fs", self.dimensions, time.monotonic() - started
                )
        return self._model

    def embed(self, items: list[Any]) -> list[list[float]]:
        """编码一批输入;每条可以是 PIL 图、图片路径,或纯文本。"""
        if not items:
            return []
        model = self._ensure()
        with self._lock, torch.inference_mode():
            vectors = model.encode(
                items,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return _unit_vectors(vectors)


class ChatModel:
    """生成模型:Qwen3-VL-4B-Instruct,可选挂 LoRA 适配器,base ↔ ft 热切。

    热切靠 peft 的 ``disable_adapter()`` 上下文管理器——同一个模型实例,
    同一份权重,只是临时把适配器摘掉。比起载两份模型,显存省一半,也保证
    两边比的是**同一个基座**。
    """

    def __init__(self, path: Path, adapter_path: Path | None) -> None:
        self.path = path
        self.adapter_path = adapter_path
        self._processor: Any = None
        self._model: Any = None
        self._lock = threading.Lock()
        self._supports_thinking: bool | None = None
        self._vocab_info: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def has_adapter(self) -> bool:
        return self.adapter_path is not None

    def _ensure(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._processor, self._model
        with self._lock:
            if self._model is None:
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

                LOGGER.info("加载生成模型: %s", self.path)
                started = time.monotonic()
                processor = AutoProcessor.from_pretrained(str(self.path))
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.path), dtype=torch.bfloat16, device_map="cuda"
                )
                if self.adapter_path is not None:
                    from peft import PeftModel

                    LOGGER.info("挂载 LoRA 适配器: %s", self.adapter_path)
                    model = PeftModel.from_pretrained(model, str(self.adapter_path))
                model.eval()
                self._processor, self._model = processor, model
                self._supports_thinking = "enable_thinking" in (
                    processor.chat_template or ""
                )
                LOGGER.info(
                    "生成模型就绪(适配器=%s), 耗时 %.1fs",
                    "有" if self.adapter_path else "无",
                    time.monotonic() - started,
                )
        return self._processor, self._model

    def preload(self) -> None:
        """只加载,不生成。"""
        self._ensure()

    def _build_inputs(
        self, processor: Any, messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], int]:
        """把 Ollama 风格的 messages 转成 Qwen3-VL 的 processor 输入。

        差别在图片的摆法:Ollama 把图片放在 message 的 ``images`` 字段
        (裸 base64 列表),Qwen3-VL 要把图片作为 content 的一段、和文字一起
        按模板排。这里做转换,并单独收集图片对象交给 processor。
        """
        converted: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            raw_images = message.get("images") or []
            if raw_images:
                parts: list[dict[str, Any]] = []
                for raw in raw_images:
                    # Ollama 的 images 字段就是 base64,不用猜——按 auto 的
                    # 长度规则判会把短的测试图误当成路径。
                    image = _load_image(raw, source="base64" if isinstance(raw, str) else "auto")
                    images.append(image)
                    parts.append({"type": "image", "image": image})
                parts.append({"type": "text", "text": str(content)})
                converted.append({"role": role, "content": parts})
            else:
                converted.append({"role": role, "content": str(content)})

        template_kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if self._supports_thinking:
            # 关掉思考链:与 ai.py 的 generate_chat/generate_structured 对齐。
            template_kwargs["enable_thinking"] = False
        text = processor.apply_chat_template(converted, **template_kwargs)

        if images:
            inputs = processor(text=[text], images=images, return_tensors="pt")
        else:
            inputs = processor(text=[text], return_tensors="pt")
        return inputs, int(inputs["input_ids"].shape[1])

    def _tokenizer_info(self, processor: Any) -> Any:
        """给 xgrammar 用的词表描述。构造一次缓存起来,不必每次请求重来。"""
        if self._vocab_info is None:
            import xgrammar as xgr

            tokenizer = getattr(processor, "tokenizer", processor)
            self._vocab_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
            LOGGER.info("xgrammar 词表就绪: vocab_size=%d", self._vocab_info.vocab_size)
        return self._vocab_info

    def _schema_logits_processor(self, processor: Any, schema: Any) -> Any:
        """按 JSON Schema 造一个 logits 处理器:每一步挡掉不合文法的 token。

        这是硬约束,不是提示词软约束。为什么不用 ``lm-format-enforcer``:
        它 0.10.11 里写死了 ``from transformers.tokenization_utils import
        PreTrainedTokenizerBase``,而 transformers 5.x 把这个符号搬走了,一
        导入就报"transformers is not installed"。降 transformers 不可行
        (Qwen3-VL 要 5.x),所以换 ``xgrammar``——它只对 logits 做位掩码,
        不依赖那些被搬走的内部符号。

        拿不到就返回 None,由调用方退回提示词约束,并在响应里如实标
        ``guided: false``,不假装约束生效了。
        """
        if not schema:
            return None
        try:
            import xgrammar as xgr
            from xgrammar.contrib.hf import LogitsProcessor

            compiler = xgr.GrammarCompiler(self._tokenizer_info(processor))
            compiled = compiler.compile_json_schema(schema)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("xgrammar 约束解码不可用,退回提示词约束: %s", exc)
            return None
        # 这个处理器是有状态的(内部维护 matcher),xgrammar 明确要求每次
        # generate 新建一个,不能跨请求复用。
        return LogitsProcessor(compiled)

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        use_adapter: bool,
        max_new_tokens: int = 512,
        schema: Any = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """跑一次生成。``use_adapter=False`` 时在同一实例上临时摘掉 LoRA。"""
        processor, model = self._ensure()
        options = options or {}
        temperature = float(options.get("temperature") or 0.0)
        want_adapter = use_adapter and self.adapter_path is not None

        prompt_messages = list(messages)
        logits_processor = self._schema_logits_processor(processor, schema)
        if schema and logits_processor is None:
            # 约束解码不可用时的兜底:把 schema 写进系统提示,求一个软约束。
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only one valid JSON object matching this JSON Schema. "
                        "No markdown, no code fences, no explanation.\n"
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
                *prompt_messages,
            ]

        started = time.monotonic()
        with self._lock, torch.inference_mode():
            inputs, prompt_len = self._build_inputs(processor, prompt_messages)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = float(options.get("top_p") or 0.8)
                generate_kwargs["top_k"] = int(options.get("top_k") or 20)
            if logits_processor is not None:
                generate_kwargs["logits_processor"] = [logits_processor]

            if want_adapter:
                output = model.generate(**inputs, **generate_kwargs)
            elif self.adapter_path is not None:
                with model.disable_adapter():
                    output = model.generate(**inputs, **generate_kwargs)
            else:
                output = model.generate(**inputs, **generate_kwargs)

            new_tokens = output[0][prompt_len:]
            content = processor.decode(new_tokens, skip_special_tokens=True).strip()
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
            eos_reached = bool(new_tokens.numel() and int(new_tokens[-1]) in eos_ids)
            done_reason = 'length' if not eos_reached and len(new_tokens) >= max_new_tokens else 'stop'

        return {
            "content": content,
            "prompt_eval_count": prompt_len,
            "eval_count": int(new_tokens.shape[0]),
            "total_duration": int((time.monotonic() - started) * 1e9),
            "guided": logits_processor is not None,
            "adapter": want_adapter,
            "eos_reached": eos_reached,
            "done_reason": done_reason,
        }


TEXT_EMBEDDER = TextEmbedder(TEXT_EMBED_PATH)
IMAGE_EMBEDDER = ImageEmbedder(IMAGE_EMBED_PATH)
CHAT_MODEL = ChatModel(CHAT_PATH, CHAT_ADAPTER_PATH)


def active_text_embedder() -> TextEmbedder | ImageEmbedder:
    """文字索引该用哪个 encoder。默认统一到 VL 模型(与图同空间)。"""
    return IMAGE_EMBEDDER if EMBED_BACKEND == "vl" else TEXT_EMBEDDER


def active_text_embed_name() -> str:
    """上面那个 encoder 对外的名字,免得响应里报错模型。"""
    return IMAGE_EMBED_NAME if EMBED_BACKEND == "vl" else TEXT_EMBED_NAME


# --------------------------------------------------------------------------
# HTTP 层
# --------------------------------------------------------------------------

app = FastAPI(title="local-qwen-models", version="1.0.0")


def _now_ns() -> int:
    return time.time_ns()


@app.get("/")
def root() -> dict[str, Any]:
    """人看的自述页:加载了什么、允许读图的根目录在哪。"""
    return {
        "service": "local-qwen-models",
        "text_embed_backend": EMBED_BACKEND,
        "text_embed": {
            "model": active_text_embed_name(),
            "path": str(IMAGE_EMBED_PATH if EMBED_BACKEND == "vl" else TEXT_EMBED_PATH),
            "loaded": active_text_embedder().loaded,
            "dimensions": active_text_embedder().dimensions,
        },
        "image_embed": {
            "model": IMAGE_EMBED_NAME,
            "path": str(IMAGE_EMBED_PATH),
            "loaded": IMAGE_EMBEDDER.loaded,
            "dimensions": IMAGE_EMBEDDER.dimensions,
        },
        "chat": {
            "base": CHAT_ALIAS,
            "ft": CHAT_FT_ALIAS if CHAT_MODEL.has_adapter else None,
            "path": str(CHAT_PATH),
            "adapter_path": str(CHAT_ADAPTER_PATH) if CHAT_ADAPTER_PATH else None,
            "loaded": CHAT_MODEL.loaded,
        },
        "image_roots": [str(root) for root in IMAGE_ROOTS],
    }


@app.get("/healthz")
@app.get("/health")
def healthz() -> dict[str, Any]:
    """给探活用:哪个模型已经在显存里。

    ``embedding_loaded`` 这个键名是 ``run_multimodal_pilot.py`` 等待后端
    就绪时读的,保留它免得那个脚本(或用它的习惯)探测失败。
    """
    embedding_loaded = active_text_embedder().loaded
    return {
        "status": "ok",
        "embedding_loaded": embedding_loaded,
        "text_embed_loaded": embedding_loaded,
        "image_embed_loaded": IMAGE_EMBEDDER.loaded,
        "chat_loaded": CHAT_MODEL.loaded,
        "text_embed_backend": EMBED_BACKEND,
        "embedding_dimensions": active_text_embedder().dimensions,
    }


@app.get("/api/embedding_identity")
def api_embedding_identity() -> dict[str, Any]:
    """Return the identity bound to the loaded encoder; never load or infer."""
    embedder = active_text_embedder()
    if not embedder.loaded or embedder.asset_identity is None or embedder.dimensions is None:
        raise HTTPException(status_code=503, detail="embedding_identity_unverified")
    return {"provider": "ollama", "model": active_text_embed_name(),
        "dimensions": embedder.dimensions, **embedder.asset_identity}


@app.get("/api/tags")
def api_tags() -> dict[str, Any]:
    """Ollama 的模型清单。三个名字都在这儿,供客户端探测。"""
    names = [TEXT_EMBED_NAME, IMAGE_EMBED_NAME, CHAT_ALIAS]
    if CHAT_MODEL.has_adapter:
        names.append(CHAT_FT_ALIAS)
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "modified_at": "2026-09-24T00:00:00Z",
                "size": 0,
                "digest": "",
                "details": {"family": "qwen", "parameter_size": "", "quantization_level": "bf16"},
            }
            for name in names
        ]
    }


@app.get("/api/ps")
def api_ps() -> dict[str, Any]:
    """常驻模型不提供卸载,所以这里如实返回空列表。

    ``ai.py`` 的 ``unload_loaded_models`` 会读这个列表再逐个发
    ``keep_alive=0``。返回空 = 它什么都不做,不会误把常驻模型卸掉。
    """
    return {"models": []}


@app.post("/api/generate")
def api_generate(payload: dict[str, Any]) -> dict[str, Any]:
    """只收下 ``keep_alive=0``,不真的卸载。见模块开头的说明。"""
    if payload.get("keep_alive") in (0, "0"):
        LOGGER.info("收到 keep_alive=0(模型名 %s);本服务模型常驻,忽略卸载请求", payload.get("model"))
    return {
        "model": str(payload.get("model") or CHAT_ALIAS),
        "created_at": "2026-09-24T00:00:00Z",
        "response": "",
        "done": True,
        "done_reason": "load",
    }


@app.post("/api/embed")
def api_embed(payload: dict[str, Any]) -> dict[str, Any]:
    """文本嵌入。契约与 Ollama ``/api/embed`` 一致:``{model,input,...}`` → ``{embeddings}``。

    走哪个 encoder 取决于 ``LOCAL_MODEL_EMBED_BACKEND``:默认 ``vl``,
    也就是**和图片同一个模型、同一个空间**——这样文字块和图块可以直接比。
    """
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        texts = [item if isinstance(item, str) else str(item) for item in raw_input]
    else:
        raise HTTPException(status_code=400, detail="input 必须是字符串或字符串数组")
    if not texts:
        return {"model": active_text_embed_name(), "embeddings": []}

    started = _now_ns()
    embeddings = active_text_embedder().embed(texts)
    return {
        "model": active_text_embed_name(),
        "embeddings": embeddings,
        "total_duration": _now_ns() - started,
        "prompt_eval_count": len(texts),
    }


@app.post("/api/embed_image")
def api_embed_image(payload: dict[str, Any]) -> dict[str, Any]:
    """图像/文本嵌入,2048 维。图与文字落在同一空间,所以两边都用它。

    输入::

        {"input": ["<图片路径 或 一段文字>", ...], "kind": "image"|"text"|"auto"}

    ``kind`` 缺省为 ``auto``:像路径且文件存在就当图,其余当文字。
    """
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        items_in = [raw_input]
    elif isinstance(raw_input, list):
        items_in = list(raw_input)
    else:
        raise HTTPException(status_code=400, detail="input 必须是字符串或数组")
    if not items_in:
        return {"model": IMAGE_EMBED_NAME, "embeddings": []}

    kind = str(payload.get("kind") or "auto").strip().lower()
    if kind not in {"auto", "image", "text"}:
        raise HTTPException(status_code=400, detail="kind 只能是 image / text / auto")

    resolved: list[Any] = []
    for item in items_in:
        if not isinstance(item, str):
            # 直接给的就是 bytes / PIL 图,当图处理。
            resolved.append(_load_image(item, source="auto") if not isinstance(item, Image.Image) else _load_image(item))
            continue
        if kind == "text":
            resolved.append(item)
            continue
        if kind == "image":
            # kind 明确说是图,那 base64 还是路径就交给 auto 按长度判。
            resolved.append(_load_image(item, source="auto"))
            continue
        # auto:像路径当图,否则当文字。
        resolved.append(_load_image(item, source="auto") if _looks_like_path(item) else item)

    started = _now_ns()
    embeddings = IMAGE_EMBEDDER.embed(resolved)
    return {
        "model": IMAGE_EMBED_NAME,
        "embeddings": embeddings,
        "total_duration": _now_ns() - started,
        "prompt_eval_count": len(resolved),
    }


@app.post("/api/chat")
def api_chat(payload: dict[str, Any]) -> dict[str, Any]:
    """生成。``model`` 决定走 base 还是 ft;``format`` 给 schema 就约束解码。

    - ``messages``:Ollama 格式,图片放在 message 的 ``images``(base64)。
    - ``format``:JSON Schema 字典,或字符串 ``"json"``;不传即自由文本。
    - ``options.num_predict`` → ``max_new_tokens``;``num_ctx`` 只记日志。
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    requested = str(payload.get("model") or CHAT_ALIAS)
    use_adapter = CHAT_MODEL.has_adapter and requested == CHAT_FT_ALIAS
    allowed_models = {CHAT_ALIAS, CHAT_FT_ALIAS} if CHAT_MODEL.has_adapter else {CHAT_ALIAS}
    if requested not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Unknown/unavailable model alias: {requested}")

    raw_format = payload.get("format")
    schema: Any = None
    if isinstance(raw_format, dict) and raw_format:
        schema = raw_format
    elif isinstance(raw_format, str) and raw_format.strip().lower() == "json":
        schema = {"type": "object"}

    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    max_new_tokens = int(options.get("num_predict") or 512)

    started = _now_ns()
    result = CHAT_MODEL.generate(
        messages=messages,
        use_adapter=use_adapter,
        max_new_tokens=max_new_tokens,
        schema=schema,
        options=options,
    )
    return {
        "model": CHAT_FT_ALIAS if result["adapter"] else requested,
        "created_at": "2026-09-24T00:00:00Z",
        "message": {"role": "assistant", "content": result["content"]},
        "done": True,
        "done_reason": result.get("done_reason", "stop"),
        "prompt_eval_count": result["prompt_eval_count"],
        "eval_count": result["eval_count"],
        "total_duration": _now_ns() - started,
        # 两个自定义字段:本次是否真用了 schema 约束解码、是否挂了 LoRA。
        # 不放进 message,免得干扰 Ollama 客户端解析。
        "guided": result["guided"],
        "adapter": result["adapter"],
        "eos_reached": result.get("eos_reached"),
    }


@app.exception_handler(HTTPException)
def _http_exception_handler(_request: Any, exc: HTTPException) -> JSONResponse:
    """把错误也包成 JSON,方便 curl 直接看。"""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# --------------------------------------------------------------------------
# 启动
# --------------------------------------------------------------------------


def _preload() -> None:
    """按 ``LOCAL_MODEL_PRELOAD`` 预载模型,免得第一个请求等太久。"""
    if PRELOAD.strip().lower() == "none":
        LOGGER.info("PRELOAD=none,不预载,首个请求触发加载")
        return
    wanted = {item.strip().lower() for item in PRELOAD.split(",") if item.strip()}
    preload_all = "all" in wanted
    if preload_all or "text" in wanted:
        # vl 模式下文字和图共用一个模型,热一次两个端点都就绪。
        active_text_embedder().embed(["预热"])
    if preload_all or "image" in wanted:
        IMAGE_EMBEDDER.embed(["预热"])
    if preload_all or "chat" in wanted:
        CHAT_MODEL.preload()


def main() -> None:
    """入口:起 uvicorn。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    LOGGER.info("监听 http://%s:%d", HOST, PORT)
    LOGGER.info("允许读图的根目录: %s", ", ".join(str(root) for root in IMAGE_ROOTS))
    if CHAT_ADAPTER_PATH is None:
        LOGGER.warning("未配置 LOCAL_MODEL_CHAT_ADAPTER;只提供基座 %s,没有 %s", CHAT_ALIAS, CHAT_FT_ALIAS)
    _preload()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
