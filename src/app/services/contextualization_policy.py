"""上下文化策略模块。

定义哪些块类型需要进行"上下文化(Contextualization)"处理,以及如何
判定一个 chunk 的 embedding 文本是否有效。当前策略下:

- ``CONTEXTUALIZED_BLOCK_TYPES`` 为空集合,即默认对所有可检索块类型
  都不启用上下文化;
- 所有可检索块类型都走"纯文本 embedding"路径(plain embedding)。

该模块是一个可插拔的策略开关:只要把某类块加入
``CONTEXTUALIZED_BLOCK_TYPES``,即可重新开启针对该类块的上下文化
处理;与之配套的 `CONTEXTUALIZATION_FIELDS` 定义了判断"已上下文化"
时所需校验的字段集合。

对外主要入口:
- :func:`requires_contextualization`:查询某块类型是否要求上下文化;
- :func:`valid_contextualized_embedding`:校验上下文化的 chunk 是否有效;
- :func:`valid_plain_embedding`:校验纯文本 embedding 的 chunk 是否有效。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CONTEXTUALIZED_BLOCK_TYPES = frozenset()
PLAIN_EMBEDDING_BLOCK_TYPES = frozenset(
    {"narrative", "caption", "appendix", "table", "figure", "formula"}
)
RETRIEVABLE_BLOCK_TYPES = CONTEXTUALIZED_BLOCK_TYPES | PLAIN_EMBEDDING_BLOCK_TYPES

CONTEXTUALIZATION_FIELDS = (
    "contextual_prefix",
    "contextualization_model",
    "contextualization_version",
    "contextualization_prompt_version",
    "contextualized_at",
)


def _field(chunk: object, name: str) -> Any:
    """从 chunk 上读取指定字段。

    兼容两种表示方式:
    - Mapping(dict)类型:直接用 ``.get(name)`` 读取;
    - 一般对象(Pydantic 模型/dataclass 等):用 ``getattr`` 读取,
      字段不存在时返回 None。
    """
    if isinstance(chunk, Mapping):
        return chunk.get(name)
    return getattr(chunk, name, None)


def requires_contextualization(block_type: str) -> bool:
    """查询某块类型是否要求进行上下文化处理。

    先校验块类型必须是可检索类型之一,否则抛出 ValueError;
    然后判断该类型是否位于 ``CONTEXTUALIZED_BLOCK_TYPES`` 集合中。
    """
    if block_type not in RETRIEVABLE_BLOCK_TYPES:
        raise ValueError(f"unsupported retrievable block type: {block_type}")
    return block_type in CONTEXTUALIZED_BLOCK_TYPES


def valid_contextualized_embedding(chunk: object) -> bool:
    """判断一个"上下文化"chunk 的 embedding 是否完整有效。

    只有当该块类型确实要求上下文化,并且:
    - 存在非空的 ``contextual_prefix``;
    - 记录了 model / version / prompt_version / contextualized_at 元数据;
    - ``embedding_text`` 恰好等于 ``"{prefix}\\n\\n{text}"`` 的拼接形式,
    才认为其 embedding 有效。
    """
    block_type = str(_field(chunk, "block_type"))
    if not requires_contextualization(block_type):
        return False
    prefix = _field(chunk, "contextual_prefix")
    text = _field(chunk, "text")
    return bool(
        prefix
        and _field(chunk, "contextualization_model")
        and _field(chunk, "contextualization_version")
        and _field(chunk, "contextualization_prompt_version")
        and _field(chunk, "contextualized_at") is not None
        and _field(chunk, "embedding_text") == f"{prefix}\n\n{text}"
    )


def valid_plain_embedding(chunk: object) -> bool:
    """判断一个"纯文本 embedding"chunk 是否有效。

    当该块类型不要求上下文化时,要求:
    - 所有 ``CONTEXTUALIZATION_FIELDS`` 字段均为 None(即未混入任何
      上下文化元数据);
    - ``embedding_text`` 与 ``text`` 完全一致(即没有附加任何前缀)。
    满足这些条件才算有效的纯文本 embedding。
    """
    block_type = str(_field(chunk, "block_type"))
    if requires_contextualization(block_type):
        return False
    return bool(
        all(_field(chunk, name) is None for name in CONTEXTUALIZATION_FIELDS)
        and _field(chunk, "embedding_text") == _field(chunk, "text")
    )
