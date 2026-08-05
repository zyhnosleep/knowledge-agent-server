"""
logging.py —— 全局日志配置模块
================================

职责：
- 为整个应用统一初始化 Python 标准库 ``logging`` 的全局配置。
- 通过 ``basicConfig`` 设置日志级别与输出格式，供所有其他模块调用。

设计说明：
- 本模块不包含任何业务逻辑，只做一次性的基础日志环境初始化。
- 使用标准库 ``logging``，而非第三方日志框架，降低外部依赖。
- 采用 ``from __future__ import annotations`` 延迟求值类型注解，
  使模块可以兼容较新的 Python 版本且避免注解求值开销。
"""

from __future__ import annotations

import logging


def configure_logging() -> None:
    """配置全局日志系统。

    调用 ``logging.basicConfig`` 设置统一的根日志器配置：

    - ``level=logging.INFO``：只输出 INFO 及以上级别（DEBUG 会被过滤）。
    - ``format="%(asctime)s %(levelname)s [%(name)s] %(message)s"``：
      每条日志记录包含时间戳、日志级别、来源日志器名称与消息正文，
      便于在排查问题时追溯日志来自哪个模块。

    说明：
    - 该函数幂等性由 ``basicConfig`` 保证（根日志器已有 handler 时
      不会重复添加），因此可安全地被多个入口调用。
    - 调用方（如 FastAPI 启动事件、CLI 入口）应在应用启动早期调用本函数，
      以确保后续所有模块的日志都采用统一的格式。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
