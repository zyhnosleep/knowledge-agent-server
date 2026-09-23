"""Agent 生成模型选择路由器。

本模块负责为 Agent 执行链路选择"生成模型"的推理目标（InferenceTarget）。
决策完全基于配置（Settings）与上游给定的路由类型（route）做确定性映射，
不涉及任何额外的大模型调用，因此该过程零开销、无副作用、易于测试。

职责边界：
- 当 route 为 ``needs_clarification``（需要澄清）时，返回 ``profile="none"``，
  表示本次请求无需调用生成模型，返回空的推理目标（base_url / model 为空串）。
- 其余所有路由统一返回 ``profile="generation"``，指向配置中 Ollama 生成模型
  的 base_url / model / context_length，作为当前唯一的生成目标。

该模块通常被 Agent 执行链路（例如 agent_synthesizer）用来把"路由决策"
解析为具体的推理目标，使"路由"与"模型选择"两个关注点解耦。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings, get_settings

@dataclass(frozen=True)
class InferenceTarget:
    """一次推理的完整目标描述（不可变数据类）。

    - ``profile``: 推理配置档位。``"none"`` 表示无需生成（例如需要澄清）；
      ``"generation"`` 表示使用配置中的 Ollama 生成模型。
    - ``base_url``: Ollama 服务地址（调用方通常已去除末尾斜杠），
      profile 为 none 时为空字符串。
    - ``model``: 要调用的模型名称，profile 为 none 时为空字符串。
    - ``context_length``: 模型上下文窗口长度（token 数），profile 为 none 时为 0。
    - ``reason``: 人类可读的决策原因，便于日志与调试时追溯选择依据。
    """

    profile: Literal["none", "generation"]
    base_url: str
    model: str
    context_length: int
    reason: str


class AgentModelRouter:
    """生成模型选择器：在不额外发起模型调用的前提下选出配置好的生成模型。

    Select the configured generation model without another model call.

    路由字符串来自上游（如 PolicyRouter 或 Agent 执行器）的决策结果；
    本类仅执行"配置 → 推理目标"的确定性映射，保持纯函数式、无副作用，
    便于单元测试注入自定义配置。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """初始化路由器。

        :param settings: 应用配置对象；为空时自动调用 ``get_settings()``
            获取全局配置单例，测试时也可显式注入自定义配置。
        """
        self._settings = settings or get_settings()

    def select(self, route: str) -> InferenceTarget:
        """根据路由类型返回对应的推理目标。

        :param route: 上游 Agent 决策产生的路由类型字符串
            （例如 ``needs_clarification`` 或任意普通路由名）。
        :return: 不可变的 :class:`InferenceTarget` 推理目标。
        """
        # 需要澄清的请求不产生任何模型调用：返回"空"推理目标，
        # 让调用方直接跳过生成阶段（base_url / model 为空串、context_length 为 0）。
        if route == "needs_clarification":
            return InferenceTarget(
                profile="none",
                base_url="",
                model="",
                context_length=0,
                reason="needs_clarification does not require generation",
            )

        # 其余所有路由统一指向配置的生成目标。DeepSeek 是 OpenAI-compatible
        # 远程 provider，不使用本地 Ollama 地址；这里的 target 主要用于
        # 队列、trace 与 UI 可观测性，真正的请求由对应 synthesizer 发出。
        generation_provider = str(
            getattr(self._settings, "generation_provider", "ollama")
        ).strip().lower()
        if generation_provider == "deepseek":
            return InferenceTarget(
                profile="generation",
                base_url=self._settings.deepseek_base_url.rstrip("/"),
                model=self._settings.deepseek_model,
                # DeepSeek 不需要 Ollama 的 num_ctx；保留配置的上下文预算
                # 作为 trace/队列中的兼容字段，避免改变 InferenceTarget 契约。
                context_length=self._settings.ollama_generation_context_length,
                reason=f"DeepSeek API generation target for route: {route}",
            )

        # 本地兼容路径：先去除 base_url 末尾斜杠，避免后续拼接模型路径
        # 时产生双斜杠。
        return InferenceTarget(
            profile="generation",
            base_url=self._settings.ollama_generation_base_url.rstrip("/"),
            model=self._settings.ollama_generation_model,
            context_length=self._settings.ollama_generation_context_length,
            reason=f"Single generation target for route: {route}",
        )
