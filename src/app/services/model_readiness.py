"""模型就绪状态检查：探测配置的 Ollama 模型是否可用/已加载。

本模块提供 :class:`ModelReadiness`，用于健康检查与诊断——判断配置的
本地 Ollama 模型，或远程 DeepSeek/Qwen API 是否已配置，
而不对模型造成加载负担或消耗 API 额度。

探测方式：调用 Ollama 的两个轻量只读端点——
- ``GET /api/tags``：列出服务端已知的模型（可部署的模型集合）；
- ``GET /api/ps``：列出当前已加载进内存的模型。

对每个配置模型给出状态：
- ``ready``：模型存在且已加载；
- ``idle``：模型存在但尚未加载（按需加载）；
- ``missing``：模型不存在于服务端；
- ``unreachable``：Ollama 服务不可达（网络/超时/非 2xx）。

性能设计：结果在进程内缓存 ``cache_seconds``（默认 5 秒），避免健康检查
被高频调用时反复打 Ollama；多线程并发调用时用锁保护缓存，只做一次真实探测。
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.services.runtime_contract import EmbeddingIdentity, RuntimeContractError


class ModelReadiness:
    """短生命周期、不触发模型加载的就绪检查器。

    Short-lived, non-loading readiness checks for configured Ollama models.

    只读探测 + 短时缓存，专为健康检查和状态页设计：不会因为检查而加载模型，
    也不会在短时间内产生大量 HTTP 请求。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache_seconds: float = 5.0,
        timeout_seconds: float = 2.0,
    ) -> None:
        """初始化检查器。

        :param settings: 应用配置；为空时自动获取全局配置单例。
        :param cache_seconds: 探测结果缓存时长（秒），默认 5 秒。
        :param timeout_seconds: 每次 Ollama HTTP 请求的超时（秒），默认 2 秒，
            保证不可达时快速失败、不拖慢健康检查。
        """
        self._settings = settings or get_settings()
        self._cache_seconds = cache_seconds
        self._timeout_seconds = timeout_seconds
        # 锁用于保护下面两个缓存字段的并发访问（多线程健康检查）。
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None

    def check(self) -> dict[str, Any]:
        """返回总体就绪状态与各模型状态；结果按缓存时长复用。

        :return: ``{"status": "ok"|"degraded", "models": {...}}``。
        """
        now = time.monotonic()
        # 短时缓存：缓存未过期则直接返回，避免每次健康检查都打 Ollama。
        with self._lock:
            if self._cached is not None and now - self._cached_at < self._cache_seconds:
                return self._cached
            # 缓存过期或首次调用：执行真实探测，并写入缓存。
            result = self._probe()
            self._cached = result
            self._cached_at = now
            return result

    def _probe(self) -> dict[str, Any]:
        """探测一次运行时并汇总各配置模型的就绪状态。

        OpenAI-compatible embedding providers intentionally use a
        configuration-only check.  Calling ``/embeddings`` here would consume
        provider quota and would not be portable across vendors; the first
        real ingestion/query call remains the authoritative connectivity check.
        """
        generation_provider = str(
            getattr(self._settings, "generation_provider", "ollama")
        ).strip().lower()
        embedding_provider = self._settings.active_embedding_provider

        # 分别评估生成模型与嵌入模型；metadata（context_length / dimensions）
        # 一并带入，便于前端展示模型能力。远程 provider 只做配置检查。
        if generation_provider == "deepseek":
            generation = self._remote_generation_status()
            generation_runtime = None
        else:
            generation_runtime = self._probe_endpoint(
                self._settings.ollama_generation_base_url
            )
            generation = self._model_status(
                generation_runtime,
                self._settings.ollama_generation_model,
                context_length=self._settings.ollama_generation_context_length,
            )
        if embedding_provider == "openai-compatible":
            embedding = self._remote_embedding_status()
        else:
            # If generation is remote, the embedding model may still live on a
            # separate local Ollama endpoint, so probe that endpoint explicitly.
            embedding_runtime = generation_runtime
            if embedding_runtime is None or (
                self._settings.ollama_embedding_base_url.rstrip("/")
                != self._settings.ollama_generation_base_url.rstrip("/")
            ):
                embedding_runtime = self._probe_endpoint(
                    self._settings.ollama_embedding_base_url
                )
            embedding = self._model_status(
                embedding_runtime,
                self._settings.ollama_embedding_model,
                dimensions=self._settings.active_embedding_dimensions,
            )
        profiles = {
            "generation": generation,
            "embedding": embedding,
        }
        if self._settings.vector_store_strict:
            try:
                expected = EmbeddingIdentity.from_settings(self._settings)
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.get(f"{self._settings.ollama_embedding_base_url.rstrip('/')}/api/embedding_identity")
                    response.raise_for_status()
                actual = EmbeddingIdentity.from_mapping(response.json())
                if expected != actual:
                    raise RuntimeContractError("embedding_identity_mismatch")
                embedding["identity_verified"] = True
            except Exception as exc:
                embedding["status"] = "unverified"
                embedding["error"] = exc.reason if isinstance(exc, RuntimeContractError) else "embedding_identity_unreachable"
            if embedding.get("identity_verified"):
                profiles["vector_store"] = self._probe_database_contract()
        # 总体判定：所有模型状态都属于 {ready, idle}（即均可用于推理）才算 ok，
        # 否则整体降级为 degraded（例如模型缺失或服务不可达）。
        healthy_statuses = {"ready", "idle", "configured"}
        overall = (
            "ok"
            if all(item["status"] in healthy_statuses for item in profiles.values())
            else "degraded"
        )
        return {"status": overall, "models": profiles}

    def _probe_database_contract(self) -> dict[str, Any]:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from app.services.runtime_contract import check_pgvector_contract
        engine = None
        try:
            connect_args = {"connect_timeout": 5} if self._settings.database_url.startswith("postgresql") else {}
            engine = create_engine(self._settings.database_url, connect_args=connect_args)
            with Session(engine) as db:
                return check_pgvector_contract(db, self._settings)
        except Exception as exc:
            return {"status": "unverified", "backend": "pgvector", "error":
                exc.reason if isinstance(exc, RuntimeContractError) else "pgvector_inspection_failed"}
        finally:
            if engine is not None:
                engine.dispose()

    def _remote_generation_status(self) -> dict[str, Any]:
        """Report DeepSeek generation configuration without making a call."""
        base_url = (getattr(self._settings, "deepseek_base_url", "") or "").strip()
        api_key = (getattr(self._settings, "deepseek_api_key", "") or "").strip()
        result: dict[str, Any] = {
            "provider": "deepseek",
            "model": getattr(self._settings, "deepseek_model", "deepseek-chat"),
            "probe": "configuration-only",
        }
        if not base_url or not api_key or api_key.upper() in {"CHANGE_ME", "YOUR_API_KEY"}:
            result["status"] = "missing"
            result["error"] = (
                "DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY are required "
                "for the configured DeepSeek provider."
            )
        else:
            result["status"] = "configured"
        return result

    def _remote_embedding_status(self) -> dict[str, Any]:
        """Report remote embedding configuration without making a paid call."""
        base_url = (self._settings.embedding_api_base_url or "").strip()
        api_key = (self._settings.embedding_api_key or "").strip()
        result: dict[str, Any] = {
            "provider": self._settings.active_embedding_provider,
            "model": self._settings.active_embedding_model,
            "dimensions": self._settings.active_embedding_dimensions,
            "probe": "configuration-only",
        }
        if (
            not base_url
            or not api_key
            or api_key.upper() in {"CHANGE_ME", "YOUR_API_KEY"}
        ):
            result["status"] = "missing"
            result["error"] = (
                "EMBEDDING_API_BASE_URL and EMBEDDING_API_KEY are required "
                "for the configured remote embedding provider."
            )
        else:
            # ``configured`` means credentials and endpoint are present; it is
            # deliberately distinct from ``ready`` because no quota-consuming
            # embedding request was made by the health check.
            result["status"] = "configured"
        return result

    def _probe_endpoint(self, base_url: str) -> dict[str, Any]:
        """调用 /api/tags 与 /api/ps 两个只读端点，返回可用/已加载模型集合。

        任一调用失败（网络错误、超时、非 2xx）时返回带 error 信息的空结果，
        由上层判定为 unreachable，而不是把异常抛给健康检查调用方。
        """
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                # 所有可用模型（已拉取、可被加载的模型集合）。
                tags_response = client.get(f"{base_url.rstrip('/')}/api/tags")
                tags_response.raise_for_status()
                # 当前已加载进内存的模型。
                ps_response = client.get(f"{base_url.rstrip('/')}/api/ps")
                ps_response.raise_for_status()
            return {
                "available": self._model_names(tags_response.json()),
                "loaded": self._model_names(ps_response.json()),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            # 任何异常都吞掉并转为 error 字符串，保持结果结构稳定。
            return {"available": set(), "loaded": set(), "error": str(exc)}

    @staticmethod
    def _model_names(payload: dict[str, Any]) -> set[str]:
        """从 Ollama 响应 payload 中提取模型名集合。

        兼容字段名差异：优先取 ``model``，回退到 ``name``。
        """
        return {
            str(item.get("model") or item.get("name"))
            for item in payload.get("models", [])
            if item.get("model") or item.get("name")
        }

    @staticmethod
    def _model_status(
        probe: dict[str, Any], model: str, **metadata: int
    ) -> dict[str, Any]:
        """根据探测结果推导单个模型的状态。

        :param probe: :meth:`_probe_endpoint` 的返回值。
        :param model: 目标模型名。
        :param metadata: 附加元数据（context_length / dimensions 等）。
        :return: ``{"status": ..., "model": ..., **metadata, "error"?}``。
        """
        # 状态判定顺序：服务不可达 > 模型缺失 > 存在但未加载 > 就绪。
        if probe["error"]:
            status = "unreachable"
        elif model not in probe["available"]:
            status = "missing"
        elif model not in probe["loaded"]:
            status = "idle"
        else:
            status = "ready"
        result: dict[str, Any] = {"status": status, "model": model, **metadata}
        # 不可达时把底层错误一并带上，便于诊断。
        if probe["error"]:
            result["error"] = probe["error"]
        return result


@lru_cache(maxsize=1)
def get_model_readiness() -> ModelReadiness:
    """返回进程级单例 :class:`ModelReadiness`（lru_cache 缓存，仅一份）。"""
    return ModelReadiness()
