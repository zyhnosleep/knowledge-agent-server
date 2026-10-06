"""
ingestion_identity.py —— Ingestion 身份/配置快照（可复现性基础设施）模块
=====================================================================

职责：
- 计算并固化 ingestion 管线的"身份信息"：算法版本、tokenizer 资产、
  嵌入模型与语义切分配置，形成可校验、可复现的配置快照。
- 生成"解析版本键"（parse version key），用于区分同一份源文档在不同
  解析/配置下的产物版本。
- 提供本地 tokenizer 的解析与内容指纹校验，确保离线/无网络环境下
  也能安全、确定性地加载已固定的分词器。

核心概念：
- **配置快照（snapshot）**：``build_ingestion_config_snapshot`` 汇总当前
  配置生成字典；再经 ``canonical_ingestion_config_hash`` 得到稳定哈希。
  快照内容变化（如换了 tokenizer、改了切分参数）会得到不同的配置哈希，
  从而驱动新版本。
- **解析版本键**：``{canonical_pipeline_version}-{源文档哈希前12位}-
  {配置哈希前12位}``，唯一标识一次可复现的解析产物。
- **配置一致性校验**：``require_matching_ingestion_config`` 在任务执行前
  对比入队时记录的配置快照与当前配置，防止"排队的旧任务 + 新配置"
  造成的产物漂移。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from app.core.config import Settings, get_settings

# 短配置哈希长度：用于版本键中截取 12 位十六进制片段
SHORT_CONFIG_HASH_LENGTH = 12
# 本地 tokenizer 快照的身份元数据文件名
TOKENIZER_SNAPSHOT_IDENTITY_FILE = ".knowledge-agent-tokenizer-snapshot.json"
# 身份元数据文件使用的 schema 版本（用于兼容性检查）
TOKENIZER_SNAPSHOT_IDENTITY_SCHEMA = "knowledge-agent-tokenizer-snapshot-v1"
# 各算法/流程的修订版本标识：任何修订号变化都会反映到配置快照中，
# 从而让产物版本自然升级，避免新旧算法产物混淆
ALGORITHM_REVISIONS = {
    "parser": "canonical-parser-v6",  # 规范解析器
    "pdf_recovery": "pdf-recovery-v2",  # PDF 恢复
    "structured_splitting": "structured-splitting-v3",  # 结构化切分
    "source_fidelity_algorithm": "source-fidelity-v1",  # 源保真算法
    "source_fidelity_schema": "source-fidelity-schema-v1",  # 源保真 schema
}
# tokenizer 相关资产文件名集合：内容指纹只对这类文件计算
_TOKENIZER_ASSET_NAMES = {
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


class TokenizerUnavailableError(RuntimeError):
    """The configured pinned tokenizer cannot be loaded without network access.

    配置固定的 tokenizer 无法在无网络访问（仅本地缓存）的情况下加载。
    """


@dataclass(frozen=True)
class ResolvedTokenizer:
    """已解析并验证的 tokenizer 及其身份信息。

    字段：
    - ``tokenizer``：transformers 的 AutoTokenizer 实例。
    - ``identity``：身份字典 ``{"name", "revision", "content_sha256"}``。
    - ``snapshot_path``：本地快照目录的绝对路径。
    """

    tokenizer: Any
    identity: dict[str, str]
    snapshot_path: Path


def _is_tokenizer_asset(path: Path) -> bool:
    """判断文件是否属于 tokenizer 资产（参与内容指纹计算）。

    判定条件：文件名小写后命中 ``_TOKENIZER_ASSET_NAMES`` 精确集合，
    或以 ``tokenizer.``、``tokenizer_``、``vocab.``、``merges.`` 开头。
    这样可以在快照目录中精确筛选出影响 tokenizer 行为的文件。
    """
    name = path.name.lower()
    return (
        name in _TOKENIZER_ASSET_NAMES
        or name.startswith("tokenizer.")
        or name.startswith("tokenizer_")
        or name.startswith("vocab.")
        or name.startswith("merges.")
    )


def tokenizer_asset_content_hash(snapshot_path: Path | str) -> str:
    """计算 tokenizer 快照目录中全部资产文件的确定性 SHA-256 内容哈希。

    哈希构建方式（保证顺序与文件边界稳定，可复现）：
    1. 收集快照目录下所有 tokenizer 资产文件，按相对路径（POSIX 形式）
       字典序排序，保证跨平台/跨文件系统顺序一致。
    2. 对每个资产，把"相对路径长度(4字节大端) + 相对路径"以及该文件
       自身的 SHA-256 摘要依次喂给主摘要，避免路径与内容互相串位。
    3. 最终返回主摘要的 hex 字符串。

    异常：
    - 快照目录不可用或无任何 tokenizer 资产时抛
      ``TokenizerUnavailableError``。
    """
    root = Path(snapshot_path).expanduser().resolve()
    if not root.is_dir():
        raise TokenizerUnavailableError(
            f"Tokenizer snapshot directory is unavailable: {root}"
        )
    # 收集并排序所有资产文件（相对路径字典序）
    assets = sorted(
        (path for path in root.rglob("*") if path.is_file() and _is_tokenizer_asset(path)),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not assets:
        raise TokenizerUnavailableError(
            f"Tokenizer snapshot contains no tokenizer assets: {root}"
        )
    digest = hashlib.sha256()
    for path in assets:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        # 先写入相对路径长度（4 字节大端）再写路径，确保边界唯一
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        # 计算单个资产的 SHA-256（1 MiB 分块读取）
        asset_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                asset_digest.update(block)
        digest.update(asset_digest.digest())
    return digest.hexdigest()


def _verify_explicit_snapshot_identity(
    snapshot_path: Path,
    *,
    name: str,
    revision: str,
    content_sha256: str,
) -> None:
    """校验显式本地快照的身份元数据（信任锚点）。

    当使用 ``local_path`` 显式指定 tokenizer 快照时，要求该目录内存在
    ``TOKENIZER_SNAPSHOT_IDENTITY_FILE`` 元数据文件，且其内容与传入的
    期望值完全一致：
    - ``schema_version`` 必须等于 ``TOKENIZER_SNAPSHOT_IDENTITY_SCHEMA``。
    - ``name``、``revision`` 必须等于期望的模型名/修订。
    - ``content_sha256`` 必须等于实测的资产内容哈希。

    任一不一致均抛 ``TokenizerUnavailableError``，防止加载被替换/伪造的
    tokenizer。这是"显式固定"模式下的安全校验锚点。
    """
    identity_path = snapshot_path / TOKENIZER_SNAPSHOT_IDENTITY_FILE
    if not identity_path.is_file():
        raise TokenizerUnavailableError(
            f"Explicit tokenizer snapshot is missing trusted identity metadata: {identity_path}"
        )
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TokenizerUnavailableError(
            f"Explicit tokenizer snapshot identity metadata is invalid: {identity_path}"
        ) from exc
    if identity.get("schema_version") != TOKENIZER_SNAPSHOT_IDENTITY_SCHEMA:
        raise TokenizerUnavailableError(
            "Explicit tokenizer snapshot identity metadata has an unsupported schema."
        )
    if identity.get("name") != name:
        raise TokenizerUnavailableError(
            "Explicit tokenizer snapshot identity metadata has the wrong model name."
        )
    if identity.get("revision") != revision:
        raise TokenizerUnavailableError(
            "Explicit tokenizer snapshot identity metadata has the wrong revision."
        )
    if identity.get("content_sha256") != content_sha256:
        raise TokenizerUnavailableError(
            "Explicit tokenizer snapshot identity metadata SHA-256 does not match its assets."
        )


@lru_cache(maxsize=8)
def resolve_local_tokenizer(
    name: str,
    revision: str,
    *,
    local_path: Path | str | None = None,
) -> ResolvedTokenizer:
    """仅从本地缓存解析固定的 tokenizer（禁止联网下载），带缓存。

    参数：
    - ``name``：Hugging Face 模型仓库 ID。
    - ``revision``：仓库修订（分支/tag/commit）。
    - ``local_path``：可选，显式指定本地快照目录；为 None 时改用
      ``snapshot_download(local_files_only=True)`` 解析本地缓存路径。

    两种模式：
    1. **显式本地快照**（``local_path`` 给定）：校验目录存在、计算内容
       哈希、核对身份元数据（``_verify_explicit_snapshot_identity``），
       再以 ``local_files_only=True`` 加载。
    2. **缓存路径模式**：用 ``snapshot_download`` 只查本地缓存定位快照
       目录，计算内容哈希，以 ``local_files_only=True`` 加载。

    任何失败（目录缺失、资产缺失、无法加载）都会被统一包装为
    ``TokenizerUnavailableError``，消息中包含模型名/修订与根因，说明
    网络下载被禁用。

    使用 ``@lru_cache``：相同 (name, revision, local_path) 组合只解析一次，
    避免反复加载大模型文件；缓存结果不可变（frozen dataclass）。
    """
    try:
        if local_path is not None:
            snapshot_path = Path(local_path).expanduser().resolve()
            if not snapshot_path.is_dir():
                raise FileNotFoundError(snapshot_path)
            # 计算内容指纹并校验身份元数据
            content_sha256 = tokenizer_asset_content_hash(snapshot_path)
            _verify_explicit_snapshot_identity(
                snapshot_path,
                name=name,
                revision=revision,
                content_sha256=content_sha256,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str(snapshot_path),
                local_files_only=True,
            )
        else:
            # 本地缓存模式：只允许用已缓存的快照
            snapshot_path = Path(
                snapshot_download(
                    repo_id=name,
                    revision=revision,
                    local_files_only=True,
                )
            ).resolve()
            tokenizer = AutoTokenizer.from_pretrained(
                name,
                revision=revision,
                local_files_only=True,
            )
            content_sha256 = tokenizer_asset_content_hash(snapshot_path)
    except Exception as exc:
        # 统一包装为 TokenizerUnavailableError，携带上下文信息
        configured_path = (
            f" at {Path(local_path).expanduser()}" if local_path is not None else ""
        )
        raise TokenizerUnavailableError(
            f"Required tokenizer {name!r} revision {revision!r} is unavailable "
            f"from the local cache{configured_path}; network downloads are disabled. "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return ResolvedTokenizer(
        tokenizer=tokenizer,
        identity={
            "name": name,
            "revision": revision,
            "content_sha256": content_sha256,
        },
        snapshot_path=snapshot_path,
    )


def get_tokenizer_identity(settings: Settings | None = None) -> dict[str, str]:
    """按当前配置解析 tokenizer 并返回其身份信息字典。

    从 ``settings``（或全局配置）读取 ``semantic_tokenizer_name``、
    ``semantic_tokenizer_revision``、``semantic_tokenizer_local_path``，
    调用 ``resolve_local_tokenizer`` 得到 ``ResolvedTokenizer`` 后返回
    ``identity``（name / revision / content_sha256）的副本。
    """
    configured = settings or get_settings()
    resolved = resolve_local_tokenizer(
        configured.semantic_tokenizer_name,
        configured.semantic_tokenizer_revision,
        local_path=configured.semantic_tokenizer_local_path,
    )
    return dict(resolved.identity)


def build_ingestion_config_snapshot(
    settings: Settings | None = None,
    *,
    tokenizer_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """构建 ingestion 配置快照字典（确定性的可复现配置全景）。

    快照包含：
    - ``algorithm_revisions``：各算法/流程的修订版本。
    - ``tokenizer``：模型名、修订、内容哈希（默认实时解析，也可由调用
      方预先传入避免重复解析）。
    - ``embedding``：provider、嵌入模型名与维度；provider 变化会强制
      生成新的解析/索引身份，避免不同服务商的向量混用。
    - ``semantic_splitting``：语义切分模型与父/子块 token 上下限、
      切分百分位、重叠 token 数。

    说明：快照的字段顺序不保证，但经 ``canonical_ingestion_config_json``
    序列化时会排序，因此哈希是稳定的。快照参与解析版本键的计算。
    """
    configured = settings or get_settings()
    identity = dict(tokenizer_identity or get_tokenizer_identity(configured))
    return {
        "algorithm_revisions": dict(ALGORITHM_REVISIONS),
        "tokenizer": {
            "name": identity["name"],
            "revision": identity["revision"],
            "content_sha256": identity["content_sha256"],
        },
        "embedding": {
            "provider": configured.active_embedding_provider,
            "model": configured.active_embedding_model,
            "dimensions": configured.active_embedding_dimensions,
        },
        "semantic_splitting": {
            "model": configured.semantic_splitting_model,
            "break_percentile": configured.semantic_break_percentile,
            "parent_tokens": {
                "min": configured.semantic_parent_min_tokens,
                "target": configured.semantic_parent_target_tokens,
                "max": configured.semantic_parent_max_tokens,
            },
            "child_tokens": {
                "min": configured.semantic_child_min_tokens,
                "target": configured.semantic_child_target_tokens,
                "max": configured.semantic_child_max_tokens,
            },
            "overlap_tokens": configured.semantic_child_overlap_tokens,
        },
    }


def canonical_ingestion_config_json(snapshot: Mapping[str, Any]) -> bytes:
    """把配置快照序列化为"规范 JSON"字节串。

    规范化要点：
    - ``ensure_ascii=False``：保留 Unicode 字符，避免转义漂移。
    - ``allow_nan=False``：拒绝 NaN/Infinity，保证跨实现一致。
    - ``sort_keys=True``：键排序，消除字典插入顺序差异。
    - ``separators=(",", ":")``：紧凑无空格输出。

    目的是让同一快照在不同进程/平台序列化结果逐字节一致，从而哈希稳定。
    """
    return json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_ingestion_config_hash(snapshot: Mapping[str, Any]) -> str:
    """计算配置快照的规范 SHA-256 哈希。

    即 ``canonical_ingestion_config_json`` 结果的哈希 hex 字符串。
    """
    return hashlib.sha256(canonical_ingestion_config_json(snapshot)).hexdigest()


def build_parse_version_key(
    source_sha256: str,
    *,
    snapshot: Mapping[str, Any] | None = None,
    config_sha256: str | None = None,
    settings: Settings | None = None,
) -> str:
    """构建解析版本键：``{pipeline版本}-{源哈希前12位}-{配置哈希前12位}``。

    参数：
    - ``source_sha256``：源文件内容的 SHA-256（内容不同则版本不同）。
    - ``snapshot``：可选的配置快照；为 None 时用 ``config_sha256`` 或
      现构建当前配置快照。
    - ``config_sha256``：可选的配置哈希；为 None 时由快照计算。
    - ``settings``：用于构建快照的配置（当两者都未给出时）。

    用途：标识"同一源文件在特定配置下的一次解析"，作为文档解析版本的
    唯一键；源内容或配置任一变化都会得到新版本键。
    """
    configured = settings or get_settings()
    if config_sha256 is None:
        # 未给定配置哈希时，先构建（或复用传入的）快照再计算哈希
        current_snapshot = dict(snapshot or build_ingestion_config_snapshot(configured))
        config_sha256 = canonical_ingestion_config_hash(current_snapshot)
    return (
        f"{configured.canonical_pipeline_version}-{source_sha256[:12]}-"
        f"{config_sha256[:SHORT_CONFIG_HASH_LENGTH]}"
    )


def require_matching_ingestion_config(
    manifest: Mapping[str, Any] | None,
    live_snapshot: Mapping[str, Any],
) -> str:
    """校验入队时记录的配置与当前运行配置一致；不一致则拒绝执行。

    参数：
    - ``manifest``：入队/建版本时记录的配置清单（含 ``ingestion_config``
      与 ``ingestion_config_sha256`` 字段）。
    - ``live_snapshot``：当前运行环境的配置快照。

    逻辑：
    - 重新计算当前快照的规范哈希。
    - 若清单中记录的完整快照与当前快照不同，或记录的哈希与当前哈希
      不同，则抛 ``RuntimeError``，提示需要在新配置下重新建立版本。

    返回：当前配置哈希（供调用方记录/复用）。
    """
    stored = dict(manifest or {})
    live_hash = canonical_ingestion_config_hash(live_snapshot)
    if (
        stored.get("ingestion_config") != dict(live_snapshot)
        or stored.get("ingestion_config_sha256") != live_hash
    ):
        raise RuntimeError(
            "Live ingestion configuration does not match the queued ParseVersion "
            "checkpoint; start a new ParseVersion under the current configuration."
        )
    return live_hash
