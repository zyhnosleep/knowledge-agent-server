"""
filesystem.py —— 本地文件系统管理模块
=====================================

职责：
- 管理上传文件在本地磁盘上的落盘、命名与路径安全校验。
- 提供文件标题/项目名的规范化工具（slugify、去除上传前缀、判断是否为
  内部样本名等）。
- 提供计算文件 SHA-256 哈希的能力（用于内容去重/身份标识）。

安全设计（核心）：
- 本模块是用户上传文件进入系统的"第一道门"，因此路径处理非常谨慎：
  - ``_ensure_within``：解析真实路径并校验子路径未逃逸出父目录，
    防止路径穿越（Path Traversal）攻击。
  - ``safe_project_slug`` / ``_safe_upload_filename``：拒绝绝对路径、
    ``..``、路径分隔符以及非法字符，只允许受白名单约束的字符集。
  - 上传采用"先写临时文件再原子 rename"策略：写入 ``.xxx.part`` 临时
    文件，成功后再 ``replace`` 成最终文件名，避免半截文件对解析造成影响。

设计说明：
- 本模块仅处理本地磁盘；对象存储（MinIO）的上传由 ``storage.py`` 负责。
- 文件名统一使用 ``uuid4().hex`` 前缀，配合 ``strip_upload_prefix`` 在
  展示时还原可读标题。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings

# 模块级加载全局配置单例
settings = get_settings()

# 匹配上传文件名中由本模块生成的 32 位十六进制 UUID 前缀，
# 形如 "<32位hex>-<原始文件名>"；展示标题时用它还原可读文件名。
UPLOAD_PREFIX_PATTERN = re.compile(r"^[0-9a-f]{32}-(.+)$", re.IGNORECASE)

# 项目 slug 的白名单字符集：字母、数字、中文（一-鿿，CJK 区段）及 ._-，长度 1~120。
# 由于 slug 会作为文件系统目录名使用，必须严格控制字符以避免路径风险。
PROJECT_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9一-鿿][A-Za-z0-9一-鿿._-]{0,119}$")

# Heuristic for filenames that are just UUIDs, hashes, or other internal sample
# identifiers rather than human-readable document titles.
# 启发式正则：判断文件名是否只是 UUID/哈希等"内部样本标识"，
# 而非人类可读的文档标题（如 <24 位以上 hex>），用于决定是否把文件名
# 当作标题展示，还是退回使用文档内容生成标题。
INTERNAL_SAMPLE_PATTERN = re.compile(r"^[0-9a-f]{24,}$", re.IGNORECASE)

# 标准 UUID 字符串模式（8-4-4-4-12 的十六进制段）
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class StoragePathError(ValueError):
    """存储路径相关错误的基类。

    继承 ``ValueError``，目的是让 API 路由层能够捕获本模块抛出的
    各类存储异常并映射为合适的客户端错误响应（如 400）。
    """


class InvalidStoragePathError(StoragePathError):
    """路径非法（如包含路径分隔符、逃逸出根目录、字符不允许等）。"""

    pass


class UploadTooLargeError(StoragePathError):
    """上传文件超过配置的 ``MAX_UPLOAD_BYTES`` 大小限制。"""

    pass


def slugify(value: str) -> str:
    """把任意字符串规范化为安全的 slug（URL/目录友好片段）。

    流程：
    1. 将连续的非字母数字/非中文字符替换为单个 ``-``。
    2. 去掉首尾多余的 ``-``。
    3. 若结果为空（原字符串全是符号），则退回使用 ``uuid4().hex``
       生成随机字符串，保证返回结果永远非空且合法。

    用途：生成项目 slug、文件名规范化时的候选片段。
    """
    # 保留字母、数字、中文，其余字符折叠为 "-"
    normalized = re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", value.strip().lower())
    # 去除首尾连字符；为空则用随机 hex 兜底
    return normalized.strip("-") or uuid4().hex


def strip_upload_prefix(value: str) -> str:
    """去掉文件名中由 ``save_upload`` 添加的 32 位 hex UUID 前缀。

    例如 ``"a1b2...0123-论文.pdf"`` -> ``"论文.pdf"``。

    说明：
    - 若字符串不以该前缀开头，则原样返回（幂等、不报错）。
    - 该函数是展示标题/判断内部样本名的公共入口，保证无论文件名是否
      带前缀，都能得到一致的处理结果。
    """
    match = UPLOAD_PREFIX_PATTERN.match(value.strip())
    if not match:
        return value
    return match.group(1)


def display_title_from_path(path: Path) -> str:
    """从文件路径提取不带前缀的显示标题（即文件主名 stem）。

    内部实现为 ``strip_upload_prefix(path.stem)``：取文件名去掉扩展名的
    部分，再去掉 UUID 前缀。例如 ``.../a1b2...-论文.pdf`` -> ``论文``。
    """
    return strip_upload_prefix(path.stem)


def looks_like_internal_sample(value: str) -> bool:
    """Return True when *value* looks like a UUID/hash sample name, not a title.

    判断给定的名称是否"看起来像内部样本标识"（UUID 或长哈希），而不是
    人类可读的文档标题。若判断为真，上层通常不会把它直接当作标题，
    而会改用文档内容来生成标题。

    判定规则：
    1. 去除上传前缀并 strip 后若为空，视为内部样本（无可用标题）。
    2. 若整体匹配标准 UUID 模式，视为内部样本。
    3. 若去掉所有非 hex 字符后仍是 24 位以上的十六进制串，
       视为哈希类内部样本。
    """
    # 先剥掉 UUID 前缀再判断，避免前缀干扰判定
    text = strip_upload_prefix(str(value)).strip()
    if not text:
        return True
    if UUID_PATTERN.fullmatch(text):
        return True
    # 过滤掉非 hex 字符后检查是否纯长哈希
    if INTERNAL_SAMPLE_PATTERN.fullmatch(re.sub(r"[^0-9a-fA-F]", "", text)):
        return True
    return False


def readable_title_from_path(path: Path) -> str | None:
    """Return a human-readable title from *path* or None if it is just an internal id.

    从文件路径提取"可读标题"：
    - 若提取出的标题经过 ``looks_like_internal_sample`` 判定为内部样本
      （UUID/哈希等），返回 None，示意调用方需要用文档内容生成标题。
    - 否则返回可读标题字符串。
    """
    title = display_title_from_path(path)
    if looks_like_internal_sample(title):
        return None
    return title


def compute_sha256(path: Path) -> str:
    """分块计算文件的 SHA-256 十六进制摘要。

    说明：
    - 以 8192 字节为块遍历读取，避免一次性把大文件载入内存。
    - 返回的摘要字符串常被用作文档的内容身份标识（content hash），
      用于去重与版本比较。
    """
    digest = hashlib.sha256()
    # iter(lambda: handle.read(8192), b"") 构造读取器：读到空字节串即停止
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_within(child: Path, parent: Path) -> Path:
    """校验并返回 child 的真实路径，确保其位于 parent 目录之内。

    这是路径穿越防护的核心函数：
    1. 对 child 与 parent 均执行 ``expanduser()`` + ``resolve()``，
       解析符号链接与 ``..``，得到绝对真实路径。
    2. 尝试用 ``relative_to`` 判断 child 是否以 parent 为祖先目录。
    3. 若不满足则抛出 ``InvalidStoragePathError``，阻止文件写往根目录外。

    返回：解析后的 child 绝对路径（供调用方使用）。

    说明：注意这里只做"包含性"校验，未校验 child 是否存在——存在性由
    调用方按需处理。
    """
    resolved_child = child.expanduser().resolve()
    resolved_parent = parent.expanduser().resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        # relative_to 失败说明 child 不在 parent 之下，即路径逃逸
        raise InvalidStoragePathError(f"Path escapes storage root: {child}") from exc
    return resolved_child


def safe_project_slug(project_slug: str) -> str:
    """校验并规范化项目 slug，拒绝任何可能造成路径风险的输入。

    项目 slug 会作为目录名拼接进存储路径，因此需严格把关：

    1. 空值直接报错（slug 必填）。
    2. 用 ``Path`` 判断是否为绝对路径；检查 parts 中是否含 ``..``；
       检查是否包含 ``/`` 或 ``\\``（路径分隔符）。
    3. 用 ``PROJECT_SLUG_PATTERN`` 白名单正则校验字符集（字母/数字/
       中文/._-）。

    通过校验后返回去除首尾空白后的 slug 字符串。
    """
    value = (project_slug or "").strip()
    if not value:
        raise InvalidStoragePathError("Project slug is required.")
    path = Path(value)
    # 拒绝绝对路径、父目录引用、路径分隔符等一切路径成分
    if path.is_absolute() or ".." in path.parts or "/" in value or "\\" in value:
        raise InvalidStoragePathError("Project slug must not contain path components.")
    # 白名单字符集校验
    if not PROJECT_SLUG_PATTERN.fullmatch(value):
        raise InvalidStoragePathError("Project slug contains unsupported characters.")
    return value


def _safe_upload_filename(filename: str | None) -> str:
    """校验并清洗上传文件名，返回安全的纯文件名（不含任何路径成分）。

    上传文件名会被拼接到存储路径中，因此这里做了与 ``safe_project_slug``
    类似的防御，另加：

    1. 拒绝空名、``.``、``..``。
    2. 拒绝绝对路径、含路径分隔符、``path.name != value``（说明 value
       中带目录结构）等情况。
    3. 剔除 ASCII 控制字符（``\\x00``-``\\x1f``、``\\x7f``）。
    4. 拒绝含冒号（``:``，Windows 文件名非法/易混淆）的名称。

    返回：清洗后的安全文件名。
    """
    value = (filename or "").strip()
    if not value or value in {".", ".."}:
        raise InvalidStoragePathError("File name is required.")
    path = Path(value)
    # 绝对路径、或 value 与 Path 解析出的文件名不一致（含目录成分）均拒绝
    if path.is_absolute() or path.name != value or ".." in path.parts or "/" in value or "\\" in value:
        raise InvalidStoragePathError("File name must not contain path components.")
    # 剔除控制字符
    sanitized = re.sub(r"[\x00-\x1f\x7f]+", "", value).strip()
    if ":" in sanitized:
        raise InvalidStoragePathError("File name contains unsupported characters.")
    # 再次确认清洗后非空且不是 "." / ".."
    if not sanitized or sanitized in {".", ".."}:
        raise InvalidStoragePathError("File name is invalid.")
    return sanitized


def project_paths(project_slug: str) -> dict[str, Path]:
    """计算（并在必要时创建）某个项目在本地磁盘上的存储根目录。

    返回形如 ``{"raw_root": <Path>}`` 的字典，其中 ``raw_root`` 是
    ``settings.raw_dir / <slug>`` 解析后的真实目录：

    1. ``safe_project_slug`` 校验 slug 合法。
    2. 解析 ``settings.raw_dir`` 的绝对路径，并递归创建该目录。
    3. 用 ``_ensure_within`` 确保 ``raw_base / safe_slug`` 确实位于
       raw_base 之内（防御路径穿越）。
    4. 递归创建项目根目录。

    设计说明：未来若需要扩展更多目录（如解析产物、临时目录），可在此
    返回值字典中追加键，保持调用方接口稳定。
    """
    safe_slug = safe_project_slug(project_slug)
    raw_base = settings.raw_dir.expanduser().resolve()
    raw_base.mkdir(parents=True, exist_ok=True)
    # 项目目录必须位于 raw 根之下，防路径逃逸
    raw_root = _ensure_within(raw_base / safe_slug, raw_base)
    raw_root.mkdir(parents=True, exist_ok=True)
    return {"raw_root": raw_root}


async def save_upload(project_slug: str, upload: UploadFile) -> Path:
    """把 FastAPI 上传文件流式写入磁盘，返回最终落盘路径。

    流程：
    1. 解析项目的 raw 根目录。
    2. 校验并清洗上传文件名；生成 ``<uuid4().hex>-<安全文件名>``
       的最终文件名（UUID 前缀用于避免重名并便于剥离标题）。
    3. 先写入同目录下的临时文件 ``.<最终名>.part``，逐块（1 MiB）
       读取上传流，边写边累计字节数；超过 ``MAX_UPLOAD_BYTES`` 立即
       抛 ``UploadTooLargeError``。
    4. 全部写完后用 ``partial_path.replace(target_path)`` 原子重命名，
       确保最终文件完整可用。
    5. 任何异常都会清理残留的临时文件后重新抛出。
    6. 最后在 ``finally`` 中关闭上传流（``await upload.close()``），
       释放连接资源。

    返回：写入成功后的最终文件 ``Path``。

    异常：
    - ``InvalidStoragePathError``：文件名/项目 slug 非法。
    - ``UploadTooLargeError``：超过大小限制。
    """
    partial_path: Path | None = None
    try:
        # 计算项目根目录（内含安全校验）
        paths = project_paths(project_slug)
        safe_filename = _safe_upload_filename(upload.filename)
        target_name = f"{uuid4().hex}-{safe_filename}"
        # 最终路径与临时路径都需在项目根之内
        target_path = _ensure_within(paths["raw_root"] / target_name, paths["raw_root"])
        partial_path = _ensure_within(paths["raw_root"] / f".{target_name}.part", paths["raw_root"])
        total_bytes = 0
        # 分块写入：每块 1 MiB，边写边统计总量以检查大小上限
        with partial_path.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise UploadTooLargeError(f"Upload exceeds MAX_UPLOAD_BYTES ({settings.max_upload_bytes}).")
                handle.write(chunk)
        # 原子重命名：临时文件变为最终文件
        partial_path.replace(target_path)
        return target_path
    except Exception:
        # 出错时删除可能残留的临时文件，避免污染目录
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise
    finally:
        # 无论成败都要关闭上传连接
        await upload.close()
