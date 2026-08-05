"""
storage.py —— 对象存储（MinIO / S3 兼容）封装模块
=================================================

职责：
- 封装 MinIO（S3 兼容的对象存储）客户端的创建与基础文件上传能力。
- 在配置开启 ``minio_enabled`` 时，自动连接 MinIO 服务并确保目标
  bucket 存在，随后即可将本地文件（如解析后的对象）上传到对象存储。

设计说明：
- 本模块依赖 ``app.core.config.get_settings()`` 提供的全局配置单例，
  上传行为完全由配置开关控制：
  - 未启用对象存储时，上传操作是"空操作"（返回 None），不会报错，
    系统可以退化为纯本地文件存储。
  - 启用时，所有文件通过 ``fput_object`` 分块流式上传。
- 此类作为仓库模式的存储层组件，供 ingestion 管线把产物（解析结果、
  摘要、向量化文本等）持久化到远端对象存储。
"""

from __future__ import annotations

from pathlib import Path

from minio import Minio

from app.core.config import get_settings

# 模块级加载全局配置单例，后续所有方法直接读取
settings = get_settings()


class ObjectStorage:
    """MinIO 对象存储客户端封装。

    提供对象存储的连接管理与单文件上传能力。对象存储在本项目中的用途是
    把 ingestion 各阶段产生的中间/最终产物（PDF 解析结果、规范化文本、
    抽象摘要等）持久化到远端，与本地 ``filesystem`` 模块形成互补：

    - ``filesystem``：管理本地文件（原始上传文件、解析产物）。
    - ``ObjectStorage``：把文件复制到 MinIO，用于跨机器共享与备份。
    """

    def __init__(self) -> None:
        """初始化对象存储客户端。

        根据全局配置决定是否启用对象存储：

        - 读取 ``settings.minio_enabled`` 开关；未启用时 ``self.client``
          保持为 None，后续所有上传调用都会短路返回 None。
        - 启用时创建 MinIO 客户端：
          - ``settings.minio_endpoint``：服务地址（如 ``localhost:9000``）。
          - ``settings.minio_access_key`` / ``settings.minio_secret_key``：
            访问凭证。
          - ``settings.minio_secure``：是否使用 HTTPS/TLS 连接。
        - 客户端创建成功后立即调用 ``_ensure_bucket`` 确保目标 bucket
          存在，避免后续上传时因 bucket 缺失而失败。
        """
        self.enabled = settings.minio_enabled
        self.client = None
        if self.enabled:
            self.client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
            self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """确保目标 bucket 存在；不存在则创建。

        仅在本模块内部调用（初始化阶段）。使用 ``assert`` 保证前置条件：
        只有客户端已创建（即对象存储已启用）时才会走到这里。

        流程：
        1. 查询 ``settings.minio_bucket`` 指定的 bucket 是否已存在。
        2. 不存在时调用 ``make_bucket`` 创建之。
        """
        assert self.client is not None
        if not self.client.bucket_exists(settings.minio_bucket):
            self.client.make_bucket(settings.minio_bucket)

    def upload(self, local_path: Path, object_name: str) -> str | None:
        """将本地文件上传到对象存储的指定对象名。

        参数：
        - ``local_path``：本地文件路径（如解析产物的完整路径）。
        - ``object_name``：对象存储中的对象键名，通常带有前缀/目录结构。

        返回：
        - 对象存储启用并上传成功时返回 ``object_name``（对象键），
          供调用方记录或后续引用。
        - 未启用对象存储（``self.enabled`` 为 False）或客户端未初始化时
          返回 None，表示本次上传为"空操作"，调用方应据此自行判断
          是否需要退化为本地持久化。

        实现说明：使用 ``fput_object`` 从本地路径直接上传，MinIO 客户端
        内部会做分块读取，适合大文件；失败时抛出的异常交由调用方处理。
        """
        if not self.enabled or self.client is None:
            return None
        self.client.fput_object(settings.minio_bucket, object_name, str(local_path))
        return object_name
