"""
repositories.py —— 数据库仓储层模块
===================================

职责：
- 提供对数据库表（ORM 模型）的轻量级访问封装，即"仓储"（Repository）
  模式的最小实现。
- 当前仅包含 ``Project``（项目）表相关的查询与创建逻辑，将数据库
  Session 的具体操作（select / add / commit / refresh）收拢到函数中，
  使上层调用方（如 API 路由、ingestion 管线）无需直接操作 SQLAlchemy。

设计说明：
- 所有函数以显式传入的 ``db: Session`` 作为数据库会话来源，便于依赖
  注入与单元测试。
- 查询使用 SQLAlchemy 2.0 风格的 ``select`` 构造式（Core 表达式），
  而非遗留的 ``Query`` API。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.records import Project


def get_or_create_project(db: Session, slug: str, name: str) -> Project:
    """按 slug 查找项目；不存在则创建，返回持久化后的项目对象。

    这是典型的 "get or create" 幂等操作，供上层在准备上传、接入新项目
    时确保数据库中有对应的 ``Project`` 记录。

    参数：
    - ``db``：SQLAlchemy 会话（事务边界由此会话管理）。
    - ``slug``：项目唯一标识，用于在数据库中定位项目。
    - ``name``：项目的显示名称，仅当项目为新创建时使用。

    流程：
    1. 用 ``select`` 查询 slug 匹配的项目；``db.scalar`` 返回单行结果，
       无匹配时为 None。
    2. 若已存在，直接返回现有项目（不重复创建）。
    3. 若不存在，构造 ``Project(slug=slug, name=name)`` 并：
       - ``db.add`` 把对象加入会话；
       - ``db.commit`` 立即提交事务，确保主键等由数据库生成的值可用；
       - ``db.refresh`` 重新加载对象，使 ``id`` 等数据库回填字段生效。

    返回：数据库中的 ``Project`` 实例（可能是已有记录或新建记录）。
    """
    # 先按 slug 精确匹配查询现有项目
    project = db.scalar(select(Project).where(Project.slug == slug))
    if project:
        return project
    # 未找到：新建并立即提交，随后刷新以拿到数据库生成的主键
    project = Project(slug=slug, name=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project
