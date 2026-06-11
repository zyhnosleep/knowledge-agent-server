from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.records import Project


def get_or_create_project(db: Session, slug: str, name: str) -> Project:
    project = db.scalar(select(Project).where(Project.slug == slug))
    if project:
        return project
    project = Project(slug=slug, name=name)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project
