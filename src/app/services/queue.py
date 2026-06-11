from __future__ import annotations

import importlib
from collections.abc import Callable

from redis import Redis
from rq import Queue, Worker

from app.core.config import get_settings

settings = get_settings()


def redis_connection() -> Redis | None:
    if not settings.redis_url:
        return None
    return Redis.from_url(settings.redis_url)


class JobDispatcher:
    def __init__(self) -> None:
        self.connection = redis_connection()
        self.queue = Queue("ingest", connection=self.connection) if self.connection is not None else None

    def enqueue_or_run(self, func: Callable[..., object] | str, *args, **kwargs):
        if self.queue is None:
            if isinstance(func, str):
                module_name, attr_name = func.rsplit(".", 1)
                imported = getattr(importlib.import_module(module_name), attr_name)
                return imported(*args, **kwargs)
            return func(*args, **kwargs)
        return self.queue.enqueue(func, *args, job_timeout=settings.queue_job_timeout, **kwargs)


def create_worker() -> Worker:
    connection = redis_connection()
    if connection is None:
        raise RuntimeError("REDIS_URL is not configured.")
    return Worker(["ingest"], connection=connection)
