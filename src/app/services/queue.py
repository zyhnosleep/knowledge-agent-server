from __future__ import annotations

import importlib
from collections.abc import Callable

from redis import Redis
from rq import Queue, Worker

from app.core.config import get_settings
from app.services.ingestion_stages import INGESTION_STAGES, STAGE_QUEUES

settings = get_settings()


def redis_connection() -> Redis | None:
    if not settings.redis_url:
        return None
    return Redis.from_url(settings.redis_url)


class JobDispatcher:
    def __init__(self) -> None:
        self.connection = redis_connection()
        self.queue = Queue("ingest", connection=self.connection) if self.connection is not None else None
        self.stage_queues = (
            {
                stage: Queue(queue_name, connection=self.connection)
                for stage, queue_name in STAGE_QUEUES.items()
            }
            if self.connection is not None
            else {}
        )

    def enqueue_or_run(self, func: Callable[..., object] | str, *args, **kwargs):
        if self.queue is None:
            if isinstance(func, str):
                module_name, attr_name = func.rsplit(".", 1)
                imported = getattr(importlib.import_module(module_name), attr_name)
                return imported(*args, **kwargs)
            return func(*args, **kwargs)
        return self.queue.enqueue(func, *args, job_timeout=settings.queue_job_timeout, **kwargs)

    def enqueue_stage(self, document_id: str, version_key: str, stage: str):
        if stage not in STAGE_QUEUES:
            raise ValueError(f"Unknown ingestion stage {stage!r}.")
        function = "app.workers.jobs.run_ingestion_stage"
        job_id = ingestion_stage_job_id(document_id, version_key, stage)
        queue = self.stage_queues.get(stage)
        if queue is None:
            module_name, attr_name = function.rsplit(".", 1)
            imported = getattr(importlib.import_module(module_name), attr_name)
            return imported(document_id, version_key, stage)
        return queue.enqueue(
            function,
            document_id,
            version_key,
            stage,
            job_timeout=settings.queue_job_timeout,
            job_id=job_id,
        )


def ingestion_stage_job_id(document_id: str, version_key: str, stage: str) -> str:
    if stage not in STAGE_QUEUES:
        raise ValueError(f"Unknown ingestion stage {stage!r}.")
    return f"ingestion:{document_id}:{version_key}:{stage}"


def create_worker() -> Worker:
    connection = redis_connection()
    if connection is None:
        raise RuntimeError("REDIS_URL is not configured.")
    queues = [*(STAGE_QUEUES[stage] for stage in INGESTION_STAGES), "ingest"]
    return Worker(queues, connection=connection)
