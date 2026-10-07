"""后台 worker 进程的启动入口。

本模块是 RQ（Redis Queue）worker 进程的入口脚本。按 deploy/standalone.md
加载与 API 相同的私有环境后，以 ``python -m app.workers.runner`` 启动，
与 FastAPI 进程分离运行，保证后台重任务不阻塞 API 请求。

启动流程分为四步：
1. ``configure_logging``：配置统一日志格式（INFO 及以上），让后续所有模块
   的日志都带时间戳、级别与来源名称。
2. ``init_db``：建表，并对 SQLite 执行轻量兼容性迁移，确保 schema 就绪。
3. ``create_worker``：按环境变量（``INGESTION_WORKER_QUEUES`` /
   ``INGESTION_WORKER_CONCURRENCY``）校验并创建 RQ Worker 实例。
4. ``worker.work()``：进入阻塞式事件循环，消费各个 ingestion 阶段队列
   （以及 legacy 的 "ingest" 队列），在 worker 进程内执行后台任务。
"""

from __future__ import annotations

from app.core.logging import configure_logging
from app.db.session import init_db
from app.services.queue import create_worker


def main() -> None:
    """Worker 主入口：初始化运行环境并开始消费任务队列。

    依次完成日志、数据库、队列 worker 的初始化后，进入阻塞式任务消费循环；
    只要进程不被信号终止，``worker.work()`` 会持续从队列拉取并执行任务。
    其中任何一个初始化步骤抛错都会让进程启动失败（fail-fast），以便
    运维控制及时发现配置问题并重启服务。
    """
    # 1. 先配置日志：确保后续所有日志（含 init_db 中的告警/错误）都按统一格式输出
    configure_logging()
    # 2. 初始化数据库：建表，并对 SQLite 执行兼容性迁移
    init_db()
    # 3. 校验环境变量并创建 RQ Worker（队列顺序、并发数必须符合约定）
    worker = create_worker()
    # 4. 阻塞式消费队列；此调用会一直运行，直到进程收到终止信号才退出
    worker.work()


if __name__ == "__main__":
    # 仅当作为脚本直接运行（python -m app.workers.runner）时才启动 worker；
    # 被其他模块 import 时不应产生任何副作用
    main()
