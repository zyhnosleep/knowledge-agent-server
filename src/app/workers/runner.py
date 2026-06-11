from __future__ import annotations

from app.core.logging import configure_logging
from app.db.session import init_db
from app.services.queue import create_worker


def main() -> None:
    configure_logging()
    init_db()
    worker = create_worker()
    worker.work()


if __name__ == "__main__":
    main()
