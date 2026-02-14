from __future__ import annotations

import os

import redis
from rq import Queue, Worker

from mot_web.config import load_settings


def main() -> None:
    settings = load_settings()
    if not settings.redis_url:
        raise SystemExit("REDIS_URL must be set to run worker")

    redis_conn = redis.from_url(settings.redis_url)
    queue_names = (os.getenv("RQ_QUEUES") or "mot").split(",")
    queues = [Queue(name.strip(), connection=redis_conn) for name in queue_names if name.strip()]

    # RQ 2.x removed the `Connection` context manager; passing the connection
    # directly keeps this compatible with both RQ 1.x and 2.x.
    worker = Worker(queues, connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
