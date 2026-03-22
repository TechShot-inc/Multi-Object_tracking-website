from __future__ import annotations

import os
import threading
import time

import redis
from rq import Queue, Worker

from prometheus_client import start_http_server

from mot_web.config import load_settings
from mot_web.metrics import RQ_JOB_DURATION_SECONDS, RQ_JOBS_TOTAL, RQ_QUEUE_DEPTH


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _start_worker_metrics_server() -> None:
    if os.getenv("WORKER_METRICS", "1").strip() == "0":
        return
    addr = os.getenv("WORKER_METRICS_ADDR", "0.0.0.0").strip() or "0.0.0.0"
    port = _env_int("WORKER_METRICS_PORT", 9101)
    start_http_server(port, addr=addr)


def _queue_count(q: Queue) -> int:
    try:
        c = getattr(q, "count", None)
        if callable(c):
            return int(c())
        if c is not None:
            return int(c)
    except Exception:
        pass
    try:
        return int(len(q))
    except Exception:
        return 0


def _start_queue_depth_poller(queues: list[Queue]) -> None:
    interval = _env_float("WORKER_QUEUE_DEPTH_POLL_SECONDS", 5.0)

    def _loop() -> None:
        while True:
            for q in queues:
                try:
                    RQ_QUEUE_DEPTH.labels(queue=q.name).set(_queue_count(q))
                except Exception:
                    pass
            time.sleep(max(0.5, interval))

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


class MetricsWorker(Worker):
    def perform_job(self, job, queue, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = super().perform_job(job, queue, *args, **kwargs)
            try:
                RQ_JOBS_TOTAL.labels(queue=queue.name, status="success").inc()
            except Exception:
                pass
            return result
        except Exception:
            try:
                RQ_JOBS_TOTAL.labels(queue=queue.name, status="failure").inc()
            except Exception:
                pass
            raise
        finally:
            try:
                RQ_JOB_DURATION_SECONDS.labels(queue=queue.name).observe(time.perf_counter() - t0)
            except Exception:
                pass


def main() -> None:
    settings = load_settings()
    if not settings.redis_url:
        raise SystemExit("REDIS_URL must be set to run worker")

    redis_conn = redis.from_url(settings.redis_url)
    queue_names = (os.getenv("RQ_QUEUES") or "mot").split(",")
    queues = [Queue(name.strip(), connection=redis_conn) for name in queue_names if name.strip()]

    _start_worker_metrics_server()
    _start_queue_depth_poller(queues)

    # RQ 2.x removed the `Connection` context manager; passing the connection
    # directly keeps this compatible with both RQ 1.x and 2.x.
    worker = MetricsWorker(queues, connection=redis_conn)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
