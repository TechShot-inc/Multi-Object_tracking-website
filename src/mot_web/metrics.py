from __future__ import annotations

import os
import time
from typing import Any, Callable

from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


_SERVICE = os.getenv("MOT_SERVICE", "mot")

HTTP_REQUESTS_TOTAL = Counter(
    "mot_http_requests_total",
    "Total HTTP requests",
    labelnames=("service", "method", "path", "status"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "mot_http_request_duration_seconds",
    "HTTP request duration (seconds)",
    labelnames=("service", "method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

WS_CONNECTIONS_ACTIVE = Gauge(
    "mot_ws_connections_active",
    "Active WebSocket connections",
    labelnames=("service",),
)
WS_MESSAGES_TOTAL = Counter(
    "mot_ws_messages_total",
    "Total WebSocket messages",
    labelnames=("service", "direction", "type"),
)
WS_ERRORS_TOTAL = Counter(
    "mot_ws_errors_total",
    "Total WebSocket errors",
    labelnames=("service",),
)

REALTIME_FRAMES_TOTAL = Counter(
    "mot_realtime_frames_total",
    "Total realtime frames processed",
    labelnames=("service",),
)
REALTIME_FRAME_TOTAL_DURATION_SECONDS = Histogram(
    "mot_realtime_frame_total_duration_seconds",
    "Realtime end-to-end processing time per frame (seconds)",
    labelnames=("service",),
    buckets=(0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2),
)

RQ_QUEUE_DEPTH = Gauge(
    "mot_rq_queue_depth",
    "RQ queue depth (jobs)",
    labelnames=("queue",),
)
RQ_JOBS_TOTAL = Counter(
    "mot_rq_jobs_total",
    "Total RQ jobs processed",
    labelnames=("queue", "status"),
)
RQ_JOB_DURATION_SECONDS = Histogram(
    "mot_rq_job_duration_seconds",
    "RQ job duration (seconds)",
    labelnames=("queue",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


def install_metrics(app: FastAPI, *, service: str) -> None:
    """Install a Prometheus /metrics endpoint plus lightweight instrumentation."""

    # Expose default process/python metrics + our custom ones.
    app.mount("/metrics", make_asgi_app())

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next: Callable):
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = _route_path(request)

        t0 = time.perf_counter()
        status = "500"
        try:
            resp = await call_next(request)
            status = str(getattr(resp, "status_code", 200))
            return resp
        except Exception:
            status = "500"
            raise
        finally:
            dt = time.perf_counter() - t0
            HTTP_REQUESTS_TOTAL.labels(service=service, method=method, path=path, status=status).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(service=service, method=method, path=path).observe(dt)
