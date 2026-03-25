from __future__ import annotations

import os
import time
from typing import Callable

from fastapi import FastAPI, Request

try:
    import prometheus_client as _prom
except ModuleNotFoundError:  # pragma: no cover
    _prom = None


_PROMETHEUS_AVAILABLE = _prom is not None


class _NoopMetric:
    def labels(self, **kwargs) -> "_NoopMetric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None

    def dec(self, amount: float = 1.0) -> None:
        return None

    def observe(self, value: float) -> None:
        return None

    def set(self, value: float) -> None:
        return None


_SERVICE = os.getenv("MOT_SERVICE", "mot")

if _prom is not None:
    HTTP_REQUESTS_TOTAL = _prom.Counter(
        "mot_http_requests_total",
        "Total HTTP requests",
        labelnames=("service", "method", "path", "status"),
    )
    HTTP_REQUEST_DURATION_SECONDS = _prom.Histogram(
        "mot_http_request_duration_seconds",
        "HTTP request duration (seconds)",
        labelnames=("service", "method", "path"),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )

    WS_CONNECTIONS_ACTIVE = _prom.Gauge(
        "mot_ws_connections_active",
        "Active WebSocket connections",
        labelnames=("service",),
    )
    WS_MESSAGES_TOTAL = _prom.Counter(
        "mot_ws_messages_total",
        "Total WebSocket messages",
        labelnames=("service", "direction", "type"),
    )
    WS_ERRORS_TOTAL = _prom.Counter(
        "mot_ws_errors_total",
        "Total WebSocket errors",
        labelnames=("service",),
    )

    REALTIME_FRAMES_TOTAL = _prom.Counter(
        "mot_realtime_frames_total",
        "Total realtime frames processed",
        labelnames=("service",),
    )
    REALTIME_FRAME_TOTAL_DURATION_SECONDS = _prom.Histogram(
        "mot_realtime_frame_total_duration_seconds",
        "Realtime end-to-end processing time per frame (seconds)",
        labelnames=("service",),
        buckets=(0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2),
    )

    RQ_QUEUE_DEPTH = _prom.Gauge(
        "mot_rq_queue_depth",
        "RQ queue depth (jobs)",
        labelnames=("queue",),
    )
    RQ_JOBS_TOTAL = _prom.Counter(
        "mot_rq_jobs_total",
        "Total RQ jobs processed",
        labelnames=("queue", "status"),
    )
    RQ_JOB_DURATION_SECONDS = _prom.Histogram(
        "mot_rq_job_duration_seconds",
        "RQ job duration (seconds)",
        labelnames=("queue",),
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
    )
else:  # pragma: no cover
    HTTP_REQUESTS_TOTAL = _NoopMetric()
    HTTP_REQUEST_DURATION_SECONDS = _NoopMetric()
    WS_CONNECTIONS_ACTIVE = _NoopMetric()
    WS_MESSAGES_TOTAL = _NoopMetric()
    WS_ERRORS_TOTAL = _NoopMetric()
    REALTIME_FRAMES_TOTAL = _NoopMetric()
    REALTIME_FRAME_TOTAL_DURATION_SECONDS = _NoopMetric()
    RQ_QUEUE_DEPTH = _NoopMetric()
    RQ_JOBS_TOTAL = _NoopMetric()
    RQ_JOB_DURATION_SECONDS = _NoopMetric()


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return request.url.path


def install_metrics(app: FastAPI, *, service: str) -> None:
    """Install a Prometheus /metrics endpoint plus lightweight instrumentation."""

    if _prom is None:
        return None

    assert _PROMETHEUS_AVAILABLE

    # Expose default process/python metrics + our custom ones.
    app.mount("/metrics", _prom.make_asgi_app())

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
