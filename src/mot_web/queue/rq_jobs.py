from __future__ import annotations

import json
import time
from typing import Any

import redis
from rq import Queue

from mot_web.config import load_settings
from mot_web.services.tracking_service import TrackingService
from mot_web.tracking.pipeline import run_video


_STATUS_KEY_PREFIX = "mot:job-status:"


def _redis_from_url(url: str) -> redis.Redis:
    # RQ expects the underlying redis client to return bytes for list operations.
    # Using decode_responses=True can break RQ internals (e.g., intermediate queue
    # cleanup calling `.decode()` on already-decoded strings).
    return redis.from_url(url)


def _set_status_redis(r: redis.Redis, job_id: str, payload: dict[str, Any]) -> None:
    r.set(f"{_STATUS_KEY_PREFIX}{job_id}", json.dumps(payload))


def _get_status_redis(r: redis.Redis, job_id: str) -> dict[str, Any] | None:
    raw = r.get(f"{_STATUS_KEY_PREFIX}{job_id}")
    if not raw:
        return None
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return None


def process_video_job(job_id: str, params: dict[str, Any] | None = None) -> None:
    """RQ task entrypoint. Must be importable at module scope."""
    params = params or {}
    settings = load_settings()

    svc = TrackingService(settings.upload_dir, settings.results_dir)

    st = svc.get_status(job_id)
    if st.get("state") == "not_found":
        return

    r = _redis_from_url(settings.redis_url) if settings.redis_url else None

    def set_status(state: str, message: str, extra: dict[str, Any] | None = None) -> None:
        svc.set_status(job_id, state, message)
        payload = {
            "job_id": job_id,
            "state": state,
            "message": message,
            "updated_at_unix": int(time.time()),
        }
        if extra:
            payload.update(extra)
        if r is not None:
            _set_status_redis(r, job_id, payload)

    try:
        set_status("running", "processing started")
        job_dir = settings.results_dir / job_id
        run_video(input_path=settings.upload_dir / st["filename"], output_dir=job_dir, params=params)
        set_status("done", "processing complete")
    except Exception as e:
        set_status("failed", str(e), extra={"error": str(e)})
        raise


def enqueue_video_job(job_id: str, params: dict[str, Any] | None = None) -> str:
    settings = load_settings()
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is not set")

    r = _redis_from_url(settings.redis_url)
    q = Queue(name="mot", connection=r)
    job = q.enqueue(process_video_job, job_id, params or {}, job_timeout=60 * 60)

    # Write queued status immediately
    payload = {
        "job_id": job_id,
        "state": "queued",
        "message": "queued",
        "updated_at_unix": int(time.time()),
        "rq_job_id": job.id,
    }
    _set_status_redis(r, job_id, payload)

    return job.id


def get_job_status(job_id: str) -> dict[str, Any] | None:
    settings = load_settings()
    if not settings.redis_url:
        return None
    r = _redis_from_url(settings.redis_url)
    return _get_status_redis(r, job_id)
