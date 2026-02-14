from __future__ import annotations

from typing import Any

import mot_web.queue.rq_jobs as rq_jobs


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self.store[key] = value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    def get(self, key: str):
        return self.store.get(key)


class _FakeJob:
    def __init__(self, id_: str):
        self.id = id_


class _FakeQueue:
    def __init__(self, *args: Any, **kwargs: Any):
        self.enqueued: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def enqueue(self, fn, *args: Any, **kwargs: Any):
        self.enqueued.append((fn, args, kwargs))
        return _FakeJob("rq-job-123")


def test_enqueue_video_job_sets_redis_status(monkeypatch):
    fake_redis = _FakeRedis()

    # Patch redis client creation
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(rq_jobs.redis, "from_url", lambda *_args, **_kwargs: fake_redis)

    # Patch RQ Queue
    monkeypatch.setattr(rq_jobs, "Queue", _FakeQueue)

    rq_id = rq_jobs.enqueue_video_job("job-abc", params={"roi": [0, 0, 1, 1]})
    assert rq_id == "rq-job-123"

    status = rq_jobs.get_job_status("job-abc")
    assert status is not None
    assert status["state"] == "queued"
    assert status["job_id"] == "job-abc"

