from __future__ import annotations

import os
import time

import pytest
import redis
from fastapi.testclient import TestClient
from rq import Queue
from rq.worker import SimpleWorker

from mot_web.app_factory import create_app


@pytest.mark.skipif(os.getenv("RUN_QUEUE_INTEGRATION") != "1", reason="Set RUN_QUEUE_INTEGRATION=1 to enable")
def test_rq_worker_processes_job_end_to_end(tmp_path, monkeypatch):
    # Runs fully in-process (FastAPI TestClient + RQ SimpleWorker), but uses a real
    # Redis instance provided by Docker Compose.
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        pytest.skip("REDIS_URL is not set")

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("QUEUE_MODE", "rq")
    monkeypatch.setenv("REDIS_URL", redis_url)

    # Avoid heavy ML deps; still validates queue wiring + status transitions.
    monkeypatch.setenv("MOT_PIPELINE", "stub")

    app = create_app()
    client = TestClient(app)

    # IMPORTANT: RQ stores pickled/binary payloads in Redis.
    # Using decode_responses=True can cause UnicodeDecodeError inside the worker.
    r = redis.from_url(redis_url)
    r.flushdb()
    q = Queue(name="mot", connection=r)

    up = client.post(
        "/video/upload",
        files={"file": ("test.mp4", b"fake video bytes", "video/mp4")},
    )
    assert up.status_code == 200
    job_id = up.json()["job_id"]

    run = client.post(f"/video/run/{job_id}", json={})
    assert run.status_code == 200
    assert run.json()["state"] == "queued"

    worker = SimpleWorker([q], connection=r)
    worker.work(burst=True)

    deadline = time.time() + 10
    last = None
    while time.time() < deadline:
        st = client.get(f"/video/status/{job_id}")
        assert st.status_code == 200
        last = st.json()
        if last.get("state") in {"done", "failed"}:
            break
        time.sleep(0.2)

    assert last is not None
    assert last.get("state") == "done", last
    assert (tmp_path / "results" / job_id / "annotations.json").exists()
