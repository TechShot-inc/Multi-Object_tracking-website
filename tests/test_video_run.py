from __future__ import annotations

from pathlib import Path
import os
import pytest

from mot_web.app_factory import create_app
from fastapi.testclient import TestClient


@pytest.fixture()
def app(tmp_path: Path):
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    os.environ["UPLOAD_DIR"] = str(tmp_path / "var" / "uploads")
    os.environ["RESULTS_DIR"] = str(tmp_path / "var" / "results")
    os.environ["ENV"] = "dev"

    app = create_app()
    return app


def test_run_job_creates_annotations_and_marks_done(app):
    client = TestClient(app)

    # upload first
    resp = client.post("/video/upload", files={"file": ("x.mp4", b"fake video bytes", "video/mp4")})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # run
    run_resp = client.post(f"/video/run/{job_id}", json={"roi": None})
    assert run_resp.status_code == 200
    assert run_resp.json()["state"] == "done"

    # output artifact
    settings = app.state.settings
    out = settings.results_dir / job_id / "annotations.json"
    assert out.exists()

    # status
    st = client.get(f"/video/status/{job_id}").json()
    assert st["state"] == "done"
