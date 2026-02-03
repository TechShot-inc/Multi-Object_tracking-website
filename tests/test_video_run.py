from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import pytest

from mot_web.app_factory import create_app


@pytest.fixture()
def app(tmp_path: Path):
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    os.environ["UPLOAD_DIR"] = str(tmp_path / "var" / "uploads")
    os.environ["RESULTS_DIR"] = str(tmp_path / "var" / "results")
    os.environ["ENV"] = "dev"

    app = create_app()
    app.config["TESTING"] = True
    return app


def test_run_job_creates_annotations_and_marks_done(app):
    client = app.test_client()

    # upload first
    resp = client.post(
        "/video/upload",
        data={"file": (BytesIO(b"fake video bytes"), "x.mp4")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]

    # run
    run_resp = client.post(f"/video/run/{job_id}", json={"roi": None})
    assert run_resp.status_code == 200
    assert run_resp.get_json()["state"] == "done"

    # output artifact
    settings = app.config["SETTINGS"]
    out = settings.results_dir / job_id / "annotations.json"
    assert out.exists()

    # status
    st = client.get(f"/video/status/{job_id}").get_json()
    assert st["state"] == "done"
