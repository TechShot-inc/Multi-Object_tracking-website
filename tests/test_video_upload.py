from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from mot_web.app_factory import create_app


@pytest.fixture()
def app(tmp_path: Path):
    # Override dirs via env vars before app creation
    # (load_settings reads env vars)
    import os
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    os.environ["UPLOAD_DIR"] = str(tmp_path / "var" / "uploads")
    os.environ["RESULTS_DIR"] = str(tmp_path / "var" / "results")
    os.environ["ENV"] = "dev"

    app = create_app()
    app.config["TESTING"] = True
    return app


def test_video_upload_creates_job_and_saves_file(app):
    client = app.test_client()

    data = {
        "file": (BytesIO(b"fake video bytes"), "test.mp4")
    }
    resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert "job_id" in payload
    assert payload["filename"] == "test.mp4"

    job_id = payload["job_id"]

    settings = app.config["SETTINGS"]
    saved = settings.upload_dir / "test.mp4"
    assert saved.exists()
    assert saved.read_bytes() == b"fake video bytes"

    # status should exist
    st = client.get(f"/video/status/{job_id}")
    assert st.status_code == 200
    st_json = st.get_json()
    assert st_json["job_id"] == job_id
    assert st_json["filename"] == "test.mp4"
    assert st_json["state"] in {"created", "unknown"}  # created if status.json exists


def test_video_upload_rejects_bad_extension(app):
    client = app.test_client()
    data = {"file": (BytesIO(b"nope"), "bad.txt")}
    resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
