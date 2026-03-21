"""
Integration tests for the complete video processing workflow.

These tests verify the full end-to-end flow from upload through processing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mot_web.app_factory import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def app(tmp_path: Path):
    """Create app with temporary directories."""
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")
    os.environ["RESULTS_DIR"] = str(tmp_path / "results")
    os.environ["ENV"] = "dev"

    app = create_app()
    return app


class TestFullVideoWorkflow:
    """End-to-end tests for video processing workflow."""

    def test_complete_workflow_upload_run_status(self, app) -> None:
        """Test complete workflow: upload -> run -> check status."""
        client = TestClient(app)

        # Step 1: Upload video
        upload_resp = client.post(
            "/video/upload",
            files={"file": ("test.mp4", b"video content", "video/mp4")},
        )
        assert upload_resp.status_code == 200
        job_id = upload_resp.json()["job_id"]

        # Step 2: Check initial status
        status_resp = client.get(f"/video/status/{job_id}")
        assert status_resp.json()["state"] == "created"

        # Step 3: Run processing
        run_resp = client.post(f"/video/run/{job_id}", json={"roi": None})
        assert run_resp.status_code == 200
        assert run_resp.json()["state"] == "done"

        # Step 4: Verify final status
        final_status = client.get(f"/video/status/{job_id}").json()
        assert final_status["state"] == "done"
        assert final_status["filename"] == "test.mp4"

    def test_workflow_produces_annotations(self, app) -> None:
        """Workflow should produce annotations.json file."""
        client = TestClient(app)
        settings = app.state.settings

        # Upload and run
        upload_resp = client.post(
            "/video/upload",
            files={"file": ("movie.mp4", b"video", "video/mp4")},
        )
        job_id = upload_resp.json()["job_id"]
        client.post(f"/video/run/{job_id}", json={})

        # Check annotations exist
        annotations_path = settings.results_dir / job_id / "annotations.json"
        assert annotations_path.exists()

        # Verify annotations content
        annotations = json.loads(annotations_path.read_text())
        assert "tracks" in annotations
        assert "source_video" in annotations

    def test_multiple_concurrent_jobs(self, app) -> None:
        """Multiple jobs should work independently."""
        client = TestClient(app)

        # Create multiple jobs
        jobs = []
        for i in range(3):
            resp = client.post(
                "/video/upload",
                files={"file": (f"video{i}.mp4", f"video{i}".encode(), "video/mp4")},
            )
            jobs.append(resp.json()["job_id"])

        # All jobs should have unique IDs
        assert len(set(jobs)) == 3

        # Run all jobs
        for job_id in jobs:
            resp = client.post(f"/video/run/{job_id}", json={})
            assert resp.status_code == 200

        # All should be done
        for job_id in jobs:
            status = client.get(f"/video/status/{job_id}").json()
            assert status["state"] == "done"

    def test_workflow_with_roi_params(self, app) -> None:
        """Workflow should accept and preserve ROI parameters."""
        client = TestClient(app)
        settings = app.state.settings

        # Upload
        resp = client.post(
            "/video/upload",
            files={"file": ("roi_test.mp4", b"video", "video/mp4")},
        )
        job_id = resp.json()["job_id"]

        # Run with ROI
        roi = [100, 100, 500, 400]
        client.post(f"/video/run/{job_id}", json={"roi": roi})

        # Check params were preserved
        annotations_path = settings.results_dir / job_id / "annotations.json"
        annotations = json.loads(annotations_path.read_text())
        assert annotations["params"]["roi"] == roi


class TestAppFactory:
    """Tests for app factory and configuration."""

    def test_app_creates_directories(self, tmp_path: Path) -> None:
        """App should create upload and results directories."""
        upload_dir = tmp_path / "new_uploads"
        results_dir = tmp_path / "new_results"

        os.environ["PROJECT_ROOT"] = str(tmp_path)
        os.environ["UPLOAD_DIR"] = str(upload_dir)
        os.environ["RESULTS_DIR"] = str(results_dir)

        assert not upload_dir.exists()
        assert not results_dir.exists()

        create_app()

        assert upload_dir.exists()
        assert results_dir.exists()

    def test_app_has_secret_key(self, app) -> None:
        """App should have SECRET_KEY configured."""
        assert app.state.settings.secret_key is not None
        assert len(app.state.settings.secret_key) > 0

    def test_app_has_max_content_length(self, app) -> None:
        """App should have MAX_CONTENT_LENGTH configured."""
        assert app.state.max_content_length > 0

    def test_blueprints_registered(self, app) -> None:
        """Core routes should be registered."""
        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/health" in paths
        assert "/" in paths
        assert "/video/" in paths
        assert "/video/upload" in paths
        assert "/realtime/" in paths
        assert "/realtime/track" in paths


class TestFileStorage:
    """Tests for file storage behavior."""

    def test_uploaded_file_saved_to_disk(self, app) -> None:
        """Uploaded files should be saved to upload directory."""
        client = TestClient(app)
        settings = app.state.settings

        content = b"unique video content 12345"
        client.post(
            "/video/upload",
            files={"file": ("saved.mp4", content, "video/mp4")},
        )

        saved_path = settings.upload_dir / "saved.mp4"
        assert saved_path.exists()
        assert saved_path.read_bytes() == content

    def test_job_directory_structure(self, app) -> None:
        """Job should create proper directory structure."""
        client = TestClient(app)
        settings = app.state.settings

        resp = client.post(
            "/video/upload",
            files={"file": ("structure.mp4", b"video", "video/mp4")},
        )
        job_id = resp.json()["job_id"]

        job_dir = settings.results_dir / job_id
        assert job_dir.exists()
        assert (job_dir / "meta.json").exists()
        assert (job_dir / "status.json").exists()
