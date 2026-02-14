"""
Tests for error handling and edge cases in routes.
"""

from __future__ import annotations

import os
from io import BytesIO
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


class TestVideoUploadErrors:
    """Tests for video upload error handling."""

    def test_upload_without_file_returns_400(self, app) -> None:
        """Upload request without file should return 400."""
        client = TestClient(app)
        resp = client.post("/video/upload", files={})
        assert resp.status_code == 400
        assert "missing file" in resp.json()["error"]

    def test_upload_empty_filename_returns_400(self, app) -> None:
        """Upload with empty filename should return 400."""
        client = TestClient(app)
        resp = client.post(
            "/video/upload",
            files={"file": ("", b"content", "video/mp4")},
        )
        # Some FastAPI/Starlette versions treat empty filenames as a multipart parsing/validation
        # error and return 422 before the route handler runs.
        assert resp.status_code in (400, 422)

    def test_upload_unsupported_extension_returns_400(self, app) -> None:
        """Upload with unsupported file type should return 400."""
        client = TestClient(app)
        resp = client.post(
            "/video/upload",
            files={"file": ("document.pdf", b"content", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "unsupported file type" in resp.json()["error"]

    def test_upload_txt_file_returns_400(self, app) -> None:
        """Upload text file should be rejected."""
        client = TestClient(app)
        resp = client.post(
            "/video/upload",
            files={"file": ("notes.txt", b"text content", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_exe_file_returns_400(self, app) -> None:
        """Upload executable should be rejected."""
        client = TestClient(app)
        resp = client.post(
            "/video/upload",
            files={"file": ("malware.exe", b"\x00\x00", "application/octet-stream")},
        )
        assert resp.status_code == 400


class TestVideoUploadSuccess:
    """Tests for successful video uploads."""

    @pytest.mark.parametrize("extension", [".mp4", ".mov", ".avi", ".mkv"])
    def test_accepts_valid_extensions(self, app, extension: str) -> None:
        """Should accept all supported video extensions."""
        client = TestClient(app)
        resp = client.post(
            "/video/upload",
            files={"file": (f"video{extension}", b"video content", "video/mp4")},
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_accepts_uppercase_extension(self, app) -> None:
        """Should accept uppercase extensions."""
        client = TestClient(app)
        resp = client.post(
            "/video/upload",
            files={"file": ("VIDEO.MP4", b"video content", "video/mp4")},
        )
        assert resp.status_code == 200


class TestRunJobErrors:
    """Tests for video processing error handling."""

    def test_run_nonexistent_job_returns_404(self, app) -> None:
        """Running a non-existent job should return 404."""
        client = TestClient(app)
        resp = client.post("/video/run/nonexistent_job_id", json={})
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]


class TestStatusErrors:
    """Tests for status endpoint error handling."""

    def test_status_nonexistent_job(self, app) -> None:
        """Status for non-existent job should return not_found state."""
        client = TestClient(app)
        resp = client.get("/video/status/nonexistent_job_id")
        assert resp.status_code == 200  # Returns 200 with not_found state
        assert resp.json()["state"] == "not_found"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, app) -> None:
        """Health endpoint should return ok status."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_is_json(self, app) -> None:
        """Health endpoint should return JSON."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.headers["content-type"].startswith("application/json")


class TestPageRoutes:
    """Tests for page rendering routes."""

    def test_index_returns_html(self, app) -> None:
        """Index page should return HTML."""
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.content or b"<html" in resp.content

    def test_video_page_returns_html(self, app) -> None:
        """Video page should return HTML."""
        client = TestClient(app)
        resp = client.get("/video/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_realtime_page_returns_html(self, app) -> None:
        """Realtime page should return HTML."""
        client = TestClient(app)
        resp = client.get("/realtime/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")


class TestMethodNotAllowed:
    """Tests for HTTP method restrictions."""

    def test_upload_get_not_allowed(self, app) -> None:
        """GET on upload endpoint should fail."""
        client = TestClient(app)
        resp = client.get("/video/upload")
        assert resp.status_code == 405

    def test_run_get_not_allowed(self, app) -> None:
        """GET on run endpoint should fail."""
        client = TestClient(app)
        resp = client.get("/video/run/someid")
        assert resp.status_code == 405

    def test_health_post_not_allowed(self, app) -> None:
        """POST on health endpoint should fail."""
        client = TestClient(app)
        resp = client.post("/health")
        assert resp.status_code == 405
