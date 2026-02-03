"""
Tests for error handling and edge cases in routes.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pytest

from mot_web.app_factory import create_app


@pytest.fixture
def app(tmp_path: Path):
    """Create app with temporary directories."""
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")
    os.environ["RESULTS_DIR"] = str(tmp_path / "results")
    os.environ["ENV"] = "dev"

    app = create_app()
    app.config["TESTING"] = True
    return app


class TestVideoUploadErrors:
    """Tests for video upload error handling."""

    def test_upload_without_file_returns_400(self, app) -> None:
        """Upload request without file should return 400."""
        client = app.test_client()
        resp = client.post("/video/upload", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "missing file" in resp.get_json()["error"]

    def test_upload_empty_filename_returns_400(self, app) -> None:
        """Upload with empty filename should return 400."""
        client = app.test_client()
        data = {"file": (BytesIO(b"content"), "")}
        resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_unsupported_extension_returns_400(self, app) -> None:
        """Upload with unsupported file type should return 400."""
        client = app.test_client()
        data = {"file": (BytesIO(b"content"), "document.pdf")}
        resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "unsupported file type" in resp.get_json()["error"]

    def test_upload_txt_file_returns_400(self, app) -> None:
        """Upload text file should be rejected."""
        client = app.test_client()
        data = {"file": (BytesIO(b"text content"), "notes.txt")}
        resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_upload_exe_file_returns_400(self, app) -> None:
        """Upload executable should be rejected."""
        client = app.test_client()
        data = {"file": (BytesIO(b"\x00\x00"), "malware.exe")}
        resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400


class TestVideoUploadSuccess:
    """Tests for successful video uploads."""

    @pytest.mark.parametrize("extension", [".mp4", ".mov", ".avi", ".mkv"])
    def test_accepts_valid_extensions(self, app, extension: str) -> None:
        """Should accept all supported video extensions."""
        client = app.test_client()
        data = {"file": (BytesIO(b"video content"), f"video{extension}")}
        resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert "job_id" in resp.get_json()

    def test_accepts_uppercase_extension(self, app) -> None:
        """Should accept uppercase extensions."""
        client = app.test_client()
        data = {"file": (BytesIO(b"video content"), "VIDEO.MP4")}
        resp = client.post("/video/upload", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200


class TestRunJobErrors:
    """Tests for video processing error handling."""

    def test_run_nonexistent_job_returns_404(self, app) -> None:
        """Running a non-existent job should return 404."""
        client = app.test_client()
        resp = client.post("/video/run/nonexistent_job_id", json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


class TestStatusErrors:
    """Tests for status endpoint error handling."""

    def test_status_nonexistent_job(self, app) -> None:
        """Status for non-existent job should return not_found state."""
        client = app.test_client()
        resp = client.get("/video/status/nonexistent_job_id")
        assert resp.status_code == 200  # Returns 200 with not_found state
        assert resp.get_json()["state"] == "not_found"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, app) -> None:
        """Health endpoint should return ok status."""
        client = app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_health_is_json(self, app) -> None:
        """Health endpoint should return JSON."""
        client = app.test_client()
        resp = client.get("/health")
        assert resp.content_type == "application/json"


class TestPageRoutes:
    """Tests for page rendering routes."""

    def test_index_returns_html(self, app) -> None:
        """Index page should return HTML."""
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data

    def test_video_page_returns_html(self, app) -> None:
        """Video page should return HTML."""
        client = app.test_client()
        resp = client.get("/video/")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")

    def test_realtime_page_returns_html(self, app) -> None:
        """Realtime page should return HTML."""
        client = app.test_client()
        resp = client.get("/realtime/")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")


class TestMethodNotAllowed:
    """Tests for HTTP method restrictions."""

    def test_upload_get_not_allowed(self, app) -> None:
        """GET on upload endpoint should fail."""
        client = app.test_client()
        resp = client.get("/video/upload")
        assert resp.status_code == 405

    def test_run_get_not_allowed(self, app) -> None:
        """GET on run endpoint should fail."""
        client = app.test_client()
        resp = client.get("/video/run/someid")
        assert resp.status_code == 405

    def test_health_post_not_allowed(self, app) -> None:
        """POST on health endpoint should fail."""
        client = app.test_client()
        resp = client.post("/health")
        assert resp.status_code == 405
