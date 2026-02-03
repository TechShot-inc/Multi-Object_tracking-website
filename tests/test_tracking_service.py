"""
Unit tests for the TrackingService.

Tests the service layer independently from Flask routes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mot_web.services.tracking_service import TrackingService, VideoJob


@pytest.fixture
def service(tmp_path: Path) -> TrackingService:
    """Create a TrackingService with temporary directories."""
    upload_dir = tmp_path / "uploads"
    results_dir = tmp_path / "results"
    return TrackingService(upload_dir, results_dir)


class TestTrackingServiceInit:
    """Tests for TrackingService initialization."""

    def test_creates_upload_directory(self, tmp_path: Path) -> None:
        """Service should create upload directory on init."""
        upload_dir = tmp_path / "new_uploads"
        results_dir = tmp_path / "new_results"

        assert not upload_dir.exists()
        TrackingService(upload_dir, results_dir)
        assert upload_dir.exists()

    def test_creates_results_directory(self, tmp_path: Path) -> None:
        """Service should create results directory on init."""
        upload_dir = tmp_path / "uploads"
        results_dir = tmp_path / "results"

        assert not results_dir.exists()
        TrackingService(upload_dir, results_dir)
        assert results_dir.exists()

    def test_handles_existing_directories(self, tmp_path: Path) -> None:
        """Service should not fail if directories already exist."""
        upload_dir = tmp_path / "uploads"
        results_dir = tmp_path / "results"
        upload_dir.mkdir()
        results_dir.mkdir()

        # Should not raise
        service = TrackingService(upload_dir, results_dir)
        assert service.upload_dir == upload_dir


class TestCreateJob:
    """Tests for job creation."""

    def test_returns_video_job(self, service: TrackingService) -> None:
        """create_job should return a VideoJob instance."""
        job = service.create_job("test.mp4")
        assert isinstance(job, VideoJob)

    def test_job_has_unique_id(self, service: TrackingService) -> None:
        """Each job should have a unique ID."""
        job1 = service.create_job("video1.mp4")
        job2 = service.create_job("video2.mp4")
        assert job1.job_id != job2.job_id

    def test_job_id_is_hex_string(self, service: TrackingService) -> None:
        """Job ID should be a valid hex string (UUID)."""
        job = service.create_job("test.mp4")
        assert len(job.job_id) == 32  # UUID hex is 32 chars
        int(job.job_id, 16)  # Should not raise

    def test_job_preserves_filename(self, service: TrackingService) -> None:
        """Job should preserve the original filename."""
        job = service.create_job("my_video.mp4")
        assert job.filename == "my_video.mp4"

    def test_job_upload_path_in_upload_dir(self, service: TrackingService) -> None:
        """Upload path should be in the upload directory."""
        job = service.create_job("test.mp4")
        assert job.upload_path.parent == service.upload_dir

    def test_job_dir_created(self, service: TrackingService) -> None:
        """Job directory should be created."""
        job = service.create_job("test.mp4")
        assert job.job_dir.exists()
        assert job.job_dir.is_dir()

    def test_meta_json_created(self, service: TrackingService) -> None:
        """meta.json should be created with job info."""
        job = service.create_job("test.mp4")
        meta_path = job.job_dir / "meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["job_id"] == job.job_id
        assert meta["filename"] == "test.mp4"
        assert "created_at_unix" in meta

    def test_status_json_created(self, service: TrackingService) -> None:
        """status.json should be created with initial state."""
        job = service.create_job("test.mp4")
        status_path = job.job_dir / "status.json"
        assert status_path.exists()

        status = json.loads(status_path.read_text())
        assert status["state"] == "created"


class TestSetStatus:
    """Tests for status updates."""

    def test_updates_state(self, service: TrackingService) -> None:
        """set_status should update the state."""
        job = service.create_job("test.mp4")
        service.set_status(job.job_id, "running")

        status_path = job.job_dir / "status.json"
        status = json.loads(status_path.read_text())
        assert status["state"] == "running"

    def test_updates_message(self, service: TrackingService) -> None:
        """set_status should update the message."""
        job = service.create_job("test.mp4")
        service.set_status(job.job_id, "failed", "Out of memory")

        status_path = job.job_dir / "status.json"
        status = json.loads(status_path.read_text())
        assert status["message"] == "Out of memory"

    def test_updates_timestamp(self, service: TrackingService) -> None:
        """set_status should update the timestamp."""
        job = service.create_job("test.mp4")
        time.sleep(0.01)  # Ensure time difference
        service.set_status(job.job_id, "done")

        status_path = job.job_dir / "status.json"
        status = json.loads(status_path.read_text())
        assert status["updated_at_unix"] >= int(time.time()) - 1

    def test_state_transitions(self, service: TrackingService) -> None:
        """Status should transition through states correctly."""
        job = service.create_job("test.mp4")

        # created -> running -> done
        service.set_status(job.job_id, "running", "processing")
        assert service.get_status(job.job_id)["state"] == "running"

        service.set_status(job.job_id, "done", "complete")
        assert service.get_status(job.job_id)["state"] == "done"


class TestGetStatus:
    """Tests for status retrieval."""

    def test_returns_job_info(self, service: TrackingService) -> None:
        """get_status should return job metadata."""
        job = service.create_job("test.mp4")
        status = service.get_status(job.job_id)

        assert status["job_id"] == job.job_id
        assert status["filename"] == "test.mp4"
        assert status["state"] == "created"

    def test_not_found_for_missing_job(self, service: TrackingService) -> None:
        """get_status should return not_found for non-existent job."""
        status = service.get_status("nonexistent_job_id")
        assert status["state"] == "not_found"

    def test_merges_meta_and_status(self, service: TrackingService) -> None:
        """get_status should merge meta.json and status.json."""
        job = service.create_job("test.mp4")
        service.set_status(job.job_id, "running", "50% complete")

        status = service.get_status(job.job_id)
        # From meta.json
        assert "filename" in status
        assert "created_at_unix" in status
        # From status.json
        assert status["state"] == "running"
        assert status["message"] == "50% complete"
