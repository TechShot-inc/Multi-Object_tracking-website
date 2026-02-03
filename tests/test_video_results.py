"""
Tests for video results endpoints (annotations, video, analytics).
"""

from __future__ import annotations

import json
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


@pytest.fixture
def completed_job(app):
    """Create and complete a job, returning the job_id."""
    client = app.test_client()
    
    # Upload
    resp = client.post(
        "/video/upload",
        data={"file": (BytesIO(b"fake video content"), "test.mp4")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]
    
    # Run
    run_resp = client.post(f"/video/run/{job_id}", json={})
    assert run_resp.status_code == 200
    
    return job_id


class TestAnnotationsEndpoint:
    """Tests for /video/results/<job_id>/annotations endpoint."""

    def test_get_annotations_returns_json(self, app, completed_job) -> None:
        """Should return annotations.json for completed job."""
        client = app.test_client()
        resp = client.get(f"/video/results/{completed_job}/annotations")
        
        assert resp.status_code == 200
        assert resp.content_type == "application/json"
        
        data = resp.get_json()
        assert "tracks" in data
        assert "source_video" in data

    def test_get_annotations_not_found_for_missing_job(self, app) -> None:
        """Should return 404 for non-existent job."""
        client = app.test_client()
        resp = client.get("/video/results/nonexistent_job/annotations")
        
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]

    def test_get_annotations_with_download_flag(self, app, completed_job) -> None:
        """Should set Content-Disposition when download=1."""
        client = app.test_client()
        resp = client.get(f"/video/results/{completed_job}/annotations?download=1")
        
        assert resp.status_code == 200
        assert "attachment" in resp.headers.get("Content-Disposition", "")


class TestVideoEndpoint:
    """Tests for /video/results/<job_id>/video endpoint."""

    def test_get_video_returns_original_as_fallback(self, app, completed_job) -> None:
        """Should return original video if no processed video exists."""
        client = app.test_client()
        resp = client.get(f"/video/results/{completed_job}/video")
        
        # The stub doesn't create an output video, so it falls back to original
        assert resp.status_code == 200
        assert resp.content_type == "video/mp4"
        assert resp.data == b"fake video content"

    def test_get_video_not_found_for_missing_job(self, app) -> None:
        """Should return 404 for non-existent job."""
        client = app.test_client()
        resp = client.get("/video/results/nonexistent_job/video")
        
        assert resp.status_code == 404

    def test_get_video_returns_processed_if_exists(self, app, completed_job) -> None:
        """Should return processed video if it exists."""
        client = app.test_client()
        settings = app.config["SETTINGS"]
        
        # Create a fake processed video
        output_path = settings.results_dir / completed_job / "output.mp4"
        output_path.write_bytes(b"processed video content")
        
        resp = client.get(f"/video/results/{completed_job}/video")
        
        assert resp.status_code == 200
        assert resp.data == b"processed video content"


class TestAnalyticsEndpoint:
    """Tests for /video/results/<job_id>/analytics endpoint."""

    def test_get_analytics_returns_stub_data(self, app, completed_job) -> None:
        """Should return analytics data derived from annotations."""
        client = app.test_client()
        resp = client.get(f"/video/results/{completed_job}/analytics")
        
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_tracks" in data
        assert data["total_tracks"] == 0  # Stub has empty tracks

    def test_get_analytics_not_found_for_missing_job(self, app) -> None:
        """Should return 404 for non-existent job."""
        client = app.test_client()
        resp = client.get("/video/results/nonexistent_job/analytics")
        
        assert resp.status_code == 404

    def test_get_analytics_uses_file_if_exists(self, app, completed_job) -> None:
        """Should return analytics.json if it exists."""
        client = app.test_client()
        settings = app.config["SETTINGS"]
        
        # Create a pre-computed analytics file
        analytics_data = {
            "total_tracks": 5,
            "heatmap": "base64_encoded_image",
            "object_counts": {"0": 2, "1": 3, "2": 5}
        }
        analytics_path = settings.results_dir / completed_job / "analytics.json"
        analytics_path.write_text(json.dumps(analytics_data), encoding="utf-8")
        
        resp = client.get(f"/video/results/{completed_job}/analytics")
        
        assert resp.status_code == 200
        # When file exists, send_file returns it directly
        data = resp.get_json()
        assert data["total_tracks"] == 5


class TestFullUploadWorkflow:
    """Integration tests for the complete upload-process-results workflow."""

    def test_full_workflow_with_roi(self, app) -> None:
        """Test complete workflow with ROI parameter."""
        client = app.test_client()
        
        # Step 1: Upload
        upload_resp = client.post(
            "/video/upload",
            data={"file": (BytesIO(b"video data"), "roi_test.mp4")},
            content_type="multipart/form-data",
        )
        assert upload_resp.status_code == 200
        job_id = upload_resp.get_json()["job_id"]
        
        # Step 2: Run with ROI
        roi = {"x": 100, "y": 100, "width": 500, "height": 400}
        run_resp = client.post(f"/video/run/{job_id}", json={"roi": roi})
        assert run_resp.status_code == 200
        assert run_resp.get_json()["state"] == "done"
        
        # Step 3: Check status
        status_resp = client.get(f"/video/status/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.get_json()["state"] == "done"
        
        # Step 4: Get annotations
        ann_resp = client.get(f"/video/results/{job_id}/annotations")
        assert ann_resp.status_code == 200
        annotations = ann_resp.get_json()
        assert annotations["params"]["roi"] == roi
        
        # Step 5: Get video
        video_resp = client.get(f"/video/results/{job_id}/video")
        assert video_resp.status_code == 200
        
        # Step 6: Get analytics
        analytics_resp = client.get(f"/video/results/{job_id}/analytics")
        assert analytics_resp.status_code == 200

    def test_multiple_uploads_isolated(self, app) -> None:
        """Multiple uploads should be isolated from each other."""
        client = app.test_client()
        
        # Upload two videos
        job_ids = []
        for i in range(2):
            resp = client.post(
                "/video/upload",
                data={"file": (BytesIO(f"video{i}".encode()), f"video{i}.mp4")},
                content_type="multipart/form-data",
            )
            job_ids.append(resp.get_json()["job_id"])
        
        # Process both
        for job_id in job_ids:
            client.post(f"/video/run/{job_id}", json={})
        
        # Each should have its own annotations
        for i, job_id in enumerate(job_ids):
            resp = client.get(f"/video/results/{job_id}/annotations")
            assert resp.status_code == 200
            annotations = resp.get_json()
            assert f"video{i}.mp4" in annotations["source_video"]
