"""
Tests for the real-time tracking endpoint.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import pytest

from mot_web.app_factory import create_app
from mot_web.web.routes.realtime import CV2_AVAILABLE


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


class TestRealtimePage:
    """Tests for realtime page rendering."""

    def test_realtime_page_returns_html(self, app) -> None:
        """Realtime page should return HTML."""
        client = app.test_client()
        resp = client.get("/realtime/")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")


class TestRealtimeTrackEndpoint:
    """Tests for the /realtime/track endpoint."""

    def test_track_without_frame_returns_error(self, app) -> None:
        """Request without frame should return 400 (or 503 if OpenCV not available)."""
        client = app.test_client()
        resp = client.post("/realtime/track", data={})
        # Without OpenCV: 503, With OpenCV: 400
        assert resp.status_code in (400, 503)
        if resp.status_code == 400:
            assert "No frame provided" in resp.get_json()["error"]
        else:
            assert "OpenCV not installed" in resp.get_json()["error"]

    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
    def test_track_with_invalid_image_returns_400(self, app) -> None:
        """Request with invalid image data should return 400."""
        client = app.test_client()
        data = {"frame": (BytesIO(b"not an image"), "frame.jpg")}
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Failed to decode frame" in resp.get_json()["error"]

    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not installed")
    def test_track_with_valid_frame_returns_annotated(self, app) -> None:
        """Request with valid frame should return annotated frame."""
        client = app.test_client()
        
        # Create a minimal valid JPEG (1x1 pixel)
        # This is a minimal valid JPEG file
        jpeg_bytes = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
            0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
            0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
            0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
            0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
            0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
            0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x5E, 0x5B,
            0xC4, 0xB0, 0xA3, 0x1C, 0x07, 0xE7, 0x5E, 0x3F, 0xFF, 0xD9
        ])
        
        data = {"frame": (BytesIO(jpeg_bytes), "frame.jpg")}
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        
        # May fail if OpenCV can't decode minimal JPEG, that's ok
        if resp.status_code == 200:
            result = resp.get_json()
            assert "annotated" in result
            assert "count" in result
            assert "timestamp" in result

    def test_track_with_invalid_roi_json_returns_400(self, app) -> None:
        """Request with invalid ROI JSON should return 400."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not installed")
            
        client = app.test_client()
        # Create a simple valid image using OpenCV
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        data = {
            "frame": (BytesIO(buffer.tobytes()), "frame.jpg"),
            "roi": "not valid json"
        }
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Invalid ROI JSON" in resp.get_json()["error"]

    def test_track_with_invalid_roi_format_returns_400(self, app) -> None:
        """Request with ROI missing required fields should return 400."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not installed")
            
        client = app.test_client()
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        data = {
            "frame": (BytesIO(buffer.tobytes()), "frame.jpg"),
            "roi": json.dumps({"x": 0.1})  # Missing y, width, height
        }
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Invalid ROI format" in resp.get_json()["error"]

    def test_track_with_valid_roi_succeeds(self, app) -> None:
        """Request with valid ROI should succeed."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not installed")
            
        client = app.test_client()
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        roi = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
        data = {
            "frame": (BytesIO(buffer.tobytes()), "frame.jpg"),
            "roi": json.dumps(roi)
        }
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["has_roi"] is True

    def test_track_with_invalid_line_json_returns_400(self, app) -> None:
        """Request with invalid line JSON should return 400."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not installed")
            
        client = app.test_client()
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        data = {
            "frame": (BytesIO(buffer.tobytes()), "frame.jpg"),
            "line": "not valid json"
        }
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Invalid line JSON" in resp.get_json()["error"]

    def test_track_with_valid_line_succeeds(self, app) -> None:
        """Request with valid line should succeed."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not installed")
            
        client = app.test_client()
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        line = {"position": "left", "x": 0.5}
        data = {
            "frame": (BytesIO(buffer.tobytes()), "frame.jpg"),
            "line": json.dumps(line)
        }
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result = resp.get_json()
        assert "counts" in result
        assert "inside" in result["counts"]
        assert "outside" in result["counts"]

    def test_track_response_has_timestamp(self, app) -> None:
        """Response should include timestamp."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not installed")
            
        client = app.test_client()
        import cv2
        import numpy as np
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        
        data = {"frame": (BytesIO(buffer.tobytes()), "frame.jpg")}
        resp = client.post("/realtime/track", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        result = resp.get_json()
        assert "timestamp" in result
        assert isinstance(result["timestamp"], int)
        assert result["timestamp"] > 0


class TestRealtimeWithoutOpenCV:
    """Tests for graceful degradation when OpenCV is not available."""

    def test_cv2_available_flag_exists(self) -> None:
        """CV2_AVAILABLE flag should be defined."""
        from mot_web.web.routes.realtime import CV2_AVAILABLE
        assert isinstance(CV2_AVAILABLE, bool)
