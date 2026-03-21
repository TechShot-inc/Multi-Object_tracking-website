"""
Contract tests for the tracking pipeline.

These tests verify the interface contract between the web layer and the tracking pipeline,
ensuring that the pipeline produces output in the expected format regardless of implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mot_web.tracking.pipeline import run_video


@pytest.fixture
def temp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


@pytest.fixture
def fake_video(temp_dirs: tuple[Path, Path]) -> Path:
    """Create a fake video file for testing."""
    input_dir, _ = temp_dirs
    video_path = input_dir / "test_video.mp4"
    # Create a minimal fake file (pipeline stub doesn't actually read it)
    video_path.write_bytes(b"fake video content")
    return video_path


class TestPipelineContract:
    """Tests for the pipeline output contract."""

    def test_run_video_creates_output_directory(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Pipeline should create output directory if it doesn't exist."""
        _, output_dir = temp_dirs
        nested_output = output_dir / "nested" / "deep"

        run_video(input_path=fake_video, output_dir=nested_output, params={})

        assert nested_output.exists()
        assert nested_output.is_dir()

    def test_run_video_creates_annotations_file(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Pipeline must create annotations.json in output directory."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_001"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        annotations_file = job_output / "annotations.json"
        assert annotations_file.exists(), "Pipeline must create annotations.json"

    def test_annotations_is_valid_json(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Annotations file must contain valid JSON."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_002"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        annotations_file = job_output / "annotations.json"
        content = annotations_file.read_text(encoding="utf-8")

        # Should not raise
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_annotations_has_required_fields(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Annotations must contain required contract fields."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_003"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        annotations_file = job_output / "annotations.json"
        data = json.loads(annotations_file.read_text(encoding="utf-8"))

        # Required fields in the contract
        assert "source_video" in data, "Must include source_video path"
        assert "created_at_unix" in data, "Must include creation timestamp"
        assert "tracks" in data, "Must include tracks list"

    def test_annotations_tracks_is_list(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Tracks field must be a list."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_004"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        annotations_file = job_output / "annotations.json"
        data = json.loads(annotations_file.read_text(encoding="utf-8"))

        assert isinstance(data["tracks"], list), "tracks must be a list"

    def test_annotations_preserves_params(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Pipeline should store the params used for processing."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_005"
        params = {"roi": [10, 20, 100, 200], "confidence_threshold": 0.5}

        run_video(input_path=fake_video, output_dir=job_output, params=params)

        annotations_file = job_output / "annotations.json"
        data = json.loads(annotations_file.read_text(encoding="utf-8"))

        assert "params" in data, "Must include params used"
        assert data["params"] == params, "Params should be preserved exactly"

    def test_annotations_source_video_is_string(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """source_video field must be a string path."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_006"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        annotations_file = job_output / "annotations.json"
        data = json.loads(annotations_file.read_text(encoding="utf-8"))

        assert isinstance(data["source_video"], str)

    def test_annotations_created_at_is_integer(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """created_at_unix field must be an integer timestamp."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_007"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        annotations_file = job_output / "annotations.json"
        data = json.loads(annotations_file.read_text(encoding="utf-8"))

        assert isinstance(data["created_at_unix"], int)
        assert data["created_at_unix"] > 0


class TestPipelineRobustness:
    """Tests for pipeline error handling and edge cases."""

    def test_run_video_with_empty_params(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Pipeline should handle empty params dict."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_empty_params"

        # Should not raise
        run_video(input_path=fake_video, output_dir=job_output, params={})

        assert (job_output / "annotations.json").exists()

    def test_run_video_with_none_roi(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Pipeline should handle None ROI in params."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_none_roi"

        run_video(input_path=fake_video, output_dir=job_output, params={"roi": None})

        data = json.loads((job_output / "annotations.json").read_text())
        assert data["params"]["roi"] is None

    def test_run_video_idempotent_output_dir(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Running pipeline twice to same output should overwrite cleanly."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_idempotent"

        # Run twice
        run_video(input_path=fake_video, output_dir=job_output, params={"run": 1})
        run_video(input_path=fake_video, output_dir=job_output, params={"run": 2})

        data = json.loads((job_output / "annotations.json").read_text())
        assert data["params"]["run"] == 2, "Second run should overwrite"


class TestTrackFormat:
    """Tests for individual track format when tracks are present.

    Note: These tests define the expected format for when the real
    tracking implementation is connected. The stub returns empty tracks.
    """

    def test_empty_tracks_is_valid(self, fake_video: Path, temp_dirs: tuple[Path, Path]) -> None:
        """Empty tracks list is valid output."""
        _, output_dir = temp_dirs
        job_output = output_dir / "job_empty_tracks"

        run_video(input_path=fake_video, output_dir=job_output, params={})

        data = json.loads((job_output / "annotations.json").read_text())
        assert data["tracks"] == [], "Stub should return empty tracks"

    # Future tests for when real tracking is implemented:
    # - test_track_has_id
    # - test_track_has_bboxes
    # - test_track_bbox_format (x, y, w, h)
    # - test_track_has_confidence_scores
