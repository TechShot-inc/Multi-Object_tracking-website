"""
Tests for configuration and settings loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from mot_web.config import Settings, load_settings


class TestSettings:
    """Tests for the Settings dataclass."""

    def test_settings_is_frozen(self) -> None:
        """Settings should be immutable (frozen dataclass)."""
        settings = Settings(
            secret_key="test",
            environment="dev",
            host="127.0.0.1",
            port=5000,
            project_root=Path("."),
            upload_dir=Path("./uploads"),
            results_dir=Path("./results"),
            max_upload_mb=100,
            queue_mode="inline",
            redis_url=None,
            triton_url=None,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            settings.port = 8080  # type: ignore


class TestLoadSettings:
    """Tests for settings loading from environment."""

    def test_default_values(self, tmp_path: Path) -> None:
        """load_settings should use defaults when env vars not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Set PROJECT_ROOT to avoid cwd issues
            os.environ["PROJECT_ROOT"] = str(tmp_path)
            settings = load_settings()

        assert settings.environment == "dev"
        assert settings.host == "127.0.0.1"
        assert settings.port == 5000
        assert settings.max_upload_mb == 512

    def test_reads_secret_key_from_env(self, tmp_path: Path) -> None:
        """load_settings should read SECRET_KEY from environment."""
        with mock.patch.dict(
            os.environ, {"PROJECT_ROOT": str(tmp_path), "SECRET_KEY": "my-super-secret-key"}, clear=True
        ):
            settings = load_settings()

        assert settings.secret_key == "my-super-secret-key"

    def test_reads_environment_from_env(self, tmp_path: Path) -> None:
        """load_settings should read ENV from environment."""
        with mock.patch.dict(os.environ, {"PROJECT_ROOT": str(tmp_path), "ENV": "prod"}, clear=True):
            settings = load_settings()

        assert settings.environment == "prod"

    def test_reads_host_and_port_from_env(self, tmp_path: Path) -> None:
        """load_settings should read HOST and PORT from environment."""
        with mock.patch.dict(
            os.environ, {"PROJECT_ROOT": str(tmp_path), "HOST": "0.0.0.0", "PORT": "8080"}, clear=True
        ):
            settings = load_settings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8080

    def test_reads_upload_dir_from_env(self, tmp_path: Path) -> None:
        """load_settings should read UPLOAD_DIR from environment."""
        custom_upload = tmp_path / "custom_uploads"
        with mock.patch.dict(os.environ, {"PROJECT_ROOT": str(tmp_path), "UPLOAD_DIR": str(custom_upload)}, clear=True):
            settings = load_settings()

        assert settings.upload_dir == custom_upload

    def test_reads_results_dir_from_env(self, tmp_path: Path) -> None:
        """load_settings should read RESULTS_DIR from environment."""
        custom_results = tmp_path / "custom_results"
        with mock.patch.dict(
            os.environ, {"PROJECT_ROOT": str(tmp_path), "RESULTS_DIR": str(custom_results)}, clear=True
        ):
            settings = load_settings()

        assert settings.results_dir == custom_results

    def test_reads_max_upload_mb_from_env(self, tmp_path: Path) -> None:
        """load_settings should read MAX_UPLOAD_MB from environment."""
        with mock.patch.dict(os.environ, {"PROJECT_ROOT": str(tmp_path), "MAX_UPLOAD_MB": "1024"}, clear=True):
            settings = load_settings()

        assert settings.max_upload_mb == 1024

    def test_paths_are_resolved(self, tmp_path: Path) -> None:
        """Paths should be resolved to absolute paths."""
        with mock.patch.dict(
            os.environ,
            {
                "PROJECT_ROOT": str(tmp_path),
            },
            clear=True,
        ):
            settings = load_settings()

        assert settings.project_root.is_absolute()
        assert settings.upload_dir.is_absolute()
        assert settings.results_dir.is_absolute()
