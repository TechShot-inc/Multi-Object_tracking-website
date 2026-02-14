from __future__ import annotations

import pytest

from mot_web.app_factory import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ENV", "dev")
    return create_app()
