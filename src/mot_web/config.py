from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    # Flask
    secret_key: str
    environment: str  # "dev" | "prod"
    host: str
    port: int

    # Paths
    project_root: Path
    upload_dir: Path
    results_dir: Path

    # Runtime
    max_upload_mb: int


def load_settings() -> Settings:
    project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()

    upload_dir = Path(os.getenv("UPLOAD_DIR", project_root / "var" / "uploads")).resolve()
    results_dir = Path(os.getenv("RESULTS_DIR", project_root / "var" / "results")).resolve()

    return Settings(
        secret_key=os.getenv("SECRET_KEY", "dev-only-change-me"),
        environment=os.getenv("ENV", "dev"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        project_root=project_root,
        upload_dir=upload_dir,
        results_dir=results_dir,
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "512")),
    )
