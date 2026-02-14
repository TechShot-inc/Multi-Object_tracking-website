from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    # Web
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

    # Queue
    queue_mode: str  # "inline" | "rq"
    redis_url: str | None

    # Triton (optional)
    triton_url: str | None


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
        queue_mode=(os.getenv("QUEUE_MODE", "inline") or "inline").strip().lower(),
        redis_url=os.getenv("REDIS_URL"),
        triton_url=os.getenv("TRITON_URL"),
    )
