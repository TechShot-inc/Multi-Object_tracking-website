from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoJob:
    job_id: str
    filename: str
    upload_path: Path
    job_dir: Path


class TrackingService:
    """
    Owns:
      - where uploads are stored
      - where per-job results are stored
      - basic job metadata/status persistence
    """

    def __init__(self, upload_dir: Path, results_dir: Path) -> None:
        self.upload_dir = Path(upload_dir)
        self.results_dir = Path(results_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, filename: str) -> VideoJob:
        job_id = uuid.uuid4().hex
        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = VideoJob(
            job_id=job_id,
            filename=filename,
            upload_path=self.upload_dir / filename,
            job_dir=job_dir,
        )

        self._write_json(
            job_dir / "meta.json",
            {
                "job_id": job_id,
                "filename": filename,
                "created_at_unix": int(time.time()),
            },
        )
        self._write_json(
            job_dir / "status.json",
            {
                "state": "created",  # created -> running -> done/failed
                "message": "",
                "updated_at_unix": int(time.time()),
            },
        )

        return job

    def set_status(self, job_id: str, state: str, message: str = "") -> None:
        job_dir = self.results_dir / job_id
        self._write_json(
            job_dir / "status.json",
            {
                "state": state,
                "message": message,
                "updated_at_unix": int(time.time()),
            },
        )

    def get_status(self, job_id: str) -> dict[str, Any]:
        job_dir = self.results_dir / job_id
        status_path = job_dir / "status.json"
        meta_path = job_dir / "meta.json"
        if not job_dir.exists():
            return {"state": "not_found", "message": "job does not exist"}

        status = self._read_json(status_path, default={"state": "unknown", "message": ""})
        meta = self._read_json(meta_path, default={})
        return {"job_id": job_id, **meta, **status}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
