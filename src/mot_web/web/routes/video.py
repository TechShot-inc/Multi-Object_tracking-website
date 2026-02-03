from __future__ import annotations

from io import BytesIO
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from mot_web.services.tracking_service import TrackingService
from mot_web.tracking.pipeline import run_video


bp = Blueprint("video", __name__, url_prefix="/video")

# Keep this tight; expand later if needed.
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def _get_service() -> TrackingService:
    settings = current_app.config["SETTINGS"]
    # cache per app instance
    svc: TrackingService | None = current_app.extensions.get("tracking_service")
    if svc is None:
        svc = TrackingService(settings.upload_dir, settings.results_dir)
        current_app.extensions["tracking_service"] = svc
    return svc


@bp.get("/")
def video_page():
    return render_template("video.html")


@bp.post("/upload")
def upload_video():
    """
    Multipart form-data:
      - file: uploaded video
    Returns: {job_id, filename}
    """
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="missing file"), 400

    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"unsupported file type: {ext}", allowed=sorted(ALLOWED_EXT)), 400

    svc = _get_service()
    job = svc.create_job(filename)

    # Save upload
    f.save(job.upload_path)

    return jsonify(job_id=job.job_id, filename=job.filename)


@bp.get("/status/<job_id>")
def job_status(job_id: str):
    svc = _get_service()
    return jsonify(svc.get_status(job_id))


@bp.post("/run/<job_id>")
def run_job(job_id: str):
    svc = _get_service()
    st = svc.get_status(job_id)
    if st.get("state") == "not_found":
        return jsonify(error="job not found"), 404

    settings = current_app.config["SETTINGS"]
    job_dir = settings.results_dir / job_id

    # You can accept JSON params from frontend later.
    params = request.get_json(silent=True) or {}

    try:
        svc.set_status(job_id, "running", "processing started")
        run_video(input_path=settings.upload_dir / st["filename"], output_dir=job_dir, params=params)
        svc.set_status(job_id, "done", "processing complete")
        return jsonify(job_id=job_id, state="done")
    except Exception as e:
        svc.set_status(job_id, "failed", str(e))
        return jsonify(job_id=job_id, state="failed", error=str(e)), 500

