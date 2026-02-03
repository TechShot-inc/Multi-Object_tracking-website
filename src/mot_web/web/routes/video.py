from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_file
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


@bp.get("/results/<job_id>/annotations")
def get_annotations(job_id: str):
    """Return the annotations.json file for a completed job."""
    settings = current_app.config["SETTINGS"]
    annotations_path = settings.results_dir / job_id / "annotations.json"
    
    if not annotations_path.exists():
        return jsonify(error="annotations not found"), 404
    
    return send_file(
        annotations_path,
        mimetype="application/json",
        as_attachment=request.args.get("download") == "1",
        download_name=f"{job_id}_annotations.json",
    )


@bp.get("/results/<job_id>/video")
def get_result_video(job_id: str):
    """Return the processed video for a completed job."""
    settings = current_app.config["SETTINGS"]
    job_dir = settings.results_dir / job_id
    
    # Look for output video (could be various names depending on pipeline)
    video_candidates = ["annotated_video.mp4", "output.mp4", "annotated.mp4", "result.mp4"]
    video_path = None
    
    for candidate in video_candidates:
        path = job_dir / candidate
        if path.exists():
            video_path = path
            break
    
    # Fallback: return the original uploaded video if no processed video exists
    if video_path is None:
        svc = _get_service()
        st = svc.get_status(job_id)
        if st.get("state") != "not_found" and st.get("filename"):
            original_path = settings.upload_dir / st["filename"]
            if original_path.exists():
                video_path = original_path
    
    if video_path is None:
        return jsonify(error="video not found"), 404
    
    return send_file(
        video_path,
        mimetype="video/mp4",
        as_attachment=request.args.get("download") == "1",
    )


@bp.get("/results/<job_id>/analytics")
def get_analytics(job_id: str):
    """Return analytics data for a completed job."""
    settings = current_app.config["SETTINGS"]
    job_dir = settings.results_dir / job_id
    
    # Check if analytics file exists
    analytics_path = job_dir / "analytics.json"
    if analytics_path.exists():
        return send_file(analytics_path, mimetype="application/json")
    
    # Generate analytics from annotations if available
    annotations_path = job_dir / "annotations.json"
    if not annotations_path.exists():
        return jsonify(error="analytics not available"), 404
    
    try:
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        
        # Extract analytics from annotations
        tracks = annotations.get("tracks", [])
        object_counts = annotations.get("object_counts", {})
        summary = annotations.get("summary", {})
        
        # Build track durations
        track_durations = {}
        for track in tracks:
            track_id = track.get("id", 0)
            duration = track.get("duration_frames", len(track.get("detections", [])))
            track_durations[str(track_id)] = duration
        
        analytics = {
            "total_tracks": summary.get("total_tracks", len(tracks)),
            "avg_objects_per_frame": summary.get("avg_objects_per_frame", 0),
            "top_ids": summary.get("top_track_ids", [])[:5],
            "object_counts": object_counts,
            "track_durations": track_durations,
            "video_info": annotations.get("video_info", {}),
        }
        return jsonify(analytics)
    except Exception as e:
        return jsonify(error=f"failed to generate analytics: {str(e)}"), 500

