from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi import File, UploadFile

from mot_web.services.tracking_service import TrackingService
from mot_web.tracking.pipeline import run_video
from mot_web.queue import enqueue_video_job, get_job_status


router = APIRouter(prefix="/video")

# Keep this tight; expand later if needed.
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _secure_filename(name: str) -> str:
    name = (name or "").strip().replace("\\", "/")
    name = name.split("/")[-1]
    name = _FILENAME_SAFE_RE.sub("_", name)
    name = name.strip("._")
    return name or "upload"


def _get_service(request: Request) -> TrackingService:
    settings = request.app.state.settings
    svc: TrackingService | None = getattr(request.app.state, "tracking_service", None)
    if svc is None:
        svc = TrackingService(settings.upload_dir, settings.results_dir)
        request.app.state.tracking_service = svc
    return svc


@router.get("/", response_class=HTMLResponse)
def video_page(request: Request):
    return request.app.state.templates.TemplateResponse("video.html", {"request": request})


@router.post("/upload")
async def upload_video(request: Request, file: UploadFile | None = File(default=None)):
    """
    Multipart form-data:
      - file: uploaded video
    Returns: {job_id, filename}
    """
    if file is None or not file.filename:
        return JSONResponse(status_code=400, content={"error": "missing file"})

    filename = _secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse(
            status_code=400,
            content={"error": f"unsupported file type: {ext}", "allowed": sorted(ALLOWED_EXT)},
        )

    svc = _get_service(request)
    job = svc.create_job(filename)

    # Save upload
    job.upload_path.parent.mkdir(parents=True, exist_ok=True)
    with job.upload_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    return {"job_id": job.job_id, "filename": job.filename}


@router.get("/status/{job_id}")
def job_status(request: Request, job_id: str):
    svc = _get_service(request)
    settings = request.app.state.settings
    if settings.queue_mode == "rq" and settings.redis_url:
        redis_status = get_job_status(job_id)
        if redis_status is not None:
            # Merge Redis status over file-based status for consistency
            base = svc.get_status(job_id)
            return {**base, **redis_status}
    return svc.get_status(job_id)


@router.post("/run/{job_id}")
def run_job(request: Request, job_id: str, params: dict[str, Any] = Body(default_factory=dict)):
    svc = _get_service(request)
    st = svc.get_status(job_id)
    if st.get("state") == "not_found":
        return JSONResponse(status_code=404, content={"error": "job not found"})

    settings = request.app.state.settings
    job_dir = settings.results_dir / job_id

    try:
        if settings.queue_mode == "rq" and settings.redis_url:
            rq_job_id = enqueue_video_job(job_id, params=params)
            svc.set_status(job_id, "queued", "queued")
            return {"job_id": job_id, "state": "queued", "rq_job_id": rq_job_id}

        svc.set_status(job_id, "running", "processing started")
        run_video(input_path=settings.upload_dir / st["filename"], output_dir=job_dir, params=params)
        svc.set_status(job_id, "done", "processing complete")
        return {"job_id": job_id, "state": "done"}
    except Exception as e:
        svc.set_status(job_id, "failed", str(e))
        return JSONResponse(status_code=500, content={"job_id": job_id, "state": "failed", "error": str(e)})


@router.get("/results/{job_id}/annotations")
def get_annotations(request: Request, job_id: str, download: int | None = Query(default=None)):
    """Return the annotations.json file for a completed job."""
    settings = request.app.state.settings
    annotations_path = settings.results_dir / job_id / "annotations.json"

    if not annotations_path.exists():
        return JSONResponse(status_code=404, content={"error": "annotations not found"})

    filename = f"{job_id}_annotations.json" if download == 1 else None
    return FileResponse(
        path=str(annotations_path),
        media_type="application/json",
        filename=filename,
    )


@router.get("/results/{job_id}/video")
def get_result_video(request: Request, job_id: str, download: int | None = Query(default=None)):
    """Return the processed video for a completed job."""
    settings = request.app.state.settings
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
        svc = _get_service(request)
        st = svc.get_status(job_id)
        if st.get("state") != "not_found" and st.get("filename"):
            original_path = settings.upload_dir / st["filename"]
            if original_path.exists():
                video_path = original_path

    if video_path is None:
        return JSONResponse(status_code=404, content={"error": "video not found"})

    filename = video_path.name if download == 1 else None
    return FileResponse(path=str(video_path), media_type="video/mp4", filename=filename)


@router.get("/results/{job_id}/analytics")
def get_analytics(request: Request, job_id: str):
    """Return analytics data for a completed job."""
    settings = request.app.state.settings
    job_dir = settings.results_dir / job_id

    # Check if analytics file exists
    analytics_path = job_dir / "analytics.json"
    if analytics_path.exists():
        return FileResponse(path=str(analytics_path), media_type="application/json")

    # Generate analytics from annotations if available
    annotations_path = job_dir / "annotations.json"
    if not annotations_path.exists():
        return JSONResponse(status_code=404, content={"error": "analytics not available"})

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
        return analytics
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"failed to generate analytics: {str(e)}"})
