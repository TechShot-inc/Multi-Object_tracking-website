from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


def run_video(input_path: Path, output_dir: Path, params: dict[str, Any]) -> None:
    """
    Run multi-object tracking on a video using YOLO detector with built-in tracking.

    Args:
        input_path: Path to the input video file
        output_dir: Directory to save output artifacts
        params: Processing parameters (roi, speed, etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional legacy/CustomBoostTrack integration (YOLO11+YOLO12 ensemble + OSNet ReID).
    # Kept behind an env switch so contract tests (fake videos, missing deps) still pass.
    pipeline_impl = (os.getenv("MOT_PIPELINE") or "yolo").strip().lower()
    if pipeline_impl in {"stub", "noop"}:
        _create_stub_output(input_path, output_dir, params, error=None)
        return
    if pipeline_impl in {"boosttrack", "customboosttrack", "ensemble"}:
        try:
            from mot_web.tracking.boosttrack_pipeline import run_video_boosttrack

            run_video_boosttrack(input_path=input_path, output_dir=output_dir, params=params)
            return
        except Exception as e:
            logger.exception("BoostTrack pipeline failed; falling back to YOLO/stub: %s", e)

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error(f"Failed to import ultralytics: {e}")
        _create_stub_output(input_path, output_dir, params, error=str(e))
        return

    # Extract parameters
    roi = params.get("roi")
    speed = max(1, int(params.get("speed", 1)))
    try:
        output_speed = float(params.get("output_speed", 1.0))
    except (TypeError, ValueError):
        output_speed = 1.0
    output_speed = max(0.1, min(10.0, output_speed))

    conf_threshold = params.get("conf_threshold", 0.25)

    # Check for GPU availability
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # Initialize YOLO model with tracking
    # Allow overriding the model path/name (useful in Docker where weights may be baked into the image)
    model_name = os.getenv("YOLO_MODEL_PATH") or "yolov8n.pt"
    if os.getenv("YOLO_MODEL_PATH"):
        p = Path(model_name)
        if not p.exists():
            logger.warning(
                "YOLO_MODEL_PATH is set but the file does not exist: %s (will let Ultralytics resolve/download if possible)",
                model_name,
            )
    logger.info(f"Loading YOLO model: {model_name}")

    try:
        model = YOLO(model_name)
        model.to(device)  # Move model to GPU if available
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}")
        _create_stub_output(input_path, output_dir, params, error=str(e))
        return

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {input_path}")
        _create_stub_output(input_path, output_dir, params, error="Failed to open video")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

    # Setup output video writer with H.264 codec for browser compatibility
    output_video_path = output_dir / "annotated_video.mp4"

    # Try H.264 codec first (browser compatible), fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec  # pyright: ignore[reportAttributeAccessIssue]
    # `speed` reduces processing load by skipping frames. `output_speed` only affects
    # the playback rate of the output video.
    output_fps = (fps / speed) * output_speed
    out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (width, height))

    if not out.isOpened():
        # Fallback to mp4v if avc1 not available
        logger.warning("H.264 codec not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
        out = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (width, height))

    if not out.isOpened():
        logger.error("Failed to create output video writer")
        cap.release()
        _create_stub_output(input_path, output_dir, params, error="Failed to create video writer")
        return

    # Process ROI
    roi_box = None
    if roi:
        try:
            if isinstance(roi, dict):
                roi_box = [int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])]
            elif isinstance(roi, (list, tuple)) and len(roi) == 4:
                roi_box = [int(v) for v in roi]
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid ROI format: {roi}, ignoring. Error: {e}")

    # Track ID to color mapping for consistent visualization
    id_colors: dict[int, tuple[int, int, int]] = {}

    # Storage for all track data
    all_tracks: dict[int, list[dict]] = {}  # track_id -> list of frame detections
    frame_data: dict[int, list[dict]] = {}  # frame_num -> list of detections
    object_counts: dict[int, int] = {}

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames based on speed setting
        if (frame_idx - 1) % speed != 0:
            continue

        processed_frames += 1

        # Run YOLO tracking (persist=True maintains track IDs across frames)
        try:
            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                classes=[0],  # Only detect persons (class 0)
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Detection failed on frame {frame_idx}: {e}")
            out.write(frame)
            continue

        # Process results
        frame_detections = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # Get track ID (may be None if tracking failed)
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1

                if track_id < 0:
                    continue

                # Filter by ROI if specified
                if roi_box is not None:
                    roi_x, roi_y, roi_w, roi_h = roi_box
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area <= 0:
                        continue
                    # Calculate intersection with ROI
                    xi1 = max(x1, roi_x)
                    yi1 = max(y1, roi_y)
                    xi2 = min(x2, roi_x + roi_w)
                    yi2 = min(y2, roi_y + roi_h)
                    if xi2 > xi1 and yi2 > yi1:
                        intersection = (xi2 - xi1) * (yi2 - yi1)
                        overlap = intersection / box_area
                        if overlap < 0.5:  # Less than 50% overlap with ROI
                            continue
                    else:
                        continue  # No intersection

                # Validate coordinates
                x1 = max(0, min(int(x1), width))
                y1 = max(0, min(int(y1), height))
                x2 = max(0, min(int(x2), width))
                y2 = max(0, min(int(y2), height))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Get or assign color for this track
                if track_id not in id_colors:
                    id_colors[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                color = id_colors[track_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"ID {track_id}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Store detection data
                det_info = {
                    "id": track_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                    "confidence": round(conf, 3),
                }
                frame_detections.append(det_info)

                # Add to track history
                if track_id not in all_tracks:
                    all_tracks[track_id] = []
                all_tracks[track_id].append(
                    {"frame": processed_frames, "bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": round(conf, 3)}
                )

        # Draw ROI if specified
        if roi_box is not None:
            rx, ry, rw, rh = roi_box
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (45, 212, 191), 2)
            cv2.putText(frame, "ROI", (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (45, 212, 191), 2)

        # Store frame data
        frame_data[processed_frames] = frame_detections
        object_counts[processed_frames] = len(frame_detections)

        # Write annotated frame
        out.write(frame)

        # Log progress periodically
        if processed_frames % 100 == 0:
            logger.info(f"Processed {processed_frames} frames ({frame_idx}/{total_frames})")

    # Release resources
    cap.release()
    out.release()

    # Convert video to browser-compatible format using ffmpeg if available
    _convert_video_for_browser(output_video_path)

    # Build analytics
    track_durations = {tid: len(dets) for tid, dets in all_tracks.items()}
    top_track_ids = sorted(track_durations.keys(), key=lambda t: track_durations[t], reverse=True)[:10]
    avg_objects = sum(object_counts.values()) / max(1, len(object_counts))

    # Create annotations output
    annotations = {
        "source_video": str(input_path),
        "created_at_unix": int(time.time()),
        "params": params,
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
        },
        "summary": {
            "total_tracks": len(all_tracks),
            "avg_objects_per_frame": round(avg_objects, 2),
            "top_track_ids": top_track_ids,
        },
        "tracks": [{"id": tid, "duration_frames": len(dets), "detections": dets} for tid, dets in all_tracks.items()],
        "frame_data": {str(k): v for k, v in frame_data.items()},
        "object_counts": {str(k): v for k, v in object_counts.items()},
        "output_video": str(output_video_path.name),
    }

    # Write annotations
    annotations_path = output_dir / "annotations.json"
    annotations_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    try:
        from mot_web.analytics import write_analytics_files

        write_analytics_files(job_dir=output_dir, annotations=annotations)
    except Exception:
        logger.exception("Failed to write analytics exports")

    logger.info(f"Pipeline complete: {processed_frames} frames, {len(all_tracks)} tracks")
    logger.info(f"Output video: {output_video_path}")
    logger.info(f"Annotations: {annotations_path}")


def _convert_video_for_browser(video_path: Path) -> None:
    """
    Convert video to browser-compatible H.264 format using ffmpeg if available.
    This ensures the video can be played in web browsers.
    """
    import subprocess
    import shutil

    # Check if ffmpeg is available (try imageio-ffmpeg bundled version first)
    ffmpeg_path = None
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"Using imageio-ffmpeg: {ffmpeg_path}")
    except ImportError:
        ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        logger.warning(
            "ffmpeg not found, video may not play in browser. Install ffmpeg or imageio-ffmpeg for better compatibility."
        )
        return

    temp_path = video_path.with_suffix(".temp.mp4")

    try:
        # Convert to H.264 with web-compatible settings
        cmd = [
            ffmpeg_path,
            "-i",
            str(video_path),
            "-c:v",
            "libx264",  # H.264 codec
            "-preset",
            "fast",  # Balance speed/quality
            "-crf",
            "23",  # Quality (lower = better, 23 is default)
            "-pix_fmt",
            "yuv420p",  # Pixel format for browser compatibility
            "-movflags",
            "+faststart",  # Enable streaming
            "-y",  # Overwrite output
            str(temp_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and temp_path.exists():
            # Replace original with converted
            video_path.unlink()
            temp_path.rename(video_path)
            logger.info("Video converted to browser-compatible H.264 format")
        else:
            logger.warning(f"ffmpeg conversion failed: {result.stderr}")
            if temp_path.exists():
                temp_path.unlink()
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg conversion timed out")
        if temp_path.exists():
            temp_path.unlink()
    except Exception as e:
        logger.warning(f"Failed to convert video: {e}")
        if temp_path.exists():
            temp_path.unlink()


def _create_stub_output(input_path: Path, output_dir: Path, params: dict[str, Any], error: str | None = "") -> None:
    """Create stub output when tracking pipeline cannot run."""
    annotations = {
        "source_video": str(input_path),
        "created_at_unix": int(time.time()),
        "params": params,
        "error": (error or "Tracking pipeline not available"),
        "tracks": [],
        "frame_data": {},
        "object_counts": {},
        "summary": {"total_tracks": 0, "avg_objects_per_frame": 0, "top_track_ids": []},
    }

    (output_dir / "annotations.json").write_text(
        json.dumps(annotations, indent=2),
        encoding="utf-8",
    )

    try:
        from mot_web.analytics import write_analytics_files

        write_analytics_files(job_dir=output_dir, annotations=annotations)
    except Exception:
        logger.exception("Failed to write analytics exports (stub)")
