from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoostTrackArtifacts:
    mot_txt: Path
    annotated_video: Path
    annotations_json: Path


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()


def _models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", _project_root() / "models")).resolve()


def _default_model_paths() -> tuple[Path, Path, Path]:
    models = _models_dir()
    yolo11 = Path(os.getenv("YOLO11_MODEL_PATH", models / "yolo11x.pt")).resolve()
    yolo12 = Path(os.getenv("YOLO12_MODEL_PATH", models / "yolo12x.pt")).resolve()
    reid = Path(os.getenv("REID_MODEL_PATH", models / "osnet_ain_ms_m_c.pth.tar")).resolve()
    return yolo11, yolo12, reid


def _parse_roi(params: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    roi = params.get("roi")
    if roi is None:
        return None

    try:
        if isinstance(roi, dict):
            x = float(roi["x"])
            y = float(roi["y"])
            w = float(roi["width"])
            h = float(roi["height"])
        elif isinstance(roi, (list, tuple)) and len(roi) == 4:
            x, y, w, h = [float(v) for v in roi]
        else:
            return None

        # normalized [0..1] => pixels
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
            x *= width
            y *= height
            w *= width
            h *= height

        xi = max(0, min(int(x), width - 1))
        yi = max(0, min(int(y), height - 1))
        wi = max(1, min(int(w), width - xi))
        hi = max(1, min(int(h), height - yi))
        return xi, yi, wi, hi
    except Exception:
        return None


def _extract_frames(input_path: Path, frames_dir: Path, speed: int) -> tuple[float, int, int, float]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % speed != 0:
            continue
        written += 1
        out_path = frames_dir / f"{written:06d}.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()
    output_fps = fps / max(1, speed)
    return fps, width, height, output_fps


def _run_customboosttrack(
    frames_dir: Path,
    result_root: Path,
    roi_box: tuple[int, int, int, int] | None,
    conf_threshold: float,
    output_fps: float,
) -> Path:
    yolo11, yolo12, reid = _default_model_paths()
    if not yolo11.exists() or not yolo12.exists() or not reid.exists():
        missing = [str(p) for p in (yolo11, yolo12, reid) if not p.exists()]
        raise FileNotFoundError(f"Missing model weights: {missing}")

    custom_dir = (_project_root() / "CustomBoostTrack").resolve()
    script = custom_dir / "run_with_ensembler.py"
    if not script.exists():
        raise FileNotFoundError(f"CustomBoostTrack script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        "tarsh",
        "--exp_name",
        "web",
        "--result_folder",
        str(result_root),
        "--dataset_path",
        str(frames_dir),
        "--model1_path",
        str(yolo11),
        "--model1_weight",
        "0.7",
        "--model2_path",
        str(yolo12),
        "--model2_weight",
        "0.3",
        "--reid_path",
        str(reid),
        "--frame_rate",
        str(max(1, int(round(output_fps)))),
        "--conf",
        str(conf_threshold),
    ]

    if roi_box is not None:
        x, y, w, h = roi_box
        cmd += ["--roi", f"{x},{y},{w},{h}"]

    logger.info("Running CustomBoostTrack: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(custom_dir), check=True)

    base = result_root / "tarsh-val"
    candidates = [
        base / "web_post_gbi" / "data" / "test.txt",
        base / "web_post" / "data" / "test.txt",
        base / "web" / "data" / "test.txt",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"Could not find MOT results file. Looked for: {candidates}")


def _read_mot_results(mot_txt: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """Return (tracks, frames) in a JSON-friendly structure."""
    tracks: dict[int, list[dict[str, Any]]] = {}
    frames: dict[int, list[dict[str, Any]]] = {}

    for raw_line in mot_txt.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue

        try:
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            if track_id < 0:
                continue
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
        except ValueError:
            continue

        det = {
            "id": track_id,
            "bbox": [max(0, int(round(x))), max(0, int(round(y))), max(0, int(round(w))), max(0, int(round(h)))],
            "confidence": round(conf, 3),
        }

        frames.setdefault(frame_id, []).append(det)
        tracks.setdefault(track_id, []).append({"frame": frame_id, **det})

    return tracks, frames


def _write_annotated_video(
    frames_dir: Path,
    mot_frames: dict[int, list[dict[str, Any]]],
    output_path: Path,
    width: int,
    height: int,
    output_fps: float,
    roi_box: tuple[int, int, int, int] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    if not out.isOpened():
        raise RuntimeError("Failed to create output video writer")

    id_colors: dict[int, tuple[int, int, int]] = {}

    frame_files = sorted(frames_dir.glob("*.jpg"))
    for idx, frame_path in enumerate(frame_files, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        dets = mot_frames.get(idx, [])
        for det in dets:
            tid = int(det["id"])
            x, y, w, h = det["bbox"]
            x2, y2 = x + w, y + h

            if tid not in id_colors:
                id_colors[tid] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
            color = id_colors[tid]
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            label = f"ID {tid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), color, -1)
            cv2.putText(
                frame,
                label,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        if roi_box is not None:
            rx, ry, rw, rh = roi_box
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (45, 212, 191), 2)
            cv2.putText(
                frame,
                "ROI",
                (rx + 5, ry + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (45, 212, 191),
                2,
            )

        out.write(frame)

    out.release()


def run_video_boosttrack(input_path: Path, output_dir: Path, params: dict[str, Any]) -> BoostTrackArtifacts:
    """Run the CustomBoostTrack ensemble pipeline and emit web artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep runs idempotent
    frames_dir = output_dir / "frames"
    result_root = output_dir / "boosttrack_results"
    for p in (frames_dir, result_root):
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    speed = max(1, int(params.get("speed", 1)))
    conf_threshold = float(params.get("conf_threshold", params.get("confidence_threshold", 0.25)))

    fps, width, height, output_fps = _extract_frames(input_path, frames_dir, speed)
    roi_box = _parse_roi(params, width=width, height=height)

    start = time.time()
    mot_txt = _run_customboosttrack(
        frames_dir=frames_dir,
        result_root=result_root,
        roi_box=roi_box,
        conf_threshold=conf_threshold,
        output_fps=output_fps,
    )
    logger.info("CustomBoostTrack finished in %.2fs", time.time() - start)

    tracks, frames = _read_mot_results(mot_txt)

    annotated_video = output_dir / "annotated_video.mp4"
    _write_annotated_video(
        frames_dir=frames_dir,
        mot_frames=frames,
        output_path=annotated_video,
        width=width,
        height=height,
        output_fps=output_fps,
        roi_box=roi_box,
    )

    annotations_json = output_dir / "annotations.json"
    annotations = {
        "source_video": str(input_path),
        "created_at_unix": int(time.time()),
        "params": params,
        "tracks": [
            {"id": tid, "detections": dets}
            for tid, dets in sorted(tracks.items(), key=lambda kv: kv[0])
        ],
        "frames": {str(k): v for k, v in sorted(frames.items(), key=lambda kv: kv[0])},
        "analytics": {
            "video": {"width": width, "height": height, "fps": fps, "output_fps": output_fps},
            "num_tracks": len(tracks),
            "num_frames": len(frames),
        },
        "artifacts": {
            "mot_results": str(mot_txt),
            "annotated_video": str(annotated_video),
        },
    }
    annotations_json.write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    return BoostTrackArtifacts(
        mot_txt=mot_txt,
        annotated_video=annotated_video,
        annotations_json=annotations_json,
    )
