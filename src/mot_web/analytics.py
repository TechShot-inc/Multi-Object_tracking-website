from __future__ import annotations

import base64
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RoiBox:
    x: int
    y: int
    w: int
    h: int

    def contains_point(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_video_info(annotations: dict[str, Any]) -> dict[str, Any]:
    vi = annotations.get("video_info") or {}
    if isinstance(vi, dict) and vi:
        return vi

    legacy = (
        (annotations.get("analytics") or {}).get("video") if isinstance(annotations.get("analytics"), dict) else None
    ) or {}
    if isinstance(legacy, dict) and legacy:
        out: dict[str, Any] = {}
        if "width" in legacy:
            out["width"] = legacy.get("width")
        if "height" in legacy:
            out["height"] = legacy.get("height")
        if "fps" in legacy:
            out["fps"] = legacy.get("fps")
        if "output_fps" in legacy:
            out["output_fps"] = legacy.get("output_fps")
        return out

    return {}


def _extract_roi(params: Any) -> RoiBox | None:
    if not isinstance(params, dict):
        return None
    roi = params.get("roi")
    if not isinstance(roi, dict):
        return None
    x = _as_int(roi.get("x"), default=0)
    y = _as_int(roi.get("y"), default=0)
    w = _as_int(roi.get("width"), default=0)
    h = _as_int(roi.get("height"), default=0)
    if w <= 0 or h <= 0:
        return None
    return RoiBox(x=x, y=y, w=w, h=h)


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    x = _as_float(bbox[0], default=0.0)
    y = _as_float(bbox[1], default=0.0)
    w = _as_float(bbox[2], default=0.0)
    h = _as_float(bbox[3], default=0.0)
    return (x + w / 2.0, y + h / 2.0)


_DIR_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def _direction_label(dx: float, dy: float) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "stationary"
    # Note: image y increases downward; convert to math coords by flipping dy.
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
    idx = int((angle + 22.5) // 45.0) % 8
    return _DIR_8[idx]


def _grid_to_png_base64(grid: np.ndarray, out_w: int, out_h: int) -> str:
    if out_w <= 0 or out_h <= 0:
        return ""
    if grid.size == 0:
        return ""
    mx = float(grid.max())
    if not (mx > 0.0):
        return ""

    scaled = (grid / mx) * 255.0
    scaled_u8 = scaled.astype(np.uint8)
    img = cv2.resize(scaled_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    ok, buf = cv2.imencode(".png", img_color)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _frame_counts_from_annotations(annotations: dict[str, Any]) -> dict[int, int]:
    object_counts = annotations.get("object_counts")
    if isinstance(object_counts, dict) and object_counts:
        out: dict[int, int] = {}
        for k, v in object_counts.items():
            out[_as_int(k)] = _as_int(v)
        return out

    # Fallback: try frame_data / frames (list of detections per frame)
    frame_data = annotations.get("frame_data")
    if isinstance(frame_data, dict) and frame_data:
        out = {}
        for k, dets in frame_data.items():
            out[_as_int(k)] = len(dets) if isinstance(dets, list) else 0
        return out

    frames = annotations.get("frames")
    if isinstance(frames, dict) and frames:
        out = {}
        for k, dets in frames.items():
            out[_as_int(k)] = len(dets) if isinstance(dets, list) else 0
        return out

    # Last resort: count detections by scanning tracks.
    counts: dict[int, int] = {}
    tracks = annotations.get("tracks")
    if isinstance(tracks, list):
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            dets = tr.get("detections")
            if not isinstance(dets, list):
                continue
            for det in dets:
                if not isinstance(det, dict):
                    continue
                frame = det.get("frame")
                if frame is None:
                    continue
                f = _as_int(frame)
                counts[f] = counts.get(f, 0) + 1
    return counts


def compute_analytics_from_annotations(annotations: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (analytics_json, per_track_rows_for_csv)."""

    raw_tracks = annotations.get("tracks")
    tracks: list[dict[str, Any]] = (
        [t for t in raw_tracks if isinstance(t, dict)] if isinstance(raw_tracks, list) else []
    )
    raw_summary = annotations.get("summary")
    summary: dict[str, Any] = raw_summary if isinstance(raw_summary, dict) else {}
    raw_params = annotations.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    roi = _extract_roi(params)
    video_info = _extract_video_info(annotations)

    fps = _as_float(video_info.get("fps") or video_info.get("output_fps"), default=30.0)
    if not (fps > 0.0):
        fps = 30.0

    width = _as_int(video_info.get("width"), default=0)
    height = _as_int(video_info.get("height"), default=0)

    frame_counts = _frame_counts_from_annotations(annotations)
    frames_sorted = sorted(frame_counts)
    avg_objects = 0.0
    if frame_counts:
        avg_objects = sum(frame_counts.values()) / max(1, len(frame_counts))

    # Occupancy per second
    occ_per_sec: dict[int, list[int]] = {}
    for f in frames_sorted:
        sec = int(math.floor(f / fps))
        occ_per_sec.setdefault(sec, []).append(frame_counts.get(f, 0))
    occupancy_per_second = [
        {"t_s": sec, "avg_count": round(sum(v) / max(1, len(v)), 3), "max_count": max(v) if v else 0}
        for sec, v in sorted(occ_per_sec.items(), key=lambda kv: kv[0])
    ]

    # Track-level metrics
    per_track_rows: list[dict[str, Any]] = []
    track_durations: dict[str, int] = {}
    direction_hist: dict[str, int] = {k: 0 for k in _DIR_8}
    direction_hist["stationary"] = 0

    entry_per_sec: dict[int, int] = {}

    density_grid: np.ndarray | None = None
    vel_sum_grid: np.ndarray | None = None
    vel_cnt_grid: np.ndarray | None = None
    avg_speed_samples: list[float] = []  # px/frame

    grid_size_px = 20
    if width > 0 and height > 0:
        grid_w = max(1, int(math.ceil(width / grid_size_px)))
        grid_h = max(1, int(math.ceil(height / grid_size_px)))
        density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        vel_sum_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        vel_cnt_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    def clamp_grid(cx: float, cy: float) -> tuple[int, int] | None:
        if density_grid is None:
            return None
        gx = int(cx // grid_size_px)
        gy = int(cy // grid_size_px)
        gx = max(0, min(gx, density_grid.shape[1] - 1))
        gy = max(0, min(gy, density_grid.shape[0] - 1))
        return gx, gy

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        track_id = tr.get("id")
        tid = _as_int(track_id, default=0)
        dets = tr.get("detections")
        if not isinstance(dets, list) or not dets:
            track_durations[str(tid)] = 0
            continue

        dets_sorted = [
            d for d in dets if isinstance(d, dict) and d.get("frame") is not None and d.get("bbox") is not None
        ]
        dets_sorted.sort(key=lambda d: _as_int(d.get("frame")))
        if not dets_sorted:
            track_durations[str(tid)] = 0
            continue

        frames = [_as_int(d.get("frame")) for d in dets_sorted]
        start_f, end_f = frames[0], frames[-1]
        span_frames = max(1, end_f - start_f + 1)
        duration_frames = len(frames)
        track_durations[str(tid)] = duration_frames

        gaps = 0
        missing = 0
        for prev, cur in zip(frames, frames[1:]):
            if cur - prev > 1:
                gaps += 1
                missing += cur - prev - 1
        lost_rate = (missing / span_frames) if span_frames > 0 else 0.0

        centers: list[tuple[float, float]] = []
        for d in dets_sorted:
            c = _bbox_center(d.get("bbox"))
            if c is not None:
                centers.append(c)

        direction = "stationary"
        if len(centers) >= 2:
            dx = centers[-1][0] - centers[0][0]
            dy = centers[-1][1] - centers[0][1]
            direction = _direction_label(dx, dy)
        direction_hist[direction] = direction_hist.get(direction, 0) + 1

        entry_sec = int(math.floor(start_f / fps))
        entry_per_sec[entry_sec] = entry_per_sec.get(entry_sec, 0) + 1

        # Heatmap accumulation
        if density_grid is not None:
            assert vel_sum_grid is not None
            assert vel_cnt_grid is not None
            for cx, cy in centers:
                g = clamp_grid(cx, cy)
                if g is None:
                    continue
                gx, gy = g
                density_grid[gy, gx] += 1.0

            for (f0, c0), (f1, c1) in zip(zip(frames, centers), zip(frames[1:], centers[1:])):
                df = max(1, f1 - f0)
                sp = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / df  # px/frame
                avg_speed_samples.append(sp)
                g = clamp_grid(c1[0], c1[1])
                if g is None:
                    continue
                gx, gy = g
                vel_sum_grid[gy, gx] += float(sp)
                vel_cnt_grid[gy, gx] += 1.0

        dwell_seconds = duration_frames / fps
        avg_speed = 0.0
        max_speed = 0.0
        if len(centers) >= 2:
            speeds = []
            for (f0, c0), (f1, c1) in zip(zip(frames, centers), zip(frames[1:], centers[1:])):
                df = max(1, f1 - f0)
                sp = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / df
                speeds.append(sp)
            if speeds:
                avg_speed = float(sum(speeds) / len(speeds))
                max_speed = float(max(speeds))

        per_track_rows.append(
            {
                "track_id": tid,
                "start_frame": start_f,
                "end_frame": end_f,
                "duration_frames": duration_frames,
                "span_frames": span_frames,
                "gaps": gaps,
                "lost_rate": round(lost_rate, 6),
                "dwell_seconds": round(dwell_seconds, 6),
                "direction": direction,
                "avg_speed_px_per_frame": round(avg_speed, 6),
                "max_speed_px_per_frame": round(max_speed, 6),
            }
        )

    # ROI occupancy (only if ROI specified and we have per-detection data)
    occupancy_roi_per_second: list[dict[str, Any]] = []
    if roi is not None:
        inside_counts: dict[int, int] = {}
        outside_counts: dict[int, int] = {}
        # Prefer frame_data/frames for fast per-frame counts; otherwise scan tracks.
        frame_dets = annotations.get("frame_data")
        if not isinstance(frame_dets, dict) or not frame_dets:
            frame_dets = annotations.get("frames") if isinstance(annotations.get("frames"), dict) else {}

        if isinstance(frame_dets, dict) and frame_dets:
            for k, dets in frame_dets.items():
                f = _as_int(k)
                ins = 0
                out = 0
                if isinstance(dets, list):
                    for det in dets:
                        if not isinstance(det, dict):
                            continue
                        c = _bbox_center(det.get("bbox"))
                        if c is None:
                            continue
                        if roi.contains_point(c[0], c[1]):
                            ins += 1
                        else:
                            out += 1
                inside_counts[f] = ins
                outside_counts[f] = out
        else:
            per_frame: dict[int, list[tuple[float, float]]] = {}
            for tr in tracks:
                if not isinstance(tr, dict):
                    continue
                dets = tr.get("detections")
                if not isinstance(dets, list):
                    continue
                for det in dets:
                    if not isinstance(det, dict):
                        continue
                    f = det.get("frame")
                    if f is None:
                        continue
                    c = _bbox_center(det.get("bbox"))
                    if c is None:
                        continue
                    per_frame.setdefault(_as_int(f), []).append(c)
            for f, cs in per_frame.items():
                ins = sum(1 for (cx, cy) in cs if roi.contains_point(cx, cy))
                inside_counts[f] = ins
                outside_counts[f] = max(0, len(cs) - ins)

        by_sec: dict[int, dict[str, list[int]]] = {}
        all_frames = sorted(set(inside_counts) | set(outside_counts))
        for f in all_frames:
            sec = int(math.floor(f / fps))
            d = by_sec.setdefault(sec, {"inside": [], "outside": []})
            d["inside"].append(inside_counts.get(f, 0))
            d["outside"].append(outside_counts.get(f, 0))
        occupancy_roi_per_second = [
            {
                "t_s": sec,
                "inside_avg": round(sum(v["inside"]) / max(1, len(v["inside"])), 3),
                "outside_avg": round(sum(v["outside"]) / max(1, len(v["outside"])), 3),
            }
            for sec, v in sorted(by_sec.items(), key=lambda kv: kv[0])
        ]

    # Dwell time distribution
    dwell_seconds_all = [r["dwell_seconds"] for r in per_track_rows if r.get("dwell_seconds") is not None]
    dwell_seconds_all_sorted = sorted(dwell_seconds_all)

    def pct(p: float) -> float:
        if not dwell_seconds_all_sorted:
            return 0.0
        idx = int(round((p / 100.0) * (len(dwell_seconds_all_sorted) - 1)))
        idx = max(0, min(idx, len(dwell_seconds_all_sorted) - 1))
        return float(dwell_seconds_all_sorted[idx])

    dwell_stats = {
        "count": len(dwell_seconds_all_sorted),
        "mean_s": round(float(sum(dwell_seconds_all_sorted) / len(dwell_seconds_all_sorted)), 6)
        if dwell_seconds_all_sorted
        else 0.0,
        "p50_s": round(pct(50.0), 6),
        "p90_s": round(pct(90.0), 6),
        "max_s": round(float(dwell_seconds_all_sorted[-1]), 6) if dwell_seconds_all_sorted else 0.0,
    }

    dwell_bins = [1.0, 2.0, 5.0, 10.0, 30.0]
    dwell_hist: dict[str, int] = {"0-1": 0, "1-2": 0, "2-5": 0, "5-10": 0, "10-30": 0, "30+": 0}
    for v in dwell_seconds_all_sorted:
        if v < dwell_bins[0]:
            dwell_hist["0-1"] += 1
        elif v < dwell_bins[1]:
            dwell_hist["1-2"] += 1
        elif v < dwell_bins[2]:
            dwell_hist["2-5"] += 1
        elif v < dwell_bins[3]:
            dwell_hist["5-10"] += 1
        elif v < dwell_bins[4]:
            dwell_hist["10-30"] += 1
        else:
            dwell_hist["30+"] += 1

    # Track quality summary
    lost_rates = [float(r.get("lost_rate", 0.0)) for r in per_track_rows]
    gaps_all = [_as_int(r.get("gaps"), default=0) for r in per_track_rows]
    avg_lost_rate = (sum(lost_rates) / len(lost_rates)) if lost_rates else 0.0
    avg_gaps = (sum(gaps_all) / len(gaps_all)) if gaps_all else 0.0

    # Heatmaps + average velocity
    heatmap_b64 = ""
    velocity_heatmap_b64 = ""
    avg_velocity_px_per_frame = 0.0
    if avg_speed_samples:
        avg_velocity_px_per_frame = float(sum(avg_speed_samples) / len(avg_speed_samples))

    if density_grid is not None and width > 0 and height > 0:
        heatmap_b64 = _grid_to_png_base64(density_grid, out_w=width, out_h=height)
        if vel_sum_grid is not None and vel_cnt_grid is not None:
            vel_avg = np.divide(vel_sum_grid, np.maximum(vel_cnt_grid, 1.0))
            velocity_heatmap_b64 = _grid_to_png_base64(vel_avg, out_w=width, out_h=height)

    # Top IDs (longest tracks)
    top_ids = [_as_int(k) for k, _v in sorted(track_durations.items(), key=lambda kv: kv[1], reverse=True)[:5]]

    peak_occ = max(frame_counts.values()) if frame_counts else 0
    peak_throughput = max(entry_per_sec.values()) if entry_per_sec else 0

    analytics: dict[str, Any] = {
        "generated_at_unix": int(time.time()),
        "video_info": video_info,
        "total_tracks": summary.get("total_tracks", len(tracks)),
        "avg_objects_per_frame": summary.get("avg_objects_per_frame", round(avg_objects, 3)),
        "top_ids": top_ids,
        "object_counts": {str(k): int(v) for k, v in sorted(frame_counts.items(), key=lambda kv: kv[0])},
        "track_durations": {str(k): int(v) for k, v in sorted(track_durations.items(), key=lambda kv: _as_int(kv[0]))},
        "avg_velocity": round(avg_velocity_px_per_frame, 6),
        "heatmap": heatmap_b64,
        "velocity_heatmap": velocity_heatmap_b64,
        "occupancy": {
            "peak": int(peak_occ),
            "per_second": occupancy_per_second,
            "roi_per_second": occupancy_roi_per_second,
        },
        "dwell_time": {
            "stats": dwell_stats,
            "histogram": dwell_hist,
        },
        "flow": {
            "direction_histogram": direction_hist,
            "peak_throughput_tracks_per_second": int(peak_throughput),
        },
        "track_quality": {
            "avg_lifetime_s": round(dwell_stats.get("mean_s", 0.0), 6),
            "avg_fragmentation_gaps": round(float(avg_gaps), 6),
            "avg_lost_rate": round(float(avg_lost_rate), 6),
        },
    }

    return analytics, per_track_rows


def write_analytics_files(job_dir: Path, annotations: dict[str, Any]) -> dict[str, Any]:
    """Compute and persist analytics.json + analytics.csv; returns analytics JSON dict."""
    job_dir.mkdir(parents=True, exist_ok=True)

    analytics, rows = compute_analytics_from_annotations(annotations)

    try:
        raw_top_ids = analytics.get("top_ids")
        top_ids_list: list[Any] = raw_top_ids if isinstance(raw_top_ids, list) else []
        analytics["top_tracks"] = _build_top_tracks_with_thumbnails(
            job_dir=job_dir,
            annotations=annotations,
            top_ids=top_ids_list,
            fps=_as_float((analytics.get("video_info") or {}).get("fps"), default=30.0),
        )
    except Exception:
        # Thumbnails are best-effort; never fail analytics export.
        analytics.setdefault("top_tracks", [])
    analytics["artifacts"] = {
        "analytics_json": "analytics.json",
        "analytics_csv": "analytics.csv",
    }

    json_path = job_dir / "analytics.json"
    csv_path = job_dir / "analytics.csv"

    tmp_json = job_dir / ".analytics.json.tmp"
    tmp_csv = job_dir / ".analytics.csv.tmp"

    tmp_json.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
    tmp_json.replace(json_path)

    fieldnames = [
        "track_id",
        "start_frame",
        "end_frame",
        "duration_frames",
        "span_frames",
        "gaps",
        "lost_rate",
        "dwell_seconds",
        "direction",
        "avg_speed_px_per_frame",
        "max_speed_px_per_frame",
    ]
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp_csv.replace(csv_path)

    return analytics


def _resolve_video_path(job_dir: Path, annotations: dict[str, Any]) -> Path | None:
    artifacts = annotations.get("artifacts")
    if isinstance(artifacts, dict):
        p = artifacts.get("annotated_video")
        if isinstance(p, str) and p:
            path = Path(p)
            if not path.is_absolute():
                path = job_dir / p
            if path.exists():
                return path

    out_name = annotations.get("output_video")
    if isinstance(out_name, str) and out_name:
        p = job_dir / out_name
        if p.exists():
            return p

    candidates = [
        job_dir / "annotated_video.mp4",
        job_dir / "output.mp4",
        job_dir / "annotated.mp4",
        job_dir / "result.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _encode_jpeg_base64(img_bgr: np.ndarray, max_size: int = 160, quality: int = 80) -> str:
    if img_bgr.size == 0:
        return ""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return ""

    scale = min(1.0, max_size / float(max(h, w)))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_track_detection_for_id(annotations: dict[str, Any], track_id: int) -> tuple[int | None, list[int] | None]:
    """Return (frame_index, bbox[x,y,w,h]) for a representative detection of a track."""
    raw_tracks = annotations.get("tracks")
    if not isinstance(raw_tracks, list):
        return None, None
    for tr in raw_tracks:
        if not isinstance(tr, dict):
            continue
        if _as_int(tr.get("id"), default=-1) != track_id:
            continue
        dets = tr.get("detections")
        if not isinstance(dets, list) or not dets:
            return None, None
        dets_sorted = [d for d in dets if isinstance(d, dict) and d.get("bbox") is not None]
        dets_sorted.sort(key=lambda d: _as_int(d.get("frame"), default=0))
        if not dets_sorted:
            return None, None

        mid = dets_sorted[len(dets_sorted) // 2]
        bbox = mid.get("bbox")
        frame = mid.get("frame")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None, None
        bb = [
            _as_int(bbox[0], default=0),
            _as_int(bbox[1], default=0),
            _as_int(bbox[2], default=0),
            _as_int(bbox[3], default=0),
        ]
        return (_as_int(frame, default=0) if frame is not None else None), bb

    return None, None


def _build_top_tracks_with_thumbnails(
    job_dir: Path,
    annotations: dict[str, Any],
    top_ids: list[Any],
    fps: float,
) -> list[dict[str, Any]]:
    if not (fps > 0.0):
        fps = 30.0

    # Durations
    durations: dict[int, int] = {}
    raw_tracks = annotations.get("tracks")
    if isinstance(raw_tracks, list):
        for tr in raw_tracks:
            if not isinstance(tr, dict):
                continue
            tid = _as_int(tr.get("id"), default=-1)
            if tid < 0:
                continue
            dets = tr.get("detections")
            if isinstance(dets, list):
                durations[tid] = len(dets)
            else:
                durations[tid] = _as_int(tr.get("duration_frames"), default=0)

    ids = [_as_int(x, default=-1) for x in top_ids]
    ids = [i for i in ids if i >= 0]

    # Open video best-effort.
    video_path = _resolve_video_path(job_dir, annotations)
    cap = cv2.VideoCapture(str(video_path)) if video_path is not None else None

    out: list[dict[str, Any]] = []
    try:
        for tid in ids:
            duration_frames = int(durations.get(tid, 0))
            dwell_seconds = (duration_frames / fps) if fps > 0 else 0.0

            thumb_b64 = ""
            if cap is not None and cap.isOpened():
                frame_idx, bbox = _extract_track_detection_for_id(annotations, track_id=tid)
                if frame_idx is not None and bbox is not None:
                    # Track pipelines differ: some are 0-based frames, others 1-based.
                    tried = [int(frame_idx), int(frame_idx) - 1]
                    frame_img = None
                    for pos in tried:
                        if pos < 0:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
                        ok, frame_img = cap.read()
                        if ok and frame_img is not None:
                            break
                        frame_img = None

                    if frame_img is not None:
                        x, y, w, h = bbox
                        x = max(0, x)
                        y = max(0, y)
                        w = max(1, w)
                        h = max(1, h)
                        H, W = frame_img.shape[:2]
                        x2 = max(0, min(W, x + w))
                        y2 = max(0, min(H, y + h))
                        x = max(0, min(W - 1, x))
                        y = max(0, min(H - 1, y))
                        if x2 > x and y2 > y:
                            crop = frame_img[y:y2, x:x2]
                            thumb_b64 = _encode_jpeg_base64(crop)

            out.append(
                {
                    "id": tid,
                    "duration_frames": duration_frames,
                    "dwell_seconds": round(float(dwell_seconds), 3),
                    "thumbnail_jpeg": thumb_b64,
                }
            )
    finally:
        if cap is not None:
            cap.release()

    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
