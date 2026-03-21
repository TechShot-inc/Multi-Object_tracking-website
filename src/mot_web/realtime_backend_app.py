from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect


def _default_model_paths() -> dict[str, str]:
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    models_dir = os.getenv("MODELS_DIR", os.path.join(project_root, "models"))
    return {
        "yolo1": os.getenv("YOLO11_MODEL_PATH", os.path.join(models_dir, "yolo11x.pt")),
        "yolo2": os.getenv("YOLO12_MODEL_PATH", os.path.join(models_dir, "yolo12x.pt")),
        "reid": os.getenv("REID_MODEL_PATH", os.path.join(models_dir, "osnet_ain_ms_m_c.pth.tar")),
    }


def _load_tracker():
    from CustomBoostTrack.realtime_ensembling import RealTimeTracker

    mp = _default_model_paths()
    return RealTimeTracker(
        model1_path=mp["yolo1"],
        model2_path=mp["yolo2"],
        reid_path=mp["reid"],
        frame_rate=30,
    )


_http_tracker = None
_http_tracker_error: str | None = None
_http_tracker_lock = threading.Lock()
_http_frame_id = 0


def _get_http_tracker():
    global _http_tracker, _http_tracker_error
    if _http_tracker is not None:
        return _http_tracker
    with _http_tracker_lock:
        if _http_tracker is not None:
            return _http_tracker
        try:
            _http_tracker = _load_tracker()
            _http_tracker_error = None
        except Exception as e:
            _http_tracker = None
            _http_tracker_error = str(e)
        return _http_tracker


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _encode_jpeg(frame_bgr, *, quality: int | None = None) -> bytes:
    import cv2

    if quality is None:
        quality = _env_int("REALTIME_PREVIEW_JPEG_QUALITY", 60)

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    ok, buffer = cv2.imencode(".jpg", frame_bgr, params)
    if not ok:
        raise ValueError("failed to encode jpeg")
    return buffer.tobytes()


def _downscale_max_w(frame_bgr, max_w: int):
    import cv2

    if int(max_w) <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= int(max_w):
        return frame_bgr
    scale = float(max_w) / float(w)
    new_w = int(max_w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _decode_jpeg(jpeg_bytes: bytes):
    import cv2
    import numpy as np

    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("failed to decode jpeg")
    return decoded


def _draw_overlays(frame_bgr, detections: list[dict[str, Any]], roi: dict | None, line: dict | None):
    import cv2

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    if roi and isinstance(roi, dict):
        try:
            x1 = int(float(roi["x"]) * w)
            y1 = int(float(roi["y"]) * h)
            x2 = int((float(roi["x"]) + float(roi["width"])) * w)
            y2 = int((float(roi["y"]) + float(roi["height"])) * h)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "ROI", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

    if line and isinstance(line, dict) and "x" in line and "position" in line:
        try:
            line_x = int(float(line["x"]) * w)
            cv2.line(annotated, (line_x, 0), (line_x, h), (255, 0, 0), 2)
            side = "L" if line["position"] == "left" else "R"
            cv2.putText(
                annotated,
                f"Count Line ({side})",
                (line_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        except Exception:
            pass

    for det in detections:
        try:
            x1 = int(float(det["x"]))
            y1 = int(float(det["y"]))
            x2 = x1 + int(float(det["width"]))
            y2 = y1 + int(float(det["height"]))
            tid = int(det["id"])
        except Exception:
            continue
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated,
            f"ID {tid}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        annotated,
        "Tracking Active",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return annotated


def _compute_line_counts(
    detections: list[dict[str, Any]],
    line: dict[str, Any],
    width: int,
) -> dict[str, int]:
    """Compute current inside/outside occupancy relative to a vertical line.

    This is intentionally stateless so the UI updates immediately after the first
    line is set, and clearing the line resets counts to 0.
    """

    line_x = int(float(line.get("x", 0.5)) * width)
    inside_side = "left" if line.get("position") == "left" else "right"

    inside = 0
    outside = 0
    for det in detections:
        try:
            x = float(det["x"])
            w = float(det["width"])
        except Exception:
            continue
        cx = x + (w / 2.0)
        side = "left" if cx < line_x else "right"
        if side == inside_side:
            inside += 1
        else:
            outside += 1

    return {"inside": int(inside), "outside": int(outside)}


app = FastAPI(title="MOT Realtime Backend")


@app.get("/health")
def health():
    return {"ok": True, "service": "mot-realtime"}


@app.post("/realtime/track")
async def track_http(frame: UploadFile):
    try:
        jpeg_bytes = await frame.read()
        frame_bgr = _decode_jpeg(jpeg_bytes)

        tracker = _get_http_tracker()
        detections: list[dict[str, Any]] = []
        if tracker is not None:
            global _http_frame_id
            _http_frame_id += 1
            detections = tracker.update(frame=frame_bgr, frame_id=_http_frame_id, roi=None)

        annotated = _draw_overlays(frame_bgr, detections=detections, roi=None, line=None)
        out_jpeg = _encode_jpeg(annotated, quality=80)
        # Keep HTTP response compatible with existing UI fallback (base64 JSON)
        import base64

        return {
            "annotated": base64.b64encode(out_jpeg).decode("utf-8"),
            "count": len(detections),
            "timestamp": int(time.time() * 1000),
            "counts": {"inside": 0, "outside": 0},
            "tracker_active": tracker is not None,
            "tracker_error": _http_tracker_error,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "tracker_active": False})


@app.websocket("/realtime/ws")
async def track_ws(ws: WebSocket):
    await ws.accept()

    # One tracker per websocket connection (correct isolation, per plan.md)
    try:
        tracker = _load_tracker()
        tracker_active = True
        tracker_error = None
    except Exception as e:
        tracker = None
        tracker_active = False
        tracker_error = str(e)

    roi_obj: dict | None = None
    line_obj: dict | None = None

    frame_counter = 0

    # Backpressure: this endpoint processes frames sequentially; the client should wait for responses.
    # If a client spams frames, we drop them while busy.
    busy = False

    async def _handle_frame(frame_bytes: bytes):
        nonlocal frame_counter, busy
        busy = True
        try:
            t0 = time.perf_counter()
            frame_counter += 1
            frame_bgr = _decode_jpeg(frame_bytes)
            t_decode = time.perf_counter()

            detections: list[dict[str, Any]] = []
            if tracker is not None:
                detections = tracker.update(frame=frame_bgr, frame_id=frame_counter, roi=roi_obj)
            t_track = time.perf_counter()

            annotated = _draw_overlays(frame_bgr, detections=detections, roi=roi_obj, line=line_obj)
            t_draw = time.perf_counter()

            preview_max_w = _env_int("REALTIME_PREVIEW_MAX_W", 640)
            annotated_preview = _downscale_max_w(annotated, preview_max_w)

            t_enc0 = time.perf_counter()
            out = _encode_jpeg(annotated_preview)
            t_enc1 = time.perf_counter()

            w = annotated.shape[1]
            line_counts = {"inside": 0, "outside": 0}
            if line_obj and isinstance(line_obj, dict):
                line_counts = _compute_line_counts(detections, line=line_obj, width=w)

            meta = {
                "type": "result",
                "count": len(detections),
                "timestamp": int(time.time() * 1000),
                "has_roi": roi_obj is not None,
                "counts": line_counts,
                "tracker_active": tracker_active,
            }
            if os.getenv("REALTIME_TIMING", "1").strip() != "0":
                t_done = time.perf_counter()
                meta["timing_ms"] = {
                    "decode": int(round((t_decode - t0) * 1000)),
                    "track": int(round((t_track - t_decode) * 1000)),
                    "draw": int(round((t_draw - t_track) * 1000)),
                    "encode": int(round((t_enc1 - t_enc0) * 1000)),
                    "total": int(round((t_done - t0) * 1000)),
                }
            if not tracker_active and tracker_error:
                meta["tracker_error"] = tracker_error

            # Send small JSON metadata then binary annotated JPEG
            await ws.send_text(json.dumps(meta))
            await ws.send_bytes(out)
        finally:
            busy = False

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await ws.send_text(json.dumps({"type": "error", "error": "invalid json"}))
                    continue

                if isinstance(payload, dict) and payload.get("type") == "config":
                    roi_obj = payload.get("roi")
                    line_obj = payload.get("line")
                    await ws.send_text(json.dumps({"type": "ack"}))
                continue

            frame_bytes = message.get("bytes")
            if frame_bytes is None:
                continue

            if busy:
                # Drop frame if still processing previous one.
                continue

            await _handle_frame(frame_bytes)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass
        return
