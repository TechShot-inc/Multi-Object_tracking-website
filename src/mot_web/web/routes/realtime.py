from __future__ import annotations

import base64
import asyncio
import json
import os
import threading
import time

from fastapi import APIRouter, Form, Request, WebSocket, WebSocketDisconnect
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# OpenCV/numpy are optional - graceful degradation for testing
try:
    import cv2 as _cv2
    import numpy as _np

    cv2 = _cv2
    np = _np
except ImportError:
    cv2 = None
    np = None

CV2_AVAILABLE = cv2 is not None and np is not None

router = APIRouter(prefix="/realtime")


_REALTIME_BACKEND_WS = os.getenv("REALTIME_BACKEND_WS")


def _backend_http_base() -> str | None:
    """Best-effort HTTP base URL for the external realtime backend.

    If REALTIME_BACKEND_HTTP is set, use it.
    Otherwise, derive from REALTIME_BACKEND_WS when present.
    """

    raw = (os.getenv("REALTIME_BACKEND_HTTP") or "").strip()
    if raw:
        return raw.rstrip("/")
    ws = (_REALTIME_BACKEND_WS or "").strip()
    if not ws:
        return None
    if ws.startswith("ws://"):
        return "http://" + ws[len("ws://") :].split("/", 1)[0]
    if ws.startswith("wss://"):
        return "https://" + ws[len("wss://") :].split("/", 1)[0]
    return None


_tracker_lock = threading.Lock()
_tracker = None
_tracker_error: str | None = None
_frame_counter = 0


def _default_model_paths() -> dict[str, str]:
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    models_dir = os.getenv("MODELS_DIR", os.path.join(project_root, "models"))
    return {
        "yolo1": os.getenv("YOLO11_MODEL_PATH", os.path.join(models_dir, "yolo11x.pt")),
        "yolo2": os.getenv("YOLO12_MODEL_PATH", os.path.join(models_dir, "yolo12x.pt")),
        "reid": os.getenv("REID_MODEL_PATH", os.path.join(models_dir, "osnet_ain_ms_m_c.pth.tar")),
    }


def _get_tracker():
    global _tracker, _tracker_error
    if _tracker is not None:
        return _tracker

    with _tracker_lock:
        if _tracker is not None:
            return _tracker
        try:
            from CustomBoostTrack.realtime_ensembling import RealTimeTracker

            model_paths = _default_model_paths()
            if not (os.path.exists(model_paths["yolo1"]) and os.path.exists(model_paths["yolo2"])):
                raise FileNotFoundError(
                    "YOLO weights not found. Set YOLO11_MODEL_PATH/YOLO12_MODEL_PATH or place them under MODELS_DIR."
                )
            if not os.path.exists(model_paths["reid"]):
                raise FileNotFoundError("ReID weights not found. Set REID_MODEL_PATH or place it under MODELS_DIR.")

            _tracker = RealTimeTracker(
                model1_path=model_paths["yolo1"],
                model2_path=model_paths["yolo2"],
                reid_path=model_paths["reid"],
                frame_rate=30,
            )
        except Exception as e:
            _tracker_error = str(e)
            _tracker = None
        return _tracker


def _tracker_status() -> dict:
    tracker = _get_tracker()
    if tracker is not None:
        return {"tracker_active": True}
    # Surface a short, user-visible reason in dev; helps debug missing weights/deps.
    err = (_tracker_error or "tracker unavailable").strip()
    if len(err) > 240:
        err = err[:240] + "…"
    return {"tracker_active": False, "tracker_error": err}


def _line_occupancy_counts(detections: list[dict], line: dict, width: int) -> dict[str, int]:
    """Compute current inside/outside occupancy relative to a vertical line.

    This is stateless on purpose: counts update immediately and never "stack".
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


@router.get("/", response_class=HTMLResponse)
def realtime_page_fastapi(request: Request):
    return request.app.state.templates.TemplateResponse("realtime.html", {"request": request})


@router.post("/track")
async def realtime_track_fastapi(
    request: Request,
    frame: UploadFile | None = File(default=None),
    roi: str | None = Form(default=None),
    line: str | None = Form(default=None),
):
    # If we have an external realtime backend configured (mot-realtime), proxy the HTTP
    # fallback to it instead of trying to import/run CustomBoostTrack inside the web container.
    # This avoids confusing "No module named 'CustomBoostTrack'" errors in dev.
    backend = _backend_http_base()
    if backend:
        try:
            import httpx

            if frame is None:
                return JSONResponse(status_code=400, content={"error": "No frame provided"})

            jpeg_bytes = await frame.read()
            files = {"frame": (frame.filename or "frame.jpg", jpeg_bytes, frame.content_type or "image/jpeg")}
            data: dict[str, str] = {}
            if roi is not None:
                data["roi"] = roi
            if line is not None:
                data["line"] = line

            # Keep a fairly small timeout so the UI can recover by retrying.
            timeout = httpx.Timeout(connect=5.0, read=20.0, write=20.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{backend}/realtime/track", files=files, data=data)
            # Pass through response JSON (keeps frontend contract intact).
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": f"realtime backend proxy failed: {str(e)}"})

    if cv2 is None or np is None:
        return JSONResponse(status_code=503, content={"error": "OpenCV not installed - realtime tracking unavailable"})

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "No frame provided"})

    frame_data = await frame.read()

    nparr = np.frombuffer(frame_data, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if decoded is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode frame"})

    # Parse optional ROI
    roi_obj = None
    if roi is not None:
        try:
            roi_obj = json.loads(roi)
            if not isinstance(roi_obj, dict) or not all(k in roi_obj for k in ["x", "y", "width", "height"]):
                return JSONResponse(status_code=400, content={"error": "Invalid ROI format"})
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid ROI JSON: {e}"})

    # Parse optional line
    line_obj = None
    if line is not None:
        try:
            line_obj = json.loads(line)
            if not isinstance(line_obj, dict) or "position" not in line_obj or "x" not in line_obj:
                return JSONResponse(status_code=400, content={"error": "Invalid line format"})
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid line JSON: {e}"})

    annotated_frame = decoded.copy()
    h, w = annotated_frame.shape[:2]

    tracker = _get_tracker()
    detections: list[dict] = []
    model_paths = None
    if tracker is not None:
        global _frame_counter
        _frame_counter += 1
        try:
            detections = tracker.update(frame=decoded, frame_id=_frame_counter, roi=roi_obj)
            model_paths = _default_model_paths()
        except Exception:
            detections = []

    # Draw ROI if provided
    if roi_obj:
        x1 = int(roi_obj["x"] * w)
        y1 = int(roi_obj["y"] * h)
        x2 = int((roi_obj["x"] + roi_obj["width"]) * w)
        y2 = int((roi_obj["y"] + roi_obj["height"]) * h)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw counting line if provided
    if line_obj:
        line_x = int(float(line_obj["x"]) * w)
        cv2.line(annotated_frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)
        side = "L" if line_obj["position"] == "left" else "R"
        cv2.putText(
            annotated_frame,
            f"Count Line ({side})",
            (line_x + 5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    for det in detections:
        try:
            x1 = int(float(det["x"]))
            y1 = int(float(det["y"]))
            x2 = x1 + int(float(det["width"]))
            y2 = y1 + int(float(det["height"]))
            tid = int(det["id"])
        except Exception:
            continue
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            annotated_frame,
            f"ID {tid}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        annotated_frame,
        "Tracking Active" if tracker is not None else "Tracking Active (stub)",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    ok, buffer = cv2.imencode(".jpg", annotated_frame)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    encoded_frame = base64.b64encode(buffer.tobytes()).decode("utf-8")

    counts = {"inside": 0, "outside": 0}
    if line_obj:
        counts = _line_occupancy_counts(detections, line=line_obj, width=w)

    resp = {
        "annotated": encoded_frame,
        "count": len(detections),
        "timestamp": int(time.time() * 1000),
        "has_roi": roi_obj is not None,
        "counts": counts,
    }
    resp.update(_tracker_status())
    if model_paths is not None:
        resp["model_paths"] = model_paths
    return resp


@router.websocket("/ws")
async def realtime_ws(ws: WebSocket):
    await ws.accept()

    # Production path: proxy this WebSocket to the dedicated realtime backend.
    # This keeps the web container CPU-only (no Torch/Ultralytics).
    if _REALTIME_BACKEND_WS:
        try:
            import websockets  # provided by uvicorn[standard]

            async with websockets.connect(_REALTIME_BACKEND_WS, max_size=50_000_000) as backend:

                async def client_to_backend():
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if msg.get("text") is not None:
                            await backend.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await backend.send(msg["bytes"])

                async def backend_to_client():
                    async for msg in backend:
                        if isinstance(msg, (bytes, bytearray)):
                            await ws.send_bytes(bytes(msg))
                        else:
                            await ws.send_text(str(msg))

                t1 = asyncio.create_task(client_to_backend())
                t2 = asyncio.create_task(backend_to_client())
                _, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
        except Exception as e:
            await ws.send_json({"type": "error", "error": f"realtime backend proxy failed: {e}"})
        finally:
            try:
                await ws.close()
            except Exception:
                pass
        return

    if cv2 is None or np is None:
        await ws.send_json({"error": "OpenCV not installed - realtime tracking unavailable"})
        await ws.close(code=1011)
        return
    roi_obj: dict | None = None
    line_obj: dict | None = None

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                # Config message: {type:"config", roi:<dict|null>, line:<dict|null>}
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await ws.send_json({"error": "invalid json"})
                    continue
                if isinstance(payload, dict) and payload.get("type") == "config":
                    roi_obj = payload.get("roi")
                    line_obj = payload.get("line")
                    await ws.send_json({"type": "ack"})
                continue

            frame_bytes = message.get("bytes")
            if frame_bytes is None:
                continue

            nparr = np.frombuffer(frame_bytes, np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if decoded is None:
                await ws.send_json({"error": "Failed to decode frame"})
                continue

            # Reuse the HTTP handler's logic by calling into the same code path.
            # This is intentionally duplicated lightly to avoid heavy refactors.
            annotated_frame = decoded.copy()
            h, w = annotated_frame.shape[:2]

            tracker = _get_tracker()
            detections: list[dict] = []
            model_paths = None
            if tracker is not None:
                global _frame_counter
                _frame_counter += 1
                try:
                    detections = tracker.update(frame=decoded, frame_id=_frame_counter, roi=roi_obj)
                    model_paths = _default_model_paths()
                except Exception:
                    detections = []

            if roi_obj and isinstance(roi_obj, dict):
                try:
                    x1 = int(float(roi_obj["x"]) * w)
                    y1 = int(float(roi_obj["y"]) * h)
                    x2 = int((float(roi_obj["x"]) + float(roi_obj["width"])) * w)
                    y2 = int((float(roi_obj["y"]) + float(roi_obj["height"])) * h)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception:
                    pass

            if line_obj and isinstance(line_obj, dict) and "x" in line_obj and "position" in line_obj:
                try:
                    line_x = int(float(line_obj["x"]) * w)
                    cv2.line(annotated_frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)
                    side = "L" if line_obj["position"] == "left" else "R"
                    cv2.putText(
                        annotated_frame,
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
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    annotated_frame, f"ID {tid}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

            cv2.putText(
                annotated_frame,
                "Tracking Active" if tracker is not None else "Tracking Active (stub)",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            ok, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ok:
                await ws.send_json({"error": "cv2.imencode failed"})
                continue
            encoded_frame = base64.b64encode(buffer.tobytes()).decode("utf-8")

            counts = {"inside": 0, "outside": 0}
            if line_obj and isinstance(line_obj, dict):
                counts = _line_occupancy_counts(detections, line=line_obj, width=w)

            resp = {
                "annotated": encoded_frame,
                "count": len(detections),
                "timestamp": int(time.time() * 1000),
                "has_roi": roi_obj is not None,
                "counts": counts,
            }
            resp.update(_tracker_status())
            if model_paths is not None:
                resp["model_paths"] = model_paths

            await ws.send_json(resp)
    except WebSocketDisconnect:
        return
