from __future__ import annotations

import base64
import json
import time
from typing import TYPE_CHECKING

from flask import Blueprint, jsonify, render_template, request

# OpenCV/numpy are optional - graceful degradation for testing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

bp = Blueprint("realtime", __name__, url_prefix="/realtime")


@bp.get("/")
def realtime_page():
    return render_template("realtime.html")


@bp.post("/track")
def realtime_track():
    """
    Process a webcam frame for real-time tracking.
    
    Expects multipart form data:
      - frame: JPEG image blob
      - roi (optional): JSON string with {x, y, width, height} normalized 0-1
      - line (optional): JSON string with {position: 'left'|'right', x: 0-1}
    
    Returns:
      - annotated: base64 JPEG of annotated frame
      - count: number of detected objects
      - timestamp: server timestamp in ms
      - counts: {inside: N, outside: M} if line is provided
    """
    if not CV2_AVAILABLE:
        return jsonify(error="OpenCV not installed - realtime tracking unavailable"), 503

    if "frame" not in request.files:
        return jsonify(error="No frame provided"), 400

    frame_file = request.files["frame"]
    frame_data = frame_file.read()

    # Decode frame
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Failed to decode frame"), 400

    # Parse optional ROI
    roi = None
    if "roi" in request.form:
        try:
            roi = json.loads(request.form["roi"])
            if not isinstance(roi, dict) or not all(k in roi for k in ["x", "y", "width", "height"]):
                return jsonify(error="Invalid ROI format"), 400
        except json.JSONDecodeError as e:
            return jsonify(error=f"Invalid ROI JSON: {e}"), 400

    # Parse optional line
    line = None
    if "line" in request.form:
        try:
            line = json.loads(request.form["line"])
            if not isinstance(line, dict) or "position" not in line or "x" not in line:
                return jsonify(error="Invalid line format"), 400
        except json.JSONDecodeError as e:
            return jsonify(error=f"Invalid line JSON: {e}"), 400

    # === STUB IMPLEMENTATION ===
    # TODO: Replace with actual tracking pipeline (CustomBoostTrack integration)
    # For now, just return the frame with a simple overlay to prove the endpoint works
    
    annotated_frame = frame.copy()
    h, w = annotated_frame.shape[:2]
    
    # Draw ROI if provided
    if roi:
        x1 = int(roi["x"] * w)
        y1 = int(roi["y"] * h)
        x2 = int((roi["x"] + roi["width"]) * w)
        y2 = int((roi["y"] + roi["height"]) * h)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw counting line if provided
    if line:
        line_x = int(line["x"] * w)
        cv2.line(annotated_frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)
        side = "L" if line["position"] == "left" else "R"
        cv2.putText(annotated_frame, f"Count Line ({side})", (line_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Add status overlay
    cv2.putText(
        annotated_frame,
        "Tracking Active (stub)",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # Encode annotated frame
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    encoded_frame = base64.b64encode(buffer).decode("utf-8")

    return jsonify(
        annotated=encoded_frame,
        count=0,  # Stub: no detections
        timestamp=int(time.time() * 1000),
        has_roi=roi is not None,
        counts={"inside": 0, "outside": 0},
    )
