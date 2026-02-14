from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient


def _make_jpeg_bytes(width: int = 160, height: int = 120) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (60, 60), (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def test_realtime_websocket_roundtrip(app):
    client = TestClient(app)

    with client.websocket_connect("/realtime/ws") as ws:
        ws.send_json({"type": "config", "roi": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}})
        ack = ws.receive_json()
        assert ack.get("type") == "ack"

        ws.send_bytes(_make_jpeg_bytes())

        # Receive the first result payload.
        msg = ws.receive_json()

    assert msg is not None
    if msg.get("error"):
        raise AssertionError(f"websocket returned error: {msg['error']}")

    assert "counts" in msg
    assert "annotated" in msg
    # Ensure base64 is decodable
    raw = base64.b64decode(msg["annotated"], validate=False)
    assert len(raw) > 100
