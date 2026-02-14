# API reference

This project exposes HTTP endpoints for offline processing and a WebSocket API for realtime.

Base URL (default Docker):

- `http://localhost:5000`

## Health

### GET /health

Returns:

```json
{ "status": "ok" }
```

## Web pages

These return HTML templates:

- `GET /` → home page
- `GET /video/` → video upload UI
- `GET /realtime/` → realtime UI

## Offline video API

All endpoints are under `/video`.

### POST /video/upload

Multipart form-data:

- `file`: video file (`.mp4`, `.mov`, `.avi`, `.mkv`)

Response:

```json
{ "job_id": "...", "filename": "..." }
```

### GET /video/status/{job_id}

Response shape is a simple status contract, e.g.

```json
{ "job_id": "...", "state": "running", "message": "..." }
```

Notes:

- If `QUEUE_MODE=rq`, the API may include queue job status fields.

### POST /video/run/{job_id}

JSON body `params` controls pipeline behavior.

Response:

```json
{ "job_id": "...", "state": "done" }
```

Or error:

```json
{ "job_id": "...", "state": "failed", "error": "..." }
```

### GET /video/results/{job_id}/annotations

Returns `application/json` (job annotations).

Query:

- `download=1` to force a download filename.

### GET /video/results/{job_id}/video

Returns `video/mp4`.

Query:

- `download=1` to force a download filename.

### GET /video/results/{job_id}/analytics

Returns `application/json`.

- If an `analytics.json` exists, it is returned.
- Otherwise, the server may derive analytics from `annotations.json`.

## Realtime API (HTTP)

### POST /realtime/track

This is an HTTP fallback path that accepts a single frame.

Multipart form-data:

- `frame`: JPEG bytes
- `roi` (optional): JSON string, normalized coordinates
- `line` (optional): JSON string `{ "x": 0..1, "position": "left"|"right" }`

Response:

```json
{
  "annotated": "<base64-jpeg>",
  "count": 0,
  "timestamp": 0,
  "has_roi": true,
  "counts": {"inside": 0, "outside": 0},
  "tracker_active": true
}
```

## Realtime API (WebSocket)

### WS /realtime/ws

In Docker Compose, the browser connects to the web app at `WS /realtime/ws`.

- If `REALTIME_BACKEND_WS` is set (Compose default), the web app **proxies** this WebSocket to the dedicated realtime backend (`mot-realtime`).
- If it is not set, the web app runs a **fallback** implementation locally.

Protocol summary (shared):

1) Client connects.
2) Client sends a JSON config message:

```json
{ "type": "config", "roi": {"x":0,"y":0,"width":1,"height":1}, "line": null }
```

3) Server replies:

```json
{ "type": "ack" }
```

4) Client sends binary messages: each is a single JPEG frame.
5) For each frame, responses differ depending on deployment mode:

### Mode A (Compose default): dedicated realtime backend

For each input frame, server replies with:

1) JSON metadata (text frame):

```json
{
  "type": "result",
  "count": 0,
  "timestamp": 0,
  "has_roi": false,
  "counts": {"inside": 0, "outside": 0},
  "tracker_active": true,
  "timing_ms": {"decode": 0, "track": 0, "draw": 0, "encode": 0, "total": 0}
}
```

2) Binary annotated JPEG preview.

### Mode B (fallback): web app realtime

For each input frame, server replies with a single JSON message containing a base64 JPEG:

```json
{
  "annotated": "<base64-jpeg>",
  "count": 0,
  "timestamp": 0,
  "has_roi": false,
  "counts": {"inside": 0, "outside": 0},
  "tracker_active": true
}
```

Backpressure:

- Frames are processed sequentially; if the client sends too fast, latency will increase.
- Some paths may drop frames under load to keep UI responsive.

## Dedicated realtime backend service

In Docker, `mot-web-dev` proxies WS to a separate backend. That backend runs on port `5001` inside the Docker network.

- Health: `GET http://mot-realtime:5001/health` (from inside Docker)
- WS: `ws://mot-realtime:5001/realtime/ws`

The proxy route exists to keep the web container CPU-only.
