# Multi-Object Tracking Solution

A web interface for multi-object tracking with two main, separated features:
- Video upload and offline processing (video.html)
- Real-time camera tracking (realtime.html)


---

## Features

- Video upload and server-side tracking pipeline
  - Upload MP4/AVI/MOV/MKV
  - Server processing (detection + tracking)
  - Download annotated video and annotations
- Real-time tracking using device camera
  - Webcam access and ROI selection
  - Server-side detection/tracking endpoint for realtime inference
  - Live overlay of annotated frames on a canvas
- Analytics & reporting
  - Heatmaps, charts and PDF report export
- Clean UI with Tailwind and modular JS/CSS assets

---

## Repository layout (relevant files)

- app/
  - app.py — Flask application factory / routes
  - realtime.py — Realtime tracking Flask endpoint(s)
  - video.py — Video upload/processing endpoints
  - realtimetracking.py — runtime for realtime
  - templates/
    - home.html — landing page selector (choose Video vs Realtime)
    - video.html — video upload & offline processing UI
    - realtime.html — realtime webcam tracking UI
  - static/
    - js/
      - common.js
      - upload.js
      - realtime.js
    - css/
      - style.css
      - premium.css
- CustomBoostTrack/ 
- run.py — start script
- requirements.txt

---



## Installation 

1. Clone the repository:
   ```
   git clone https://github.com/TechShot-inc/Multi-Object_tracking-website.git
   cd Multi-Object_tracking-website
   ```

2. Create and activate environment (conda recommended):
   ```
   conda create -n AICV python=3.11 -y
   conda activate AICV
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```



## Run the app

From the repository root:

```
python run.py
```

Open your browser to:
```
http://localhost:5001
```

(If your Flask config uses a different port, follow the printed URL in the terminal.)

---

## Usage

### Homepage
- The homepage (home.html) acts as a selector. Choose either "Video Upload" or "Real-Time Tracking". Each option opens a dedicated page:
  - `/video` -> video.html (video upload & processing)
  - `/realtime` -> realtime.html (camera tracking)

### Video Upload (video.html)
1. Upload a supported video file.
2. Optionally select ROI on the preview.
3. Configure frame extraction / output speed sliders.
4. Click "Upload and Process".
5. After processing, view annotated video, download annotations or PDF report.

### Real-Time Tracking (realtime.html)
1. Allow webcam access when prompted by browser.
2. Optionally select ROI before starting.
3. Click "Start Tracking" to begin real-time processing (frames are sent to `/realtime-track`).
4. Click "Stop Tracking" to end the session and release the camera.

---

## Frontend notes

- The UI uses separate templates:
  - `video.html` contains the whole video upload and report flow.
  - `realtime.html` contains webcam, ROI canvas, overlay canvas, and start/stop controls.
- Real-time overlay is drawn to a canvas (`tracking-canvas`) sized to match the video element. If annotated frames flicker or appear delayed, check:
  - webcam permissions
  - canvas sizing code (resizeRoiCanvasRealtime)
  - that `/realtime-track` returns `annotated` base64 JPEG for each frame
- JS files:
  - `static/js/realtime.js` — core realtime logic (start/stop, frame capture, ROI)
  - `static/js/upload.js` — video upload handling
  - `static/js/common.js` — shared helpers (modal, image utilities)
  
---


## Contributors

- Domadios Morcos — https://github.com/DomaMorcos  
- Ahmed Walaaeldin — https://github.com/Ahmed-Walaaeldin  
- Omar Hekal — https://github.com/omar-hekal

