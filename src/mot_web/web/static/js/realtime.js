let roiRealtime = null;
let isSelectingRoi = false;
let isTracking = false;
let videoStream = null;
let startX, startY;
let linePosition = 'left'; // Default to left line
let lineX = null; // User-defined line x-coordinate (normalized)
let isDrawingLine = false;

// Realtime performance tuning.
// Override via URL params (no rebuild needed):
//   rtFps=12&rtJpegQ=0.7&rtSendW=960&rtSendH=540&rtCapW=1280&rtCapH=720&rtDebug=1
function _rtParamFloat(name, fallback) {
    try {
        const v = new URLSearchParams(window.location.search).get(name);
        if (v === null || v === undefined || String(v).trim() === '') return fallback;
        const n = parseFloat(v);
        return Number.isFinite(n) ? n : fallback;
    } catch (e) {
        return fallback;
    }
}

function _rtParamInt(name, fallback) {
    try {
        const v = new URLSearchParams(window.location.search).get(name);
        if (v === null || v === undefined || String(v).trim() === '') return fallback;
        const n = parseInt(v, 10);
        return Number.isFinite(n) ? n : fallback;
    } catch (e) {
        return fallback;
    }
}

function _rtParamBool(name, fallback) {
    try {
        const v = new URLSearchParams(window.location.search).get(name);
        if (v === null || v === undefined || String(v).trim() === '') return fallback;
        return ['1', 'true', 'yes', 'on'].includes(String(v).trim().toLowerCase());
    } catch (e) {
        return fallback;
    }
}

const REALTIME_DEBUG = _rtParamBool('rtDebug', false);
function dlog(...args) {
    if (REALTIME_DEBUG) console.log(...args);
}

const CAPTURE_WIDTH_IDEAL = _rtParamInt('rtCapW', 1280);
const CAPTURE_HEIGHT_IDEAL = _rtParamInt('rtCapH', 720);
// What we actually send over the wire.
const SEND_MAX_WIDTH = _rtParamInt('rtSendW', 960);
const SEND_MAX_HEIGHT = _rtParamInt('rtSendH', 540);
const TARGET_FPS = Math.max(1, _rtParamInt('rtFps', 12));
const JPEG_QUALITY = Math.min(0.95, Math.max(0.3, _rtParamFloat('rtJpegQ', 0.7)));

let captureCanvas = null;
let captureCtx = null;

const video = document.getElementById('realtime-video');
const trackingCanvas = document.getElementById('tracking-canvas');
const trackingCtx = trackingCanvas.getContext('2d');
let roiCanvasRealtime = document.getElementById('roi-canvas-realtime');
const lineCanvas = document.getElementById('line-canvas');
const lineCtx = lineCanvas.getContext('2d');
const selectRoiButton = document.getElementById('select-roi-realtime');
const clearRoiButton = document.getElementById('clear-roi-realtime');
const drawLineButton = document.getElementById('draw-line');
const clearLineButton = document.getElementById('clear-line');
const personsInsideSpan = document.getElementById('persons-inside');
const personsOutsideSpan = document.getElementById('persons-outside');

// Phase 6: realtime stats strip + inline error UX
const statFpsEl = document.getElementById('rt-stat-fps');
const statLatencyEl = document.getElementById('rt-stat-latency');
const statObjectsEl = document.getElementById('rt-stat-objects');

const errorPanelEl = document.getElementById('realtime-error');
const errorKindEl = document.getElementById('realtime-error-kind');
const errorMessageEl = document.getElementById('realtime-error-message');

let _rtLastLatencyMs = null;
let _rtLastObjects = null;
let _rtFrameTimes = [];

function _rtSetStatText(el, value) {
    if (!el) return;
    el.textContent = (value === null || value === undefined || value === '') ? '-' : String(value);
}

function _rtUpdateStatsUI() {
    // FPS from sliding window of successful renders
    let fpsText = '-';
    if (_rtFrameTimes.length >= 2) {
        const dtMs = _rtFrameTimes[_rtFrameTimes.length - 1] - _rtFrameTimes[0];
        if (dtMs > 0) {
            const fps = ((_rtFrameTimes.length - 1) * 1000) / dtMs;
            fpsText = fps.toFixed(1);
        }
    }

    _rtSetStatText(statFpsEl, fpsText);
    _rtSetStatText(statObjectsEl, (_rtLastObjects === null ? '-' : _rtLastObjects));
    _rtSetStatText(statLatencyEl, (_rtLastLatencyMs === null ? '-' : `${_rtLastLatencyMs} ms`));
}

function _rtRecordRenderedFrame() {
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    _rtFrameTimes.push(now);
    // Keep ~2 seconds of history at typical FPS
    while (_rtFrameTimes.length > 30) _rtFrameTimes.shift();
    _rtUpdateStatsUI();
}

function _rtClearStats() {
    _rtLastLatencyMs = null;
    _rtLastObjects = null;
    _rtFrameTimes = [];
    _rtUpdateStatsUI();
}

function _rtSetError(kind, message) {
    if (!errorPanelEl || !errorKindEl || !errorMessageEl) return;
    const k = (kind || 'Error').trim();
    const msg = (message || '').trim();
    if (!msg) return;
    errorKindEl.textContent = k;
    errorMessageEl.textContent = msg;
    errorPanelEl.classList.remove('hidden');
}

function _rtClearError() {
    if (!errorPanelEl) return;
    errorPanelEl.classList.add('hidden');
    if (errorKindEl) errorKindEl.textContent = 'Error';
    if (errorMessageEl) errorMessageEl.textContent = '';
}

function resetCountersUI() {
    personsInsideSpan.textContent = '0';
    personsOutsideSpan.textContent = '0';
}

function initializeWebcam() {
    dlog('Initializing webcam UI');
    resetRoiState();
    resetLineState();
    resizeCanvases();
    _rtClearError();
    _rtClearStats();
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Ready to start tracking';
    // Set up line position radio buttons
    document.querySelectorAll('input[name="line-position"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            linePosition = e.target.value;
            dlog('Line position changed to:', linePosition);
            drawLine();
        });
    });
}

function startTracking() {
    dlog('Starting tracking');
    document.getElementById('start-tracking').disabled = true;
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Starting camera...';
    _rtClearError();
    _rtClearStats();
    roiCanvasRealtime.style.pointerEvents = 'none';

    // Camera access requires a secure context in most browsers. HTTP is allowed for localhost
    // but NOT for arbitrary LAN IPs / 0.0.0.0.
    const host = window.location.hostname;
    const isLocalhost = (host === 'localhost' || host === '127.0.0.1' || host === '::1');
    if (!window.isSecureContext && !isLocalhost) {
        console.error('Insecure context for camera:', window.location.origin);
        document.getElementById('start-tracking').disabled = false;
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Camera requires HTTPS or localhost';
        _rtSetError('Camera', 'Camera requires HTTPS or localhost');
        showModal(
            'Camera access is blocked on insecure origins. Open this page via https://, or use http://localhost:5000 instead of ' +
                window.location.origin
        );
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('Webcam access not supported');
        document.getElementById('start-tracking').disabled = false;
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Webcam access not supported';
        _rtSetError('Camera', 'Webcam access not supported');
        showModal('Webcam access not supported in this browser.');
        return;
    }

    navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: CAPTURE_WIDTH_IDEAL },
            height: { ideal: CAPTURE_HEIGHT_IDEAL },
            facingMode: "user"
        }
    }).then(stream => {
        dlog('Webcam stream acquired');
        video.srcObject = stream;
        videoStream = stream;
        isTracking = true;
        
        trackingCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
        roiCanvasRealtime.getContext('2d').clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
        lineCtx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);

        video.onloadeddata = function() {
            dlog('Video data loaded, width:', video.videoWidth || 1280, 'height:', video.videoHeight || 720);
            video.play().then(() => {
                dlog('Video playback started');
                resizeCanvases();
                drawLine();
                document.getElementById('stop-tracking').disabled = false;
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Camera active, starting tracking...';
                requestAnimationFrame(processFrames);
            }).catch(err => {
                console.error('Video play error:', err);
                document.getElementById('start-tracking').disabled = false;
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error playing video';
                _rtSetError('Camera', (err && (err.message || err.name)) ? (err.message || err.name) : String(err));
                showModal('Error playing webcam video: ' + err.message);
                stopTracking();
            });
        };
    }).catch(err => {
        console.error('Webcam access error:', err);
        document.getElementById('start-tracking').disabled = false;
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error accessing webcam';
        const msg = (err && (err.message || err.name)) ? (err.message || err.name) : String(err);
        _rtSetError('Camera', msg);
        showModal('Error accessing webcam: ' + msg);
    });
}

function stopTracking() {
    dlog('Stopping tracking');
    isTracking = false;
    document.getElementById('start-tracking').disabled = false;
    document.getElementById('stop-tracking').disabled = true;
    document.getElementById('realtime-loading').classList.add('hidden');
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Ready to start tracking';
    _rtClearError();
    _rtClearStats();
    
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    video.srcObject = null;
    video.poster = '';
    
    trackingCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
    roiCanvasRealtime.getContext('2d').clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
    lineCtx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);
    personsInsideSpan.textContent = '0';
    personsOutsideSpan.textContent = '0';
}

let lastFrameTime = 0;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

let ws = null;
let wsReady = false;
let lastConfigKey = null;

function getWsUrl() {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // If the user copied the bind address (0.0.0.0) into the browser, websocket and camera
    // will fail. Keep the server bound to 0.0.0.0, but connect via localhost.
    const hostname = (window.location.hostname === '0.0.0.0') ? 'localhost' : window.location.hostname;
    const port = window.location.port ? `:${window.location.port}` : '';
    return `${proto}://${hostname}${port}/realtime/ws`;
}

function ensureWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }
    try {
        ws = new WebSocket(getWsUrl());
        ws.binaryType = 'arraybuffer';
        wsReady = false;
        ws.onopen = () => {
            wsReady = true;
            lastConfigKey = null;
            dlog('WebSocket connected');
        };
        ws.onclose = () => {
            wsReady = false;
            dlog('WebSocket closed');
        };
        ws.onerror = (e) => {
            wsReady = false;
            console.warn('WebSocket error', e);
        };
    } catch (e) {
        wsReady = false;
        ws = null;
    }
}

function sendWsConfigIfChanged() {
    if (!wsReady || !ws) return;

    const normalizedRoi = roiRealtime ? {
        x: roiRealtime.x / (video.videoWidth || 1280),
        y: roiRealtime.y / (video.videoHeight || 720),
        width: roiRealtime.width / (video.videoWidth || 1280),
        height: roiRealtime.height / (video.videoHeight || 720)
    } : null;

    const lineData = (lineX !== null) ? { position: linePosition, x: lineX } : null;

    const key = JSON.stringify({ roi: normalizedRoi, line: lineData });
    if (key === lastConfigKey) return;
    lastConfigKey = key;

    ws.send(JSON.stringify({ type: 'config', roi: normalizedRoi, line: lineData }));
}

function processFrames(timestamp) {
    if (!isTracking) {
        dlog('Tracking stopped, exiting processFrames');
        return;
    }

    if (timestamp - lastFrameTime < FRAME_INTERVAL) {
        requestAnimationFrame(processFrames);
        return;
    }

    if (video.readyState < 2) {
        console.warn('Video not ready (readyState:', video.readyState, '), retrying...');
        requestAnimationFrame(processFrames);
        return;
    }

    lastFrameTime = timestamp;
    dlog('Processing frame at timestamp:', timestamp);

    if (trackingCanvas.width !== (video.videoWidth || 1280) || trackingCanvas.height !== (video.videoHeight || 720)) {
        resizeCanvases();
        dlog('Resized canvases to match video:', video.videoWidth || 1280, 'x', video.videoHeight || 720);
        drawLine();
    }

    if (!captureCanvas) {
        captureCanvas = document.createElement('canvas');
        captureCtx = captureCanvas.getContext('2d');
    }

    const srcW = video.videoWidth || 1280;
    const srcH = video.videoHeight || 720;
    const scale = Math.min(SEND_MAX_WIDTH / srcW, SEND_MAX_HEIGHT / srcH, 1.0);
    const sendW = Math.max(1, Math.round(srcW * scale));
    const sendH = Math.max(1, Math.round(srcH * scale));

    if (captureCanvas.width !== sendW) captureCanvas.width = sendW;
    if (captureCanvas.height !== sendH) captureCanvas.height = sendH;

    try {
        captureCtx.drawImage(video, 0, 0, sendW, sendH);
        dlog('Captured video frame', { srcW, srcH, sendW, sendH, jpegQ: JPEG_QUALITY, fps: TARGET_FPS });

        captureCanvas.toBlob(async blob => {
            if (!blob) {
                console.error('Failed to create blob from canvas');
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error capturing frame';
                showModal('Failed to create frame data');
                if (isTracking) {
                    setTimeout(() => requestAnimationFrame(processFrames), FRAME_INTERVAL);
                }
                return;
            }

            ensureWebSocket();
            if (wsReady && ws) {
                sendWsConfigIfChanged();
            }

            try {
                let data = null;
                if (wsReady && ws) {
                    const arrayBuffer = await blob.arrayBuffer();

                    data = await new Promise((resolve, reject) => {
                        const timeout = setTimeout(() => reject(new Error('WebSocket timeout')), 7000);
                        let meta = null;
                        let jpeg = null;
                        let isDone = false;

                        const cleanup = () => {
                            if (isDone) return;
                            isDone = true;
                            try {
                                ws.removeEventListener('message', handler);
                            } catch (e) {
                                // ignore
                            }
                            clearTimeout(timeout);
                        };

                        const tryResolve = () => {
                            if (isDone) return;
                            if (meta && meta.annotated) {
                                cleanup();
                                resolve(meta);
                                return;
                            }
                            if (meta && jpeg) {
                                cleanup();
                                resolve({ meta, jpeg });
                            }
                        };

                        const handler = (event) => {
                            try {
                                // New protocol (production): server sends JSON metadata, then binary JPEG.
                                // Legacy protocol: server sends JSON with {annotated: base64}.
                                if (typeof event.data === 'string') {
                                    const parsed = JSON.parse(event.data);
                                    if (parsed && parsed.type === 'error') {
                                        cleanup();
                                        reject(new Error(parsed.error || 'realtime error'));
                                        return;
                                    }
                                    if (parsed && parsed.error) {
                                        cleanup();
                                        reject(new Error(parsed.error));
                                        return;
                                    }

                                    // Legacy base64 path
                                    if (parsed && parsed.annotated) {
                                        meta = parsed;
                                        tryResolve();
                                        return;
                                    }

                                    // Metadata for binary path
                                    if (parsed && parsed.type === 'result') {
                                        meta = parsed;
                                        tryResolve();
                                    }
                                    // Ignore ack or other JSON messages.
                                    return;
                                }

                                if (event.data instanceof ArrayBuffer) {
                                    jpeg = event.data;
                                    tryResolve();
                                    return;
                                }

                                // Some browsers/proxies still deliver Blob even when binaryType is set.
                                if (event.data instanceof Blob) {
                                    event.data
                                        .arrayBuffer()
                                        .then((buf) => {
                                            jpeg = buf;
                                            tryResolve();
                                        })
                                        .catch((err) => {
                                            cleanup();
                                            reject(err);
                                        });
                                    return;
                                }
                            } catch (e) {
                                // ignore non-json / unexpected payload
                            }
                        };

                        ws.addEventListener('message', handler);
                        ws.send(arrayBuffer);
                    });
                } else {
                    const formData = new FormData();
                    formData.append('frame', blob, 'frame.jpg');
                    if (roiRealtime) {
                        const normalizedRoi = {
                            x: roiRealtime.x / (video.videoWidth || 1280),
                            y: roiRealtime.y / (video.videoHeight || 720),
                            width: roiRealtime.width / (video.videoWidth || 1280),
                            height: roiRealtime.height / (video.videoHeight || 720)
                        };
                        formData.append('roi', JSON.stringify(normalizedRoi));
                    }
                    if (lineX !== null) {
                        const lineData = { position: linePosition, x: lineX };
                        formData.append('line', JSON.stringify(lineData));
                    }

                    dlog('Sending frame to /realtime/track (HTTP fallback)');
                    const response = await fetch('/realtime/track', { method: 'POST', body: formData });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    data = await response.json();
                }
                dlog('Received tracking response');

                const meta = (data && data.meta) ? data.meta : data;
                const annotatedBytes = (data && data.jpeg) ? data.jpeg : null;
                const annotatedBase64 = (meta && meta.annotated) ? meta.annotated : null;

                if (meta && meta.error) {
                    throw new Error(meta.error);
                }
                if (!annotatedBytes && !annotatedBase64) {
                    throw new Error('No annotated frame in response');
                }

                // Hide loading overlay once we have a good response.
                document.getElementById('realtime-loading').classList.add('hidden');

                // Update counts
                if (lineX === null) {
                    resetCountersUI();
                } else if (meta && meta.counts) {
                    personsInsideSpan.textContent = meta.counts.inside ?? 0;
                    personsOutsideSpan.textContent = meta.counts.outside ?? 0;
                }

                // Update stats strip (object count + server latency)
                if (meta && (meta.count === 0 || meta.count)) {
                    _rtLastObjects = meta.count;
                }
                if (meta && meta.timing_ms && (meta.timing_ms.total === 0 || meta.timing_ms.total)) {
                    _rtLastLatencyMs = meta.timing_ms.total;
                } else {
                    _rtLastLatencyMs = null;
                }
                if (meta && meta.tracker_active === false && meta.tracker_error) {
                    _rtSetError('Tracker init', meta.tracker_error);
                }

                // Surface whether the real BoostTrack+YOLO tracker is running.
                // Important: do not let a *past* stub-mode error stick forever.
                const statusEl = document.getElementById('realtime-status').querySelector('.status-message');
                if (meta && meta.tracker_active === false && meta.tracker_error) {
                    statusEl.textContent = `Tracking (stub mode): ${meta.tracker_error}`;
                } else if (meta && meta.tracker_active === true) {
                    // Clear any previous stub-mode message.
                    statusEl.textContent = `Tracking... (${(meta && meta.count) || 0} objects)`;
                }

                // Display model paths if available
                if (meta && meta.model_paths) {
                    const modelInfo = `Models: General2 (${meta.model_paths.yolo1.split('/').pop()}), 12General1 (${meta.model_paths.yolo2.split('/').pop()}), ReID (${meta.model_paths.reid.split('/').pop()})`;
                    document.getElementById('realtime-status').querySelector('.status-message').textContent = 
                        `Tracking... (${meta.count || 0} objects) | ${modelInfo}`;
                } else {
                    // Only overwrite status if we didn't already set a stub-mode error above.
                    if (!String(statusEl.textContent || '').startsWith('Tracking (stub mode):')) {
                        statusEl.textContent = `Tracking... (${(meta && meta.count) || 0} objects)`;
                    }
                }

                const img = new Image();
                img.onload = () => {
                    trackingCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
                    trackingCtx.drawImage(img, 0, 0, trackingCanvas.width, trackingCanvas.height);
                    trackingCanvas.style.visibility = 'visible';
                    _rtRecordRenderedFrame();
                    dlog('Rendered annotated frame, object count:', (meta && meta.count) || 0);
                    if (isTracking) {
                        requestAnimationFrame(processFrames);
                    }
                };

                img.onerror = () => {
                    console.error('Failed to load annotated frame');
                    document.getElementById('realtime-loading').classList.add('hidden');
                    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error rendering frame';
                    if (isTracking) {
                        setTimeout(() => requestAnimationFrame(processFrames), 1000 / 30);
                    }
                };

                if (annotatedBytes) {
                    const blob = new Blob([annotatedBytes], { type: 'image/jpeg' });
                    const url = URL.createObjectURL(blob);
                    img.onload = () => {
                        trackingCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
                        trackingCtx.drawImage(img, 0, 0, trackingCanvas.width, trackingCanvas.height);
                        trackingCanvas.style.visibility = 'visible';
                        URL.revokeObjectURL(url);
                        _rtRecordRenderedFrame();
                        dlog('Rendered annotated frame (binary), object count:', (meta && meta.count) || 0);
                        if (isTracking) {
                            requestAnimationFrame(processFrames);
                        }
                    };
                    img.src = url;
                } else {
                    img.src = 'data:image/jpeg;base64,' + annotatedBase64;
                }
            } catch (error) {
                console.error('Error processing frame:', error);
                // If the WebSocket is flaky (common on some proxies), fall back to HTTP instead of getting stuck.
                if (String(error && error.message || '').includes('WebSocket timeout')) {
                    try {
                        if (ws) ws.close();
                    } catch (e) {
                        // ignore
                    }
                    ws = null;
                    wsReady = false;
                    lastConfigKey = null;
                    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'WebSocket slow; using HTTP fallback...';
                    document.getElementById('realtime-loading').classList.add('hidden');
                    _rtSetError('WebSocket timeout', 'WebSocket was slow; switched to HTTP fallback');
                } else {
                    document.getElementById('realtime-loading').classList.remove('hidden');
                    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Tracking error';
                    _rtSetError('Backend', (error && error.message) ? error.message : String(error));
                }
                if (isTracking) {
                    setTimeout(() => requestAnimationFrame(processFrames), FRAME_INTERVAL);
                }
            }
        }, 'image/jpeg', JPEG_QUALITY);
    } catch (error) {
        console.error('Error capturing frame:', error);
        document.getElementById('realtime-loading').classList.remove('hidden');
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error capturing frame';
        _rtSetError('Capture', (error && error.message) ? error.message : String(error));
        if (isTracking) {
            setTimeout(() => requestAnimationFrame(processFrames), FRAME_INTERVAL);
        }
    }
}

function resizeCanvases() {
    const videoWidth = video.videoWidth || 1280;
    const videoHeight = video.videoHeight || 720;
    dlog('Resizing canvases to:', videoWidth, 'x', videoHeight);
    
    video.width = videoWidth;
    video.height = videoHeight;
    video.style.width = videoWidth + 'px';
    video.style.height = videoHeight + 'px';
    
    const canvases = [roiCanvasRealtime, trackingCanvas, lineCanvas];
    canvases.forEach(canvas => {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        canvas.style.width = videoWidth + 'px';
        canvas.style.height = videoHeight + 'px';
        canvas.style.border = 'none';
    });

    if (roiRealtime && !isSelectingRoi) {
        const ctx = roiCanvasRealtime.getContext('2d');
        ctx.clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
        ctx.strokeStyle = '#2DD4BF';
        ctx.lineWidth = 3;
        ctx.fillStyle = 'rgba(45, 212, 191, 0.2)';
        ctx.fillRect(roiRealtime.x, roiRealtime.y, roiRealtime.width, roiRealtime.height);
        ctx.strokeRect(roiRealtime.x, roiRealtime.y, roiRealtime.width, roiRealtime.height);
    }
    drawLine();
}

function drawLine() {
    lineCtx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);
    if (lineX !== null) {
        const videoWidth = video.videoWidth || 1280;
        const x = lineX * videoWidth;
        lineCtx.strokeStyle = '#FF0000';
        lineCtx.lineWidth = 3;
        lineCtx.beginPath();
        lineCtx.moveTo(x, 0);
        lineCtx.lineTo(x, lineCanvas.height);
        lineCtx.stroke();
        dlog('Drew line at x:', x);
    }
}

video.addEventListener('loadedmetadata', () => {
    dlog('Video metadata loaded, resizing canvases');
    resizeCanvases();
});

window.addEventListener('resize', () => {
    resizeCanvases();
});

function resetRoiState() {
    dlog('Resetting ROI state');
    isSelectingRoi = false;
    roiRealtime = null;
    const ctx = roiCanvasRealtime.getContext('2d');
    ctx.clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
    roiCanvasRealtime.style.pointerEvents = 'none';
    roiCanvasRealtime.style.cursor = 'default';
    clearRoiButton.disabled = true;
    const newCanvas = roiCanvasRealtime.cloneNode(true);
    roiCanvasRealtime.parentNode.replaceChild(newCanvas, roiCanvasRealtime);
    roiCanvasRealtime = newCanvas;
    roiCanvasRealtime.setAttribute('tabindex', '0');
    roiCanvasRealtime.addEventListener('mousedown', handleMouseDown);
    roiCanvasRealtime.addEventListener('mousemove', handleMouseMove);
    roiCanvasRealtime.addEventListener('mouseup', handleMouseUp);
    roiCanvasRealtime.addEventListener('click', handleLineClick);
}

function resetLineState() {
    dlog('Resetting line state');
    isDrawingLine = false;
    lineX = null;
    lineCtx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);
    roiCanvasRealtime.style.pointerEvents = 'none';
    roiCanvasRealtime.style.cursor = 'default';
    clearLineButton.disabled = true;
}

function handleMouseDown(e) {
    if (!isSelectingRoi) {
        dlog('Ignoring mousedown: not in ROI selection mode');
        return;
    }
    dlog('Mouse down detected');
    const rect = roiCanvasRealtime.getBoundingClientRect();
    const scaleX = roiCanvasRealtime.width / rect.width;
    const scaleY = roiCanvasRealtime.height / rect.height;
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    dlog('ROI drawing started at:', startX, startY);
}

function handleMouseMove(e) {
    if (!isSelectingRoi || startX === undefined || startY === undefined) return;
    const ctx = roiCanvasRealtime.getContext('2d');
    const rect = roiCanvasRealtime.getBoundingClientRect();
    const scaleX = roiCanvasRealtime.width / rect.width;
    const scaleY = roiCanvasRealtime.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    ctx.clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
    ctx.strokeStyle = '#2DD4BF';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(45, 212, 191, 0.15)';
    const rectX = Math.min(startX, x);
    const rectY = Math.min(startY, y);
    const rectW = Math.abs(x - startX);
    const rectH = Math.abs(y - startY);
    ctx.fillRect(rectX, rectY, rectW, rectH);
    ctx.strokeRect(rectX, rectY, rectW, rectH);
}

function handleMouseUp(e) {
    if (!isSelectingRoi || startX === undefined || startY === undefined) {
        dlog('Ignoring mouseup: not in ROI selection mode');
        return;
    }
    dlog('Mouse up detected');
    isSelectingRoi = false;
    roiCanvasRealtime.style.pointerEvents = 'none';
    roiCanvasRealtime.style.cursor = 'default';
    document.body.focus();

    const rect = roiCanvasRealtime.getBoundingClientRect();
    const scaleX = roiCanvasRealtime.width / rect.width;
    const scaleY = roiCanvasRealtime.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const roiWidth = Math.abs(x - startX);
    const roiHeight = Math.abs(y - startY);
    roiRealtime = {
        x: Math.min(startX, x),
        y: Math.min(startY, y),
        width: roiWidth,
        height: roiHeight
    };

    const videoWidth = video.videoWidth || 1280;
    const videoHeight = video.videoHeight || 720;
    const minWidth = videoWidth * 0.05;
    const minHeight = videoHeight * 0.05;

    if (video.readyState >= 2 && (roiWidth < minWidth || roiHeight < minHeight)) {
        dlog('ROI too small:', roiWidth, 'x', roiHeight, 'min:', minWidth, 'x', minHeight);
        showModal(`ROI is too small. Please select a larger area (minimum ${minWidth.toFixed(0)}x${minHeight.toFixed(0)} pixels).`);
        resetRoiState();
        video.play();
        return;
    }

    const ctx = roiCanvasRealtime.getContext('2d');
    ctx.clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
    ctx.strokeStyle = '#2DD4BF';
    ctx.lineWidth = 3;
    ctx.fillStyle = 'rgba(45, 212, 191, 0.2)';
    ctx.fillRect(roiRealtime.x, roiRealtime.y, roiRealtime.width, roiRealtime.height);
    ctx.strokeRect(roiRealtime.x, roiRealtime.y, roiRealtime.width, roiRealtime.height);
    clearRoiButton.disabled = false;
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 
        `ROI selected: x=${roiRealtime.x.toFixed(0)}, y=${roiRealtime.y.toFixed(0)}, w=${roiRealtime.width.toFixed(0)}, h=${roiRealtime.height.toFixed(0)}`;
    dlog('ROI selected:', roiRealtime);
    startX = undefined;
    startY = undefined;
    video.play();
}

function handleLineClick(e) {
    if (!isDrawingLine) {
        dlog('Ignoring click: not in line drawing mode');
        return;
    }
    dlog('Line click detected');
    const rect = roiCanvasRealtime.getBoundingClientRect();
    const scaleX = roiCanvasRealtime.width / rect.width;
    const x = (e.clientX - rect.left) * scaleX;
    const videoWidth = video.videoWidth || 1280;
    lineX = x / videoWidth; // Normalize x-coordinate
    isDrawingLine = false;
    roiCanvasRealtime.style.pointerEvents = 'none';
    roiCanvasRealtime.style.cursor = 'default';
    document.body.focus();
    drawLine();
    clearLineButton.disabled = false;
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 
        `Line selected at x=${(lineX * videoWidth).toFixed(0)}`;
    dlog('Line selected at normalized x:', lineX);
    video.play();
}

selectRoiButton.addEventListener('click', () => {
    dlog('Select ROI clicked');
    resetRoiState();
    isSelectingRoi = true;
    roiCanvasRealtime.style.pointerEvents = 'auto';
    roiCanvasRealtime.style.cursor = 'crosshair';
    roiCanvasRealtime.getContext('2d').clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Draw ROI on video';
    roiCanvasRealtime.focus();
});

clearRoiButton.addEventListener('click', () => {
    dlog('Clear ROI clicked');
    resetRoiState();
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'ROI cleared';
});

drawLineButton.addEventListener('click', () => {
    dlog('Draw Line clicked');
    resetLineState();
    isDrawingLine = true;
    roiCanvasRealtime.style.pointerEvents = 'auto';
    roiCanvasRealtime.style.cursor = 'crosshair';
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Click to set vertical line';
    roiCanvasRealtime.focus();
});

clearLineButton.addEventListener('click', () => {
    dlog('Clear Line clicked');
    resetLineState();
    resetCountersUI();
    // Force config to be re-sent even if the current state matches the last sent state.
    lastConfigKey = null;
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Line cleared';
});

document.addEventListener('DOMContentLoaded', initializeWebcam);