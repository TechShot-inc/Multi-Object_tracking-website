let roiRealtime = null;
let isSelectingRoi = false;
let isTracking = false;
let videoStream = null;
let startX, startY;
let linePosition = 'left'; // Default to left line
let lineX = null; // User-defined line x-coordinate (normalized)
let isDrawingLine = false;

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

function initializeWebcam() {
    console.log('Initializing webcam UI');
    resetRoiState();
    resetLineState();
    resizeCanvases();
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Ready to start tracking';
    // Set up line position radio buttons
    document.querySelectorAll('input[name="line-position"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            linePosition = e.target.value;
            console.log('Line position changed to:', linePosition);
            drawLine();
        });
    });
}

function startTracking() {
    console.log('Starting tracking');
    document.getElementById('start-tracking').disabled = true;
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Starting camera...';
    roiCanvasRealtime.style.pointerEvents = 'none';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('Webcam access not supported');
        document.getElementById('start-tracking').disabled = false;
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Webcam access not supported';
        showModal('Webcam access not supported in this browser.');
        return;
    }

    navigator.mediaDevices.getUserMedia({ 
        video: { 
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            facingMode: "user"
        } 
    }).then(stream => {
        console.log('Webcam stream acquired');
        video.srcObject = stream;
        videoStream = stream;
        isTracking = true;
        
        trackingCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
        roiCanvasRealtime.getContext('2d').clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
        lineCtx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);

        video.onloadeddata = function() {
            console.log('Video data loaded, width:', video.videoWidth || 1280, 'height:', video.videoHeight || 720);
            video.play().then(() => {
                console.log('Video playback started');
                resizeCanvases();
                drawLine();
                document.getElementById('stop-tracking').disabled = false;
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Camera active, starting tracking...';
                requestAnimationFrame(processFrames);
            }).catch(err => {
                console.error('Video play error:', err);
                document.getElementById('start-tracking').disabled = false;
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error playing video';
                showModal('Error playing webcam video: ' + err.message);
                stopTracking();
            });
        };
    }).catch(err => {
        console.error('Webcam access error:', err);
        document.getElementById('start-tracking').disabled = false;
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error accessing webcam';
        showModal('Error accessing webcam: ' + err.message);
    });
}

function stopTracking() {
    console.log('Stopping tracking');
    isTracking = false;
    document.getElementById('start-tracking').disabled = false;
    document.getElementById('stop-tracking').disabled = true;
    document.getElementById('realtime-loading').classList.add('hidden');
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Ready to start tracking';
    
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
const FRAME_INTERVAL = 1000 / 30;

function processFrames(timestamp) {
    if (!isTracking) {
        console.log('Tracking stopped, exiting processFrames');
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
    console.log('Processing frame at timestamp:', timestamp);

    if (trackingCanvas.width !== (video.videoWidth || 1280) || trackingCanvas.height !== (video.videoHeight || 720)) {
        resizeCanvases();
        console.log('Resized canvases to match video:', video.videoWidth || 1280, 'x', video.videoHeight || 720);
        drawLine();
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth || 1280;
    tempCanvas.height = video.videoHeight || 720;
    const tempCtx = tempCanvas.getContext('2d');

    try {
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        console.log('Captured video frame');

        tempCanvas.toBlob(async blob => {
            if (!blob) {
                console.error('Failed to create blob from canvas');
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error capturing frame';
                showModal('Failed to create frame data');
                if (isTracking) {
                    setTimeout(() => requestAnimationFrame(processFrames), 1000 / 30);
                }
                return;
            }

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
                console.log('Sending normalized ROI:', normalizedRoi);
            }
            if (lineX !== null) {
                const lineData = {
                    position: linePosition,
                    x: lineX
                };
                formData.append('line', JSON.stringify(lineData));
                console.log('Sending line data:', lineData);
            }

            try {
                console.log('Sending frame to /realtime-track');
                const response = await fetch('/realtime-track', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                console.log('Received tracking response:', data);

                if (data.error) {
                    throw new Error(data.error);
                }

                if (!data.annotated) {
                    throw new Error('No annotated frame in response');
                }

                // Update counts
                if (data.counts) {
                    personsInsideSpan.textContent = data.counts.inside || 0;
                    personsOutsideSpan.textContent = data.counts.outside || 0;
                }

                // Display model paths if available
                if (data.model_paths) {
                    const modelInfo = `Models: General2 (${data.model_paths.yolo1.split('/').pop()}), 12General1 (${data.model_paths.yolo2.split('/').pop()}), ReID (${data.model_paths.reid.split('/').pop()})`;
                    document.getElementById('realtime-status').querySelector('.status-message').textContent = 
                        `Tracking... (${data.count || 0} objects) | ${modelInfo}`;
                } else {
                    document.getElementById('realtime-status').querySelector('.status-message').textContent = 
                        `Tracking... (${data.count || 0} objects)`;
                }

                const img = new Image();
                img.onload = () => {
                    trackingCtx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
                    trackingCtx.drawImage(img, 0, 0);
                    trackingCanvas.style.visibility = 'visible';
                    console.log('Rendered annotated frame, object count:', data.count || 0);
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

                img.src = 'data:image/jpeg;base64,' + data.annotated;
            } catch (error) {
                console.error('Error processing frame:', error);
                document.getElementById('realtime-loading').classList.remove('hidden');
                document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Tracking error';
                showModal('Tracking error: ' + error.message);
                if (isTracking) {
                    setTimeout(() => requestAnimationFrame(processFrames), 1000 / 30);
                }
            }
        }, 'image/jpeg', 0.9);
    } catch (error) {
        console.error('Error capturing frame:', error);
        document.getElementById('realtime-loading').classList.remove('hidden');
        document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Error capturing frame';
        showModal('Error capturing frame: ' + error.message);
        if (isTracking) {
            setTimeout(() => requestAnimationFrame(processFrames), 1000 / 30);
        }
    }
}

function resizeCanvases() {
    const videoWidth = video.videoWidth || 1280;
    const videoHeight = video.videoHeight || 720;
    console.log('Resizing canvases to:', videoWidth, 'x', videoHeight);
    
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
        console.log('Drew line at x:', x);
    }
}

video.addEventListener('loadedmetadata', () => {
    console.log('Video metadata loaded, resizing canvases');
    resizeCanvases();
});

window.addEventListener('resize', () => {
    resizeCanvases();
});

function resetRoiState() {
    console.log('Resetting ROI state');
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
    console.log('Resetting line state');
    isDrawingLine = false;
    lineX = null;
    lineCtx.clearRect(0, 0, lineCanvas.width, lineCanvas.height);
    roiCanvasRealtime.style.pointerEvents = 'none';
    roiCanvasRealtime.style.cursor = 'default';
    clearLineButton.disabled = true;
}

function handleMouseDown(e) {
    if (!isSelectingRoi) {
        console.log('Ignoring mousedown: not in ROI selection mode');
        return;
    }
    console.log('Mouse down detected');
    const rect = roiCanvasRealtime.getBoundingClientRect();
    const scaleX = roiCanvasRealtime.width / rect.width;
    const scaleY = roiCanvasRealtime.height / rect.height;
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    console.log('ROI drawing started at:', startX, startY);
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
        console.log('Ignoring mouseup: not in ROI selection mode');
        return;
    }
    console.log('Mouse up detected');
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
        console.log('ROI too small:', roiWidth, 'x', roiHeight, 'min:', minWidth, 'x', minHeight);
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
    console.log('ROI selected:', roiRealtime);
    startX = undefined;
    startY = undefined;
    video.play();
}

function handleLineClick(e) {
    if (!isDrawingLine) {
        console.log('Ignoring click: not in line drawing mode');
        return;
    }
    console.log('Line click detected');
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
    console.log('Line selected at normalized x:', lineX);
    video.play();
}

selectRoiButton.addEventListener('click', () => {
    console.log('Select ROI clicked');
    resetRoiState();
    isSelectingRoi = true;
    roiCanvasRealtime.style.pointerEvents = 'auto';
    roiCanvasRealtime.style.cursor = 'crosshair';
    roiCanvasRealtime.getContext('2d').clearRect(0, 0, roiCanvasRealtime.width, roiCanvasRealtime.height);
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Draw ROI on video';
    roiCanvasRealtime.focus();
});

clearRoiButton.addEventListener('click', () => {
    console.log('Clear ROI clicked');
    resetRoiState();
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'ROI cleared';
});

drawLineButton.addEventListener('click', () => {
    console.log('Draw Line clicked');
    resetLineState();
    isDrawingLine = true;
    roiCanvasRealtime.style.pointerEvents = 'auto';
    roiCanvasRealtime.style.cursor = 'crosshair';
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Click to set vertical line';
    roiCanvasRealtime.focus();
});

clearLineButton.addEventListener('click', () => {
    console.log('Clear Line clicked');
    resetLineState();
    document.getElementById('realtime-status').querySelector('.status-message').textContent = 'Line cleared';
});

document.addEventListener('DOMContentLoaded', initializeWebcam);