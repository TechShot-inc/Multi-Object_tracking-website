document.addEventListener('DOMContentLoaded', async () => {
    // DOM elements
    const cameraSelect = document.getElementById('camera-select');
    const startTrackingBtn = document.getElementById('start-tracking');
    const stopTrackingBtn = document.getElementById('stop-tracking');
    const videoElement = document.getElementById('camera-feed');
    const canvas = document.getElementById('tracking-overlay');
    const objectCountElement = document.getElementById('object-count');
    const fpsElement = document.getElementById('fps');
    
    // Canvas context for drawing
    const ctx = canvas.getContext('2d');
    
    // Track state
    let isTracking = false;
    let model = null;
    let videoStream = null;
    let animationFrameId = null;
    let trackers = [];
    let lastFrameTime = 0;
    
    // Colors for different object IDs
    const colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', 
        '#FF00FF', '#00FFFF', '#FFA500', '#800080',
        '#008000', '#000080', '#800000', '#808000'
    ];
    
    // Kalman filter settings
    const kalmanParams = {
        R: 0.01, // Process noise
        Q: 0.1,  // Measurement noise
        A: 1,    // State transition matrix
        B: 0,    // Control matrix
        C: 1     // Measurement matrix
    };
    
    // Initialize cameras dropdown
    async function initializeCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            // Clear previous options
            cameraSelect.innerHTML = '';
            
            if (videoDevices.length === 0) {
                const option = document.createElement('option');
                option.text = 'No cameras found';
                cameraSelect.add(option);
                startTrackingBtn.disabled = true;
            } else {
                videoDevices.forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.text = device.label || `Camera ${cameraSelect.length + 1}`;
                    cameraSelect.add(option);
                });
                startTrackingBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error accessing media devices:', error);
            const option = document.createElement('option');
            option.text = 'Error loading cameras';
            cameraSelect.add(option);
            startTrackingBtn.disabled = true;
        }
    }
    
    // Load TensorFlow.js model
    async function loadModel() {
        try {
            // Use TensorFlow.js COCO-SSD model (pre-trained)
            // This is a placeholder - in production you'd use your specific model
            model = await cocoSsd.load();
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load model:', error);
            return false;
        }
    }
    
    // Start camera and tracking
    async function startTracking() {
        if (isTracking) return;
        
        try {
            // Ensure model is loaded
            if (!model) {
                // Dynamically load COCO-SSD model script if not already loaded
                if (!window.cocoSsd) {
                    await new Promise((resolve, reject) => {
                        const script = document.createElement('script');
                        script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd';
                        script.onload = resolve;
                        script.onerror = reject;
                        document.head.appendChild(script);
                    });
                }
                
                const modelLoaded = await loadModel();
                if (!modelLoaded) {
                    alert('Failed to load detection model. Please try again.');
                    return;
                }
            }
            
            // Get selected camera
            const deviceId = cameraSelect.value;
            
            // Start video stream
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            // Set video source
            videoElement.srcObject = videoStream;
            
            // Wait for video to be ready
            await new Promise(resolve => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play();
                    resolve();
                };
            });
            
            // Set canvas size to match video
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            
            // Update UI
            isTracking = true;
            startTrackingBtn.disabled = true;
            stopTrackingBtn.disabled = false;
            trackers = [];
            
            // Start detection loop
            lastFrameTime = performance.now();
            detectFrame();
            
        } catch (error) {
            console.error('Error starting tracking:', error);
            alert(`Error starting camera: ${error.message}`);
        }
    }
    
    // Stop tracking and release camera
    function stopTracking() {
        if (!isTracking) return;
        
        // Stop animation frame
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        
        // Stop video stream
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
            videoElement.srcObject = null;
        }
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Update UI
        isTracking = false;
        startTrackingBtn.disabled = false;
        stopTrackingBtn.disabled = true;
        objectCountElement.textContent = '0';
        fpsElement.textContent = '0';
    }
    
    // Calculate IoU (Intersection over Union) for bounding boxes
    function calculateIoU(box1, box2) {
        const x1 = Math.max(box1.x, box2.x);
        const y1 = Math.max(box1.y, box2.y);
        const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
        const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
        
        if (x2 < x1 || y2 < y1) return 0;
        
        const intersection = (x2 - x1) * (y2 - y1);
        const box1Area = box1.width * box1.height;
        const box2Area = box2.width * box2.height;
        
        return intersection / (box1Area + box2Area - intersection);
    }
    
    // Hungarian algorithm for assignment
    function hungarianAssignment(costMatrix) {
        // This is a simplified version of the Hungarian algorithm
        // In a real application, you'd use a proper implementation like munkres-js
        
        const assignments = [];
        const rows = costMatrix.length;
        const cols = costMatrix[0]?.length || 0;
        
        // Simple greedy assignment - not optimal but works for this demo
        const assignedCols = new Set();
        
        for (let r = 0; r < rows; r++) {
            let minCost = Infinity;
            let bestCol = -1;
            
            for (let c = 0; c < cols; c++) {
                if (!assignedCols.has(c) && costMatrix[r][c] < minCost) {
                    minCost = costMatrix[r][c];
                    bestCol = c;
                }
            }
            
            // Only assign if IoU is above threshold (represented by cost below threshold)
            if (minCost < 0.5) {  // IoU threshold of 0.5
                assignedCols.add(bestCol);
                assignments.push([r, bestCol]);
            }
        }
        
        return assignments;
    }
    
    // Main detection and tracking function
    async function detectFrame() {
        if (!isTracking) return;
        
        try {
            // Calculate FPS
            const now = performance.now();
            const fps = 1000 / (now - lastFrameTime);
            lastFrameTime = now;
            fpsElement.textContent = Math.round(fps);
            
            // Predict using the model
            const predictions = await model.detect(videoElement);
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Prepare detected objects
            const detections = predictions.map(pred => ({
                x: pred.bbox[0],
                y: pred.bbox[1],
                width: pred.bbox[2],
                height: pred.bbox[3],
                class: pred.class,
                score: pred.score
            }));
            
            // Filter low-confidence detections
            const filteredDetections = detections.filter(d => d.score > 0.5);
            
            // Predict next positions of existing trackers (Kalman filter prediction step)
            trackers.forEach(tracker => {
                // Simple prediction (would use Kalman filter in production)
                tracker.predicted = {
                    x: tracker.x + tracker.vx,
                    y: tracker.y + tracker.vy,
                    width: tracker.width,
                    height: tracker.height
                };
            });
            
            // Calculate cost matrix (1 - IoU)
            const costMatrix = trackers.map(tracker => 
                filteredDetections.map(detection => 
                    1 - calculateIoU(tracker.predicted, detection)
                )
            );
            
            // Perform assignment using Hungarian algorithm
            const assignments = costMatrix.length > 0 && costMatrix[0]?.length > 0 ? 
                hungarianAssignment(costMatrix) : [];
            
            // Mark assigned trackers and detections
            const assignedTrackerIndices = new Set();
            const assignedDetectionIndices = new Set();
            
            // Update assigned trackers
            assignments.forEach(([trackerIdx, detectionIdx]) => {
                assignedTrackerIndices.add(trackerIdx);
                assignedDetectionIndices.add(detectionIdx);
                
                const tracker = trackers[trackerIdx];
                const detection = filteredDetections[detectionIdx];
                
                // Calculate velocity
                const vx = (detection.x - tracker.x) * 0.5; // Smooth velocity
                const vy = (detection.y - tracker.y) * 0.5;
                
                // Update tracker with detection
                tracker.x = detection.x;
                tracker.y = detection.y;
                tracker.width = detection.width;
                tracker.height = detection.height;
                tracker.vx = vx;
                tracker.vy = vy;
                tracker.age++;
                tracker.totalVisible++;
                tracker.consecutiveInvisible = 0;
            });
            
            // Handle unassigned trackers
            trackers.forEach((tracker, idx) => {
                if (!assignedTrackerIndices.has(idx)) {
                    // If not assigned, increment invisible count and predict position
                    tracker.x += tracker.vx;
                    tracker.y += tracker.vy;
                    tracker.consecutiveInvisible++;
                    tracker.age++;
                }
            });
            
            // Create new trackers for unassigned detections
            filteredDetections.forEach((detection, idx) => {
                if (!assignedDetectionIndices.has(idx)) {
                    trackers.push({
                        id: Date.now() + idx, // Unique ID
                        x: detection.x,
                        y: detection.y,
                        width: detection.width,
                        height: detection.height,
                        vx: 0,
                        vy: 0,
                        class: detection.class,
                        age: 1,
                        totalVisible: 1,
                        consecutiveInvisible: 0
                    });
                }
            });
            
            // Remove trackers that have been invisible for too long
            trackers = trackers.filter(tracker => tracker.consecutiveInvisible < 10);
            
            // Draw trackers
            trackers.forEach(tracker => {
                // Get color by ID
                const colorIdx = tracker.id % colors.length;
                ctx.strokeStyle = colors[colorIdx];
                ctx.lineWidth = 2;
                
                // Draw bounding box
                ctx.strokeRect(tracker.x, tracker.y, tracker.width, tracker.height);
                
                // Draw ID and class
                ctx.fillStyle = colors[colorIdx];
                ctx.font = '16px Arial';
                ctx.fillText(`ID: ${tracker.id % 1000} ${tracker.class}`, tracker.x, tracker.y - 5);
            });
            
            // Update tracker count
            objectCountElement.textContent = trackers.length.toString();
            
        } catch (error) {
            console.error('Error in detection frame:', error);
        }
        
        // Request next frame
        animationFrameId = requestAnimationFrame(detectFrame);
    }
    
    // Event listeners
    startTrackingBtn.addEventListener('click', startTracking);
    stopTrackingBtn.addEventListener('click', stopTracking);
    
    // Initialize cameras on page load
    initializeCameras();
    
    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        stopTracking();
    });
});