document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const statusContainer = document.getElementById('upload-status');
    const statusMessage = statusContainer.querySelector('.status-message');
    const progressBar = statusContainer.querySelector('.progress-bar');
    const resultsContainer = document.getElementById('results-container');
    const resultVideo = document.getElementById('result-video');
    const replayButton = document.getElementById('replay-video');
    const downloadVideoButton = document.getElementById('download-video');
    const downloadAnnotationsButton = document.getElementById('download-annotations');
    const analyticsSection = document.getElementById('analytics-section');
    const heatmapImage = document.getElementById('heatmap-image');
    const heatmapInfo = document.getElementById('heatmap-info');
    const velocityHeatmapImage = document.getElementById('velocity-heatmap-image');
    const velocityHeatmapInfo = document.getElementById('velocity-heatmap-info');
    const objectCountChartCanvas = document.getElementById('object-count-chart');
    const trackDurationChartCanvas = document.getElementById('track-duration-chart');
    const topIdsContainer = document.getElementById('top-ids-container');
    const avgVelocityElement = document.getElementById('avg-velocity');
    const downloadReportButton = document.getElementById('download-report');
    const roiCanvasUpload = document.getElementById('roi-canvas-upload');
    const uploadPreview = document.getElementById('upload-preview');
    const roiInstruction = document.getElementById('roi-instruction');
    const roiContainerUpload = document.getElementById('roi-container-upload');
    const clearRoiButton = document.getElementById('clear-roi-upload');
    let resultId = null;
    let objectCountChart = null;
    let trackDurationChart = null;
    let roiContextUpload = null;
    let videoWidth = 0;
    let videoHeight = 0;

    // Setup clear ROI button
    if (clearRoiButton) {
        clearRoiButton.addEventListener('click', () => {
            if (roiContextUpload && uploadPreview) {
                window.roiUpload = null;
                roiContextUpload.clearRect(0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
                if (uploadPreview.videoWidth) {
                    roiContextUpload.drawImage(uploadPreview, 0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
                }
                clearRoiButton.disabled = true;
                if (roiInstruction) {
                    roiInstruction.textContent = 'ROI cleared. Draw a new ROI or submit without ROI.';
                }
                console.log('ROI cleared');
            }
        });
    }

    // Initialize ROI canvas for upload
    if (roiCanvasUpload && uploadPreview) {
        roiContextUpload = roiCanvasUpload.getContext('2d');
        roiCanvasUpload.addEventListener('mousedown', startDrawingROI);
        roiCanvasUpload.addEventListener('mousemove', drawROI);
        roiCanvasUpload.addEventListener('mouseup', finishDrawingROI);
        console.log('ROI canvas for upload initialized');

        uploadPreview.addEventListener('loadedmetadata', () => {
            console.log('Video metadata loaded');
            resizeRoiCanvas();
        });

        function resizeRoiCanvas() {
            if (uploadPreview.videoWidth && uploadPreview.videoHeight) {
                roiCanvasUpload.width = uploadPreview.videoWidth;
                roiCanvasUpload.height = uploadPreview.videoHeight;
                roiCanvasUpload.style.width = uploadPreview.offsetWidth + 'px';
                roiCanvasUpload.style.height = uploadPreview.offsetHeight + 'px';
                videoWidth = uploadPreview.videoWidth;
                videoHeight = uploadPreview.videoHeight;
                console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
                setTimeout(() => {
                    roiContextUpload.drawImage(uploadPreview, 0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
                }, 100);
            }
        }

        const videoFileInput = document.getElementById('video-file');
        if (videoFileInput) {
            videoFileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    const file = e.target.files[0];
                    if (file.size > 500 * 1024 * 1024) { // 500MB limit
                        showModal('Video file is too large. Please select a file under 500MB.');
                        e.target.value = '';
                        return;
                    }
                    uploadPreview.src = URL.createObjectURL(file);
                    roiContainerUpload.classList.remove('hidden');
                    window.roiUpload = null;
                    if (roiInstruction) {
                        roiInstruction.textContent = 'Click "Select ROI" to draw on the video.';
                    }
                }
            });
        }
    }

    let drawing = false;
    let roiStart = null;
    window.roiUpload = null;

    function startDrawingROI(e) {
        if (!uploadPreview.videoWidth) {
            showModal('Please wait for the video to load before selecting ROI.');
            return;
        }
        const canvasRect = roiCanvasUpload.getBoundingClientRect();
        const scaleX = roiCanvasUpload.width / canvasRect.width;
        const scaleY = roiCanvasUpload.height / canvasRect.height;
        const rawX = (e.clientX - canvasRect.left) * scaleX;
        const rawY = (e.clientY - canvasRect.top) * scaleY;
        roiStart = {
            x: Math.floor(rawX),
            y: Math.floor(rawY)
        };
        drawing = true;
        console.log('Started drawing ROI at:', roiStart);
        uploadPreview.pause();
    }

    function drawROI(e) {
        if (!drawing) return;
        const canvasRect = roiCanvasUpload.getBoundingClientRect();
        const scaleX = roiCanvasUpload.width / canvasRect.width;
        const scaleY = roiCanvasUpload.height / canvasRect.height;
        const rawX = (e.clientX - canvasRect.left) * scaleX;
        const rawY = (e.clientY - canvasRect.top) * scaleY;
        const x = Math.floor(rawX);
        const y = Math.floor(rawY);
        roiContextUpload.clearRect(0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
        if (uploadPreview.videoWidth) {
            roiContextUpload.drawImage(uploadPreview, 0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
        }
        roiContextUpload.strokeStyle = '#2DD4BF';
        roiContextUpload.lineWidth = 2;
        roiContextUpload.fillStyle = 'rgba(45, 212, 191, 0.15)';
        const rectX = Math.min(roiStart.x, x);
        const rectY = Math.min(roiStart.y, y);
        const rectW = Math.abs(x - roiStart.x);
        const rectH = Math.abs(y - roiStart.y);
        roiContextUpload.fillRect(rectX, rectY, rectW, rectH);
        roiContextUpload.strokeRect(rectX, rectY, rectW, rectH);
    }

    function finishDrawingROI(e) {
        if (!drawing) return;
        drawing = false;
        const canvasRect = roiCanvasUpload.getBoundingClientRect();
        const scaleX = roiCanvasUpload.width / canvasRect.width;
        const scaleY = roiCanvasUpload.height / canvasRect.height;
        const rawX = (e.clientX - canvasRect.left) * scaleX;
        const rawY = (e.clientY - canvasRect.top) * scaleY;
        const x = Math.floor(rawX);
        const y = Math.floor(rawY);
        const roiWidth = Math.abs(x - roiStart.x);
        const roiHeight = Math.abs(y - roiStart.y);
        window.roiUpload = {
            x: Math.floor(Math.min(roiStart.x, x)),
            y: Math.floor(Math.min(roiStart.y, y)),
            width: Math.floor(roiWidth),
            height: Math.floor(roiHeight)
        };
        const minWidth = Math.floor(videoWidth * 0.05);
        const minHeight = Math.floor(videoHeight * 0.05);
        if (window.roiUpload.width < minWidth || window.roiUpload.height < minHeight) {
            showModal(`ROI is too small. Please select a larger area (minimum ${minWidth}x${minHeight} pixels).`);
            window.roiUpload = null;
            clearRoiButton.disabled = true;
            return;
        }
        roiContextUpload.clearRect(0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
        if (uploadPreview.videoWidth) {
            roiContextUpload.drawImage(uploadPreview, 0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
        }
        roiContextUpload.strokeStyle = '#2DD4BF';
        roiContextUpload.lineWidth = 3;
        roiContextUpload.fillStyle = 'rgba(45, 212, 191, 0.2)';
        roiContextUpload.fillRect(window.roiUpload.x, window.roiUpload.y, window.roiUpload.width, window.roiUpload.height);
        roiContextUpload.strokeRect(window.roiUpload.x, window.roiUpload.y, window.roiUpload.width, window.roiUpload.height);
        console.log('ROI selected:', window.roiUpload);
        clearRoiButton.disabled = false;
        if (roiInstruction) {
            roiInstruction.textContent = 'ROI selected. Submit to process video.';
        }
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);
        if (window.roiUpload && window.roiUpload.width > 0 && window.roiUpload.height > 0) {
            const roiData = {
                x: window.roiUpload.x,
                y: window.roiUpload.y,
                width: window.roiUpload.width,
                height: window.roiUpload.height
            };
            console.log('Sending ROI data:', roiData);
            formData.append('roi', JSON.stringify(roiData));
        } else {
            console.log('No valid ROI selected');
        }
        uploadButton.disabled = true;
        statusMessage.textContent = 'Uploading...';
        progressBar.style.width = '10%';
        progressBar.setAttribute('aria-valuenow', 10);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            const data = await response.json();
            console.log('Upload response:', data);
            if (data.status === 'success') {
                statusMessage.textContent = 'Processing video...';
                progressBar.style.width = '30%';
                progressBar.setAttribute('aria-valuenow', 30);
                await pollStatus(data.filename);
            } else {
                throw new Error(data.message || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            showModal(`Error uploading video: ${error.message}`);
            resetUploadForm();
        }
    });

    async function pollStatus(filename) {
        try {
            const response = await fetch(`/results/${filename}/status`);
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.statusText}`);
            }
            const data = await response.json();
            console.log('Status:', data);
            if (data.status === 'completed') {
                const videoResponse = await fetch(`/results/${filename}/video`);
                if (!videoResponse.ok) {
                    console.log('Video not ready yet, continuing to poll...');
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    return pollStatus(filename);
                }
                statusMessage.textContent = 'Processing complete!';
                progressBar.style.width = '100%';
                progressBar.setAttribute('aria-valuenow', 100);
                await displayResults(filename);
            } else if (data.status === 'error') {
                throw new Error(data.message || 'Processing failed');
            } else if (data.status === 'processing') {
                statusMessage.textContent = data.message || 'Processing...';
                const progress = data.progress || 0;
                progressBar.style.width = `${progress * 100}%`;
                progressBar.setAttribute('aria-valuenow', progress * 100);
                await new Promise(resolve => setTimeout(resolve, 2000));
                await pollStatus(filename);
            } else {
                throw new Error(data.message || 'Unknown processing status');
            }
        } catch (error) {
            console.error('Status check error:', error);
            if (error.message.includes('Video not ready') || error.message.includes('Video processing not complete')) {
                await new Promise(resolve => setTimeout(resolve, 2000));
                await pollStatus(filename);
            } else {
                showModal(`Error processing video: ${error.message}`);
                resetUploadForm();
            }
        }
    }

    async function displayResults(filename) {
        try {
            resultsContainer.classList.remove('hidden');
            resultVideo.src = `/results/${filename}/video`;
            resultVideo.load();
            resultVideo.controls = true;
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Video loading timed out after 60 seconds'));
                }, 60000);
                resultVideo.onloadeddata = () => {
                    clearTimeout(timeout);
                    resolve();
                };
                resultVideo.onerror = () => {
                    clearTimeout(timeout);
                    reject(new Error('Failed to load video'));
                };
            });
            replayButton.onclick = () => {
                resultVideo.currentTime = 0;
                resultVideo.play();
            };
            downloadVideoButton.onclick = (e) => {
                e.preventDefault();
                const link = document.createElement('a');
                link.href = `/results/${filename}/video?download=1`;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };
            downloadAnnotationsButton.onclick = (e) => {
                e.preventDefault();
                const link = document.createElement('a');
                link.href = `/results/${filename}/annotations?download=1`;
                link.download = `${filename.split('.')[0]}_annotations.json`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };
            await fetchAnalytics(filename);
            resetUploadForm();
        } catch (error) {
            console.error('Error displaying results:', error);
            if (error.message.includes('Video loading timed out') || error.message.includes('Failed to load video')) {
                await new Promise(resolve => setTimeout(resolve, 2000));
                await pollStatus(filename);
            } else {
                showModal(`Error displaying results: ${error.message}`);
                resetUploadForm();
            }
        }
    }

    async function fetchAnalytics(filename) {
        try {
            const response = await fetch(`/results/${filename}/analytics`);
            if (!response.ok) {
                throw new Error(`Analytics fetch failed: ${response.statusText}`);
            }
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
            if (data.heatmap) {
                heatmapImage.src = `data:image/png;base64,${data.heatmap}`;
                heatmapInfo.classList.remove('hidden');
            }
            if (data.velocity_heatmap) {
                velocityHeatmapImage.src = `data:image/png;base64,${data.velocity_heatmap}`;
                velocityHeatmapInfo.classList.remove('hidden');
            }
            if (data.object_counts) {
                const frames = Object.keys(data.object_counts).map(Number);
                const counts = Object.values(data.object_counts);
                const maxPoints = 1000;
                let sampledFrames = frames;
                let sampledCounts = counts;
                if (frames.length > maxPoints) {
                    const step = Math.floor(frames.length / maxPoints);
                    sampledFrames = frames.filter((_, i) => i % step === 0);
                    sampledCounts = counts.filter((_, i) => i % step === 0);
                }
                if (objectCountChart) {
                    objectCountChart.destroy();
                }
                objectCountChart = new Chart(objectCountChartCanvas, {
                    type: 'line',
                    data: {
                        labels: sampledFrames,
                        datasets: [{
                            label: 'Objects per Frame',
                            data: sampledCounts,
                            borderColor: '#2DD4BF',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Number of Objects'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Frame Number'
                                }
                            }
                        }
                    }
                });
            }
            if (data.track_durations) {
                const ids = Object.keys(data.track_durations);
                const durations = Object.values(data.track_durations);
                if (trackDurationChart) {
                    trackDurationChart.destroy();
                }
                trackDurationChart = new Chart(trackDurationChartCanvas, {
                    type: 'bar',
                    data: {
                        labels: ids,
                        datasets: [{
                            label: 'Track Duration (frames)',
                            data: durations,
                            backgroundColor: '#2DD4BF'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Duration (frames)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Track ID'
                                }
                            }
                        }
                    }
                });
            }
            if (data.top_ids && data.crops) {
                topIdsContainer.innerHTML = '';
                data.top_ids.forEach((id, index) => {
                    const idDiv = document.createElement('div');
                    idDiv.className = 'top-id-item';
                    idDiv.innerHTML = `
                        <h4>Track ID: ${id}</h4>
                        <div class="crops-container">
                            ${data.crops[index].map(crop => `
                                <img src="data:image/jpeg;base64,${crop}" alt="Track ${id} crop">
                            `).join('')}
                        </div>
                    `;
                    topIdsContainer.appendChild(idDiv);
                });
            }
            if (data.avg_velocity !== undefined) {
                avgVelocityElement.textContent = data.avg_velocity.toFixed(2);
            }
            analyticsSection.classList.remove('hidden');
        } catch (error) {
            console.error('Error fetching analytics:', error);
            showModal(`Error loading analytics: ${error.message}`);
            analyticsSection.classList.add('hidden');
        }
    }

    function resetUploadForm() {
        uploadButton.disabled = false;
        statusMessage.textContent = 'Ready to upload';
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        uploadForm.reset();
        roiContainerUpload.classList.add('hidden');
        window.roiUpload = null;
        clearRoiButton.disabled = true;
        if (roiInstruction) {
            roiInstruction.textContent = 'Upload a video to select ROI.';
        }
        if (uploadPreview.src) {
            URL.revokeObjectURL(uploadPreview.src);
            uploadPreview.src = '';
            roiContextUpload.clearRect(0, 0, roiCanvasUpload.width, roiCanvasUpload.height);
        }
    }

    function showModal(message) {
        document.getElementById('modal-message').textContent = message;
        document.getElementById('modal').classList.remove('hidden');
    }
});