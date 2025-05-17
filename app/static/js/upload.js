document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const videoFile = document.getElementById('video-file');
    const uploadButton = document.getElementById('upload-button');
    const statusContainer = document.getElementById('upload-status');
    const statusMessage = statusContainer.querySelector('.status-message');
    const progressBar = statusContainer.querySelector('.progress-bar');
    const resultsContainer = document.getElementById('results-container');
    const resultVideo = document.getElementById('result-video');
    const downloadVideoBtn = document.getElementById('download-video');
    const downloadAnnotationsBtn = document.getElementById('download-annotations');
    
    let resultId = null;
    let statusCheckInterval = null;
    
    videoFile.addEventListener('change', () => {
        const fileName = videoFile.files[0] ? videoFile.files[0].name : 'Choose a video file';
        const fileLabel = videoFile.nextElementSibling;
        fileLabel.textContent = fileName;
    });
    
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!videoFile.files[0]) {
            alert('Please select a video file first.');
            return;
        }
        
        const file = videoFile.files[0];
        const formData = new FormData();
        formData.append('video', file);
        const speedValue = document.getElementById('speed').value;
        formData.append('speed', speedValue);
        const outputSpeedValue = document.getElementById('output_speed').value;
        formData.append('output_speed', outputSpeedValue);
        
        uploadButton.disabled = true;
        uploadButton.textContent = 'Uploading...';
        
        statusMessage.textContent = 'Uploading video...';
        progressBar.style.width = '0%';
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progress <= 100) {
                    progressBar.style.width = `${progress}%`;
                }
                if (progress >= 100) {
                    clearInterval(progressInterval);
                }
            }, 200);
            
            const data = await response.json();
            
            if (data.status === 'success') {
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                statusMessage.textContent = 'Video uploaded. Processing...';
                
                resultId = data.result_id;
                checkProcessingStatus();
            } else {
                throw new Error(data.message || 'Failed to upload video');
            }
        } catch (error) {
            statusMessage.textContent = `Error: ${error.message}`;
            uploadButton.disabled = false;
            uploadButton.textContent = 'Upload & Process';
        }
    });
    
    function checkProcessingStatus() {
        if (!resultId) return;
        
        statusCheckInterval = setInterval(async () => {
            try {
                const response = await fetch(`/results/${resultId}/status`);
                const data = await response.json();
                
                if (data.status === 'completed') {
                    statusMessage.textContent = 'Processing complete!';
                    clearInterval(statusCheckInterval);
                    showResults();
                } else if (data.status === 'failed') {
                    statusMessage.textContent = `Processing failed: ${data.error || 'Unknown error'}`;
                    clearInterval(statusCheckInterval);
                    uploadButton.disabled = false;
                    uploadButton.textContent = 'Upload & Process';
                }
            } catch (error) {
                console.error('Error checking status:', error);
            }
        }, 2000);
    }
    
    async function showResults() {
        if (!resultId) return;
        
        resultsContainer.style.display = 'block';
        resultVideo.src = `/results/${resultId}/video`;
        
        try {
            const response = await fetch(`/results/${resultId}/annotations`);
            const annotations = await response.json();
            console.log(`Loaded annotations with ${Object.keys(annotations).length} frames`);
            overlayAnnotations(annotations);
        } catch (error) {
            console.error("Error loading annotations:", error);
            statusMessage.textContent = "Error loading annotations.";
        }
        
        resultVideo.onerror = function() {
            console.error("Error loading video");
            statusMessage.textContent = "Error loading video. Trying alternative player...";
            showAltPlayer(`/results/${resultId}/video`);
        };
        
        resultVideo.onloadeddata = function() {
            console.log("Video loaded successfully");
        };
        
        resultVideo.load();
        
        uploadButton.disabled = false;
        uploadButton.textContent = 'Upload & Process';
        
        downloadVideoBtn.onclick = () => {
            window.open(`/results/${resultId}/video`, '_blank');
        };
        downloadAnnotationsBtn.onclick = () => {
            window.open(`/results/${resultId}/annotations`, '_blank');
        };
    }
    
    function overlayAnnotations(annotations) {
        const canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = resultVideo.offsetTop + 'px';
        canvas.style.left = resultVideo.offsetLeft + 'px';
        canvas.width = resultVideo.videoWidth;
        canvas.height = resultVideo.videoHeight;
        resultVideo.parentElement.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        resultVideo.addEventListener('timeupdate', () => {
            const frameId = Math.floor(resultVideo.currentTime * 30) + 1; // Assuming 30 FPS
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (annotations[frameId]) {
                annotations[frameId].forEach(det => {
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(det.x, det.y, det.width, det.height);
                    ctx.fillStyle = 'red';
                    ctx.font = '16px Arial';
                    ctx.fillText(`ID: ${det.id}`, det.x, det.y - 5);
                });
            }
        });
    }
});