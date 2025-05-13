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
    
    // Update the file input label with the selected filename
    videoFile.addEventListener('change', () => {
        const fileName = videoFile.files[0] ? videoFile.files[0].name : 'Choose a video file';
        const fileLabel = videoFile.nextElementSibling;
        fileLabel.textContent = fileName;
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!videoFile.files[0]) {
            alert('Please select a video file first.');
            return;
        }
        
        const file = videoFile.files[0];
        const formData = new FormData();
        formData.append('video', file);
        
        // Disable the upload button during upload
        uploadButton.disabled = true;
        uploadButton.textContent = 'Uploading...';
        
        // Update status message
        statusMessage.textContent = 'Uploading video...';
        progressBar.style.width = '0%';
        
        try {
            // Upload the video
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            // Show progress during upload (simulated)
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
                
                // Store the result ID for checking status and retrieving results
                resultId = data.result_id;
                
                // Start checking processing status
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
    
    // Function to check video processing status
    function checkProcessingStatus() {
        if (!resultId) return;
        
        statusCheckInterval = setInterval(async () => {
            try {
                const response = await fetch(`/results/${resultId}/status`);
                const data = await response.json();
                
                if (data.status === 'completed') {
                    // Processing complete, show results
                    statusMessage.textContent = 'Processing complete!';
                    clearInterval(statusCheckInterval);
                    showResults();
                } else if (data.status === 'failed') {
                    // Processing failed
                    statusMessage.textContent = `Processing failed: ${data.error || 'Unknown error'}`;
                    clearInterval(statusCheckInterval);
                    uploadButton.disabled = false;
                    uploadButton.textContent = 'Upload & Process';
                }
                // If status is 'processing', just keep waiting
            } catch (error) {
                console.error('Error checking status:', error);
            }
        }, 2000); // Check every 2 seconds
    }
    
    // Function to display results
    function showResults() {
        if (!resultId) return;
        
        // Show the results container
        resultsContainer.style.display = 'block';
        
        // Set the video source
        resultVideo.src = `/results/${resultId}/video`;
        
        // Reset the upload button
        uploadButton.disabled = false;
        uploadButton.textContent = 'Upload & Process';
        
        // Configure download buttons
        downloadVideoBtn.onclick = () => window.location.href = `/results/${resultId}/video`;
        downloadAnnotationsBtn.onclick = () => window.location.href = `/results/${resultId}/annotations`;
    }
}); 