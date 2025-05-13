# Multi-Object Tracking Solution

This project provides a web interface for multi-object tracking in videos, with two main features:
1. Video upload and offline processing
2. Real-time object tracking using device cameras

## Features

### Video Upload and Processing
- Upload video files (MP4, AVI, MOV, MKV)
- Server-side processing with a tracking pipeline
- View annotated videos with tracking results
- Download processed videos and annotations

### Real-Time Tracking
- Access and select from available camera devices
- Real-time object detection using TensorFlow.js
- Object tracking with ID assignment and persistence
- Performance metrics (FPS, object count)

## Technology Stack

### Server-Side
- Flask (Python web framework)
- Threading for asynchronous video processing
- RESTful API endpoints for video management

### Client-Side
- HTML5, CSS3, and JavaScript
- TensorFlow.js for in-browser object detection
- Simplified tracking algorithm (IoU-based with velocity estimation)
- HTML5 Canvas for visualization

## Setup and Installation

### Prerequisites
- Python 3.7+
- Node.js (optional, for development)

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/multi-object-tracking.git
cd multi-object-tracking
```

2. Install Python dependencies
```
pip install -r requirements.txt
```

3. Run the application
```
python3 -m app.app
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Video Upload
1. Navigate to the "Video Upload" section
2. Click "Choose a video file" and select a video from your device
3. Click "Upload & Process" and wait for processing to complete
4. View the processed video and download results as needed

### Real-Time Tracking
1. Navigate to the "Real-Time Tracking" section
2. Select a camera from the dropdown menu
3. Click "Start Tracking" to begin real-time detection and tracking
4. View object IDs, classes, and tracking statistics
5. Click "Stop Tracking" when finished

## Development

### Project Structure
```
app/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── upload.js
│   │   └── tracking.js
│   └── models/       # For custom TensorFlow.js models
├── templates/
│   └── index.html
├── uploads/          # Temporary storage for uploaded videos
├── results/          # Storage for processed videos and annotations
└── app.py            # Flask application
```

### Customization
- To integrate your own tracking pipeline, modify the `process_video()` function in `app.py`
- To use a custom detection model, replace the COCO-SSD model loading in `tracking.js`

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- TensorFlow.js team for providing pre-trained models
- Flask team for the excellent web framework

## Prizes Day Presentation Tips
- Prepare sample videos for quick demonstrations
- Test the real-time tracking on the presentation device beforehand
- Have backup pre-processed videos ready in case of connection issues 