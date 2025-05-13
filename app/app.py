from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, send_file
from app.tracking import main as process_tracking
import os
import uuid
import json
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)

# Get the base directory of the app
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Move up one level to get project root if needed
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Configure paths
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(PROJECT_ROOT, 'results')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Print paths for debugging
print(f"BASE_DIR: {BASE_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
print(f"RESULTS_FOLDER: {app.config['RESULTS_FOLDER']}")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Processing status dictionary
processing_status = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(video_path, result_id):
    """
    Process the uploaded video with the tracking pipeline.
    This is a placeholder for the actual tracking implementation.
    """
    try:
        processing_status[result_id] = {'status': 'processing'}
        
        # Create results directory for this upload
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Set output path for the annotated video
        annotated_video_path = os.path.join(result_dir, 'annotated_video.mp4')
        
        # Call the tracking pipeline
        process_tracking(video_path, annotated_video_path)
        
        # Extract tracking data from the results folder
        # This depends on what your tracking pipeline outputs
        annotations = {}  # You'll need to convert your tracking data to this format
        
        # Save annotations to file
        with open(os.path.join(result_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f)
        
        processing_status[result_id] = {'status': 'completed'}
        
    except Exception as e:
        processing_status[result_id] = {'status': 'failed', 'error': str(e)}
        print(f"Error processing video: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file found'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this upload
        result_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}_{filename}")
        file.save(video_path)
        
        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_video, 
            args=(video_path, result_id)
        )
        processing_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded successfully. Processing started.',
            'result_id': result_id
        })
    
    return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/results/<result_id>/status')
def get_status(result_id):
    if result_id in processing_status:
        return jsonify(processing_status[result_id])
    return jsonify({'status': 'not_found'}), 404

@app.route('/results/<result_id>/video')
def get_video(result_id):
    # Use direct path calculation based on our base directory
    # Manually construct the absolute path to avoid Flask path handling issues
    video_filename = f"{result_id}/annotated_video.mp4"
    
    # Try multiple path variants
    possible_paths = [
        os.path.join(app.config['RESULTS_FOLDER'], result_id, 'annotated_video.mp4'),
        os.path.join(BASE_DIR, 'results', result_id, 'annotated_video.mp4'),
        os.path.join(PROJECT_ROOT, 'results', result_id, 'annotated_video.mp4'),
        os.path.join('/Users/ahmedwalaa/Desktop/coding/AI/fawry website/app/results', result_id, 'annotated_video.mp4'),
        os.path.join('/Users/ahmedwalaa/Desktop/coding/AI/fawry website/results', result_id, 'annotated_video.mp4')
    ]
    
    # Find the first path that exists
    video_path = None
    for path in possible_paths:
        print(f"Checking path: {path}")
        if os.path.exists(path):
            video_path = path
            print(f"Found video at: {video_path}")
            break
    
    if not video_path:
        print("No valid video path found!")
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Try direct file read and serve
        print(f"Sending file: {video_path}")
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving file: {e}")
        return jsonify({'error': 'Error serving video file'}), 500

@app.route('/results/<result_id>/annotations')
def get_annotations(result_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
    return send_from_directory(result_dir, 'annotations.json')

if __name__ == '__main__':
    app.run(debug=True) 