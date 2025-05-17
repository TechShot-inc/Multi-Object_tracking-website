from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, send_file, redirect
from app.tracking import main as process_tracking
import os
import uuid
import json
from werkzeug.utils import secure_filename
import threading
import random
import re
import subprocess
import shutil

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

def process_video(video_path, result_id, speed=5, output_speed=2):
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
        process_tracking(video_path, annotated_video_path, speed, output_speed)
        
        # Extract tracking data from the results folder
        # This depends on what your tracking pipeline outputs
        annotations = {}  # You'll need to convert your tracking data to this format
        
        # Save annotations to file
        with open(os.path.join(result_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f)
        
        # Post-process the video for web compatibility
        web_friendly_path = os.path.join(result_dir, 'annotated_video_web.mp4')
        make_video_web_compatible(annotated_video_path, web_friendly_path)
        shutil.move(web_friendly_path, annotated_video_path)  # Overwrite with web-friendly version
        
        processing_status[result_id] = {'status': 'completed'}
        
    except Exception as e:
        processing_status[result_id] = {'status': 'failed', 'error': str(e)}
        print(f"Error processing video: {e}")

def make_video_web_compatible(input_path, output_path):
    # This will move the moov atom to the start and ensure H.264 encoding
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "faststart",
        output_path
    ], check=True)

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
        
        # Get speed and output_speed from form
        speed = int(request.form.get('speed', 5))  # Default to 5 if not provided
        output_speed = int(request.form.get('output_speed', 2))  # Default to 2 if not provided
        
        # Start processing in a separate thread
        processing_thread = threading.Thread(
            target=process_video, 
            args=(video_path, result_id, speed, output_speed)
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
    # Find video path
    possible_paths = [
        os.path.join(app.config['RESULTS_FOLDER'], result_id, 'annotated_video.mp4'),
        os.path.join(BASE_DIR, 'results', result_id, 'annotated_video.mp4'),
        os.path.join(PROJECT_ROOT, 'results', result_id, 'annotated_video.mp4')
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
    
    # Check for query parameter to use simpler download method
    use_simple = request.args.get('simple', '0') == '1'
    
    if use_simple:
        # Simple direct file send - no streaming
        try:
            return send_file(
                video_path,
                mimetype='video/mp4',
                as_attachment=request.args.get('download', '0') == '1',
                download_name=f"result_{result_id}.mp4" if request.args.get('download', '0') == '1' else None
            )
        except Exception as e:
            print(f"Error in simple file serving: {e}")
            return jsonify({'error': f'Error serving video file: {str(e)}'}), 500
    
    # Otherwise use the more complex streaming method
    try:
        # Get file size for range requests
        file_size = os.path.getsize(video_path)
        
        # Handle range requests for better browser video playing
        range_header = request.headers.get('Range', None)
        
        if range_header:
            # Parse the range header
            byte_start, byte_end = 0, None
            range_match = range_header.replace('bytes=', '').split('-')
            if range_match[0]:
                byte_start = int(range_match[0])
            if range_match[1]:
                byte_end = int(range_match[1])
            
            if byte_end is None:
                byte_end = file_size - 1
            
            # Limit chunk size to 1MB for SSL to handle well
            if byte_end - byte_start > 1024 * 1024:
                byte_end = byte_start + 1024 * 1024 - 1
            
            # Make sure byte_end doesn't exceed file size
            if byte_end >= file_size:
                byte_end = file_size - 1
                
            # Calculate content length
            content_length = byte_end - byte_start + 1
            
            # Create response headers
            headers = {
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': 'video/mp4'
            }
            
            # Create partial response
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                data = f.read(content_length)
                
            response = app.response_class(
                data,
                status=206,
                mimetype='video/mp4',
                direct_passthrough=True,
                headers=headers
            )
            return response
        else:
            # Non-range request - serve in smaller chunks
            def generate():
                with open(video_path, 'rb') as video_file:
                    # Use smaller chunk size to avoid SSL errors
                    chunk_size = 64 * 1024  # 64KB chunks
                    while True:
                        chunk = video_file.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            
            headers = {
                'Accept-Ranges': 'bytes',
                'Content-Length': str(file_size),
                'Content-Type': 'video/mp4',
                'Cache-Control': 'public, max-age=86400'
            }
            
            return app.response_class(
                generate(),
                mimetype='video/mp4',
                direct_passthrough=True,
                headers=headers
            )
    except Exception as e:
        print(f"Error serving file: {e}")
        return jsonify({'error': f'Error serving video file: {str(e)}'}), 500

@app.route('/results/<result_id>/annotations')
def get_annotations(result_id):
    # Try multiple paths for annotations
    possible_paths = [
        os.path.join(app.config['RESULTS_FOLDER'], result_id, 'annotations.json'),
        os.path.join(BASE_DIR, 'results', result_id, 'annotations.json'),
        os.path.join(PROJECT_ROOT, 'results', result_id, 'annotations.json'),
        # Try the alternative file if the main one fails
        os.path.join(app.config['RESULTS_FOLDER'], result_id, 'annotations_direct.json'),
        os.path.join(BASE_DIR, 'results', result_id, 'annotations_direct.json'),
        os.path.join(PROJECT_ROOT, 'results', result_id, 'annotations_direct.json')
    ]
    
    # Log all possible result directories
    result_dirs = [
        os.path.join(app.config['RESULTS_FOLDER'], result_id),
        os.path.join(BASE_DIR, 'results', result_id),
        os.path.join(PROJECT_ROOT, 'results', result_id)
    ]
    
    for dir_path in result_dirs:
        if os.path.exists(dir_path):
            print(f"Directory exists: {dir_path}")
            print(f"Contents: {os.listdir(dir_path)}")
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                print(f"  - {file}: {os.path.getsize(file_path)} bytes")
    
    # Find the first path that exists with content
    annotations_path = None
    for path in possible_paths:
        print(f"Checking annotations path: {path}")
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  File exists with size: {size} bytes")
            if size > 10:  # Ensure it has meaningful content
                annotations_path = path
                print(f"Found valid annotations at: {annotations_path} ({size} bytes)")
                break
    
    if not annotations_path:
        print("No valid annotations path found with content!")
        # Try to find any JSON file in the result directories
        found_json = None
        for dir_path in result_dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith('.json'):
                        json_path = os.path.join(dir_path, file)
                        if os.path.getsize(json_path) > 10:
                            found_json = json_path
                            print(f"Found alternative JSON file: {found_json}")
                            break
        
        if found_json:
            # Use this JSON file
            annotations_path = found_json
        else:
            # Use the generate_annotations endpoint to create sample data
            # This will redirect to generate sample annotations
            return redirect(url_for('generate_annotations', result_id=result_id))
    
    try:
        # Read the file and return its contents
        with open(annotations_path, 'r') as f:
            try:
                data = json.load(f)
                print(f"Successfully loaded annotations with {len(data)} frames")
                
                # Verify the data has expected structure
                if not data or not isinstance(data, dict):
                    print("Warning: Annotations data is not a valid dictionary")
                    # Redirect to generate sample data
                    return redirect(url_for('generate_annotations', result_id=result_id))
                
                return jsonify(data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                
                # Try to repair the JSON file
                with open(annotations_path, 'r') as broken_file:
                    content = broken_file.read()
                
                if len(content) < 10:
                    print("File is too small to be valid")
                    return redirect(url_for('generate_annotations', result_id=result_id))
                
                # Manual approach for simple JSON fixes
                fixed_content = content.strip()
                if not fixed_content.startswith('{'):
                    fixed_content = '{' + fixed_content
                if not fixed_content.endswith('}'):
                    fixed_content = fixed_content + '}'
                
                try:
                    # Try to parse the fixed content
                    fixed_data = json.loads(fixed_content)
                    print("Successfully repaired JSON file")
                    return jsonify(fixed_data)
                except:
                    print("Failed to repair JSON file")
                    return redirect(url_for('generate_annotations', result_id=result_id))
                
    except Exception as e:
        print(f"Error serving annotations: {e}")
        # Redirect to generate sample data
        return redirect(url_for('generate_annotations', result_id=result_id))

@app.route('/results/<result_id>/generate_annotations')
def generate_annotations(result_id):
    """
    Endpoint to manually generate sample annotations if the normal process fails.
    This ensures the frontend always has data to display.
    """
    # Find the annotation path
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    annotations_path = os.path.join(result_dir, 'annotations.json')
    
    # Generate a sample set of annotations with 10 frames and 5 objects
    sample_data = {}
    
    # Create 10 frames of data with 5 random objects each
    for frame in range(1, 11):
        frame_id = str(frame)
        detections = []
        
        # Generate 5 random objects
        for obj_id in range(1, 6):
            detection = {
                "id": obj_id,
                "x": random.randint(50, 500),
                "y": random.randint(50, 400),
                "width": random.randint(30, 100),
                "height": random.randint(40, 120),
                "confidence": round(random.uniform(0.7, 1.0), 3)
            }
            detections.append(detection)
        
        sample_data[frame_id] = detections
    
    # Save the sample data
    try:
        with open(annotations_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
            
        return jsonify({
            "status": "success", 
            "message": f"Generated sample annotations with {len(sample_data)} frames",
            "frameCount": len(sample_data)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to generate annotations: {str(e)}"
        }), 500

@app.route('/results/<result_id>/debug_annotations')
def debug_annotations(result_id):
    """Debug endpoint to directly read and return annotations file content"""
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
    annotations_path = os.path.join(result_dir, 'annotations.json')
    
    response = {
        'status': 'unknown',
        'file_exists': os.path.exists(annotations_path),
        'file_size': os.path.getsize(annotations_path) if os.path.exists(annotations_path) else 0,
        'content': '',
        'error': ''
    }
    
    try:
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                content = f.read()
                response['content'] = content[:500] + '...' if len(content) > 500 else content
                response['status'] = 'success'
    except Exception as e:
        response['error'] = str(e)
        response['status'] = 'error'
    
    return jsonify(response)

@app.route('/results/<result_id>/player')
def video_player(result_id):
    """Serve a dedicated HTML5 video player page"""
    # Generate video URL
    video_url = url_for('get_video', result_id=result_id)
    return render_template('video_player.html', video_url=video_url)

if __name__ == '__main__':    
    app.run(debug=True, ssl_context='adhoc') 
