from flask import Flask, render_template, request, jsonify, url_for, send_file, Response
from .tracking import main as process_tracking
from .realtimetracking import RealTimeTrackingService, draw_detections
from .CustomBoostTrack.realtime_ensembling import RealTimeTracker
import os
import uuid
import json
from werkzeug.utils import secure_filename
import threading
import subprocess
import shutil
import base64
import cv2
import numpy as np
import time
from .CustomBoostTrack.utils import generate_mot_heatmap, get_longest_staying_ids, generate_velocity_heatmap, get_track_durations, get_average_velocity
import pandas as pd
from ultralytics import YOLO

app = Flask(__name__)

# Directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
APP_DIR = BASE_DIR
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'Uploads')
app.config['RESULTS_FOLDER'] = os.path.join(PROJECT_ROOT, 'results')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Logging paths
print(f"BASE_DIR: {BASE_DIR}")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
print(f"RESULTS_FOLDER: {app.config['RESULTS_FOLDER']}")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Processing status
processing_status = {}

# Initialize real-time tracker
realtime_tracker = RealTimeTrackingService(camera_index=0, frame_rate=30)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def make_video_web_compatible(input_path, output_path, fps):
    """Convert video to web-compatible format using ffmpeg with specified frame rate."""
    try:
        command = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-r', str(fps),  # Explicitly set frame rate
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ]
        subprocess.run(command, check=True, capture_output=True)
        print(f"Video converted successfully: {output_path} at {fps} FPS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error in video conversion: {e}")
        return False

def process_video(video_path, output_path, filename, speed=1.0, output_speed=1.0, roi=None):
    """Process video with tracking and ROI"""
    try:
        print(f"Starting video processing for: {video_path}")
        
        # Create results directory
        results_dir = os.path.dirname(output_path)
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
        
        # Create analytics directory
        analytics_dir = os.path.join(results_dir, 'analytics')
        os.makedirs(analytics_dir, exist_ok=True)
        print(f"Created analytics directory: {analytics_dir}")
        
        # Create frames directory for tracking
        frames_dir = os.path.join(results_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Created frames directory: {frames_dir}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Extract frames with ROI if provided
        frame_count = 0
        extracted_frames = 0
        print(f"Starting frame extraction with speed={speed}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Modulo-based frame sampling
            if frame_count % int(speed) == 0:
                if roi:
                    x, y, w, h = roi
                    frame = frame[y:y+h, x:x+w]
                frame_path = os.path.join(frames_dir, f"frame_{extracted_frames:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames += 1
            frame_count += 1
            
            # Update progress (25% for frame extraction)
            progress = frame_count / total_frames
            processing_status[filename] = {
                'status': 'processing',
                'progress': progress * 0.25,
                'message': f'Extracting frames: {frame_count}/{total_frames}'
            }
        
        cap.release()
        print(f"Frame extraction complete. Extracted {extracted_frames} frames")
        
        # Validate frame count
        expected_frames = total_frames // speed
        if abs(extracted_frames - expected_frames) > 1:
            print(f"Warning: Expected ~{expected_frames} frames, got {extracted_frames}")
        
        # Run tracking with ensembler
        output_folder = os.path.join(results_dir, 'results')
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder for tracking: {output_folder}")
        
        # Calculate frame rate for tracking
        tracking_fps = max(1, int(fps / speed))
        print(f"Using tracking FPS: {tracking_fps}")
        
        # Update status for tracking (25% to 75%)
        processing_status[filename] = {
            'status': 'processing',
            'progress': 0.25,
            'message': 'Running tracking...'
        }
        
        # Prepare tracking command
        tracking_script = os.path.join(APP_DIR, "CustomBoostTrack", "run_with_ensembler.py")
        if not os.path.exists(tracking_script):
            raise Exception(f"Tracking script not found: {tracking_script}")
            
        print(f"Tracking script exists: {os.path.exists(tracking_script)}")
        
        # Run tracking process with real-time terminal logging
        try:
            print("Starting tracking process...")
            tracking_command = [
                "python", tracking_script,
                "--dataset", "custom",
                "--exp_name", "tracking",
                "--result_folder", output_folder,
                "--frame_rate", str(tracking_fps),
                "--dataset_path", frames_dir,
                "--model1_path", os.path.join(PROJECT_ROOT, "app", "models", "General1.pt"),
                "--model1_weight", "0.7",
                "--model2_path", os.path.join(PROJECT_ROOT, "app", "models", "YOLO12Final.pt"),
                "--model2_weight", "0.3",
                "--conf_thresh", "0.1"
            ]
            print(f"Tracking command: {' '.join(tracking_command)}")
            tracking_process = subprocess.Popen(
                tracking_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            def stream_logs():
                while True:
                    stdout_line = tracking_process.stdout.readline()
                    stderr_line = tracking_process.stderr.readline()
                    if stdout_line:
                        print(stdout_line.strip())  # Print to terminal
                    if stderr_line:
                        print(stderr_line.strip())  # Print to terminal
                    if tracking_process.poll() is not None:
                        # Capture any remaining output
                        for line in tracking_process.stdout:
                            print(line.strip())
                        for line in tracking_process.stderr:
                            print(line.strip())
                        break
            
            log_thread = threading.Thread(target=stream_logs)
            log_thread.start()
            
            try:
                tracking_process.wait(timeout=600)  # 10 minutes
                log_thread.join()
                if tracking_process.returncode != 0:
                    raise Exception(f"Tracking failed with return code {tracking_process.returncode}")
            except subprocess.TimeoutExpired:
                tracking_process.kill()
                log_thread.join()
                raise Exception("Tracking process timed out after 10 minutes")
                
        except Exception as e:
            print(f"Error running tracking process: {str(e)}")
            raise
        
        print("Tracking process completed successfully")
        
        # Update status for post-processing (75% to 90%)
        processing_status[filename] = {
            'status': 'processing',
            'progress': 0.75,
            'message': 'Processing results...'
        }
        
        # Find tracking results
        tracking_results_path = None
        for root, dirs, files in os.walk(output_folder):
            if "test.txt" in files:
                tracking_results_path = os.path.join(root, "test.txt")
                break
        
        if not tracking_results_path:
            raise Exception("Tracking results not found")
        
        print(f"Found tracking results at: {tracking_results_path}")
        
        # Copy tracking results to analytics directory
        mot_file = os.path.join(analytics_dir, 'test.txt')
        shutil.copy2(tracking_results_path, mot_file)
        print(f"Copied tracking results to: {mot_file}")
        
        # Process frames with tracking results
        cap = cv2.VideoCapture(video_path)
        output_fps = max(1, (fps / speed) / output_speed)  # Correct output frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Read tracking results
        tracking_df = pd.read_csv(tracking_results_path, header=None)
        if tracking_df.shape[1] >= 7:
            tracking_df.columns = ['frame', 'track_id', 'x', 'y', 'width', 'height', 'confidence'] + \
                                [f'col_{i}' for i in range(tracking_df.shape[1] - 7)]
        
        print(f"Loaded tracking data with {len(tracking_df)} detections")
        
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        if len(frame_files) != extracted_frames:
            print(f"Warning: Expected {extracted_frames} frames, found {len(frame_files)} in {frames_dir}")
        
        tracking_data = {}
        
        print(f"Processing {len(frame_files)} extracted frames with tracking results...")
        for frame_idx, frame_file in enumerate(frame_files, 1):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            # Create full-size frame if ROI was applied
            if roi:
                x, y, w, h = roi
                full_frame = np.zeros((height, width, 3), dtype=np.uint8)
                full_frame[y:y+h, x:x+w] = frame
                frame = full_frame
            
            # Get detections for this frame
            frame_detections = tracking_df[tracking_df['frame'] == frame_idx]
            detections_list = []
            
            for _, det in frame_detections.iterrows():
                det_x = int(det['x'])
                det_y = int(det['y'])
                det_w = int(det['width'])
                det_h = int(det['height'])
                track_id = int(det['track_id'])
                conf = float(det['confidence'])
                
                # Adjust coordinates if ROI was applied
                if roi:
                    det_x += x
                    det_y += y
                
                # Draw detection
                cv2.rectangle(frame, (det_x, det_y), (det_x + det_w, det_y + det_h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (det_x, det_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detections_list.append({
                    'id': track_id,
                    'x': det_x,
                    'y': det_y,
                    'width': det_w,
                    'height': det_h,
                    'confidence': conf
                })
            
            # Draw ROI if applied
            if roi:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = "Active ROI"
                cv2.putText(frame, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            print(f"Writing frame {frame_idx} at {output_fps} FPS")
            
            tracking_data[str(frame_idx)] = detections_list
            
            # Update progress (75% to 90%)
            progress = 0.75 + (frame_idx / len(frame_files)) * 0.15
            processing_status[filename] = {
                'status': 'processing',
                'progress': progress,
                'message': f'Processing frame {frame_idx}/{len(frame_files)}'
            }
        
        # Release resources
        cap.release()
        out.release()
        print(f"Finished processing {len(frame_files)} frames, output FPS: {output_fps}")
        
        # Save tracking data
        tracking_file = os.path.join(analytics_dir, 'tracking_data.json')
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f)
        print(f"Saved tracking data to: {tracking_file}")
        
        # Update status for final conversion (90% to 100%)
        processing_status[filename] = {
            'status': 'processing',
            'progress': 0.90,
            'message': 'Converting video...'
        }
        
        # Convert to web-compatible format
        web_output = output_path.replace('.mp4', '_web.mp4')
        print(f"Converting video to web format: {web_output}")
        
        if not make_video_web_compatible(output_path, web_output, output_fps):
            raise Exception("Video conversion failed")
        
        print("Video conversion complete")
        
        # Clean up original file
        os.remove(output_path)
        
        # Clean up temporary files
        shutil.rmtree(frames_dir)
        shutil.rmtree(output_folder)
        print("Cleaned up temporary files")
        
        # Verify all required files exist before marking as complete
        required_files = [
            web_output,
            tracking_file,
            mot_file
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise Exception(f"Required file not found: {file_path}")
        
        print("All required files verified")
        
        # Update final status
        processing_status[filename] = {
            'status': 'completed',
            'progress': 1.0,
            'message': 'Processing complete'
        }
        
        print("Video processing completed successfully")
        return web_output
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        processing_status[filename] = {
            'status': 'error',
            'progress': 0.0,
            'message': f'Error: {str(e)}'
        }
        raise

def process_frame(frame, model_paths):
    """Process a single frame with tracking"""
    try:
        yolo11 = YOLO(model_paths['yolo11'])
        yolo12 = YOLO(model_paths['yolo12'])
        results11 = yolo11(frame, conf=0.1)[0]
        results12 = yolo12(frame, conf=0.1)[0]
        detections = []
        for det in results11.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': int(cls)
            })
        for det in results12.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': int(cls)
            })
        tracker = BYTETracker()
        tracks = tracker.update(detections)
        results = []
        for track in tracks:
            results.append({
                'id': track.track_id,
                'box': track.tlbr.tolist(),
                'confidence': track.score
            })
        return results
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload, generate thumbnail, and start processing."""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file found'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            base_name = os.path.splitext(filename)[0]
            thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_thumb.jpg")
            result_dir = os.path.join(app.config['RESULTS_FOLDER'], base_name)
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted existing video: {video_path}")
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                print(f"Deleted existing thumbnail: {thumbnail_path}")
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
                print(f"Deleted existing results directory: {result_dir}")
        except Exception as e:
            print(f"Error deleting existing files: {e}")
            return jsonify({'status': 'error', 'message': f'Error deleting existing files: {str(e)}'}), 500
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(video_path)
        print(f"Video uploaded: {video_path}")
        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_thumb.jpg")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            with open(thumbnail_path, 'wb') as f:
                f.write(buffer)
            print(f"Thumbnail generated: {thumbnail_path}")
        else:
            print(f"Failed to generate thumbnail for: {video_path}")
        cap.release()
        speed = float(request.form.get('speed', 5))
        output_speed = float(request.form.get('output_speed', 1))
        print(f"Received parameters: speed={speed}, output_speed={output_speed}")
        roi = None
        if 'roi' in request.form:            
            try:
                roi = json.loads(request.form['roi'])
                if not all(k in roi for k in ['x', 'y', 'width', 'height']):
                    print("Invalid ROI format received")
                    return jsonify({'status': 'error', 'message': 'Invalid ROI format'}), 400
                for key in ['x', 'y', 'width', 'height']:
                    if not isinstance(roi[key], (int, float)) or roi[key] < 0:
                        raise ValueError(f"Invalid ROI {key} value: {roi[key]}")
                print(f"ROI received: {roi}")
            except json.JSONDecodeError as e:
                print(f"ROI JSON decode error: {e}")
                return jsonify({'status': 'error', 'message': 'Invalid ROI JSON'}), 400
            except ValueError as e:
                print(f"ROI value error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 400
        processing_status[filename] = {
            'status': 'processing',
            'progress': 0.0,
            'message': 'Starting video processing...'
        }
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], os.path.splitext(filename)[0])
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, 'annotated_video.mp4')
        thread = threading.Thread(
            target=process_video,
            args=(video_path, output_path, filename, speed, output_speed, roi)
        )
        thread.daemon = True
        thread.start()
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded successfully. Processing started.',
            'filename': filename,
            'thumbnail': f"/Uploads/{os.path.splitext(filename)[0]}_thumb.jpg"
        })
    return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/Uploads/<filename>')
def serve_uploaded_file(filename):
    """Serve uploaded files (e.g., thumbnails)."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/results/<filename>/status')
def get_status(filename):
    """Check video processing status with detailed progress information."""
    if filename not in processing_status:
        return jsonify({'status': 'not_found', 'progress': 0.0, 'message': 'Result not found'}), 404
    status_info = processing_status.get(filename, {})
    video_web_path = os.path.join(app.config['RESULTS_FOLDER'], os.path.splitext(filename)[0], 'annotated_video_web.mp4')
    if os.path.exists(video_web_path):
        if os.path.getsize(video_web_path) > 0:
            processing_status[filename] = {'status': 'completed', 'progress': 1.0, 'message': 'Processing complete'}
            return jsonify({
                'status': 'completed',
                'progress': 1.0,
                'message': 'Processing complete'
            })
    return jsonify({
        'status': status_info.get('status', 'processing'),
        'progress': status_info.get('progress', 0.0),
        'message': status_info.get('message', 'Processing...')
    }), 200

@app.route('/results/<filename>/video')
def get_video(filename):
    """Serve processed video with range-based streaming."""
    video_path = os.path.join(app.config['RESULTS_FOLDER'], os.path.splitext(filename)[0], 'annotated_video_web.mp4')
    if not os.path.exists(video_path):
        print(f"No video found at: {video_path}")
        return jsonify({'error': 'Video not found'}), 404
    if os.path.getsize(video_path) == 0:
        print(f"Video file is empty: {video_path}")
        return jsonify({'error': 'Video processing not complete'}), 404
    print(f"Serving video: {video_path}")
    try:
        file_size = os.path.getsize(video_path)
        range_header = request.headers.get('Range')
        if range_header:
            byte_start, byte_end = 0, None
            range_match = range_header.replace('bytes=', '').split('-')
            if range_match[0]:
                byte_start = int(range_match[0])
            if range_match[1]:
                byte_end = int(range_match[1])
            if byte_end is None:
                byte_end = file_size - 1
            content_length = byte_end - byte_start + 1
            headers = {
                'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': 'video/mp4',
                'Cache-Control': 'no-cache'
            }
            with open(video_path, 'rb') as f:
                f.seek(byte_start)
                data = f.read(content_length)
            return Response(data, status=206, mimetype='video/mp4', headers=headers)
        def generate():
            with open(video_path, 'rb') as video_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = video_file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        headers = {
            'Accept-Ranges': 'bytes',
            'Content-Length': str(file_size),
            'Content-Type': 'video/mp4',
            'Cache-Control': 'no-cache'
        }
        return Response(generate(), mimetype='video/mp4', headers=headers)
    except Exception as e:
        print(f"Error serving video: {e}")
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500

@app.route('/results/<filename>/annotations')
def get_annotations(filename):
    """Serve or download annotations JSON."""
    annotations_path = os.path.join(app.config['RESULTS_FOLDER'], os.path.splitext(filename)[0], 'annotated_video_tracking.json')
    print(f"Checking annotations: {annotations_path}")
    if not os.path.exists(annotations_path):
        print(f"No annotations found at: {annotations_path}")
        return jsonify({'error': 'Annotations not found'}), 404
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded annotations with {len(data)} frames")
        if request.args.get('download', '0') == '1':
            return send_file(
                annotations_path,
                mimetype='application/json',
                as_attachment=True,
                download_name=f"annotations_{filename}.json"
            )
        return jsonify(data)
    except Exception as e:
        print(f"Error serving annotations: {e}")
        return jsonify({'error': f'Error serving annotations: {str(e)}'}), 500

@app.route('/results/<filename>/analytics')
def serve_analytics(filename):
    """Generate and serve analytics data."""
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], os.path.splitext(filename)[0])
    mot_file = os.path.join(result_dir, 'test.txt')
    frames_folder = os.path.join(result_dir, 'frames')
    if not (os.path.exists(mot_file) and os.path.exists(frames_folder)):
        print(f"Analytics data missing: MOT={os.path.exists(mot_file)}, Frames={os.path.exists(frames_folder)}")
        return jsonify({'error': 'Analytics data not found'}), 404
    video_path = os.path.join(result_dir, 'annotated_video_web.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return jsonify({'error': 'Could not open video for dimensions'}), 500
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    heatmap_path = os.path.join(result_dir, 'heatmap.png')
    velocity_heatmap_path = os.path.join(result_dir, 'velocity_heatmap.png')
    heatmap_success = generate_mot_heatmap(mot_file, width, height, grid_size=20, output_file=heatmap_path)
    velocity_heatmap_success = generate_velocity_heatmap(mot_file, width, height, grid_size=20, output_file=velocity_heatmap_path)
    heatmap_base64 = ''
    velocity_heatmap_base64 = ''
    if heatmap_success:
        with open(heatmap_path, 'rb') as f:
            heatmap_base64 = base64.b64encode(f.read()).decode('utf-8')
        print(f"Generated heatmap: {heatmap_path}")
    if velocity_heatmap_success:
        with open(velocity_heatmap_path, 'rb') as f:
            velocity_heatmap_base64 = base64.b64encode(f.read()).decode('utf-8')
        print(f"Generated velocity heatmap: {velocity_heatmap_path}")
    top_ids, crops = get_longest_staying_ids(mot_file, frames_folder, top_n=5)
    crops_base64 = []
    for id_crops in crops:
        id_crops_base64 = []
        for crop in id_crops:
            _, buffer = cv2.imencode('.jpg', crop)
            crop_base64 = base64.b64encode(buffer).decode('utf-8')
            id_crops_base64.append(crop_base64)
        crops_base64.append(id_crops_base64)
    try:
        df = pd.read_csv(mot_file, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_', 'y_', 'z_'])
        object_counts = df.groupby('frame').size().to_dict()
        track_durations = get_track_durations(mot_file)
        avg_velocity = get_average_velocity(mot_file)
        print(f"Analytics generated: {len(object_counts)} frames, {len(track_durations)} tracks")
    except Exception as e:
        print(f"Error processing MOT file: {e}")
        object_counts = {}
        track_durations = {}
        avg_velocity = 0.0
    return jsonify({
        'heatmap': heatmap_base64,
        'velocity_heatmap': velocity_heatmap_base64,
        'top_ids': top_ids,
        'crops': crops_base64,
        'object_counts': object_counts,
        'track_durations': track_durations,
        'avg_velocity': float(avg_velocity)
    })

@app.route('/realtime-track', methods=['POST'])
def realtime_track():
    """Process webcam frame for real-time tracking."""
    print('Received /realtime-track request')
    if 'frame' not in request.files:
        print('Error: No frame provided')
        return jsonify({'error': 'No frame provided'}), 400
    frame_file = request.files['frame']
    frame_data = frame_file.read()
    print(f'Frame received, size: {len(frame_data)} bytes')
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        print('Error: Failed to decode frame')
        return jsonify({'error': 'Failed to decode frame'}), 400
    print(f'Frame decoded, shape: {frame.shape}')
    roi = None
    if 'roi' in request.form:
        try:
            roi_data = request.form['roi']
            print(f'Raw ROI data: {roi_data}')
            roi = json.loads(roi_data)
            if not isinstance(roi, dict) or not all(k in roi for k in ['x', 'y', 'width', 'height']):
                print('Error: Invalid ROI format')
                return jsonify({'error': 'Invalid ROI format'}), 400
            for key in ['x', 'y', 'width', 'height']:
                roi[key] = float(roi[key])
            if roi['width'] <= 0 or roi['height'] <= 0:
                print('Error: ROI must have positive width and height')
                return jsonify({'error': 'ROI must have positive width and height'}), 400
            print(f'Validated ROI: {roi}')
        except json.JSONDecodeError as e:
            print(f'Error: Invalid ROI JSON: {e}')
            return jsonify({'error': f'Invalid ROI JSON: {str(e)}'}), 400
        except (ValueError, TypeError) as e:
            print(f'Error: Invalid ROI values: {e}')
            return jsonify({'error': f'Invalid ROI values: {str(e)}'}), 400
    try:
        annotated_frame, detections = realtime_tracker.process_frame(frame, roi=roi)
        print(f'Processed frame, detections: {len(detections)}')
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'annotated': encoded_frame,
            'count': len(detections),
            'timestamp': int(time.time() * 1000),
            'has_roi': roi is not None
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, ssl_context='adhoc')