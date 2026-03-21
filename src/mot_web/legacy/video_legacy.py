from flask import request, jsonify, url_for, send_file, Response
import subprocess
from app.app import app
from .tracking import process_video_with_tracking as process_tracking
import os
import json
from werkzeug.utils import secure_filename
import threading
import shutil
import base64
import cv2
import numpy as np
import time
from .CustomBoostTrack.utils import (
    generate_mot_heatmap,
    get_longest_staying_ids,
    generate_velocity_heatmap,
    get_track_durations,
    get_average_velocity,
)
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Processing status
processing_status = {}

# Allowed video extensions
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_video_web_compatible(input_path, output_path, fps):
    """Convert video to web-compatible format using ffmpeg with specified frame rate."""
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            output_path,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        logger.error(f"Video converted successfully: {output_path} at {fps} FPS")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Video conversion timed out after 300 seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting video: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error in video conversion: {e}")
        return False


def extract_frames(video_path, output_folder, speed=5):
    """Extract frames from video at specified speed."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(speed))  # speed directly controls sampling
    saved_frame_number = 0
    original_frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if original_frame_number % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_number += 1
        original_frame_number += 1
    cap.release()
    expected_frames = (total_frames + frame_interval - 1) // frame_interval
    if abs(saved_frame_number - expected_frames) > 1:
        logger.error(f"Expected ~{expected_frames} frames, extracted {saved_frame_number} for speed={speed}")
    logger.error(f"Extracted {saved_frame_number} frames to {output_folder} for speed={speed}")


def process_video(video_path, output_path, filename, speed=1.0, output_speed=1.0, roi=None):
    """Process video with tracking and ROI."""
    try:
        logger.error(f"Starting video processing for: {video_path}, speed={speed}, output_speed={output_speed}")

        # Create results directory
        result_dir = os.path.dirname(output_path)
        os.makedirs(result_dir, exist_ok=True)
        logger.error(f"Created results directory: {result_dir}")

        # Create analytics directory
        analytics_dir = os.path.join(result_dir, "analytics")
        os.makedirs(analytics_dir, exist_ok=True)
        logger.error(f"Created analytics directory: {analytics_dir}")

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.error(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        cap.release()

        # Validate ROI against video dimensions
        if roi:
            required_keys = ["x", "y", "width", "height"]
            if not all(k in roi for k in required_keys):
                logger.error("Invalid ROI format")
                raise ValueError("Invalid ROI format")
            for key in required_keys:
                if not isinstance(roi[key], (int, float)) or roi[key] <= 0:
                    logger.error(f"Invalid ROI {key} value: {roi[key]}")
                    raise ValueError(f"Invalid ROI {key} value: {roi[key]}")
            if roi["x"] + roi["width"] > width or roi["y"] + roi["height"] > height:
                logger.error("ROI exceeds video dimensions")
                raise ValueError("ROI exceeds video dimensions")
            logger.error(f"Validated ROI: {roi}")

        # Update status for frame extraction (0% to 25%)
        processing_status[filename] = {"status": "processing", "progress": 0.0, "message": "Extracting frames..."}

        # Extract frames
        frames_path = os.path.join(result_dir, "frames")
        extract_frames(video_path, frames_path, speed=speed)

        # Verify frames were extracted
        frame_files = [f for f in os.listdir(frames_path) if f.endswith(".jpg")]
        if not frame_files:
            raise Exception(f"No frames extracted to {frames_path}")
        logger.error(f"Found {len(frame_files)} frames in {frames_path}")

        # Update status for tracking (25% to 75%)
        processing_status[filename] = {"status": "processing", "progress": 0.25, "message": "Running tracking..."}

        # Run ensembler as subprocess
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        tracking_dir = os.path.join(result_dir, "tracking")
        os.makedirs(tracking_dir, exist_ok=True)
        script_path = os.path.join(app.config["APP_DIR"], "CustomBoostTrack", "run_with_ensembler.py")
        command = [
            "python",
            script_path,
            "--dataset",
            "tarsh",
            "--exp_name",
            "BTPP",
            "--result_folder",
            tracking_dir,
            "--frame_rate",
            str(int(fps / speed)),
            "--dataset_path",
            frames_path,
            "--model1_path",
            os.path.join(app.config["APP_DIR"], "models", "General2.pt"),
            "--model1_weight",
            "0.7",
            "--model2_path",
            os.path.join(app.config["APP_DIR"], "models", "12General1.pt"),
            "--model2_weight",
            "0.3",
            "--reid_path",
            os.path.join(app.config["APP_DIR"], "models", "osnet_ain_ms_m_c.pth.tar"),
            "--conf_thresh",
            "0.1",
        ]
        if roi:
            roi_str = f"{int(roi['x'])},{int(roi['y'])},{int(roi['width'])},{int(roi['height'])}"
            command.extend(["--roi", roi_str])

        # Verify file paths
        file_paths = [
            script_path,
            command[13],  # model1_path
            command[17],  # model2_path
            command[21],  # reid_path
            frames_path,
            command[7],  # tracking_dir
        ]
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file or directory not found: {path}")

        logger.error(f"Running command: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.error(f"Subprocess output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess failed with exit code {e.returncode}")
            logger.error(f"Subprocess stdout: {e.stdout}")
            logger.error(f"Subprocess stderr: {e.stderr}")
            raise Exception(f"Subprocess failed: {e.stderr}")

        # Find tracking results (test.txt)
        tracking_data_path = os.path.join(tracking_dir, "tarsh-val", "BTPP_post_gbi", "data", "test.txt")
        if not os.path.exists(tracking_data_path):
            raise Exception(f"Tracking data not found: {tracking_data_path}")

        # Parse tracking results into detections
        detections = {}
        with open(tracking_data_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                if frame_id not in detections:
                    detections[frame_id] = []
                detections[frame_id].append([x, y, x + w, y + h, conf, 0, track_id])
        logger.error(f"Parsed {len(detections)} frames of detections for speed={speed}")

        # Update status for analytics processing (75% to 90%)
        processing_status[filename] = {"status": "processing", "progress": 0.75, "message": "Processing analytics..."}

        # Run tracking for analytics
        output_video, analytics_file = process_tracking(
            video_path=video_path,
            output_dir=result_dir,
            detections=detections,
            speed=speed,
            output_speed=output_speed,
            roi=roi,
        )
        logger.error(f"Analytics completed, video: {output_video}, analytics: {analytics_file}")

        # Copy tracking results to analytics directory as test.txt
        mot_file = os.path.join(analytics_dir, "test.txt")
        shutil.copy2(tracking_data_path, mot_file)
        logger.error(f"Copied tracking results to: {mot_file}")

        # Update status for final conversion (90% to 100%)
        processing_status[filename] = {"status": "processing", "progress": 0.90, "message": "Converting video..."}

        # Convert to web-compatible format
        web_output = output_path.replace(".mp4", "_web.mp4")
        logger.error(f"Converting video to web format: {web_output}")

        output_fps = max(1, fps * output_speed)
        if not make_video_web_compatible(output_video, web_output, output_fps):
            raise Exception("Video conversion failed")

        logger.error("Video conversion complete")

        # Clean up original file
        os.remove(output_video)
        logger.error(f"Removed original video: {output_video}")

        # Verify all required files exist before marking as complete
        required_files = [web_output, analytics_file, mot_file]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise Exception(f"Required file not found: {file_path}")

        logger.error("All required files verified")

        # Update final status
        processing_status[filename] = {"status": "completed", "progress": 1.0, "message": "Processing complete"}

        logger.error("Video processing completed successfully")
        return web_output

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        processing_status[filename] = {"status": "error", "progress": 0.0, "message": f"Error: {str(e)}"}
        raise


@app.route("/upload", methods=["POST"])
def upload_video():
    """Handle video upload, generate thumbnail, and start processing."""
    if "video" not in request.files:
        logger.error("No video file found in request")
        return jsonify({"status": "error", "message": "No video file found"}), 400
    file = request.files["video"]
    if file.filename == "":
        logger.error("No video selected")
        return jsonify({"status": "error", "message": "No video selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            base_name = os.path.splitext(filename)[0]
            thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base_name}_thumb.jpg")
            result_dir = os.path.join(app.config["RESULTS_FOLDER"], base_name)
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.error(f"Deleted existing video: {video_path}")
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                logger.error(f"Deleted existing thumbnail: {thumbnail_path}")
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
                logger.error(f"Deleted existing results directory: {result_dir}")
        except Exception as e:
            logger.error(f"Error deleting existing files: {e}")
            return jsonify({"status": "error", "message": f"Error deleting existing files: {str(e)}"}), 500
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(video_path)
        logger.error(f"Video uploaded: {video_path}")
        thumbnail_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{os.path.splitext(filename)[0]}_thumb.jpg")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            with open(thumbnail_path, "wb") as f:
                f.write(buffer)
            logger.error(f"Thumbnail generated: {thumbnail_path}")
        else:
            logger.error(f"Failed to generate thumbnail for: {video_path}")
        cap.release()
        speed = float(request.form.get("speed", 5))
        output_speed = float(request.form.get("output_speed", 1))
        logger.error(f"Received parameters: speed={speed}, output_speed={output_speed}")
        roi = None
        if "roi" in request.form:
            try:
                roi = json.loads(request.form["roi"])
                if not all(k in roi for k in ["x", "y", "width", "height"]):
                    logger.error("Invalid ROI format received")
                    return jsonify({"status": "error", "message": "Invalid ROI format"}), 400
                for key in ["x", "y", "width", "height"]:
                    if not isinstance(roi[key], (int, float)) or roi[key] < 0:
                        raise ValueError(f"Invalid ROI {key} value: {roi[key]}")
                logger.error(f"ROI received: {roi}")
            except json.JSONDecodeError as e:
                logger.error(f"ROI JSON decode error: {e}")
                return jsonify({"status": "error", "message": "Invalid ROI JSON"}), 400
            except ValueError as e:
                logger.error(f"ROI value error: {e}")
                return jsonify({"status": "error", "message": str(e)}), 400
        processing_status[filename] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Starting video processing...",
        }
        result_dir = os.path.join(app.config["RESULTS_FOLDER"], os.path.splitext(filename)[0])
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, "annotated_video.mp4")
        thread = threading.Thread(
            target=process_video, args=(video_path, output_path, filename, speed, output_speed, roi)
        )
        thread.daemon = True
        thread.start()
        return jsonify(
            {
                "status": "success",
                "message": "Video uploaded successfully. Processing started.",
                "filename": filename,
                "thumbnail": f"/Uploads/{os.path.splitext(filename)[0]}_thumb.jpg",
            }
        )
    logger.error(f"File type not allowed: {file.filename}")
    return jsonify({"status": "error", "message": "File type not allowed"}), 400


@app.route("/Uploads/<filename>")
def serve_uploaded_file(filename):
    """Serve uploaded files (e.g., thumbnails)."""
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
        logger.error(f"Serving file: {file_path}")
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500


@app.route("/results/<filename>/status")
def get_status(filename):
    """Check video processing status with detailed progress information."""
    try:
        if filename not in processing_status:
            logger.error(f"Result not found for filename: {filename}")
            return jsonify({"status": "not_found", "progress": 0.0, "message": "Result not found"}), 404
        status_info = processing_status.get(filename, {})
        video_web_path = os.path.join(
            app.config["RESULTS_FOLDER"], os.path.splitext(filename)[0], "annotated_video_web.mp4"
        )
        if os.path.exists(video_web_path):
            if os.path.getsize(video_web_path) > 0:
                processing_status[filename] = {"status": "completed", "progress": 1.0, "message": "Processing complete"}
                logger.error(f"Processing complete for {filename}")
                return jsonify({"status": "completed", "progress": 1.0, "message": "Processing complete"})
        logger.error(f"Status for {filename}: {status_info}")
        return jsonify(
            {
                "status": status_info.get("status", "processing"),
                "progress": status_info.get("progress", 0.0),
                "message": status_info.get("message", "Processing..."),
            }
        ), 200
    except Exception as e:
        logger.error(f"Error checking status for {filename}: {e}")
        return jsonify({"status": "error", "message": f"Error checking status: {str(e)}"}), 500


@app.route("/results/<filename>/video")
def get_video(filename):
    """Serve processed video with range-based streaming."""
    try:
        video_path = os.path.join(
            app.config["RESULTS_FOLDER"], os.path.splitext(filename)[0], "annotated_video_web.mp4"
        )
        if not os.path.exists(video_path):
            logger.error(f"No video found at: {video_path}")
            return jsonify({"error": "Video not found"}), 404
        if os.path.getsize(video_path) == 0:
            logger.error(f"Video file is empty: {video_path}")
            return jsonify({"error": "Video processing not complete"}), 404
        logger.error(f"Serving video: {video_path}")
        file_size = os.path.getsize(video_path)
        range_header = request.headers.get("Range")
        if range_header:
            byte_start, byte_end = 0, None
            range_match = range_header.replace("bytes=", "").split("-")
            if range_match[0]:
                byte_start = int(range_match[0])
            if range_match[1]:
                byte_end = int(range_match[1])
            if byte_end is None:
                byte_end = file_size - 1
            content_length = byte_end - byte_start + 1
            headers = {
                "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
                "Cache-Control": "no-cache",
            }
            with open(video_path, "rb") as f:
                f.seek(byte_start)
                data = f.read(content_length)
            return Response(data, status=206, mimetype="video/mp4", headers=headers)

        def generate():
            with open(video_path, "rb") as video_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = video_file.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
            "Cache-Control": "no-cache",
        }
        return Response(generate(), mimetype="video/mp4", headers=headers)
    except Exception as e:
        logger.error(f"Error serving video {filename}: {e}")
        return jsonify({"error": f"Error serving video: {str(e)}"}), 500


@app.route("/results/<filename>/annotations")
def get_annotations(filename):
    """Serve or download analytics JSON."""
    try:
        analytics_path = os.path.join(app.config["RESULTS_FOLDER"], os.path.splitext(filename)[0], "analytics.json")
        logger.error(f"Checking analytics: {analytics_path}")
        if not os.path.exists(analytics_path):
            logger.error(f"No analytics found at: {analytics_path}")
            return jsonify({"error": "Analytics not found"}), 404
        with open(analytics_path, "r") as f:
            data = json.load(f)
        logger.error(f"Loaded analytics with {len(data.get('object_counts', {}))} frames")
        if request.args.get("download", "0") == "1":
            return send_file(
                analytics_path,
                mimetype="application/json",
                as_attachment=True,
                download_name=f"analytics_{filename}.json",
            )
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error serving analytics {filename}: {e}")
        return jsonify({"error": f"Error serving analytics: {str(e)}"}), 500


@app.route("/results/<filename>/analytics")
def serve_analytics(filename):
    """Generate and serve analytics data."""
    try:
        result_dir = os.path.join(app.config["RESULTS_FOLDER"], os.path.splitext(filename)[0])
        mot_file = os.path.join(result_dir, "analytics/test.txt")
        frames_folder = os.path.join(result_dir, "frames")
        if not os.path.exists(mot_file):
            logger.error(f"Analytics data missing: MOT={os.path.exists(mot_file)}")
            return jsonify({"error": "Analytics data not found"}), 404
        video_path = os.path.join(result_dir, "annotated_video_web.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return jsonify({"error": "Could not open video for dimensions"}), 500
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        heatmap_path = os.path.join(result_dir, "heatmap.png")
        velocity_heatmap_path = os.path.join(result_dir, "velocity_heatmap.png")
        heatmap_success = generate_mot_heatmap(mot_file, width, height, grid_size=20, output_file=heatmap_path)
        velocity_heatmap_success = generate_velocity_heatmap(
            mot_file, width, height, grid_size=20, output_file=velocity_heatmap_path
        )
        heatmap_base64 = ""
        velocity_heatmap_base64 = ""
        if heatmap_success:
            with open(heatmap_path, "rb") as f:
                heatmap_base64 = base64.b64encode(f.read()).decode("utf-8")
            logger.error(f"Generated heatmap: {heatmap_path}")
        if velocity_heatmap_success:
            with open(velocity_heatmap_path, "rb") as f:
                velocity_heatmap_base64 = base64.b64encode(f.read()).decode("utf-8")
            logger.error(f"Generated velocity heatmap: {velocity_heatmap_path}")
        top_ids, crops = get_longest_staying_ids(mot_file, frames_folder, top_n=5)
        crops_base64 = []
        for id_crops in crops:
            id_crops_base64 = []
            for crop in id_crops:
                _, buffer = cv2.imencode(".jpg", crop)
                crop_base64 = base64.b64encode(buffer).decode("utf-8")
                id_crops_base64.append(crop_base64)
            crops_base64.append(id_crops_base64)
        try:
            df = pd.read_csv(mot_file, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "x_", "y_", "z_"])
            object_counts = df.groupby("frame").size().to_dict()
            track_durations = get_track_durations(mot_file)
            avg_velocity = get_average_velocity(mot_file)
            logger.error(f"Analytics generated: {len(object_counts)} frames, {len(track_durations)} tracks")
        except Exception as e:
            logger.error(f"Error processing MOT file: {e}")
            object_counts = {}
            track_durations = {}
            avg_velocity = 0.0
        return jsonify(
            {
                "heatmap": heatmap_base64,
                "velocity_heatmap": velocity_heatmap_base64,
                "top_ids": top_ids,
                "crops": crops_base64,
                "object_counts": object_counts,
                "track_durations": track_durations,
                "avg_velocity": float(avg_velocity),
            }
        )
    except Exception as e:
        logger.error(f"Error generating analytics for {filename}: {e}")
        return jsonify({"error": f"Error generating analytics: {str(e)}"}), 500
