from flask import request, jsonify
from app.app import app
from .realtimetracking import RealTimeTrackingService, draw_detections
import json
import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Initialize real-time tracker
realtime_tracker = RealTimeTrackingService(camera_index=0, frame_rate=30)

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
        tracker = BYTETracker()  # Note: BYTETracker import is assumed to be available
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