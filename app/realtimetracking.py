import os
import cv2
import json
import argparse
import numpy as np
import time
from datetime import datetime
from app.CustomBoostTrack.realtime_ensembling import RealTimeTracker
# from CustomBoostTrack.realtime_ensembling import RealTimeTracker

# Directory setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
WORKING_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(WORKING_DIR, exist_ok=True)

def draw_detections(frame, detections):
    """Draw bounding boxes and track IDs on the frame."""
    for det in detections:
        x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
        track_id = det['id']
        conf = det['confidence']
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw track ID and confidence
        label = f"ID: {track_id} ({conf:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

class RealTimeTrackingService:
    def __init__(self, camera_index=0, frame_rate=30):
        # Model paths
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = os.path.dirname(self.APP_DIR)
        self.reid_path = os.path.join(self.APP_DIR, "models", "osnet_ain_ms_m_c.pth.tar")
        self.model1_path = os.path.join(self.APP_DIR, "models", "YOLO11.pt")
        self.model2_path = os.path.join(self.APP_DIR, "models", "YOLO12.pt")
        self.frame_rate = frame_rate
        # Load tracker once
        self.tracker = RealTimeTracker(self.model1_path, self.model2_path, self.reid_path, self.frame_rate)
        self.frame_id = 1

    def process_frame(self, frame):
        # frame: numpy array (BGR)
        detections = self.tracker.update(frame, self.frame_id)
        self.frame_id += 1
        # Draw detections
        annotated = draw_detections(frame.copy(), detections)
        return annotated, detections

def main(camera_index=0, frame_rate=30):
    # Model paths
    reid_path = os.path.join(APP_DIR, "models", "osnet_ain_ms_m_c.pth.tar")
    model1_path = os.path.join(APP_DIR, "models", "YOLO11.pt")
    model2_path = os.path.join(APP_DIR, "models", "YOLO12.pt")

    for path in [reid_path, model1_path, model2_path]:
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return

    # Initialize tracker
    tracker = RealTimeTracker(model1_path, model2_path, reid_path, frame_rate)

    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera (index {camera_index})")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        print("Error: Invalid camera resolution")
        cap.release()
        return

    # Tracking data
    tracking_data = {}
    frame_id = 1
    result_filename = os.path.join(PROJECT_ROOT, 'output', 'results', 'realtime-val', 'BTPP', 'data', 'test.txt')
    if os.path.exists(result_filename):
        os.remove(result_filename)

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Resize frame for efficiency
            frame = cv2.resize(frame, (640, 480))

            # Process frame
            detections = tracker.update(frame, frame_id)
            tracking_data[str(frame_id)] = detections

            # Draw visualizations
            frame = draw_detections(frame, detections)
            cv2.imshow('Real-Time Tracking', frame)

            # Optional: Write to test.txt
            # tracker.write_results_to_file(result_filename, frame_id, detections)

            frame_id += 1
            elapsed_time = time.time() - start_time
            fps = 1 / (elapsed_time + 1e-9)
            print(f"Frame {frame_id-1} FPS: {fps:.1f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Save annotations on exit
        annotations_path = os.path.join(WORKING_DIR, 'annotations.json')
        try:
            with open(annotations_path, 'w') as f:
                json.dump(tracking_data, f, indent=2)
        except Exception as e:
            print(f"Final annotations save failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time multi-object tracking with webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    args = parser.parse_args()

    main(args.camera, args.fps)