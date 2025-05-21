import os
import cv2
import json
import argparse
import numpy as np
import time
from datetime import datetime
import torch
from app.CustomBoostTrack.realtime_ensembling import RealTimeTracker

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
        self.model1_path = os.path.join(self.APP_DIR, "models", "General2.pt")
        self.model2_path = os.path.join(self.APP_DIR, "models", "12General1.pt")
        self.frame_rate = frame_rate
        # Load tracker once
        self.tracker = RealTimeTracker(self.model1_path, self.model2_path, self.reid_path, self.frame_rate)
        # Initialize frame counter and tracking state
        self.frame_id = 1
        self.active = False
        # Line crossing counters
        self.persons_inside = 0
        self.persons_outside = 0
        # Color settings for visualization        
        self.colors = [
            (66, 135, 245),   # Premium blue
            (0, 196, 180),    # Teal
            (255, 107, 107),  # Coral
            (156, 39, 176),   # Purple
            (255, 193, 7),    # Amber
            (76, 175, 80),    # Green
            (33, 150, 243),   # Blue
            (233, 30, 99),    # Pink
        ]
            
    def process_frame(self, frame, roi=None, line=None):
        """Process frame with optional ROI and line crossing detection."""
        frame_height, frame_width = frame.shape[:2]
        processing_frame = np.ascontiguousarray(frame)
        original_frame = frame.copy()
    
        # Increment frame ID for continuous tracking
        if not self.active:
            self.frame_id = 1
            self.active = True
            self.persons_inside = 0
            self.persons_outside = 0
        else:
            self.frame_id += 1
    
        # Handle ROI if provided
        roi_applied = False
        x_roi, y_roi, w_roi, h_roi = 0, 0, frame_width, frame_height
        
        if roi and isinstance(roi, dict) and all(k in roi for k in ['x', 'y', 'width', 'height']):
            try:
                is_relative = all(0 <= float(roi[k]) <= 1.0 for k in ['x', 'y', 'width', 'height'])
                if is_relative:
                    x_roi = int(float(roi['x']) * frame_width)
                    y_roi = int(float(roi['y']) * frame_height)
                    w_roi = int(float(roi['width']) * frame_width)
                    h_roi = int(float(roi['height']) * frame_height)
                else:
                    x_roi = int(float(roi['x']))
                    y_roi = int(float(roi['y']))
                    w_roi = int(float(roi['width']))
                    h_roi = int(float(roi['height']))
                
                x_roi = max(0, min(x_roi, frame_width - 1))
                y_roi = max(0, min(y_roi, frame_height - 1))
                w_roi = max(1, min(w_roi, frame_width - x_roi))
                h_roi = max(1, min(h_roi, frame_height - y_roi))
                
                min_width = int(frame_width * 0.05)
                min_height = int(frame_height * 0.05)
                
                if w_roi >= min_width and h_roi >= min_height:
                    processing_frame = np.ascontiguousarray(frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi])
                    roi_applied = True
                    print(f"Processing with ROI: x={x_roi}, y={y_roi}, w={w_roi}, h={h_roi}")
                else:
                    print(f"ROI too small (w={w_roi}, h={h_roi}), minimum required: {min_width}x{min_height}")
            except Exception as e:
                print(f"Error applying ROI: {e}")
                processing_frame = frame
        
        try:
            if processing_frame.shape[2] != 3:
                print("Warning: Converting frame to BGR format")
                processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_RGB2BGR)
            
            detections = self.tracker.update(processing_frame, self.frame_id, roi=roi)
            
            if roi_applied:
                for det in detections:
                    det['x'] += x_roi
                    det['y'] += y_roi
            
            # Debug detections
            for det in detections:
                print(f"Detection: id={det['id']}, class={det.get('class')}, x={det['x']}, y={det['y']}, w={det['width']}, h={det['height']}, conf={det['confidence']}")
            
            # Line crossing detection
            counts = {'inside': self.persons_inside, 'outside': self.persons_outside}
            print(f"Line data received: {line}")
            if line and isinstance(line, dict) and all(k in line for k in ['position', 'x']):
                try:
                    line_x = int(float(line['x']) * frame_width)
                    line_x = max(0, min(line_x, frame_width - 1))
                    position = line['position']
                    print(f"Line: position={position}, x={line_x}")
                    
                    # Reset counts for this frame
                    inside_count = 0
                    outside_count = 0
                    
                    # Process detections for line crossing
                    for det in detections:
                        x, w = det['x'], det['width']
                        left_edge = x
                        right_edge = x + w
                        # Check if detection is a person (assume person if class is None, since YOLO detects person)
                        det_class = det.get('class')
                        is_person = det_class is None or det_class == 0 or str(det_class).lower() == 'person'
                        print(f"Checking detection: id={det['id']}, class={det_class}, is_person={is_person}, left_edge={left_edge}, right_edge={right_edge}")
                        if is_person:
                            if position == 'left':
                                # Inside is left side, outside is right side
                                if right_edge <= line_x:  # Fully left of line
                                    inside_count += 1
                                    print(f"Person fully left of line_x={line_x}, inside_count={inside_count}")
                                elif left_edge >= line_x:  # Fully right of line
                                    outside_count += 1
                                    print(f"Person fully right of line_x={line_x}, outside_count={outside_count}")
                            else:  # position == 'right'
                                # Inside is right side, outside is left side
                                if left_edge >= line_x:  # Fully right of line
                                    inside_count += 1
                                    print(f"Person fully right of line_x={line_x}, inside_count={inside_count}")
                                elif right_edge <= line_x:  # Fully left of line
                                    outside_count += 1
                                    print(f"Person fully left of line_x={line_x}, outside_count={outside_count}")
                        else:
                            print(f"Skipping non-person detection: id={det['id']}, class={det_class}")
                    
                    # Update persistent counts
                    self.persons_inside = inside_count
                    self.persons_outside = outside_count
                    counts = {'inside': self.persons_inside, 'outside': self.persons_outside}
                    print(f"Updated counts: inside={self.persons_inside}, outside={self.persons_outside}")
                except Exception as e:
                    print(f"Error processing line crossing: {e}")
            else:
                print("No valid line data for counting")
            
            # Start with clean original frame for drawing
            frame = original_frame.copy()
            
            # Draw ROI if provided
            if roi and isinstance(roi, dict) and all(k in roi for k in ['x', 'y', 'width', 'height']):
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
                cv2.rectangle(overlay, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 0, 0), -1)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                border_color = self.colors[0]
                cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), border_color, 2)
                label = "Active ROI" if roi_applied else "Selecting ROI"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x_roi, y_roi - label_size[1] - 5), 
                             (x_roi + label_size[0] + 10, y_roi), border_color, -1)
                cv2.putText(frame, label, (x_roi + 5, y_roi - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw line if provided
            if line and isinstance(line, dict) and all(k in line for k in ['position', 'x']):
                line_x = int(float(line['x']) * frame_width)
                cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)
                label = f"Line: {line['position'].capitalize()}"
                cv2.putText(frame, label, (line_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw detections
            for det in detections:
                color = self.colors[det['id'] % len(self.colors)]
                x1, y1 = int(det['x']), int(det['y'])
                x2, y2 = x1 + int(det['width']), y1 + int(det['height'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {det['id']:d} ({det['confidence']:.2f})"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - baseline - 2),
                            (x1 + label_size[0] + 2, y1), color, -1)
                cv2.putText(frame, label, (x1 + 1, y1 - baseline - 1),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add stats overlay
            stats_height = 40
            stats_margin = 10
            cv2.rectangle(frame, 
                         (stats_margin, frame_height - stats_height - stats_margin),
                         (frame_width - stats_margin, frame_height - stats_margin), 
                         (0, 0, 0), -1)
            cv2.rectangle(frame, 
                         (stats_margin, frame_height - stats_height - stats_margin),
                         (frame_width - stats_margin, frame_height - stats_margin), 
                         (255, 255, 255), 1)
            
            objects_text = f"Objects: {len(detections)}"
            frame_text = f"Frame: {self.frame_id}"
            time_text = datetime.now().strftime("%H:%M:%S")
            
            cv2.putText(frame, objects_text, 
                      (stats_margin + 10, frame_height - stats_margin - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, frame_text, 
                      (frame_width // 2 - 40, frame_height - stats_margin - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, time_text, 
                      (frame_width - stats_margin - 100, frame_height - stats_margin - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame, detections, counts
            
        except Exception as e:
            print(f"Error in tracking: {e}")
            return original_frame, [], counts

def main(camera_index=0, frame_rate=30, roi=None):
    reid_path = os.path.join(APP_DIR, "models", "osnet_ain_ms_m_c.pth.tar")
    model1_path = os.path.join(APP_DIR, "models", "General2.pt")
    model2_path = os.path.join(APP_DIR, "models", "12General1.pt")

    for path in [reid_path, model1_path, model2_path]:
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return

    tracker = RealTimeTracker(model1_path, model2_path, reid_path, frame_rate)

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

    roi_dict = None
    if roi:
        try:
            if isinstance(roi, list) and len(roi) == 4:
                roi_dict = {
                    'x': max(0, float(roi[0])),
                    'y': max(0, float(roi[1])),
                    'width': max(0, float(roi[2])),
                    'height': max(0, float(roi[3]))
                }
                if roi_dict['width'] > width - roi_dict['x'] or roi_dict['height'] > height - roi_dict['y']:
                    print("Warning: ROI exceeds frame bounds, adjusting")
                    roi_dict['width'] = min(roi_dict['width'], width - roi_dict['x'])
                    roi_dict['height'] = min(roi_dict['height'], height - roi_dict['y'])
                print(f"Using ROI: {roi_dict}")
            else:
                print("Error: ROI must be a list of four numbers")
                roi_dict = None
        except (ValueError, TypeError) as e:
            print(f"Error parsing ROI: {e}")
            roi_dict = None

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

            frame = cv2.resize(frame, (640, 480))

            original_frame = frame.copy()
            if roi_dict:
                try:
                    x, y, w, h = int(roi_dict['x']), int(roi_dict['y']), int(roi_dict['width']), int(roi_dict['height'])
                    if w > 0 and h > 0:
                        frame = frame[y:y+h, x:x+w]
                        print(f"Main: Cropped to ROI, shape: {frame.shape}")
                    else:
                        print("Main: Invalid ROI dimensions, using full frame")
                        frame = original_frame
                except (KeyError, TypeError) as e:
                    print(f"Error cropping frame in main: {e}")
                    frame = original_frame

            detections = tracker.update(frame, frame_id, roi=roi_dict)
            if roi_dict and w > 0 and h > 0:
                for det in detections:
                    try:
                        det['x'] = float(det['x']) + x
                        det['y'] = float(det['y']) + y
                    except (KeyError, TypeError) as e:
                        print(f"Error adjusting detection in main: {e}")

            tracking_data[str(frame_id)] = detections

            frame = draw_detections(original_frame, detections)
            
            frame_id += 1
            elapsed_time = time.time() - start_time
            fps = 1 / (elapsed_time + 1e-9)
            print(f"Frame {frame_id-1} FPS: {fps:.1f}")

    finally:
        cap.release()

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
    parser.add_argument("--roi", type=str, help="ROI coordinates as x,y,w,h (e.g., 100,100,200,200)")
    args = parser.parse_args()

    roi = None
    if args.roi:
        try:
            roi_coords = args.roi.split(',')
            if len(roi_coords) == 4:
                roi = [float(coord) for coord in roi_coords]
            else:
                print("Error: ROI must be four numbers separated by commas")
                roi = None
        except ValueError:
            print("Error: ROI must be four numbers separated by commas")
            roi = None

    main(args.camera, args.fps, roi=roi)