import os
import cv2
import json
import argparse
import numpy as np
import time
from datetime import datetime
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
        self.model1_path = os.path.join(self.APP_DIR, "models", "YOLO11.pt")
        self.model2_path = os.path.join(self.APP_DIR, "models", "YOLO12.pt")
        self.frame_rate = frame_rate
        # Load tracker once
        self.tracker = RealTimeTracker(self.model1_path, self.model2_path, self.reid_path, self.frame_rate)
        # Initialize frame counter and tracking state
        self.frame_id = 1
        self.active = False
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
        
    def process_frame(self, frame, roi=None):
        """Process frame with optional ROI (coordinates can be either absolute or in relative 0-1 range)"""
        frame_height, frame_width = frame.shape[:2]
        processing_frame = np.ascontiguousarray(frame)
        original_frame = frame.copy()
    
        # Increment frame ID for continuous tracking
        if not self.active:
            self.frame_id = 1
            self.active = True
        else:
            self.frame_id += 1
    
        # Handle ROI if provided
        roi_applied = False
        x, y, w, h = 0, 0, 0, 0
        
        if roi and isinstance(roi, dict) and all(k in roi for k in ['x', 'y', 'width', 'height']):
            try:
                # Check if coordinates appear to be relative (all values <= 1.0)
                is_relative = all(0 <= float(roi[k]) <= 1.0 for k in ['x', 'y', 'width', 'height'])
                
                if is_relative:
                    # Convert relative coordinates (0-1) to absolute pixels
                    x = int(float(roi['x']) * frame_width)
                    y = int(float(roi['y']) * frame_height)
                    w = int(float(roi['width']) * frame_width)
                    h = int(float(roi['height']) * frame_height)
                else:
                    # Use absolute coordinates
                    x = int(float(roi['x']))
                    y = int(float(roi['y']))
                    w = int(float(roi['width']))
                    h = int(float(roi['height']))
                
                # Ensure coordinates are valid
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = max(1, min(w, frame_width - x))
                h = max(1, min(h, frame_height - y))
                
                # Only process ROI if it's large enough
                min_width = int(frame_width * 0.05)  # 5% of frame width
                min_height = int(frame_height * 0.05)  # 5% of frame height
                
                if w >= min_width and h >= min_height:
                    # Create a contiguous copy of the ROI
                    processing_frame = np.ascontiguousarray(frame[y:y+h, x:x+w])
                    roi_applied = True
                    print(f"Processing with ROI: x={x}, y={y}, w={w}, h={h}")
                else:
                    print(f"ROI too small (w={w}, h={h}), minimum required: {min_width}x{min_height}")
            except Exception as e:
                print(f"Error applying ROI: {e}")
                processing_frame = frame
        
        try:
            # Ensure frame is in BGR format for OpenCV operations
            if processing_frame.shape[2] != 3:
                print("Warning: Converting frame to BGR format")
                processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_RGB2BGR)
            
            # Pass processing_frame to tracker
            detections = self.tracker.update(processing_frame, self.frame_id, roi=roi)
            
            # If ROI was applied, adjust coordinates back to original frame
            if roi_applied:
                for det in detections:
                    det['x'] += x
                    det['y'] += y
            
            # Start with clean original frame for drawing
            frame = original_frame.copy()
            
            # Always draw ROI if coordinates are provided, even during selection
            if roi and isinstance(roi, dict) and all(k in roi for k in ['x', 'y', 'width', 'height']):
                # Draw ROI with semi-transparent overlay
                overlay = frame.copy()
                # Darken the area outside ROI
                cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
                # Cut out the ROI area
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
                # Apply the overlay with transparency
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Draw ROI border with animation effect
                border_color = self.colors[0]
                border_thickness = 2
                # Draw main border
                cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_thickness)
                
                # Add ROI label with background
                label = "Active ROI" if roi_applied else "Selecting ROI"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 5), 
                             (x + label_size[0] + 10, y), border_color, -1)
                cv2.putText(frame, label, (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw detections with premium styling
            for det in detections:
                color = self.colors[det['id'] % len(self.colors)]
                x1, y1 = int(det['x']), int(det['y'])
                x2, y2 = x1 + int(det['width']), y1 + int(det['height'])
                
                # Draw premium bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and confidence with better styling
                label = f"ID: {det['id']:d} ({det['confidence']:.2f})"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - baseline - 2),
                            (x1 + label_size[0] + 2, y1), color, -1)
                # Draw text in white
                cv2.putText(frame, label, (x1 + 1, y1 - baseline - 1),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add stats overlay at the bottom of frame
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
            
            # Add text
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
            
            return frame, detections
            
        except Exception as e:
            print(f"Error in tracking: {e}")
            return original_frame, []

def main(camera_index=0, frame_rate=30, roi=None):
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

    # Convert ROI list to dictionary
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

            # Draw visualizations
            frame = draw_detections(original_frame, detections)
            
            # Comment out the window display
            # cv2.imshow('Real-Time Tracking', frame)

            frame_id += 1
            elapsed_time = time.time() - start_time
            fps = 1 / (elapsed_time + 1e-9)
            print(f"Frame {frame_id-1} FPS: {fps:.1f}")

            # Comment out the waitKey
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        cap.release()
        # Comment out destroyAllWindows
        # cv2.destroyAllWindows()

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