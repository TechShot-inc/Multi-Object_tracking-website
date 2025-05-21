import os
import cv2
import json
import argparse
import numpy as np
import time
from datetime import datetime
from app.CustomBoostTrack.realtime_ensembling import RealTimeTracker
import shutil
import pandas as pd
import random
import subprocess

# Directory setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
WORKING_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(WORKING_DIR, exist_ok=True)

def draw_detections(frame, detections):
    """Draw bounding boxes and track IDs on the frame."""
    for det in detections:
        x, y, w, h = int(det['x']), int(det['y']), int(det['width']), int(det['height'])
        track_id = det['id']
        conf = det['confidence']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"ID: {track_id} ({conf:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

class RealTimeTrackingService:
    def __init__(self, camera_index=0, frame_rate=30):
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_ROOT = os.path.dirname(self.APP_DIR)
        self.reid_path = os.path.join(self.APP_DIR, "models", "osnet_ain_ms_m_c.pth.tar")
        self.model1_path = os.path.join(self.APP_DIR, "models", "YOLO11.pt")
        self.model2_path = os.path.join(self.APP_DIR, "models", "YOLO12.pt")
        self.frame_rate = frame_rate
        self.tracker = RealTimeTracker(self.model1_path, self.model2_path, self.reid_path, self.frame_rate)
        self.frame_id = 1
        self.active = False
        self.colors = [
            (66, 135, 245),
            (0, 196, 180),
            (255, 107, 107),
            (156, 39, 176),
            (255, 193, 7),
            (76, 175, 80),
            (33, 150, 243),
            (233, 30, 99),
        ]
        
    def process_frame(self, frame, roi=None):
        """Process frame with optional ROI (coordinates are absolute pixels)"""
        frame_height, frame_width = frame.shape[:2]
        processing_frame = frame.copy()
        original_frame = frame.copy()
        if not self.active:
            self.frame_id = 1
            self.active = True
        else:
            self.frame_id += 1
        roi_applied = False
        x, y, w, h = 0, 0, 0, 0
        if roi and isinstance(roi, dict) and all(k in roi for k in ['x', 'y', 'width', 'height']):
            try:
                x = int(float(roi['x']))
                y = int(float(roi['y']))
                w = int(float(roi['width']))
                h = int(float(roi['height']))
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = max(1, min(w, frame_width - x))
                h = max(1, min(h, frame_height - y))
                if w >= 20 and h >= 20:
                    processing_frame = frame[y:y+h, x:x+w]
                    roi_applied = True
                    print(f"Processing with ROI: x={x}, y={y}, w={w}, h={h}")
                else:
                    print(f"ROI too small (w={w}, h={h}), using full frame")
            except Exception as e:
                print(f"Error applying ROI: {e}")
                processing_frame = frame
        try:
            detections = self.tracker.update(processing_frame, self.frame_id)
            if roi_applied:
                for det in detections:
                    det['x'] += x
                    det['y'] += y
            frame = original_frame.copy()
            if roi_applied:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (45, 212, 191), 2)  # Match GUI color #2DD4BF
                label = "Active ROI"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 5), 
                             (x + label_size[0] + 10, y), (45, 212, 191), -1)
                cv2.putText(frame, label, (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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
            return frame, detections
        except Exception as e:
            print(f"Error in tracking: {e}")
            return original_frame, []

def extract_frames(video_path, output_folder, speed=5):
    """Extract frames using modulo-based sampling."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    frame_count = 0
    saved_frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(speed) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_number += 1
        frame_count += 1
    cap.release()
    expected_frames = total_frames // speed
    if abs(saved_frame_number - expected_frames) > 1:
        print(f"Warning: Expected ~{expected_frames} frames, extracted {saved_frame_number}")
    print(f"Extracted {saved_frame_number} frames to {output_folder}")

def extract_frames_with_roi(video_path, output_folder, speed=5, x=0, y=0, w=0, h=0):
    """Extract frames with ROI using modulo-based sampling."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    frame_count = 0
    saved_frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(speed) == 0:
            roi_frame = frame[y:y+h, x:x+w]
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, roi_frame)
            saved_frame_number += 1
        frame_count += 1
    cap.release()
    expected_frames = total_frames // speed
    if abs(saved_frame_number - expected_frames) > 1:
        print(f"Warning: Expected ~{expected_frames} frames, extracted {saved_frame_number}")
    print(f"Extracted {saved_frame_number} frames with ROI to {output_folder}")

def main(video_path, output_path, speed=5, output_speed=1, roi=None):
    print(f"Processing video with speed={speed}, output_speed={output_speed}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_path = os.path.join(WORKING_DIR, base_name, 'frames')
    result_dir = os.path.dirname(output_path)
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    roi_applied = False
    x, y, w, h = 0, 0, 0, 0
    if roi and isinstance(roi, dict) and all(k in roi for k in ['x', 'y', 'width', 'height']):
        try:
            x = int(float(roi['x']))
            y = int(float(roi['y']))
            w = int(float(roi['width']))
            h = int(float(roi['height']))
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))
            min_width = int(frame_width * 0.05)
            min_height = int(frame_height * 0.05)
            if w >= min_width and h >= min_height:
                roi_applied = True
                print(f"Processing with ROI: x={x}, y={y}, w={w}, h={h}")
            else:
                print(f"ROI too small (w={w}, h={h}), minimum required: {min_width}x{min_height}")
        except Exception as e:
            print(f"Error applying ROI: {e}")
    if roi_applied:
        extract_frames_with_roi(video_path, frames_path, speed=speed, x=x, y=y, w=w, h=h)
    else:
        extract_frames(video_path, frames_path, speed=speed)
    frame_rate = max(1, fps / speed)
    output_frame_rate = max(1, (fps / speed) / output_speed)
    dataset_name = "tarsh-val"
    output_folder = os.path.join(PROJECT_ROOT, 'results', base_name, 'tracking')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    dataset_path = frames_path
    reid_path = os.path.join(APP_DIR, "models", "osnet_ain_ms_m_c.pth.tar")
    model1_path = os.path.join(APP_DIR, "models", "YOLO11.pt")
    model1_weight = 0.7
    model2_path = os.path.join(APP_DIR, "models", "YOLO12.pt")
    model2_weight = 0.3
    custom_boost_track_script = os.path.join(APP_DIR, "CustomBoostTrack", "run_with_ensembler.py")
    tracking_command = [
        "python", custom_boost_track_script, 
        "--dataset", dataset_name, 
        "--exp_name", "BTPP", 
        "--result_folder", output_folder,
        "--frame_rate", str(frame_rate), 
        "--reid_path", reid_path, 
        "--dataset_path", dataset_path,
        "--model1_path", model1_path,
        "--model1_weight", str(model1_weight),
        "--model2_path", model2_path,
        "--model2_weight", str(model2_weight),
        "--conf_thresh", "0.1"
    ]
    print(f"Tracking command: {' '.join(tracking_command)}")
    try:
        subprocess.run(tracking_command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Tracking failed: {e}")
    tracking_results_path = os.path.join(output_folder, "BTPP_post_gbi", "data", "test.txt")
    if not os.path.exists(tracking_results_path):
        found = False
        for root, dirs, files in os.walk(output_folder):
            if "test.txt" in files:
                tracking_results_path = os.path.join(root, "test.txt")
                found = True
                break
        if not found:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    if file.endswith(".txt"):
                        tracking_results_path = os.path.join(root, file)
                        found = True
                        break
                if found:
                    break
        if not found:
            raise FileNotFoundError("Tracking results file (test.txt) not found")
    def draw_tracking_results_with_roi(frames_folder_path, annnots_path, annoted_video_path, frame_rate, output_speed):
        images = [os.path.join(frames_folder_path, path) for path in os.listdir(frames_folder_path) if path.endswith('.jpg')]
        images = sorted(images)
        gt_df = pd.read_csv(annnots_path, header=None)
        if gt_df.shape[1] == 10:
            gt_df.columns = ['frame', 'track_id', 'x', 'y', 'width', 'height', 'confidence', 'x_', 'y_', 'z_']
        elif gt_df.shape[1] == 9:
            gt_df.columns = ['frame', 'track_id', 'x', 'y', 'width', 'height', 'confidence', 'x_', 'y_']
            gt_df['z_'] = -1
        else:
            default_columns = ['frame', 'track_id', 'x', 'y', 'width', 'height', 'confidence']
            gt_df.columns = default_columns + [f'col_{i}' for i in range(gt_df.shape[1] - len(default_columns))]
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        output_frame_rate = max(1, (fps / speed) / output_speed)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(annoted_video_path, fourcc, output_frame_rate, (width, height))
        id_to_color = {}
        tracking_data = {}
        print(f"Writing {len(images)} frames at {output_frame_rate} FPS")
        for frame_id in range(1, len(images)+1):
            frame_path = images[frame_id-1]
            roi_img = cv2.imread(frame_path)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            if roi_applied:
                img[y:y+h, x:x+w] = roi_img
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.rectangle(img, (x, y), (x + w, y + h), (45, 212, 191), 2)  # Match GUI color #2DD4BF
                label = "Active ROI"
                cv2.putText(img, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                img = roi_img
            detections = gt_df[gt_df['frame'] == frame_id]
            frame_detections = []
            for _, row in detections.iterrows():
                track_id = int(row['track_id'])
                det_x = int(row['x'])
                det_y = int(row['y'])
                det_width = int(row['width'])
                det_height = int(row['height'])
                confidence = float(row['confidence'])
                if roi_applied:
                    det_x += x
                    det_y += y
                if track_id not in id_to_color:
                    id_to_color[track_id] = [random.randint(0, 255) for _ in range(3)]
                color = tuple(id_to_color[track_id])
                cv2.rectangle(img, (det_x, det_y), (det_x + det_width, det_y + det_height), color, 2)
                cv2.putText(img, f"ID: {track_id}", (det_x, det_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                frame_detections.append({
                    "id": track_id,
                    "x": det_x,
                    "y": det_y,
                    "width": det_width,
                    "height": det_height,
                    "confidence": confidence
                })
            tracking_data[str(frame_id)] = frame_detections
            video_writer.write(img)
            print(f"Writing frame {frame_id} at {output_frame_rate} FPS")
        video_writer.release()
        print(f"Finished writing video at {annoted_video_path}")
        return tracking_data
    try:
        tracking_data = draw_tracking_results_with_roi(frames_path, tracking_results_path, output_path, frame_rate, output_speed)
    except Exception as e:
        raise RuntimeError(f"Failed to generate annotated video: {e}")
    annotations_path = os.path.join(result_dir, 'annotations.json')
    final_tracking_data = dict(tracking_data)
    try:
        with open(annotations_path, 'w') as f:
            json_data = json.dumps(final_tracking_data, indent=2)
            f.write(json_data)
        file_size = os.path.getsize(annotations_path)
        if file_size <= 2:
            direct_path = os.path.join(result_dir, 'annotations_direct.json')
            with open(direct_path, 'w') as f:
                f.write("{\n")
                for i, (frame_id, detections) in enumerate(final_tracking_data.items()):
                    f.write(f'  "{frame_id}": {json.dumps(detections)}')
                    if i < len(final_tracking_data) - 1:
                        f.write(",\n")
                    else:
                        f.write("\n")
                f.write("}\n")
            if os.path.getsize(direct_path) > 10:
                shutil.copy2(direct_path, annotations_path)
    except Exception:
        try:
            if tracking_data and len(tracking_data) > 0:
                first_frame = next(iter(tracking_data))
                fallback_data = {first_frame: tracking_data[first_frame]}
                with open(annotations_path, 'w') as f:
                    json.dump(fallback_data, f)
        except:
            with open(annotations_path, 'w') as f:
                f.write('{"1":[]}')
    try:
        shutil.rmtree(frames_path)
        shutil.rmtree(output_folder)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary folders: {e}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video multi-object tracking")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output video file")
    parser.add_argument("--speed", type=float, default=5, help="Frame extraction speed")
    parser.add_argument("--output_speed", type=float, default=1, help="Output video speed")
    parser.add_argument("--roi", type=str, help="ROI coordinates as x,y,w,h (e.g., 100,100,200,200)")
    args = parser.parse_args()
    roi = None
    if args.roi:
        try:
            roi_coords = args.roi.split(',')
            if len(roi_coords) == 4:
                roi = {
                    'x': float(roi_coords[0]),
                    'y': float(roi_coords[1]),
                    'width': float(roi_coords[2]),
                    'height': float(roi_coords[3])
                }
            else:
                print("Error: ROI must be four numbers separated by commas")
                roi = None
        except ValueError:
            print("Error: ROI must be four numbers separated by commas")
            roi = None
    main(args.video, args.output, args.speed, args.output_speed, roi=roi)