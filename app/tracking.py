import cv2
import numpy as np
import os
import json
import base64
from datetime import datetime
import logging
import shutil
import pandas as pd
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoTracker:
    def __init__(self, video_path, output_dir, speed=1, output_speed=1.0, roi=None):
        """Initialize video tracker."""
        self.video_path = video_path
        self.output_dir = output_dir
        self.speed = max(1, int(speed))  # Ensure speed is an integer >= 1
        self.output_speed = output_speed
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise Exception("Failed to open video")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_fps = max(1, (self.fps / self.speed) * output_speed)
        self.frame_interval = max(1, int(self.fps / (self.fps / self.speed)))
        logger.info(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames, speed: {self.speed}, output_speed: {output_speed}, output FPS: {self.output_fps}, frame_interval: {self.frame_interval}")
        
        self.roi = None
        if roi:
            if isinstance(roi, dict):
                required_keys = ['x', 'y', 'width', 'height']
                if not all(k in roi for k in required_keys):
                    logger.error(f"Invalid ROI dictionary: {roi}")
                    raise ValueError("ROI dictionary must contain x, y, width, height")
                try:
                    self.roi = [int(roi['x']), int(roi['y']), 
                               int(roi['width']), int(roi['height'])]
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid ROI values: {roi}, error: {e}")
                    raise ValueError(f"ROI values must be numeric: {roi}")
            elif isinstance(roi, (list, tuple)) and len(roi) == 4:
                try:
                    self.roi = [int(v) for v in roi]
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid ROI values: {roi}, error: {e}")
                    raise ValueError(f"ROI values must be numeric: {roi}")
            else:
                logger.error(f"Invalid ROI format: {roi}")
                raise ValueError("ROI must be a list/tuple of [x, y, w, h] or a dictionary")
            
            roi_x, roi_y, roi_w, roi_h = self.roi
            if roi_x < 0 or roi_y < 0 or roi_w <= 0 or roi_h <= 0:
                logger.error(f"Invalid ROI dimensions: {self.roi}")
                raise ValueError("ROI dimensions must be positive")
            if roi_x + roi_w > self.width or roi_y + roi_h > self.height:
                logger.error(f"ROI exceeds video dimensions: {self.roi}")
                raise ValueError(f"ROI exceeds video dimensions: {self.width}x{self.height}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = os.path.join(output_dir, f"annotated_video_web_{timestamp}.mp4")
        self.frames_path = os.path.join(output_dir, f"frames_{timestamp}")
        os.makedirs(self.frames_path, exist_ok=True)
        self.analytics_data = {
            "object_counts": {},
            "track_durations": {},
            "heatmap": None,
            "velocity_heatmap": None,
            "top_ids": [],
            "crops": [],
            "avg_velocity": 0.0
        }
        self.tracks = {}
        self.velocities = {}
        self.density_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.velocity_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.frame_number = 0
        self.id_to_color = {}
        self.cap.release()

    def extract_frames(self):
        """Extract every nth frame based on speed."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            raise Exception("Failed to open video")
        saved_frame_number = 0
        original_frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if original_frame_number % self.speed == 0:
                frame_filename = os.path.join(self.frames_path, f"frame_{saved_frame_number:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_number += 1
            original_frame_number += 1
        cap.release()
        logger.info(f"Extracted {saved_frame_number} frames to {self.frames_path}")

    def calculate_intersection_area(self, box, roi):
        """Calculate intersection area between box and ROI."""
        x1, y1, x2, y2 = box
        rx, ry, rw, rh = roi
        rx2, ry2 = rx + rw, ry + rh
        xi1 = max(x1, rx)
        yi1 = max(y1, ry)
        xi2 = min(x2, rx2)
        yi2 = min(y2, ry2)
        if xi2 > xi1 and yi2 > yi1:
            return (xi2 - xi1) * (yi2 - yi1)
        return 0

    def clip_box_to_roi(self, box, roi):
        """Clip box to ROI boundaries."""
        x1, y1, x2, y2 = box
        rx, ry, rw, rh = roi
        rx2, ry2 = rx + rw, ry + rh
        x1 = max(x1, rx)
        y1 = max(y1, ry)
        x2 = min(x2, rx2)
        y2 = min(y2, ry2)
        if x2 > x1 and y2 > y1:
            return [int(x1), int(y1), int(x2), int(y2)]
        return None

    def apply_roi(self, detections):
        """Filter and clip detections based on ROI (75% overlap)."""
        if not self.roi or not detections:
            return detections
        roi_x, roi_y, roi_w, roi_h = self.roi
        filtered_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls, track_id = det
            box = [x1, y1, x2, y2]
            box_area = (x2 - x1) * (y2 - y1)
            if box_area <= 0:
                continue
            intersection_area = self.calculate_intersection_area(box, self.roi)
            overlap = intersection_area / box_area if box_area > 0 else 0
            if overlap >= 0.75:
                clipped_box = self.clip_box_to_roi(box, self.roi)
                if clipped_box:
                    filtered_detections.append(clipped_box + [conf, cls, track_id])
        return filtered_detections

    def draw_detections(self, frame, detections, frame_id):
        """Draw detections and ROI on frame."""
        if self.roi:
            roi_x, roi_y, roi_w, roi_h = self.roi
            try:
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                             (45, 212, 191), 2)
                cv2.putText(frame, "ROI", (roi_x + 5, roi_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                logger.error(f"Error drawing ROI {self.roi}: {e}")
                raise

        frame_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls, track_id = det
            try:
                x1 = max(0, min(float(x1), self.width))
                y1 = max(0, min(float(y1), self.height))
                x2 = max(0, min(float(x2), self.width))
                y2 = max(0, min(float(y2), self.height))
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                if x2 <= x1 or y2 <= y1:
                    continue
                if track_id not in self.id_to_color:
                    self.id_to_color[track_id] = [random.randint(0, 255) for _ in range(3)]
                color = tuple(self.id_to_color[track_id])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {int(track_id)}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                self.density_map[y1:y2, x1:x2] += 1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id not in self.velocities:
                    self.velocities[track_id] = []
                if track_id in self.tracks and frame_id > 1:
                    prev_x, prev_y, prev_frame = self.tracks[track_id]
                    frame_gap = frame_id - prev_frame
                    if frame_gap <= self.frame_interval:
                        dx = center_x - prev_x
                        dy = center_y - prev_y
                        velocity = np.sqrt(dx**2 + dy**2) / frame_gap
                        self.velocities[track_id].append(velocity)
                        self.velocity_map[y1:y2, x1:x2] += velocity
                self.tracks[track_id] = (center_x, center_y, frame_id)
                frame_detections.append({
                    "id": int(track_id),
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": conf
                })
            except Exception as e:
                logger.error(f"Error drawing detection {det}: {str(e)}")
                continue
        return frame, frame_detections

    def generate_heatmap(self, data_map):
        """Generate heatmap from density or velocity map."""
        if np.max(data_map) == 0:
            return None
        normalized = cv2.normalize(data_map, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.png', heatmap)
        return base64.b64encode(buffer).decode('utf-8')

    def process_video(self, detections_list):
        """Process video using sampled frames and GBI detections."""
        # Extract sampled frames
        self.extract_frames()
        images = [os.path.join(self.frames_path, path) for path in os.listdir(self.frames_path) if path.endswith('.jpg')]
        images = sorted(images)
        if not images:
            logger.error(f"No frames extracted in {self.frames_path}")
            raise Exception("No frames extracted")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, self.output_fps,
                             (self.width, self.height))
        if not out.isOpened():
            logger.error(f"Failed to open video writer: {self.output_video_path}")
            raise Exception("Failed to open video writer")

        # Process each sampled frame
        tracking_data = {}
        for frame_id, frame_path in enumerate(images, 1):
            self.frame_number = frame_id - 1
            img = cv2.imread(frame_path)
            if img is None:
                logger.warning(f"Failed to read frame {frame_path}")
                continue
            # Get GBI detections for this frame
            detections = detections_list.get(frame_id, [])
            detections = self.apply_roi(detections)
            img, frame_detections = self.draw_detections(img, detections, frame_id)
            out.write(img)
            self.analytics_data["object_counts"][frame_id] = len(detections)
            tracking_data[str(frame_id)] = frame_detections

        out.release()

        # Update track durations
        for track_id, (x, y, last_frame) in self.tracks.items():
            self.analytics_data["track_durations"][int(track_id)] = last_frame

        # Generate heatmaps
        self.analytics_data["heatmap"] = self.generate_heatmap(self.density_map)
        self.analytics_data["velocity_heatmap"] = self.generate_heatmap(self.velocity_map)

        # Calculate average velocity
        total_velocity = sum([sum(v) for v in self.velocities.values()])
        total_count = sum([len(v) for v in self.velocities.values()])
        self.analytics_data["avg_velocity"] = total_velocity / total_count if total_count > 0 else 0.0

        # Get top track IDs
        top_ids = sorted(self.analytics_data["track_durations"],
                        key=self.analytics_data["track_durations"].get, reverse=True)[:5]
        self.analytics_data["top_ids"] = [int(id) for id in top_ids]

        # Generate crops for top tracks
        cap = cv2.VideoCapture(self.video_path)
        for track_id in self.analytics_data["top_ids"]:
            crops = []
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            saved_frame_number = 0
            while len(crops) < 3 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.speed == 0:
                    saved_frame_number += 1
                    frame_key = saved_frame_number
                    detections = detections_list.get(frame_key, [])
                    detections = self.apply_roi(detections)
                    for det in detections:
                        if int(det[6]) == track_id:
                            x1, y1, x2, y2 = map(int, [det[0], det[1], det[2], det[3]])
                            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > self.width or y2 > self.height:
                                continue
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                _, buffer = cv2.imencode('.jpg', crop)
                                crops.append(base64.b64encode(buffer).decode('utf-8'))
                frame_idx += 1
            self.analytics_data["crops"].append(crops)
        cap.release()

        # Save analytics
        analytics_path = os.path.join(self.output_dir, "analytics.json")
        try:
            with open(analytics_path, 'w') as f:
                json.dump(self.analytics_data, f, indent=4)
            file_size = os.path.getsize(analytics_path)
            if file_size <= 2:
                direct_path = os.path.join(self.output_dir, 'analytics_direct.json')
                with open(direct_path, 'w') as f:
                    f.write(json.dumps(self.analytics_data, indent=2))
                if os.path.getsize(direct_path) > 10:
                    shutil.copy2(direct_path, analytics_path)
        except Exception:
            logger.warning("Failed to save full analytics, saving fallback")
            fallback_data = {"1": tracking_data.get("1", [])}
            with open(analytics_path, 'w') as f:
                json.dump(fallback_data, f)

        # Clean up frames
        if os.path.exists(self.frames_path):
            shutil.rmtree(self.frames_path)

        return self.output_video_path, analytics_path

    def cleanup(self):
        """Clean up resources."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

def process_video_with_tracking(video_path, output_dir, detections, speed=1, output_speed=1.0, roi=None):
    """Main function to process video with tracking."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        tracker = VideoTracker(video_path, output_dir, speed, output_speed, roi)
        output_video, analytics_file = tracker.process_video(detections)
        return output_video, analytics_file
    except Exception as e:
        logger.error(f"Error in process_video_with_tracking: {e}")
        raise