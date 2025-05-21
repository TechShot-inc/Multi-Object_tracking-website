import time
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO
import cv2

# Add CustomBoostTrack to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracker.boost_track import BoostTrack
from detectors import YoloDetector, EnsembleDetector
import dataset
from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings

class RealTimeTracker:
    def __init__(self, model1_path, model2_path, reid_path, frame_rate=30):
        """Initialize tracker with YOLO models and BoostTrack."""
        self.frame_rate = frame_rate
        self.frame_id = 0
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize YOLO models with FP16
        self.model1 = YoloDetector("yolo11x.pt", conf=0.3)
        self.model2 = YoloDetector("yolo12x.pt", conf=0.3)
        try:
            self.model1.model.float16 = True  # Enable FP16
            self.model2.model.float16 = True
            # Move models to GPU
            self.model1.model.to(self.device)
            self.model2.model.to(self.device)
        except AttributeError:
            print("Warning: FP16 not supported by ultralytics, using FP32")

        self.detector = EnsembleDetector(
            self.model1, self.model2,
            model1_weight=0.7, model2_weight=0.3,
            iou_thresh=0.5, conf_thresh=0.2
        )

        # Optimize BoostTrack settings
        GeneralSettings.values['dataset'] = 'realtime'
        GeneralSettings.values['use_embedding'] = True
        GeneralSettings.values['use_ecc'] = True
        GeneralSettings.values['reid_path'] = reid_path
        GeneralSettings.values['min_hits'] = 1
        GeneralSettings.values['iou_threshold'] = 0.5
        GeneralSettings.values['min_box_area'] = 0
        GeneralSettings.values['aspect_ratio_thresh'] = 100.0
        GeneralSettings.values['max_age'] = 30
        GeneralSettings.values['det_thresh'] = 0.2

        BoostTrackSettings.values['lambda_iou'] = 1.0
        BoostTrackSettings.values['lambda_mhd'] = 0.0
        BoostTrackSettings.values['lambda_shape'] = 0.0
        BoostTrackSettings.values['use_dlo_boost'] = False
        BoostTrackSettings.values['use_duo_boost'] = False
        BoostTrackSettings.values['s_sim_corr'] = False

        BoostTrackPlusPlusSettings.values['use_rich_s'] = False
        BoostTrackPlusPlusSettings.values['use_sb'] = False
        BoostTrackPlusPlusSettings.values['use_vt'] = False

        # Initialize BoostTrack
        self.tracker = BoostTrack(video_name="realtime")
        self.preproc = dataset.ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def update(self, frame, frame_id, roi=None):
        """Process a single frame and return detections."""
        self.frame_id = frame_id
        start_time = time.time()

        # Validate and apply ROI
        original_frame = frame.copy()
        roi_applied = False
        x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]

        if roi:
            try:
                if not isinstance(roi, dict) or not all(k in roi for k in ['x', 'y', 'width', 'height']):
                    print(f"Invalid ROI format: {roi}")
                    roi = None
                else:
                    # Convert relative coordinates to absolute if needed
                    if all(0 <= float(roi[k]) <= 1.0 for k in ['x', 'y', 'width', 'height']):
                        x = int(float(roi['x']) * frame.shape[1])
                        y = int(float(roi['y']) * frame.shape[0])
                        w = int(float(roi['width']) * frame.shape[1])
                        h = int(float(roi['height']) * frame.shape[0])
                    else:
                        x = int(float(roi['x']))
                        y = int(float(roi['y']))
                        w = int(float(roi['width']))
                        h = int(float(roi['height']))
                    
                    # Ensure coordinates are valid
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    # Calculate minimum size (5% of frame dimensions)
                    min_width = int(frame.shape[1] * 0.05)
                    min_height = int(frame.shape[0] * 0.05)
                    
                    if w >= min_width and h >= min_height:
                        # Create a contiguous copy of the ROI
                        frame = np.ascontiguousarray(frame[y:y+h, x:x+w])
                        roi_applied = True
                        print(f"Tracker: Using ROI with dimensions {w}x{h} (min: {min_width}x{min_height})")
                    else:
                        print(f"Tracker: ROI too small ({w}x{h}), minimum required: {min_width}x{min_height}")
                        roi = None
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error processing ROI: {e}")
                roi = None

        # Ensure frame is in BGR format for OpenCV operations
        if frame.shape[2] != 3:
            print("Warning: Converting frame to BGR format")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Create a copy for detection and ensure it's in RGB format for YOLO
        detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess frame for tracking
        height, width = frame.shape[:2]
        img, _ = self.preproc(frame, None, (height, width))
        img = img.reshape(1, *img.shape)
        img = torch.from_numpy(img).to(self.device)  # Move to GPU

        # Run detection with ensembling for every frame
        dets = self.detector(detection_frame)
        # Ensure dets is a numpy array on CPU before any numpy operation
        if torch.is_tensor(dets):
            dets = dets.cpu().numpy()
        if len(dets) == 0:
            dets = np.empty((0, 5))
        else:
            # Filter detections to only include those within the ROI
            if roi_applied:
                # Convert detections to absolute coordinates
                dets_abs = dets.copy()
                dets_abs[:, 0] += x  # x1
                dets_abs[:, 1] += y  # y1
                dets_abs[:, 2] += x  # x2
                dets_abs[:, 3] += y  # y2
                # Check if detection is within ROI
                in_roi = (
                    (dets_abs[:, 0] >= x) &  # x1 >= roi_x
                    (dets_abs[:, 1] >= y) &  # y1 >= roi_y
                    (dets_abs[:, 2] <= x + w) &  # x2 <= roi_x + roi_w
                    (dets_abs[:, 3] <= y + h)  # y2 <= roi_y + roi_h
                )
                dets = dets[in_roi]
                print(f"Filtered detections: {len(dets)} within ROI")

        # Run BoostTrack
        if len(dets) > 0:
            try:
                # Use the original BGR frame for the tracker
                # Ensure img is on CPU if BoostTrack expects numpy
                img_for_tracker = img
                if isinstance(img_for_tracker, torch.Tensor):
                    if img_for_tracker.device.type != 'cpu':
                        img_for_tracker = img_for_tracker.cpu()
                    img_for_tracker = img_for_tracker.numpy()
                targets = self.tracker.update(dets, img_for_tracker, frame, f"realtime:{frame_id}")
                # Minimal filtering: confidence > 0.2
                if isinstance(targets, torch.Tensor):
                    targets = targets.cpu().numpy()
                if len(targets) > 0:
                    mask = targets[:, 5] > 0.2
                    targets = targets[mask]
                tlwhs, ids, confs = self._process_targets(targets)
            except Exception as e:
                print(f"Tracker error: {e}")
                tlwhs, ids, confs = [], [], []
        else:
            tlwhs, ids, confs = [], [], []

        # Adjust detections for ROI offset
        if roi_applied:
            for tlwh in tlwhs:
                tlwh[0] += x  # Adjust x
                tlwh[1] += y  # Adjust y

        # Format detections for output
        detections = []
        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            x, y, w, h = tlwh
            detections.append({
                "id": int(track_id),
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "confidence": float(conf)
            })

        elapsed_time = time.time() - start_time
        fps = 1 / (elapsed_time + 1e-9)
        print(f"Frame {frame_id} processed in {elapsed_time:.3f}s ({fps:.1f} FPS)")

        return detections

    def _process_targets(self, targets):
        """Convert BoostTrack output to tlwh, ids, confs."""
        tlwhs, ids, confs = [], [], []
        # Ensure targets is on CPU if it's a tensor
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        for t in targets:
            x1, y1, x2, y2, track_id, conf = t
            tlwh = [x1, y1, x2 - x1, y2 - y1]
            tlwhs.append(tlwh)
            ids.append(track_id)
            confs.append(conf)
        return tlwhs, ids, confs

    def write_results_to_file(self, filename, frame_id, detections):
        """Write detections to file in MOT format."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a') as f:
            if not detections:
                f.write(f"{frame_id},-1,0,0,0,0,0,-1,-1,-1\n")
            else:
                for det in detections:
                    x, y, w, h = det['x'], det['y'], det['width'], det['height']
                    f.write(f"{frame_id},{det['id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{det['confidence']:.2f},-1,-1,-1\n")