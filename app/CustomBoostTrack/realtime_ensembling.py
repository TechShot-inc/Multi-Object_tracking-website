import time
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

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

        # Initialize YOLO models with FP16
        self.model1 = YoloDetector(model1_path, conf=0.3)
        self.model2 = YoloDetector(model2_path, conf=0.3)
        # self.model1 = YoloDetector("yolo11x.pt", conf=0.3)
        # self.model2 = YoloDetector("yolo12x.pt", conf=0.3)
        try:
            self.model1.model.float16 = True  # Enable FP16
            self.model2.model.float16 = True
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

        # Cache for detections
        self.cached_detections = None
        self.model1_interval = 1  # Increased for performance
        self.model2_interval = 2  # Increased for performance

    def update(self, frame, frame_id):
        """Process a single frame and return detections."""
        self.frame_id = frame_id
        start_time = time.time()

        # Preprocess frame
        height, width = frame.shape[:2]
        img, _ = self.preproc(frame, None, (height, width))
        img = img.reshape(1, *img.shape)

        # Run detection based on schedule
        use_model1 = (frame_id % self.model1_interval == 1)
        use_model2 = (frame_id % self.model2_interval == 1)

        if use_model1 or use_model2:
            dets = self.detector(frame)
            if len(dets) > 0:
                self.cached_detections = dets.numpy()
            else:
                self.cached_detections = np.empty((0, 5))
        else:
            dets = self.cached_detections if self.cached_detections is not None else np.empty((0, 5))

        # Run BoostTrack
        if dets is not None and len(dets) > 0:
            try:
                targets = self.tracker.update(dets, img, frame, f"realtime:{frame_id}")
                # Minimal filtering: confidence > 0.2
                if len(targets) > 0:
                    mask = targets[:, 5] > 0.2
                    targets = targets[mask]
                tlwhs, ids, confs = self._process_targets(targets)
            except Exception as e:
                print(f"Tracker error: {e}")
                tlwhs, ids, confs = [], [], []
        else:
            tlwhs, ids, confs = [], [], []

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