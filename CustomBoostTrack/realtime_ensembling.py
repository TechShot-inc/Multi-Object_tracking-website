import time
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO
import cv2


def _env_str(name: str) -> str | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def _env_float(*names: str, default: str | float) -> float:
    for n in names:
        v = _env_str(n)
        if v is not None:
            return float(v)
    return float(default)


def _env_int(*names: str, default: str | int) -> int:
    for n in names:
        v = _env_str(n)
        if v is not None:
            return int(v)
    return int(default)

# Add CustomBoostTrack to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracker.boost_trackrt import BoostTrack
from detectors import EnsembleDetector, TritonYoloDetector, YoloDetector
import dataset
from default_settings import GeneralSettings, BoostTrackSettings, BoostTrackPlusPlusSettings
import utils

class RealTimeTracker:
    def __init__(self, model1_path, model2_path, reid_path, frame_rate=30):
        """Initialize tracker with YOLO models and BoostTrack."""
        self.frame_rate = frame_rate
        self.frame_id = 0
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        use_cuda = self.device.type == "cuda"

        model1_path = model1_path or os.getenv("YOLO11_MODEL_PATH", "")
        model2_path = model2_path or os.getenv("YOLO12_MODEL_PATH", "")
        reid_path = reid_path or os.getenv("REID_MODEL_PATH", reid_path or "")

        if not model1_path or not model2_path:
            raise ValueError(
                "RealTimeTracker requires YOLO11/YOLO12 weights. Provide model1_path/model2_path or set YOLO11_MODEL_PATH/YOLO12_MODEL_PATH."
            )

        detector_backend = (os.getenv("DETECTOR_BACKEND") or "local").strip().lower()

        # Realtime tuning profiles.
        # "legacy" aligns with the legacy scripts in src/mot_web/legacy and CustomBoostTrack/run_with_ensembler.py.
        profile = (os.getenv("REALTIME_PROFILE") or "").strip().lower()
        is_legacy = profile in {"legacy", "legacy_realtime", "legacy_webcam"}

        # Detection knobs.
        # NOTE: Triton-exported person-only models in this repo emit relatively low confidence scores.
        # For Triton we use a lower YOLO conf default, even in legacy profile.
        if detector_backend == "triton":
            # Triton person-only exports have low conf scale; legacy still needs some floor to avoid noisy boxes.
            default_yolo_conf = "0.07" if is_legacy else "0.01"
        else:
            default_yolo_conf = "0.10" if is_legacy else "0.10"

        default_ensemble_conf = "0.12" if is_legacy else ("0.02" if detector_backend == "triton" else "0.08")
        default_ensemble_iou = "0.60" if is_legacy else "0.50"

        yolo_conf = _env_float("REALTIME_YOLO_CONF", "YOLO_CONF", default=default_yolo_conf)
        ensemble_conf = _env_float("REALTIME_ENSEMBLE_CONF", "ENSEMBLE_CONF", default=default_ensemble_conf)
        ensemble_iou = _env_float("REALTIME_ENSEMBLE_IOU", "ENSEMBLE_IOU", default=default_ensemble_iou)
        default_det_thresh = str(ensemble_conf) if is_legacy else str(min(yolo_conf, ensemble_conf))
        det_thresh = _env_float("REALTIME_DET_THRESH", "DET_THRESH", default=default_det_thresh)

        if detector_backend == "triton":
            model1_name = os.getenv("TRITON_YOLO11_MODEL") or "yolo11"
            model2_name = os.getenv("TRITON_YOLO12_MODEL") or "yolo12"
            self.model1 = TritonYoloDetector(model1_name, conf=yolo_conf)
            self.model2 = TritonYoloDetector(model2_name, conf=yolo_conf)
        else:
            # Initialize YOLO models with FP16
            self.model1 = YoloDetector(model1_path, conf=yolo_conf)
            self.model2 = YoloDetector(model2_path, conf=yolo_conf)
            if use_cuda:
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
            iou_thresh=ensemble_iou, conf_thresh=ensemble_conf
        )

        # BoostTrack settings
        # Defaults come from CustomBoostTrack/default_settings.py; legacy profile uses those defaults.
        GeneralSettings.values['dataset'] = 'realtime'
        GeneralSettings.values['use_embedding'] = use_cuda
        GeneralSettings.values['use_ecc'] = (os.getenv("REALTIME_USE_ECC", "1").strip() != "0")
        if reid_path:
            GeneralSettings.values['reid_path'] = reid_path

        min_hits_default = "3" if is_legacy else "1"
        iou_threshold_default = "0.35" if is_legacy else "0.50"
        # Legacy video pipeline used 10px^2, but realtime webcams often benefit from a higher floor to cut noise.
        min_box_area_default = "40" if is_legacy else "0"
        aspect_ratio_thresh_default = "1.6" if is_legacy else "100.0"
        max_age_default = str(self.frame_rate if is_legacy else 30)

        GeneralSettings.values['min_hits'] = _env_int("REALTIME_MIN_HITS", default=min_hits_default)
        GeneralSettings.values['iou_threshold'] = _env_float("REALTIME_IOU_THRESHOLD", default=iou_threshold_default)
        GeneralSettings.values['min_box_area'] = _env_float("REALTIME_MIN_BOX_AREA", default=min_box_area_default)
        GeneralSettings.values['aspect_ratio_thresh'] = _env_float("REALTIME_ASPECT_RATIO_THRESH", default=aspect_ratio_thresh_default)
        GeneralSettings.values['max_age'] = _env_int("REALTIME_MAX_AGE", default=max_age_default)
        GeneralSettings.values['det_thresh'] = det_thresh

        lambda_iou_default = "0.5" if is_legacy else "1.0"
        lambda_mhd_default = "0.25" if is_legacy else "0.0"
        lambda_shape_default = "0.25" if is_legacy else "0.0"
        use_dlo_default = "1" if is_legacy else "0"
        use_duo_default = "1" if is_legacy else "0"
        s_sim_corr_default = "0"

        BoostTrackSettings.values['lambda_iou'] = _env_float("REALTIME_LAMBDA_IOU", default=lambda_iou_default)
        BoostTrackSettings.values['lambda_mhd'] = _env_float("REALTIME_LAMBDA_MHD", default=lambda_mhd_default)
        BoostTrackSettings.values['lambda_shape'] = _env_float("REALTIME_LAMBDA_SHAPE", default=lambda_shape_default)
        BoostTrackSettings.values['use_dlo_boost'] = (os.getenv("REALTIME_USE_DLO_BOOST", use_dlo_default).strip() != "0")
        BoostTrackSettings.values['use_duo_boost'] = (os.getenv("REALTIME_USE_DUO_BOOST", use_duo_default).strip() != "0")
        BoostTrackSettings.values['s_sim_corr'] = (os.getenv("REALTIME_S_SIM_CORR", s_sim_corr_default).strip() != "0")

        btpp_default = "1" if is_legacy else "0"
        BoostTrackPlusPlusSettings.values['use_rich_s'] = (os.getenv("REALTIME_USE_RICH_S", btpp_default).strip() != "0")
        BoostTrackPlusPlusSettings.values['use_sb'] = (os.getenv("REALTIME_USE_SB", btpp_default).strip() != "0")
        BoostTrackPlusPlusSettings.values['use_vt'] = (os.getenv("REALTIME_USE_VT", btpp_default).strip() != "0")

        # Initialize BoostTrack
        self.tracker = BoostTrack(video_name="realtime")
        # BoostTrack.update() only needs img_tensor.shape for an internal scale calculation.
        # For our detectors, detections are already in original image coordinates, so scale=1.
        # Avoid per-frame ValTransform + torch device transfers to reduce latency.
        self._dummy_img_tensor = None

    def update(self, frame, frame_id, roi=None):
        """Process a single frame and return detections."""
        self.frame_id = frame_id
        start_time = time.time()

        debug = os.getenv("REALTIME_DEBUG", "0").strip() != "0"

        # Validate and apply ROI
        roi_applied = False
        roi_x, roi_y, roi_w, roi_h = 0, 0, frame.shape[1], frame.shape[0]

        if roi:
            try:
                if not isinstance(roi, dict) or not all(k in roi for k in ['x', 'y', 'width', 'height']):
                    if debug:
                        print(f"Invalid ROI format: {roi}")
                    roi = None
                else:
                    # Convert relative coordinates to absolute if needed
                    if all(0 <= float(roi[k]) <= 1.0 for k in ['x', 'y', 'width', 'height']):
                        roi_x = int(float(roi['x']) * frame.shape[1])
                        roi_y = int(float(roi['y']) * frame.shape[0])
                        roi_w = int(float(roi['width']) * frame.shape[1])
                        roi_h = int(float(roi['height']) * frame.shape[0])
                    else:
                        roi_x = int(float(roi['x']))
                        roi_y = int(float(roi['y']))
                        roi_w = int(float(roi['width']))
                        roi_h = int(float(roi['height']))
                    
                    # Ensure coordinates are valid
                    roi_x = max(0, roi_x)
                    roi_y = max(0, roi_y)
                    roi_w = min(roi_w, frame.shape[1] - roi_x)
                    roi_h = min(roi_h, frame.shape[0] - roi_y)
                    
                    # Calculate minimum size (5% of frame dimensions)
                    min_width = int(frame.shape[1] * 0.05)
                    min_height = int(frame.shape[0] * 0.05)
                    
                    if roi_w >= min_width and roi_h >= min_height:
                        # Create a contiguous copy of the ROI
                        frame = np.ascontiguousarray(frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])
                        roi_applied = True
                        if debug:
                            print(
                                f"Tracker: Using ROI with dimensions {roi_w}x{roi_h} (min: {min_width}x{min_height})"
                            )
                    else:
                        if debug:
                            print(
                                f"Tracker: ROI too small ({roi_w}x{roi_h}), minimum required: {min_width}x{min_height}"
                            )
                        roi = None
            except (KeyError, TypeError, ValueError) as e:
                if debug:
                    print(f"Error processing ROI: {e}")
                roi = None

        # Ensure frame is in BGR format for OpenCV operations
        if getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
            if debug:
                print("Warning: Converting frame to BGR format")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run detection on BGR frames.
        # Ultralytics expects BGR for numpy/cv2 images; Triton detector wrapper also assumes BGR
        # and converts to RGB internally unless TRITON_EXPECTS_BGR=1.
        detection_frame = frame

        # Provide a lightweight dummy tensor for BoostTrack scale computation (scale=1).
        height, width = frame.shape[:2]
        if self._dummy_img_tensor is None or self._dummy_img_tensor.shape[2] != height or self._dummy_img_tensor.shape[3] != width:
            self._dummy_img_tensor = np.empty((1, 3, height, width), dtype=np.uint8)
        img_tensor = self._dummy_img_tensor

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
                dets_abs[:, 0] += roi_x  # x1
                dets_abs[:, 1] += roi_y  # y1
                dets_abs[:, 2] += roi_x  # x2
                dets_abs[:, 3] += roi_y  # y2
                # Check if detection is within ROI
                in_roi = (
                    (dets_abs[:, 0] >= roi_x) &  # x1 >= roi_x
                    (dets_abs[:, 1] >= roi_y) &  # y1 >= roi_y
                    (dets_abs[:, 2] <= roi_x + roi_w) &  # x2 <= roi_x + roi_w
                    (dets_abs[:, 3] <= roi_y + roi_h)  # y2 <= roi_y + roi_h
                )
                dets = dets[in_roi]
                if debug:
                    print(f"Filtered detections: {len(dets)} within ROI")

        # Run BoostTrack
        if len(dets) > 0:
            try:
                # Use the original BGR frame for the tracker
                targets = self.tracker.update(dets, img_tensor, frame, f"realtime:{frame_id}")
                if isinstance(targets, torch.Tensor):
                    targets = targets.cpu().numpy()
                tlwhs, ids, confs = self._process_targets(targets)
            except Exception as e:
                if debug:
                    print(f"Tracker error: {e}")
                tlwhs, ids, confs = [], [], []
        else:
            tlwhs, ids, confs = [], [], []

        # Adjust detections for ROI offset
        if roi_applied:
            for tlwh in tlwhs:
                tlwh[0] += roi_x  # Adjust x
                tlwh[1] += roi_y  # Adjust y

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
        if debug:
            print(f"Frame {frame_id} processed in {elapsed_time:.3f}s ({fps:.1f} FPS)")

        return detections

    def _process_targets(self, targets):
        """Convert BoostTrack output to tlwh, ids, confs."""
        tlwhs, ids, confs = [], [], []
        # Ensure targets is on CPU if it's a tensor
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Final output filtering (important for realtime UI):
        # BoostTrack can emit many tiny/odd boxes when detections are noisy.
        # Apply the same basic geometric filtering used in the offline pipeline.
        min_box_area = _env_float("REALTIME_MIN_BOX_AREA", default=str(GeneralSettings.values.get('min_box_area', 0)))
        aspect_ratio_thresh = _env_float(
            "REALTIME_ASPECT_RATIO_THRESH",
            default=str(GeneralSettings.values.get('aspect_ratio_thresh', 100.0)),
        )
        min_w = _env_float("REALTIME_MIN_BOX_W", default="0")
        min_h = _env_float("REALTIME_MIN_BOX_H", default="0")
        min_track_conf = _env_float("REALTIME_MIN_TRACK_CONF", default="0")

        kept_targets = []
        for t in targets:
            x1, y1, x2, y2, track_id, conf = t
            if float(conf) < min_track_conf:
                continue
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            if w < min_w or h < min_h:
                continue
            if (w * h) < min_box_area:
                continue
            # Filter out overly-wide boxes (same logic as utils.filter_targets)
            if (w / h) > aspect_ratio_thresh:
                continue
            kept_targets.append([x1, y1, x2, y2, track_id, conf])

        # Use existing helper to produce tlwh/id/conf lists
        tlwhs, ids, confs = utils.filter_targets(np.asarray(kept_targets, dtype=np.float32), aspect_ratio_thresh, min_box_area)
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