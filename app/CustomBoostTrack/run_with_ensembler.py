import os
import shutil
import time
import math
import logging

import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack
from ultralytics import YOLO
import cv2
import numpy as np
import torch

from detectors import *

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """Ensure directory exists, creating it if necessary."""
    logger.debug(f"Ensuring directory exists: {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def calculate_intersection_area(box, roi):
    """Calculate the intersection area between a bounding box and ROI."""
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

def clip_box_to_roi(box, roi):
    """Clip a bounding box to stay within ROI boundaries."""
    x1, y1, x2, y2 = box
    rx, ry, rw, rh = roi
    rx2, ry2 = rx + rw, ry + rh

    x1 = max(x1, rx)
    y1 = max(y1, ry)
    x2 = min(x2, rx2)
    y2 = min(y2, ry2)

    if x2 > x1 and y2 > y1:
        return [x1, y1, x2, y2]
    return None

def apply_roi_to_detections(preds, roi):
    """Filter and clip detections based on ROI (75% area overlap)."""
    if not roi:
        return preds
    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    filtered_preds = []
    for pred in preds:
        x1, y1, x2, y2, conf = pred.tolist()
        box = [x1, y1, x2, y2]
        box_area = (x2 - x1) * (y2 - y1)
        intersection_area = calculate_intersection_area(box, [x, y, w, h])
        if box_area > 0 and (intersection_area / box_area) >= 0.75:
            clipped_box = clip_box_to_roi(box, [x, y, w, h])
            if clipped_box:
                clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped_box
                filtered_preds.append([clipped_x1, clipped_y1, clipped_x2, clipped_y2, conf])
    return torch.tensor(filtered_preds) if filtered_preds else torch.tensor([])

def get_main_args():
    logger.debug("Entering get_main_args()")
    parser = make_parser()
    logger.debug("Parser created")
    
    existing_args = [action.dest for action in parser._actions]
    logger.debug("Existing arguments collected")
    
    if 'iou_thresh' in existing_args:
        parser.set_defaults(iou_thresh=0.6)
        logger.debug("iou_thresh default set")
    if 'min_hits' in existing_args:
        parser.set_defaults(min_hits=3)
        logger.debug("min_hits default set")
    if 'conf' in existing_args:
        parser.set_defaults(conf=0.3)
        logger.debug("conf default set")
    
    new_args = [
        ("--conf_thresh", {"type": float, "default": 0.3, "help": "Confidence threshold for fused boxes"}),
        ("--max_age", {"type": int, "default": None, "help": "Max age for tracks (overrides frame_rate if set)"}),
        ("--det_thresh", {"type": float, "default": None, "help": "Detection confidence threshold"}),
        ("--iou_threshold", {"type": float, "default": 0.35, "help": "IoU threshold for tracking"}),
        ("--min_box_area", {"type": float, "default": 10, "help": "Min box area for valid detections"}),
        ("--aspect_ratio_thresh", {"type": float, "default": 1.6, "help": "Aspect ratio threshold for detections"}),
        ("--lambda_iou", {"type": float, "default": 0.5, "help": "IoU cost weight"}),
        ("--lambda_mhd", {"type": float, "default": 0.25, "help": "Mahalanobis distance cost weight"}),
        ("--lambda_shape", {"type": float, "default": 0.25, "help": "Shape similarity cost weight"}),
        ("--use_dlo_boost", {"type": int, "default": 1, "help": "Enable DLO boost (1=True, 0=False)"}),
        ("--use_duo_boost", {"type": int, "default": 1, "help": "Enable DUO boost (1=True, 0=False)"}),
        ("--dlo_boost_coef", {"type": float, "default": None, "help": "DLO boost coefficient"}),
        ("--n_min", {"type": int, "default": 25, "help": "Min frames for interpolation"}),
        ("--n_dti", {"type": int, "default": 20, "help": "Max frame gap for linear interpolation"}),
        ("--interval", {"type": int, "default": 1000, "help": "Max frame gap for interpolation"}),
        ("--dataset", {"type": str, "default": "mot17"}),
        ("--result_folder", {"type": str, "default": "results/trackers/"}),
        ("--test_dataset", {"action": "store_true"}),
        ("--exp_name", {"type": str, "default": "test"}),
        ("--no_reid", {"action": "store_true"}),
        ("--no_cmc", {"action": "store_true"}),
        ("--s_sim_corr", {"action": "store_true"}),
        ("--btpp_arg_iou_boost", {"action": "store_true"}),
        ("--btpp_arg_no_sb", {"action": "store_true"}),
        ("--btpp_arg_no_vt", {"action": "store_true"}),
        ("--no_post", {"action": "store_true"}),
        ("--dataset_path", {"type": str}),
        ("--model1_path", {"type": str}),
        ("--model1_weight", {"type": float, "default": 0.5}),
        ("--model2_path", {"type": str}),
        ("--model2_weight", {"type": float, "default": 0.5}),
        ("--reid_path", {"type": str}),
        ("--reid_path2", {"type": str, "default": None}),
        ("--reid_weight1", {"type": float, "default": 0.5}),
        ("--reid_weight2", {"type": float, "default": 0.5}),
        ("--frame_rate", {"type": int, "default": 25}),
        ("--visualize", {"action": "store_true", "help": "Enable visualization of detections and tracks"}),
        ("--track_percent", {"type": float, "default": 1.0, "help": "Percentage of video to track (0.0 to 1.0)"}),
        ("--roi", {"type": str, "help": "ROI coordinates as x,y,w,h (e.g., 100,100,200,200)"}),
    ]
    
    for arg_name, kwargs in new_args:
        arg_dest = arg_name.lstrip('-').replace('-', '_')
        if arg_dest not in existing_args:
            parser.add_argument(arg_name, **kwargs)
            logger.debug(f"Added argument {arg_name}")
    
    logger.debug("Parsing arguments")
    args = parser.parse_args()
    logger.debug("Arguments parsed")
    
    logger.debug("Setting result_folder")
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
        logger.debug("Set result_folder to MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")
        logger.debug("Set result_folder to MOT20-val")
    elif args.dataset == "tarsh":
        args.result_folder = os.path.join(args.result_folder, "tarsh-val")
        logger.debug("Set result_folder to tarsh-val")
    
    if args.test_dataset:
        args.result_folder = args.result_folder.replace("-val", "-test")
        logger.debug("Updated result_folder for test_dataset")
    
    logger.debug("Validating track_percent")
    if not 0.0 < args.track_percent <= 1.0:
        raise ValueError("track_percent must be between 0.0 and 1.0")
    logger.debug("track_percent validated")
    
    logger.debug("Returning args")
    return args

def my_data_loader(main_path, track_percent=1.0):
    logger.debug(f"Entering my_data_loader with main_path={main_path}, track_percent={track_percent}")
    img_paths = [os.path.join(main_path, img) for img in os.listdir(main_path)]
    img_paths = sorted(img_paths)
    num_images = math.ceil(len(img_paths) * track_percent)
    img_paths = img_paths[:num_images]
    preproc = dataset.ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    for idx, img_path in enumerate(img_paths[:], 1):
        logger.debug(f"Loading image {img_path}")
        np_img = cv2.imread(img_path)
        height, width, _ = np_img.shape
        img, target = preproc(np_img, None, (height, width))
        yield ((img.reshape(1, *img.shape), np_img), target, (height, width, torch.tensor(idx), None, ["test"]), img_path)

def visualize_detections(np_img, model1_preds, model2_preds, ensemble_preds, video_name, frame_id, vis_folder, roi=None):
    logger.debug(f"Visualizing detections for {video_name}, frame {frame_id}")
    img_model1 = np_img.copy()
    img_model2 = np_img.copy()
    img_ensemble = np_img.copy()
    
    if roi:
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        for img in [img_model1, img_model2, img_ensemble]:
            cv2.rectangle(img, (x, y), (x + w, y + h), (45, 212, 191), 2)
            cv2.putText(img, "ROI", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if len(model1_preds) > 0:
        for pred in model1_preds:
            x1, y1, x2, y2, conf = pred.tolist()
            cv2.rectangle(img_model1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_model1, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    if len(model2_preds) > 0:
        for pred in model2_preds:
            x1, y1, x2, y2, conf = pred.tolist()
            cv2.rectangle(img_model2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_model2, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if len(ensemble_preds) > 0:
        for pred in model2_preds:
            x1, y1, x2, y2, conf = pred.tolist()
            cv2.rectangle(img_ensemble, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img_ensemble, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    os.makedirs(os.path.join(vis_folder, video_name), exist_ok=True)
    cv2.imwrite(os.path.join(vis_folder, video_name, f"model1_frame_{frame_id:06d}.jpg"), img_model1)
    cv2.imwrite(os.path.join(vis_folder, video_name, f"model2_frame_{frame_id:06d}.jpg"), img_model2)
    cv2.imwrite(os.path.join(vis_folder, video_name, f"ensemble_frame_{frame_id:06d}.jpg"), img_ensemble)
    logger.debug(f"Visualizations saved to {vis_folder}/{video_name}")

def visualize_gbi_tracks(dataset_path, gbi_folder, vis_folder_gbi, roi=None):
    logger.debug(f"Visualizing GBI tracks for {gbi_folder}")
    for file_name in os.listdir(gbi_folder):
        video_name = file_name.split('.')[0]
        gbi_path = os.path.join(gbi_folder, file_name)
        
        tracks = {}
        with open(gbi_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                if frame_id not in tracks:
                    tracks[frame_id] = []
                tracks[frame_id].append((track_id, x, y, w, h))
        
        video_img_path = dataset_path
        if not os.path.exists(video_img_path):
            logger.warning(f"Image path {video_img_path} not found for GBI visualization")
            continue
        
        img_paths = sorted([os.path.join(video_img_path, img) for img in os.listdir(video_img_path) if img.endswith(('.jpg', '.png'))])
        
        for frame_id in tracks:
            img_idx = frame_id - 1
            if img_idx >= len(img_paths):
                logger.warning(f"No image found for frame {frame_id} in {video_name}")
                continue
            
            np_img = cv2.imread(img_paths[img_idx])
            if np_img is None:
                logger.warning(f"Failed to load image {img_paths[img_idx]}")
                continue
            
            if roi:
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                cv2.rectangle(np_img, (x, y), (x + w, y + h), (45, 212, 191), 2)
                cv2.putText(np_img, "ROI", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            for track_id, x, y, w, h in tracks[frame_id]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(np_img, f"ID:{track_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            vis_path = os.path.join(vis_folder_gbi, video_name)
            os.makedirs(vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(vis_path, f"gbi_frame_{frame_id:06d}.jpg"), np_img)
        logger.debug(f"GBI visualizations saved for {video_name}")

def visualize_selected_frames(dataset_path, gbi_folder, stored_detections, selected_frames, video_name, selected_folder, roi=None):
    logger.debug(f"Visualizing selected frames for {video_name}")
    video_img_path = dataset_path
    if not os.path.exists(video_img_path):
        logger.warning(f"Image path {video_img_path} not found for selected frames visualization")
        return
    
    img_paths = sorted([os.path.join(video_img_path, img) for img in os.listdir(video_img_path) if img.endswith(('.jpg', '.png'))])
    
    gbi_path = os.path.join(gbi_folder, f"{video_name}.txt")
    tracks = {}
    if os.path.exists(gbi_path):
        with open(gbi_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                if frame_id not in tracks:
                    tracks[frame_id] = []
                tracks[frame_id].append((track_id, x, y, w, h))
    
    for frame_id in selected_frames:
        img_idx = frame_id - 1
        if img_idx >= len(img_paths):
            logger.warning(f"No image found for frame {frame_id} in {video_name}")
            continue
        
        np_img = cv2.imread(img_paths[img_idx])
        if np_img is None:
            logger.warning(f"Failed to load image {img_paths[img_idx]}")
            continue
        
        img_model1 = np_img.copy()
        img_model2 = np_img.copy()
        img_ensemble = np_img.copy()
        img_gbi = np_img.copy()
        
        if roi:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            for img in [img_model1, img_model2, img_ensemble, img_gbi]:
                cv2.rectangle(img, (x, y), (x + w, y + h), (45, 212, 191), 2)
                cv2.putText(img, "ROI", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if frame_id in stored_detections['model1']:
            for x1, y1, x2, y2, conf in stored_detections['model1'][frame_id]:
                cv2.rectangle(img_model1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img_model1, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if frame_id in stored_detections['model2']:
            for x1, y1, x2, y2, conf in stored_detections['model2'][frame_id]:
                cv2.rectangle(img_model2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_model2, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if frame_id in stored_detections['ensemble']:
            for x1, y1, x2, y2, conf in stored_detections['ensemble'][frame_id]:
                cv2.rectangle(img_ensemble, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img_ensemble, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if frame_id in tracks:
            for track_id, x, y, w, h in tracks[frame_id]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(img_gbi, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img_gbi, f"ID:{track_id}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        for subfolder, img in [('model1', img_model1), ('model2', img_model2), ('ensemble', img_ensemble), ('gbi', img_gbi)]:
            vis_path = os.path.join(selected_folder, video_name, subfolder)
            os.makedirs(vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(vis_path, f"frame_{frame_id:06d}.jpg"), img)
        logger.debug(f"Saved visualizations for frame {frame_id}")

def main():
    logger.debug("Entering main()")
    args = get_main_args()
    logger.debug("Arguments parsed")
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        try:
            x, y, w, h = map(int, args.roi.split(','))
            roi = {'x': x, 'y': y, 'width': w, 'height': h}
            logger.debug(f"ROI applied: x={x}, y={y}, w={w}, h={h}")
        except ValueError:
            logger.warning("Invalid ROI format, ignoring ROI")
            roi = None
    
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset
    GeneralSettings.values['reid_path'] = args.reid_path
    GeneralSettings.values['min_hits'] = args.min_hits
    GeneralSettings.values['iou_threshold'] = args.iou_threshold
    GeneralSettings.values['min_box_area'] = args.min_box_area
    GeneralSettings.values['aspect_ratio_thresh'] = args.aspect_ratio_thresh
    if args.max_age is not None:
        GeneralSettings.values['max_age'] = args.max_age
    else:
        GeneralSettings.values['max_age'] = args.frame_rate
    
    if args.det_thresh is not None:
        GeneralSettings.dataset_specific_settings[args.dataset] = {'det_thresh': args.det_thresh}
    if args.dlo_boost_coef is not None:
        BoostTrackSettings.dataset_specific_settings[args.dataset] = {'dlo_boost_coef': args.dlo_boost_coef}
    
    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr
    BoostTrackSettings.values['lambda_iou'] = args.lambda_iou
    BoostTrackSettings.values['lambda_mhd'] = args.lambda_mhd
    BoostTrackSettings.values['lambda_shape'] = args.lambda_shape
    BoostTrackSettings.values['use_dlo_boost'] = bool(args.use_dlo_boost)
    BoostTrackSettings.values['use_duo_boost'] = bool(args.use_duo_boost)
    
    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt

    tracker = None
    results = {}
    frame_count = 0
    total_time = 0
    stored_detections = {'model1': {}, 'model2': {}, 'ensemble': {}}
    first_video_name = None
    frame_ids = []

    logger.debug(f"Loading model1 from {args.model1_path}")
    model1 = YoloDetector(args.model1_path)  # No conf_thresh to use YOLO default
    logger.debug("Model1 initialized")
    
    logger.debug(f"Loading model2 from {args.model2_path}")
    model2 = YoloDetector(args.model2_path)  # No conf_thresh to use YOLO default
    logger.debug("Model2 initialized")
    
    logger.debug("Initializing EnsembleDetector")
    det = EnsembleDetector(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)
    logger.debug("EnsembleDetector initialized")
    
    vis_folder = os.path.join(args.result_folder, args.exp_name, "visualizations") if args.visualize else None
    if args.visualize:
        os.makedirs(vis_folder, exist_ok=True)
        logger.debug(f"Visualization folder created: {vis_folder}")
    
    img_paths = [f for f in os.listdir(args.dataset_path) if f.endswith(('.jpg', '.png'))]
    total_frames = math.ceil(len(img_paths) * args.track_percent)
    logger.info(f"Total frames to process: {total_frames}")
    
    total_start_time = time.time()
    
    for (img, np_img), _, info, img_path in my_data_loader(args.dataset_path, args.track_percent):
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []
            if first_video_name is None:
                first_video_name = video_name
        
        logger.info(f"Processing frame {frame_count + 1}/{total_frames} ({video_name}:{frame_id})")
        frame_start_time = time.time()
        
        if frame_id == 1:
            logger.debug(f"Initializing tracker for {video_name}")
            logger.debug(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        # Process full image, apply ROI filtering to detections
        det_img = np_img
        model1_start = time.time()
        model1_preds = model1(det_img)
        model1_preds = apply_roi_to_detections(model1_preds, roi)
        model1_time = time.time() - model1_start
        
        model2_start = time.time()
        model2_preds = model2(det_img)
        model2_preds = apply_roi_to_detections(model2_preds, roi)
        model2_time = time.time() - model2_start
        
        ensemble_start = time.time()
        ensemble_preds = det(det_img)
        ensemble_preds = apply_roi_to_detections(ensemble_preds, roi)
        ensemble_time = time.time() - ensemble_start
        
        if args.visualize and video_name == first_video_name:
            stored_detections['model1'][frame_id] = model1_preds.tolist() if len(model1_preds) > 0 else []
            stored_detections['model2'][frame_id] = model2_preds.tolist() if len(model2_preds) > 0 else []
            stored_detections['ensemble'][frame_id] = ensemble_preds.tolist() if len(ensemble_preds) > 0 else []
            frame_ids.append(frame_id)
        
        vis_start = time.time()
        if args.visualize:
            visualize_detections(np_img, model1_preds, model2_preds, ensemble_preds, video_name, frame_id, vis_folder, roi)
        vis_time = time.time() - vis_start
        
        track_start = time.time()
        if ensemble_preds is None or len(ensemble_preds) == 0:
            logger.warning("No ensemble predictions, skipping frame")
            continue
        targets = tracker.update(ensemble_preds, img, np_img, tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])
        logger.debug(f"{len(ids)} ids detected")
        track_time = time.time() - track_start
        
        save_start = time.time()
        results[video_name].append((frame_id, tlwhs, ids, confs))
        save_time = time.time() - save_start
        
        frame_count += 1
        total_time += time.time() - frame_start_time
        frame_total_time = time.time() - frame_start_time
        cumulative_time = time.time() - total_start_time
        
        logger.debug(f"Frame {frame_count}/{total_frames} completed in {frame_total_time:.2f}s "
                     f"(Model1: {model1_time:.2f}s, Model2: {model2_time:.2f}s, Ensemble: {ensemble_time:.2f}s, "
                     f"Track: {track_time:.2f}s, Visualize: {vis_time:.2f}s, Save: {save_time:.2f}s)")
        logger.debug(f"Cumulative time: {cumulative_time:.2f}s")

    logger.info(f"Tracking completed. Total time: {time.time() - total_start_time:.2f}s")
    tracker.dump_cache()
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    save_start = time.time()
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    save_time = time.time() - save_start
    logger.info(f"Results saved to {folder} in {save_time:.2f}s")
    if args.visualize:
        logger.info(f"Visualizations saved to {vis_folder}")

    if not args.no_post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            logger.info(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        utils.dti(post_folder_data, post_folder_data, n_dti=args.n_dti, n_min=args.n_min)

        logger.info(f"Linear interpolation post-processing applied, saved to {post_folder_data}")

        post_folder_gbi_root = os.path.join(args.result_folder, args.exp_name + "_post_gbi")
        post_folder_gbi = os.path.join(post_folder_gbi_root, "data")
        
        ensure_dir(post_folder_gbi_root)
        ensure_dir(post_folder_gbi)
        
        vis_folder_gbi = os.path.join(post_folder_gbi_root, "visualizations") if args.visualize else None
        if args.visualize:
            ensure_dir(vis_folder_gbi)
        
        gbi_start = time.time()
        for file_name in os.listdir(post_folder_data):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)
            
            ensure_dir(os.path.dirname(out_path2))
            
            GBInterpolation(path_in=in_path, path_out=out_path2, interval=args.interval)
        gbi_time = time.time() - gbi_start
        logger.info(f"Gradient boosting interpolation post-processing applied in {gbi_time:.2f}s, saved to {post_folder_gbi}")
        
        if args.visualize:
            vis_gbi_start = time.time()
            visualize_gbi_tracks(args.dataset_path, post_folder_gbi, vis_folder_gbi, roi)
            vis_gbi_time = time.time() - vis_gbi_start
            logger.info(f"GBI visualizations saved to {vis_folder_gbi} in {vis_gbi_time:.2f}s")
        
        if args.visualize and first_video_name:
            vis_selected_start = time.time()
            selected_folder = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "selected_frames")
            os.makedirs(selected_folder, exist_ok=True)
            if len(frame_ids) >= 10:
                step = len(frame_ids) // 10
                selected_frames = [frame_ids[i * step] for i in range(10)]
            else:
                selected_frames = frame_ids[:10]
            visualize_selected_frames(args.dataset_path, post_folder_gbi, stored_detections, selected_frames, first_video_name, selected_folder, roi)
            vis_selected_time = time.time() - vis_selected_start
            logger.info(f"Selected frame visualizations saved to {selected_folder} in {vis_selected_time:.2f}s")

if __name__ == "__main__":
    main()