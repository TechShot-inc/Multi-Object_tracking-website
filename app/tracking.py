import os
import shutil
import cv2
import pandas as pd
import random
from tqdm import tqdm
import subprocess
import numpy as np
import json

# Get application root directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Updated working directory to use app subdirectory
working_dir = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(working_dir, exist_ok=True)

def extract_frames(video_path, output_folder, speed=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    saved_frame_number = 0
    original_frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if original_frame_number % speed == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_number += 1
        original_frame_number += 1
    cap.release()

def draw_tracking_results_on_frames(frames_folder_path, annnots_path, annoted_video_path, frame_rate):
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
    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(annoted_video_path, fourcc, frame_rate, (width, height))
    id_to_color = {}
    tracking_data = {}
    for frame_id in range(1, len(images)+1):
        frame_path = images[frame_id-1]
        img = cv2.imread(frame_path)
        detections = gt_df[gt_df['frame'] == frame_id]
        frame_detections = []
        for _, row in detections.iterrows():
            track_id = int(row['track_id'])
            x = int(row['x'])
            y = int(row['y'])
            width = int(row['width'])
            height = int(row['height'])
            confidence = float(row['confidence'])
            if track_id not in id_to_color:
                id_to_color[track_id] = [random.randint(0, 255) for _ in range(3)]
            color = tuple(id_to_color[track_id])
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
            cv2.putText(img, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            frame_detections.append({
                "id": track_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "confidence": confidence
            })
        tracking_data[str(frame_id)] = frame_detections
        video_writer.write(img)
    video_writer.release()
    return tracking_data

def main(video_path, output_path, speed=5, output_speed=2):
    cwd = ""
    frames_path = os.path.join(cwd, os.path.basename(video_path).split('.')[0])
    output_video = output_path 
    os.makedirs(frames_path, exist_ok=True)
    extract_frames(video_path, frames_path, speed=speed)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = int(fps // speed) if speed > 0 else int(fps)
    if frame_rate < 1:
        frame_rate = 1
    dataset_name = "tarsh-val"
    output_folder = os.path.join(PROJECT_ROOT, 'output', 'results')
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
    subprocess.run([
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
    ], check=True)
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
    result_dir = os.path.dirname(output_path)
    os.makedirs(result_dir, exist_ok=True)
    tracking_data = draw_tracking_results_on_frames(frames_path, tracking_results_path, output_video, frame_rate * output_speed)
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
    shutil.rmtree(frames_path)
    shutil.rmtree(output_folder)
    return output_video
