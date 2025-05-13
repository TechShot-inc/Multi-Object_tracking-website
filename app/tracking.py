import os
import shutil
import cv2
import pandas as pd
import random
from tqdm import tqdm
import subprocess
import numpy as np  # Add import for numpy


# Ensure the directory exists and is accessible
working_dir = 'BoostTrack'



def extract_frames(video_path, output_folder, speed = 5): # speed like 2x, 3x, 4x
    """
    Extract frames from a video at 3x speed (keeping only 1/3 of the frames) 
    and save them as images in a folder.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the folder where frames will be saved
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info:")
    print(f" - Name: {os.path.basename(video_path)}")
    print(f" - Frames: {frame_count}")
    print(f" - FPS: {fps}")
    print(f" - Resolution: {width}x{height}")
    
    # Read and save every 3rd frame (for 3x speed)
    saved_frame_number = 0
    original_frame_number = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Only save every 3rd frame (0, 3, 6, etc.)
        if original_frame_number % speed == 0:
            # Save frame as an image file
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_number:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_number += 1
            
            # Print progress every 100 saved frames
            if saved_frame_number % 100 == 0:
                print(f"Processed frame {original_frame_number}/{frame_count} (saved {saved_frame_number})")
        
        original_frame_number += 1

    
    # Release resources
    cap.release()
    print(f"\nFinished! Saved {saved_frame_number} frames (1/{speed} of original) to {output_folder}")



def draw_tracking_results_on_frames(frames_folder_path, annnots_path, annoted_video_path, frame_rate):
    images = [os.path.join(frames_folder_path, path) for path in os.listdir(frames_folder_path)]
    images = sorted(images)
    gt_df = pd.read_csv(annnots_path, header=None)
    gt_df.columns = ['frame', 'track_id', 'x', 'y', 'width', 'height', 'confidence', 'x_', 'y_', 'z_']

    first_image = cv2.imread(images[0])
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(annoted_video_path, fourcc, frame_rate, (width, height))
    
    # Dictionary to store colors for each track ID
    id_to_color = {}

    for frame_id in range(1, len(images)+1):
    # for frame_id in gt_df['frame'].unique():
        frame_path = images[frame_id-1]
        img = cv2.imread(frame_path)

        # Get the detections for this frame
        detections = gt_df[gt_df['frame'] == frame_id]

        # Draw each detection on the image
        for _, row in detections.iterrows():
            track_id = int(row['track_id'])
            x = int(row['x'])
            y = int(row['y'])
            width = int(row['width'])
            height = int(row['height'])
            confidence = row['confidence']

            # Generate a random color for each track ID
            if track_id not in id_to_color:
                id_to_color[track_id] = [random.randint(0, 255) for _ in range(3)]

            color = tuple(id_to_color[track_id])
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
            cv2.putText(img, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        video_writer.write(img)
    video_writer.release()
    print("video is Done")


def main(video_path, output_path):
    cwd = ""
    frames_path = os.path.join(cwd, os.path.basename(video_path).split('.')[0])
    output_video = output_path 
    os.makedirs(frames_path, exist_ok=True)
    extract_frames(video_path, frames_path, speed=10)

    

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    dataset_name = "tarsh"
    output_folder = os.path.join(cwd, "results")
    dataset_path = frames_path
    reid_path = "models/OSNet_Best.pth"
    frame_rate = int(fps)
    model1_path = "models/yolo11xlast.pt"
    model1_weight = 0.7
    model2_path = "models/yolo12xlast.pt"
    model2_weight = 0.3
    print(frame_rate)

    subprocess.run([
        "python", "BoostTrack/run_with_ensembler.py", 
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

    speed_of_output_video = 2
    draw_tracking_results_on_frames(frames_path, os.path.join(output_folder, "BTPP_post_gbi/data/test.txt"), output_video, fps*speed_of_output_video)
    shutil.rmtree(frames_path)
    shutil.rmtree(output_folder)

    return output_video
