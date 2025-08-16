import glob
import os
import numpy as np
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2

def generate_mot_heatmap(mot_file, width, height, grid_size=20, output_file='heatmap.png'):
    """Generate a heatmap of annotation density from a MOT file."""
    try:
        df = pd.read_csv(mot_file, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_', 'y_', 'z_'])
    except Exception as e:
        print(f"Error reading MOT file: {e}")
        return False

    x_bins = np.linspace(0, width, grid_size + 1)
    y_bins = np.linspace(0, height, grid_size + 1)
    centers_x = df['x'] + df['w'] / 2
    centers_y = df['y'] + df['h'] / 2
    heatmap, _, _ = np.histogram2d(centers_x, centers_y, bins=[x_bins, y_bins])
    heatmap = heatmap / (heatmap.max() if heatmap.max() > 0 else 1)
    plt.figure(figsize=(10, 10 * height / width))
    sns.heatmap(heatmap.T, cmap='viridis', cbar=True)
    plt.axis('off')
    try:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        return True
    except Exception as e:
        print(f"Error saving heatmap: {e}")
        plt.close()
        return False

def generate_velocity_heatmap(mot_file, width, height, grid_size=20, output_file='velocity_heatmap.png'):
    """Generate a heatmap of average object velocity."""
    try:
        df = pd.read_csv(mot_file, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_', 'y_', 'z_'])
        df['center_x'] = df['x'] + df['w'] / 2
        df['center_y'] = df['y'] + df['h'] / 2
        velocities = []
        for obj_id in df['id'].unique():
            track = df[df['id'] == obj_id][['frame', 'center_x', 'center_y']].sort_values('frame')
            if len(track) < 2:
                continue
            dx = track['center_x'].diff().abs()
            dy = track['center_y'].diff().abs()
            dt = track['frame'].diff()
            speed = np.sqrt(dx**2 + dy**2) / dt
            track['speed'] = speed
            velocities.append(track[['center_x', 'center_y', 'speed']].dropna())
        if not velocities:
            return False
        velocities = pd.concat(velocities)
        x_bins = np.linspace(0, width, grid_size + 1)
        y_bins = np.linspace(0, height, grid_size + 1)
        heatmap, _, _ = np.histogram2d(
            velocities['center_x'], velocities['center_y'], bins=[x_bins, y_bins], weights=velocities['speed']
        )
        counts, _, _ = np.histogram2d(velocities['center_x'], velocities['center_y'], bins=[x_bins, y_bins])
        heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0)
        heatmap = heatmap / (heatmap.max() if heatmap.max() > 0 else 1)
        plt.figure(figsize=(10, 10 * height / width))
        sns.heatmap(heatmap.T, cmap='plasma', cbar=True)
        plt.axis('off')
        try:
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            return True
        except Exception as e:
            print(f"Error saving velocity heatmap: {e}")
            plt.close()
            return False
    except Exception as e:
        print(f"Error processing velocity heatmap: {e}")
        return False

def get_longest_staying_ids(mot_file, frames_folder, top_n=5):
    """Get the top N longest-staying object IDs and their image crops."""
    try:
        df = pd.read_csv(mot_file, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_', 'y_', 'z_'])
        id_counts = df.groupby('id').size().sort_values(ascending=False)
        top_ids = id_counts.head(top_n).index.tolist()
        crops = []
        for obj_id in top_ids:
            id_crops = []
            id_detections = df[df['id'] == obj_id]
            sample_frames = id_detections['frame'].sample(n=min(3, len(id_detections)), random_state=42).tolist()
            for frame in sample_frames:
                frame_file = os.path.join(frames_folder, f"frame_{frame-1:06d}.jpg")
                if os.path.exists(frame_file):
                    img = cv2.imread(frame_file)
                    if img is None:
                        continue
                    detection = id_detections[id_detections['frame'] == frame].iloc[0]
                    x, y, w, h = int(detection['x']), int(detection['y']), int(detection['w']), int(detection['h'])
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    if w > 0 and h > 0:
                        crop = img[y:y+h, x:x+w]
                        if crop.size > 0:
                            id_crops.append(crop)
            crops.append(id_crops)
        return top_ids, crops
    except Exception as e:
        print(f"Error processing longest staying IDs: {e}")
        return [], []

def get_track_durations(mot_file):
    """Calculate track durations for each object ID."""
    try:
        df = pd.read_csv(mot_file, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_', 'y_', 'z_'])
        durations = df.groupby('id')['frame'].agg(lambda x: x.max() - x.min() + 1).to_dict()
        return durations
    except Exception as e:
        print(f"Error calculating track durations: {e}")
        return {}

def get_average_velocity(mot_file):
    """Calculate average velocity of objects (pixels/frame)."""
    try:
        df = pd.read_csv(mot_file, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x_', 'y_', 'z_'])
        df['center_x'] = df['x'] + df['w'] / 2
        df['center_y'] = df['y'] + df['h'] / 2
        velocities = []
        for obj_id in df['id'].unique():
            track = df[df['id'] == obj_id][['frame', 'center_x', 'center_y']].sort_values('frame')
            if len(track) < 2:
                continue
            dx = track['center_x'].diff().abs()
            dy = track['center_y'].diff().abs()
            dt = track['frame'].diff()
            speed = np.sqrt(dx**2 + dy**2) / dt
            velocities.extend(speed.dropna())
        return np.mean(velocities) if velocities else 0.0
    except Exception as e:
        print(f"Error calculating average velocity: {e}")
        return 0.0

def write_results_no_score(filename, results):
    """Writes results in MOT style to filename."""
    save_format = "{frame},{id},{x1},{y1},{w},{h},{c},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, conf in results:
            for tlwh, track_id, c in zip(tlwhs, track_ids, conf):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    c=round(c, 2)
                )
                f.write(line)

def filter_targets(online_targets, aspect_ratio_thresh, min_box_area):
    """Removes targets not meeting threshold criteria."""
    online_tlwhs = []
    online_ids = []
    online_conf = []
    for t in online_targets:
        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
        tid = t[4]
        tc = t[5]
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_conf.append(tc)
    return online_tlwhs, online_ids, online_conf

def dti(txt_path, save_path, n_min=25, n_dti=20):
    """Performs disconnected track interpolation on MOT files."""
    def dti_write_results(filename, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)

    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]
        print(seq_name)
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)