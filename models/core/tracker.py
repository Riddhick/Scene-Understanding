# -----------------------------------------------------------------------------
#
#       Focused Video Tracking Pipeline with Camera Compensation (Fixed)
#
# -----------------------------------------------------------------------------

import cv2
import json
import math
import random
import numpy as np
from collections import defaultdict
import torch
from ultralytics import YOLO
import requests
from boxmot import StrongSort
from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
class_colors = {}

def get_class_color(class_name: str):
    """Assigns a unique and consistent color for each object class."""
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return class_colors[class_name]

# -----------------------------------------------------------------------------
# DETECTION MODEL LOADER
# -----------------------------------------------------------------------------
def load_model(model_name="D:\\Work\\RCI\\Code\\models\\weights\\yolov11_trained.pt"):
    """Loads the YOLO object detection model."""
    print(f"Loading model: {model_name}")
    if not Path(model_name).exists():
        print(f"❌ Error: Model file not found at {model_name}")
        sys.exit(1)
    return YOLO(model_name)

# --- Tracking ID helpers ---
id_remap = defaultdict(dict)
next_id_counter = defaultdict(int)
id_class_history = defaultdict(list)
last_confidence = defaultdict(float)
HISTORY_LENGTH = 5

# -----------------------------------------------------------------------------
# CAMERA COMPENSATION
# -----------------------------------------------------------------------------
def transform_point(point, matrix):
    """Applies a 2x3 affine matrix to a (x, y) point."""
    p = np.array([point[0], point[1], 1], dtype=np.float32)
    transformed_p = matrix @ p
    return (int(transformed_p[0]), int(transformed_p[1]))

# -----------------------------------------------------------------------------
# DIRECTION CALCULATION
# -----------------------------------------------------------------------------
def compute_direction(start, end):
    """Returns dominant direction (up/down/left/right or stationary)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 15:
        return "stationary"
    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360
    if 45 <= angle < 135:
        return "up"
    elif 135 <= angle < 225:
        return "left"
    elif 225 <= angle < 315:
        return "down"
    else:
        return "right"

# -----------------------------------------------------------------------------
# FILE DOWNLOADER (for ReID model)
# -----------------------------------------------------------------------------
def download_file(url, local_filename):
    """Downloads ReID model if not available."""
    print(f"Downloading ReID model from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
        print(f"✅ Download complete: {local_filename}")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# VIDEO PROCESSING
# -----------------------------------------------------------------------------
def process_video(video_path, model, output_json_path="video_temporal_output.json",
                  save_output=True, trajectory_sample_n=10):
    """Runs detection + tracking with camera compensation and temporal summarization."""

    if not Path(video_path).exists():
        print(f"❌ Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    out_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

    frame_count = 0
    object_tracks = {}
    class_map = {}

    # --- Optical Flow Parameters for Camera Motion Estimation ---
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)

    prev_gray = None
    M_global = np.eye(2, 3, dtype=np.float32)

    # --- Initialize StrongSORT ---
    reid_model_path = Path('osnet_x0_25_msmt17.pt')
    if not reid_model_path.exists():
        reid_model_url = "https://github.com/maudrun/boxmot/releases/download/v0.0.12/osnet_x0_25_msmt17.pt"
        download_file(reid_model_url, reid_model_path)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing StrongSORT on device: {device}")
    tracker = StrongSort(
        reid_weights=reid_model_path,
        device=device,
        half=True if device != 'cpu' else False
    )
    print("✅ StrongSORT initialized.")

    print("\n--- Starting video processing (temporal mode) ---")

    # -----------------------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- CAMERA COMPENSATION ---
        M_frame = np.eye(2, 3, dtype=np.float32)
        if prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                if len(good_new) > 6:
                    M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                    if M is not None:
                        M_frame = M
        M_global_3x3 = np.vstack([M_global, [0, 0, 1]])
        M_frame_3x3 = np.vstack([M_frame, [0, 0, 1]])
        M_global_updated_3x3 = M_frame_3x3 @ M_global_3x3
        M_global = M_global_updated_3x3[0:2, :]
        M_compensate = cv2.invertAffineTransform(M_global)
        prev_gray = frame_gray.copy()

        annotated_frame = frame.copy()
        detected_objects = []

        # --- DETECTION + TRACKING ---
        results = model.predict(frame, verbose=False, conf=0.40)
        dets = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf.item())
                cls_id = float(box.cls.item())
                dets.append([x1, y1, x2, y2, conf, cls_id])
        dets = np.array(dets, dtype=np.float32)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        if dets.shape[0] > 0:
            try:
                tracks_np = tracker.update(annotated_frame, dets)
            except AssertionError as e:
                print("⚠️ Tracker update failed:", e)
                tracks_np = np.empty((0, 8))
        else:
            tracks_np = np.empty((0, 8))

        # --- TRACK VISUALIZATION ---
        if tracks_np.shape[0] > 0:
            for track in tracks_np:
                x1, y1, x2, y2, orig_id, conf, cls_id, _ = track
                x1, y1, x2, y2, orig_id = map(int, [x1, y1, x2, y2, orig_id])
                cls_id = int(cls_id)
                conf = float(conf)
                raw_class = model.names[cls_id]

                id_class_history[orig_id].append(raw_class)
                if len(id_class_history[orig_id]) > HISTORY_LENGTH:
                    id_class_history[orig_id].pop(0)
                stable_class = max(set(id_class_history[orig_id]),
                                   key=id_class_history[orig_id].count)
                prev_conf = last_confidence.get(orig_id, 0)
                if conf > prev_conf + 0.1:
                    stable_class = raw_class
                    id_class_history[orig_id] = [stable_class]
                    last_confidence[orig_id] = conf
                else:
                    last_confidence[orig_id] = max(prev_conf, conf)
                if orig_id not in id_remap[stable_class]:
                    next_id_counter[stable_class] += 1
                    id_remap[stable_class][orig_id] = next_id_counter[stable_class]
                new_id = id_remap[stable_class][orig_id]
                unique_name = f"{stable_class} {new_id}"
                comp_obj = {
                    "name": unique_name,
                    "class": stable_class,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                }
                detected_objects.append(comp_obj)

                # Draw bbox
                color = get_class_color(stable_class)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, unique_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- TRACK COMPENSATION + TRAJECTORY STORAGE ---
        for obj in detected_objects:
            name = obj["name"]
            center = obj["center"]
            compensated_center = transform_point(center, M_compensate)
            if name not in object_tracks:
                object_tracks[name] = []
                class_map[name] = obj["class"]
            if frame_count % trajectory_sample_n == 0:
                object_tracks[name].append((frame_count, compensated_center))

        # --- Visualization ---
        cv2.imshow("Scene Understanding - Video", annotated_frame)
        if save_output:
            out_writer.write(annotated_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------------------------------------------------------
    # POST-PROCESS SUMMARY
    # -----------------------------------------------------------------------------
    summarized_objects = []
    for name, trajectory in object_tracks.items():
        if len(trajectory) < 2:
            continue
        entry_frame = trajectory[0][0]
        exit_frame = trajectory[-1][0]
        duration_sec_obj = (exit_frame - entry_frame) / fps
        total_dist = sum(
            math.hypot(trajectory[i][1][0] - trajectory[i-1][1][0],
                       trajectory[i][1][1] - trajectory[i-1][1][1])
            for i in range(1, len(trajectory))
        )
        avg_speed = total_dist / duration_sec_obj if duration_sec_obj > 0 else 0
        direction = compute_direction(trajectory[0][1], trajectory[-1][1])
        stationary = direction == "stationary"
        summarized_objects.append({
            "id": name,
            "class": class_map[name],
            "entry_frame": entry_frame,
            "exit_frame": exit_frame,
            "entry_time_sec": round(entry_frame / fps, 2),
            "exit_time_sec": round(exit_frame / fps, 2),
            "duration_sec": round(duration_sec_obj, 2),
            "trajectory": [{"frame": f, "x": c[0], "y": c[1]} for f, c in trajectory],
            "avg_speed_px_per_sec": round(avg_speed, 2),
            "direction": direction,
            "stationary": stationary
        })

    video_summary = {
        "video_metadata": {
            "video_name": Path(video_path).name,
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2)
        },
        "objects": summarized_objects
    }

    with open(output_json_path, "w") as f:
        json.dump(video_summary, f, indent=4)

    cap.release()
    if save_output:
        out_writer.release()
    cv2.destroyAllWindows()

    print("\n✅ Video processing complete.")
    print(f"➡ Annotated video saved as 'annotated_output.mp4'")
    print(f"➡ JSON summary saved as '{output_json_path}'")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main_video():
    video_path = "C:\\Users\\Riddhick\\Downloads\\cropped.mp4"
    model_path = "D:\\Work\\RCI\\Code\\models\\weights\\yolov11_trained.pt"
    model = load_model(model_path)
    process_video(video_path, model, "video_scene_output.json")

if __name__ == "__main__":
    main_video()
