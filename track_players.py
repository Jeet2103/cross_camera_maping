import os
import cv2
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from Yolov5_StrongSORT_OSNet.boxmot.tracker_zoo import create_tracker
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger('tracking.log')

# ========== Device Fix Block ==========
# Fix issue where 'CUDA' is incorrectly set as a device string
if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() == "cuda":
    logger.warning("Invalid CUDA_VISIBLE_DEVICES='cuda' detected. Resetting to CPU mode.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Determine if GPU is available, otherwise fallback to CPU
device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
logger.info(f"Using device: {device}")
# ======================================

# === Directory & Path Setup ===
INPUT_DIR = 'detections'  # Directory containing detection CSV files
OUTPUT_DIR = 'tracking'   # Directory to save tracking outputs
VIDEO_DIR = 'data'        # Directory containing input videos
CONFIG = r'Yolov5_StrongSORT_OSNet\boxmot\configs\strongsort.yaml'  # StrongSORT config
REID_CKPT = Path(r'osnet\osnet_x0_25_msmt17.pt')  # OSNet weights

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class label mappings
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# Color for each class for bounding box visualization
COLOR_MAP = {
    0: (255, 0, 0),      # Blue for ball
    1: (0, 255, 255),    # Yellow for goalkeeper
    2: (0, 0, 255),      # Red for player
    3: (0, 255, 0)       # Green for referee
}

FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for text on video frames

# === Main Tracking Function ===
def track_players(video_key):
    """
    Track players and other objects across frames in a given video.
    Saves annotated frames and tracking data per frame.
    """
    logger.info(f"Started tracking on video: {video_key}")

    # File paths
    det_path = f"{INPUT_DIR}/{video_key}_detections.csv"
    video_path = f"{VIDEO_DIR}/{video_key}.mp4"
    output_csv_path = f"{OUTPUT_DIR}/{video_key}_tracks.csv"

    # Validate file existence
    if not os.path.exists(det_path):
        logger.error(f"Detection file not found: {det_path}")
        return

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    # Load detection CSV
    try:
        df = pd.read_csv(det_path)
        logger.info(f"Loaded detections from {det_path}")
    except Exception as e:
        logger.error(f"Failed to load detection CSV: {e}")
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Initialize StrongSORT tracker
    try:
        tracker = create_tracker(
            tracker_type='strongsort',
            tracker_config=CONFIG,
            device=device,
            reid_weights=REID_CKPT
        )
        logger.info("StrongSORT tracker initialized.")
    except Exception as e:
        logger.error(f"Tracker initialization failed: {e}")
        return

    tracks_output = []  # List to store tracking data
    frame_id = 0        # Frame counter

    # Frame-by-frame processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info(f"End of video reached: {video_key}")
            break

        try:
            # Get detections for this frame
            frame_detections = df[df['frame_id'] == frame_id]
            detections = (
                frame_detections[['x1', 'y1', 'x2', 'y2', 'conf', 'class']].to_numpy(dtype=np.float32)
                if not frame_detections.empty else np.empty((0, 6), dtype=np.float32)
            )

            # Update tracker with current detections
            tracker_outputs = tracker.update(dets=detections, img=frame)

            for i, output in enumerate(tracker_outputs):
                # Parse tracker output
                x1, y1, x2, y2, track_id = map(int, output[:5])
                cls_id = int(detections[i][5]) if i < len(detections) else -1

                label = f"{CLASS_NAMES.get(cls_id, 'object')} ID {track_id}"
                color = COLOR_MAP.get(cls_id, (255, 255, 255))

                # Draw bounding box and label
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 4), FONT, 0.5, (0, 0, 0), 1)

                # Store tracking result
                tracks_output.append([frame_id, track_id, x1, y1, x2, y2, cls_id])
        except Exception as e:
            logger.error(f"Tracking failed at frame {frame_id}: {e}")
            continue

        # Save annotated frame
        out_frame_path = f"{OUTPUT_DIR}/{video_key}_track_{frame_id:04d}.jpg"
        try:
            cv2.imwrite(out_frame_path, frame)
            logger.debug(f"Saved frame: {out_frame_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame {frame_id}: {e}")

        frame_id += 1

    cap.release()

    # Save final tracking results as CSV
    try:
        pd.DataFrame(tracks_output, columns=["frame_id", "track_id", "x1", "y1", "x2", "y2", "class"]).to_csv(
            output_csv_path, index=False
        )
        logger.info(f"Tracking data saved: {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save tracking CSV: {e}")

# === Entry Point ===
if __name__ == '__main__':
    track_players('broadcast')
    track_players('tacticam')
