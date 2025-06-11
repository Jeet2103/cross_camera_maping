from ultralytics import YOLO
import cv2
import os
import pandas as pd
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger('detection.log')

# Paths
VIDEO_PATHS = {'broadcast': 'data/broadcast.mp4', 'tacticam': 'data/tacticam.mp4'}
MODEL_PATH = 'model/best.pt'
OUTPUT_DIR = 'detections'
COUNT_DIR = 'counts'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COUNT_DIR, exist_ok=True)

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    logger.info(f'Model loaded from {MODEL_PATH}')
except Exception as e:
    logger.error(f'Failed to load model from {MODEL_PATH}: {e}')
    raise

# Class names and drawing colors
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
COLOR_MAP = {
    0: (255, 0, 0),      # Blue
    1: (0, 255, 255),    # Yellow
    2: (0, 0, 255),      # Red
    3: (0, 255, 0)       # Green
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Detection function with logging
def detect_players(video_key):
    logger.info(f'Starting detection on video: {video_key}')
    
    if video_key not in VIDEO_PATHS:
        logger.error(f"Invalid video key: {video_key}")
        return

    cap = cv2.VideoCapture(VIDEO_PATHS[video_key])
    if not cap.isOpened():
        logger.error(f"Failed to open video: {VIDEO_PATHS[video_key]}")
        return

    frame_id = 0
    results_list, counts_list = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info(f'End of video stream for {video_key}')
            break

        try:
            results = model(frame)
        except Exception as e:
            logger.error(f"Inference failed on frame {frame_id}: {e}")
            continue

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # Count objects in current frame
        count_vals = {
            0: sum(c == 0 for c in classes),  # ball
            1: sum(c == 1 for c in classes),  # goalkeeper
            2: sum(c == 2 for c in classes),  # player
            3: sum(c == 3 for c in classes)   # referee
        }
        counts_list.append([frame_id, count_vals[0], count_vals[1], count_vals[2], count_vals[3]])

        # Draw detections and collect data
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i][:4])
            conf = float(confs[i])
            cls_id = classes[i]
            label = f"{CLASS_NAMES.get(cls_id, 'object')} {conf:.2f}"
            color = COLOR_MAP.get(cls_id, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), FONT, 0.8, (0, 0, 0), 2)

            results_list.append([frame_id, x1, y1, x2, y2, conf, cls_id])

        # Save frame
        out_path = f"{OUTPUT_DIR}/{video_key}_frame_{frame_id:04d}.jpg"
        try:
            cv2.imwrite(out_path, frame)
            logger.debug(f"Saved annotated frame: {out_path}")
        except Exception as e:
            logger.warning(f"Could not save frame {frame_id}: {e}")

        frame_id += 1

    cap.release()

    # Save results to CSV
    try:
        df = pd.DataFrame(results_list, columns=["frame_id", "x1", "y1", "x2", "y2", "conf", "class"])
        df.to_csv(f"{OUTPUT_DIR}/{video_key}_detections.csv", index=False)
        logger.info(f"Detections saved to: {OUTPUT_DIR}/{video_key}_detections.csv")
    except Exception as e:
        logger.error(f"Error saving detection CSV: {e}")

    try:
        df_count = pd.DataFrame(counts_list, columns=["frame_id", "balls", "goalkeepers", "players", "referees"])
        df_count.to_csv(f"{COUNT_DIR}/{video_key}_object_counts.csv", index=False)
        logger.info(f"Counts saved to: {COUNT_DIR}/{video_key}_object_counts.csv")
    except Exception as e:
        logger.error(f"Error saving count CSV: {e}")


if __name__ == '__main__':
    detect_players('broadcast')
    detect_players('tacticam')
