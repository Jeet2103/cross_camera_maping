import cv2
import pandas as pd
import json
import os
from logger_config.logger import get_logger

# Initialize logger for tracking events and debugging
logger = get_logger("visualize_results.log")

# Ensure the output directory exists to store final visualized videos
os.makedirs('output', exist_ok=True)

try:
    # Load ID mapping from tacticam to broadcast (computed during cross-view matching)
    id_map = json.load(open('mapping/id_mapping.json'))
    # Convert string keys to integer for safe use in visualization
    t2b = {int(k): int(v) for k, v in id_map.items()}
    logger.info("Successfully loaded ID mapping from 'mapping/id_mapping.json'.")
except Exception as e:
    logger.error(f"Failed to load ID mapping: {e}")
    raise

def vis_one(key):
    """
    Visualizes the tracking results for a given video key (either 'broadcast' or 'tacticam').
    Draws bounding boxes with IDs and exports annotated video.
    """
    try:
        # Load tracking results and open corresponding video
        df = pd.read_csv(f'tracking/{key}_tracks.csv')
        cap = cv2.VideoCapture(f'data/{key}.mp4')

        # Get video resolution and frame rate for output writer
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(5)
        out = cv2.VideoWriter(f'output/{key}_output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        logger.info(f"Started visualization for '{key}' at {w}x{h}, {fps} FPS.")
    except Exception as e:
        logger.error(f"Error opening files or video stream for '{key}': {e}")
        return

    fid = 0  # Frame index
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Finished processing all frames for '{key}'.")
            break

        # Filter tracking data for the current frame
        sub = df[df.frame_id == fid]

        for _, r in sub.iterrows():
            tid = int(r.track_id)
            x1, y1, x2, y2 = map(int, [r.x1, r.y1, r.x2, r.y2])

            # Determine final ID and label color
            if key == 'tacticam' and tid in t2b:
                cid = t2b[tid]
                label = f"ID:{cid}"
                color = (0, 255, 0)  # Green for matched IDs
            else:
                cid = tid
                label = f"NoMap:{tid}"
                color = (0, 0, 255) if key == 'tacticam' else (255, 255, 0)  # Red or Cyan

            # Draw bounding box and ID label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save annotated frame to video
        out.write(frame)
        fid += 1

    # Release resources after processing
    cap.release()
    out.release()
    logger.info(f"Visualization saved to output/{key}_output.mp4")

if __name__ == '__main__':
    # Generate visualizations for both camera views
    vis_one('broadcast')
    vis_one('tacticam')
    logger.info("All visualizations completed.")
