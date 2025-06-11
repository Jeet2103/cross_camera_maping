import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchreid.utils.feature_extractor import FeatureExtractor
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger("extract_features.log")

# === Directory to save feature files ===
FEATURES_DIR = 'mapping/features'
os.makedirs(FEATURES_DIR, exist_ok=True)

# === Load feature extractor ===
try:
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    logger.info("Feature extractor initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing feature extractor: {e}")
    raise

# === Load precomputed homography matrix ===
try:
    H = np.load('mapping/homography.npy')
    logger.info("Homography matrix loaded.")
except Exception as e:
    logger.error(f"Failed to load homography matrix: {e}")
    raise

def transform_pts(pts, H):
    """
    Apply homography transformation to a list of points.
    """
    return cv2.perspectiveTransform(np.array([pts], dtype=np.float32), H)[0]

def extract_features(video_key):
    """
    Extract appearance and spatial features from tracked players in a video.
    Saves averaged features per player in .npy format.
    """
    logger.info(f"Extracting features for {video_key}...")
    try:
        df = pd.read_csv(f"tracking/{video_key}_tracks.csv")
        cap = cv2.VideoCapture(f"data/{video_key}.mp4")
    except Exception as e:
        logger.error(f"Error loading tracking data or video for '{video_key}': {e}")
        return

    features = {}

    for tid in df['track_id'].unique():
        sub = df[df['track_id'] == tid]
        feats, centers = [], []

        for _, row in sub.iterrows():
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, row.frame_id)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Crop the player's bounding box
                x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Convert BGR to RGB for extractor
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                feats.append(crop)

                # Center point transformation using homography
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                fx, fy = transform_pts([[cx, cy]], H)[0]
                centers.append([fx, fy])

            except Exception as e:
                logger.error(f"Error processing frame {row.frame_id} for TID {tid}: {e}")

        # Compute average features and transformed center
        if feats:
            try:
                with torch.no_grad():
                    fv = extractor(feats)
                    fv = np.stack([f.cpu().numpy() for f in fv])
                    avg_fv = fv.mean(axis=0)
                    avg_ctr = np.mean(centers, axis=0)
                    features[int(tid)] = np.concatenate([avg_fv, avg_ctr])
            except Exception as e:
                logger.error(f"Error extracting features for TID {tid}: {e}")

    # Save features as .npy file
    try:
        np.save(f"{FEATURES_DIR}/{video_key}_features.npy", features)
        logger.info(f"Saved {video_key} features to {FEATURES_DIR}/{video_key}_features.npy")
    except Exception as e:
        logger.error(f"Error saving features for {video_key}: {e}")

# === Entry Point ===
if __name__ == '__main__':
    extract_features('broadcast')
    extract_features('tacticam')
    logger.info("Feature extraction completed for all videos.")
