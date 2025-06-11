import cv2
import numpy as np

# List of keypoints in broadcast view (from video)
pts_b = np.array([
    [100, 50], [1180, 50], [1180, 620], [100, 620]
], dtype=np.float32)

# Corresponding points in top-down field coordinates (e.g., standard pitch 0-1 range)
pts_field = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(pts_b, pts_field)
np.save('mapping/homography.npy', H)
print("Homography matrix saved at mapping/homography.npy")
