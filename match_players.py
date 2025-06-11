import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from logger_config.logger import get_logger

# Initialize logger
logger = get_logger("match_players.log")

try:
    # === Load extracted features for broadcast and tacticam views ===
    feat_b = np.load('mapping/features/broadcast_features.npy', allow_pickle=True).item()
    feat_t = np.load('mapping/features/tacticam_features.npy', allow_pickle=True).item()
    logger.info("Loaded broadcast and tacticam features successfully.")
except Exception as e:
    logger.error(f"Failed to load features: {e}")
    raise

# === Convert dictionary to arrays for computation ===
ids_b, Xb = zip(*feat_b.items())
ids_t, Xt = zip(*feat_t.items())
Xb = np.stack(Xb).astype('float32')
Xt = np.stack(Xt).astype('float32')

# === Normalize appearance embeddings ===
Xa = Xb[:, :2048]
Ya = Xt[:, :2048]
Xa = Xa / np.linalg.norm(Xa, axis=1, keepdims=True)
Ya = Ya / np.linalg.norm(Ya, axis=1, keepdims=True)

# === Combine appearance and spatial features ===
X = np.concatenate([Xa, Xb[:, 2048:]], axis=1)
Y = np.concatenate([Ya, Xt[:, 2048:]], axis=1)

# === Compute cost matrix ===
W_app, W_sp = 0.7, 0.3  # appearance and spatial weights
try:
    cost = -W_app * (Y[:, :2048] @ Xa.T) + W_sp * (
        np.linalg.norm(Y[:, 2048:][:, None, :] - Xb[:, 2048:], axis=2)
    )
    logger.info("Cost matrix computed using weighted appearance + spatial distance.")
except Exception as e:
    logger.error(f"Error computing cost matrix: {e}")
    raise

# === Solve linear assignment problem ===
row, col = linear_sum_assignment(cost)
mapping = {}

# === Threshold-based assignment and logging ===
for r, c in zip(row, col):
    score = -cost[r, c]
    if score > 0.5:
        mapping[int(ids_t[r])] = int(ids_b[c])
        logger.info(f"Tac {ids_t[r]} -> Broad {ids_b[c]} (score={score:.3f})")
    else:
        logger.warning(f"Tac {ids_t[r]} skipped (score={score:.3f})")

# === Save final mapping ===
try:
    with open('mapping/id_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved ID mapping to 'mapping/id_mapping.json'. Mapped {len(mapping)} players.")
except Exception as e:
    logger.error(f"Failed to save ID mapping: {e}")
