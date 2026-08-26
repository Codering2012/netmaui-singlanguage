import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import extract_per_finger_vectors, TEMPLATES_PER_FINGER, compute_local_hand_frame

# Test pose from user screenshot
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.50, 0.80, 0.00] # Wrist

pts[1] = [0.44, 0.70, -0.01]
pts[2] = [0.42, 0.64, -0.02]
pts[3] = [0.46, 0.63, 0.01]
pts[4] = [0.50, 0.64, 0.02] # Thumb tip

pts[5] = [0.44, 0.52, 0.00]; pts[6] = [0.44, 0.46, 0.02]; pts[7] = [0.44, 0.50, 0.04]; pts[8] = [0.44, 0.56, 0.03]
pts[9] = [0.50, 0.50, 0.00]; pts[10] = [0.50, 0.44, 0.02]; pts[11] = [0.50, 0.48, 0.04]; pts[12] = [0.50, 0.55, 0.03]
pts[13] = [0.56, 0.52, 0.00]; pts[14] = [0.56, 0.46, 0.02]; pts[15] = [0.56, 0.50, 0.04]; pts[16] = [0.56, 0.57, 0.03]
pts[17] = [0.62, 0.56, 0.00]; pts[18] = [0.62, 0.50, 0.02]; pts[19] = [0.62, 0.54, 0.04]; pts[20] = [0.62, 0.60, 0.03]

in_vecs = extract_per_finger_vectors(pts)
weights = {"thumb": 0.35, "index": 0.20, "middle": 0.20, "ring": 0.125, "pinky": 0.125}

print("=== SIMILARITY OF USER POSE WITH CLOSED FIST LETTERS ===")
for c in ['A', 'E', 'S', 'T', 'M', 'N', 'O']:
    templ_vecs = TEMPLATES_PER_FINGER[c]
    total_sim = 0.0
    for f_name, w in weights.items():
        u = in_vecs[f_name]
        v = templ_vecs[f_name]
        sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
        total_sim += w * sim
    print(f"  Class '{c}': Similarity = {total_sim:.4f}")
