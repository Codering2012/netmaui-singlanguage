import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, extract_per_finger_vectors, TEMPLATES_PER_FINGER, FINGER_BONES

# Real L pose from user screenshot:
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.5, 0.7, 0.0] # Wrist

pts[1] = [0.45, 0.65, 0.0]
pts[2] = [0.40, 0.63, 0.0]
pts[3] = [0.35, 0.62, 0.0]
pts[4] = [0.30, 0.62, 0.0] # Thumb tip

pts[5] = [0.48, 0.55, 0.0]
pts[6] = [0.48, 0.45, 0.0]
pts[7] = [0.48, 0.35, 0.0]
pts[8] = [0.48, 0.25, 0.0] # Index tip

pts[9] = [0.52, 0.55, 0.0]
pts[10] = [0.52, 0.50, 0.0]
pts[11] = [0.52, 0.53, 0.0]
pts[12] = [0.52, 0.56, 0.0]

pts[13] = [0.55, 0.56, 0.0]
pts[14] = [0.55, 0.52, 0.0]
pts[15] = [0.55, 0.54, 0.0]
pts[16] = [0.55, 0.57, 0.0]

pts[17] = [0.58, 0.58, 0.0]
pts[18] = [0.58, 0.55, 0.0]
pts[19] = [0.58, 0.57, 0.0]
pts[20] = [0.58, 0.60, 0.0]

in_vecs = extract_per_finger_vectors(pts)
templ_vecs = TEMPLATES_PER_FINGER['L']

print("=== FINGER-BY-FINGER SIMILARITY FOR L ===")
for f in ["thumb", "index", "middle", "ring", "pinky"]:
    u = in_vecs[f]
    v = templ_vecs[f]
    sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
    print(f"  Finger {f:6s}: sim = {sim:.4f}")
