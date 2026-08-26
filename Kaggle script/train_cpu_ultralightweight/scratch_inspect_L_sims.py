import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import extract_per_finger_vectors, TEMPLATES_PER_FINGER, compute_local_hand_frame

pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.58, 0.84, 0.0]
pts[1] = [0.51, 0.82, 0.0]; pts[2] = [0.44, 0.74, 0.0]; pts[3] = [0.39, 0.72, 0.0]; pts[4] = [0.34, 0.72, 0.0]
pts[5] = [0.48, 0.56, 0.0]; pts[6] = [0.47, 0.44, 0.0]; pts[7] = [0.46, 0.36, 0.0]; pts[8] = [0.46, 0.28, 0.0]
pts[9] = [0.52, 0.56, 0.0]; pts[10] = [0.48, 0.60, 0.0]; pts[11] = [0.48, 0.67, 0.0]; pts[12] = [0.50, 0.72, 0.0]
pts[13] = [0.55, 0.58, 0.0]; pts[14] = [0.52, 0.62, 0.0]; pts[15] = [0.52, 0.69, 0.0]; pts[16] = [0.53, 0.72, 0.0]
pts[17] = [0.60, 0.60, 0.0]; pts[18] = [0.59, 0.70, 0.0]; pts[19] = [0.55, 0.65, 0.0]; pts[20] = [0.55, 0.69, 0.0]

in_vecs = extract_per_finger_vectors(pts)
weights = {"thumb": 0.35, "index": 0.20, "middle": 0.20, "ring": 0.125, "pinky": 0.125}

print("=== DETAILED SIMILARITY BREAKDOWN FOR NEW L POSE ===")
sims = []
for c, templ_vecs in TEMPLATES_PER_FINGER.items():
    total_sim = 0.0
    f_sims = {}
    for f_name, w in weights.items():
        u = in_vecs[f_name]
        v = templ_vecs[f_name]
        sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
        total_sim += w * sim
        f_sims[f_name] = sim
    sims.append((c, total_sim, f_sims))

sims.sort(key=lambda x: x[1], reverse=True)
for c, s, fs in sims[:6]:
    print(f"Letter '{c}': Total Sim = {s:.4f} | Thumb: {fs['thumb']:.3f}, Index: {fs['index']:.3f}, Middle: {fs['middle']:.3f}, Ring: {fs['ring']:.3f}, Pinky: {fs['pinky']:.3f}")
