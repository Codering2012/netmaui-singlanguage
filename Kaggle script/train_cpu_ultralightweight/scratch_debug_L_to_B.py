import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import classify_asl_geometry, compute_local_hand_frame, extract_per_finger_vectors, TEMPLATES_PER_FINGER, FINGER_BONES

# Real L pose from user screenshot:
# Wrist: [0.55, 0.82, 0.0]
# Thumb: extended horizontally to the left (X=0.32, Y=0.68)
# Index: pointing straight up (X=0.50, Y=0.28)
# Middle: curled at (X=0.54, Y=0.55)
# Ring: curled at (X=0.57, Y=0.57)
# Pinky: curled at (X=0.60, Y=0.60)
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.55, 0.82, 0.0] # Wrist

pts[1] = [0.49, 0.77, 0.0]
pts[2] = [0.43, 0.72, 0.0]
pts[3] = [0.38, 0.70, 0.0]
pts[4] = [0.32, 0.68, 0.0] # Thumb tip (extended horizontally!)

pts[5] = [0.50, 0.58, 0.0]
pts[6] = [0.50, 0.48, 0.0]
pts[7] = [0.50, 0.38, 0.0]
pts[8] = [0.50, 0.28, 0.0] # Index tip (pointing straight up!)

pts[9] = [0.55, 0.58, 0.0]
pts[10] = [0.55, 0.52, 0.0]
pts[11] = [0.55, 0.56, 0.0]
pts[12] = [0.55, 0.60, 0.0] # Middle tip (curled!)

pts[13] = [0.58, 0.60, 0.0]
pts[14] = [0.58, 0.54, 0.0]
pts[15] = [0.58, 0.58, 0.0]
pts[16] = [0.58, 0.62, 0.0] # Ring tip (curled!)

pts[17] = [0.61, 0.62, 0.0]
pts[18] = [0.61, 0.56, 0.0]
pts[19] = [0.61, 0.60, 0.0]
pts[20] = [0.61, 0.64, 0.0] # Pinky tip (curled!)

in_vecs = extract_per_finger_vectors(pts)
weights = {"thumb": 0.35, "index": 0.20, "middle": 0.20, "ring": 0.125, "pinky": 0.125}

print("=== SIMILARITY OF USER L POSE WITH ALL TEMPLATES ===")
all_sims = []
for c, templ_vecs in TEMPLATES_PER_FINGER.items():
    total_sim = 0.0
    for f_name, w in weights.items():
        u = in_vecs[f_name]
        v = templ_vecs[f_name]
        sim = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
        total_sim += w * sim
    all_sims.append((c, total_sim))

all_sims.sort(key=lambda x: x[1], reverse=True)
for c, s in all_sims[:8]:
    print(f"  Class '{c}': Similarity = {s:.4f}")

extractor = ASLLandmarkFeatureExtractor()
c_geo, conf_geo = classify_asl_geometry(pts)
c_hyb, conf_hyb = extractor.extract_features(pts)
print(f"\nGeometric Result: '{c_geo}' ({conf_geo*100:.1f}%)")
print(f"Hybrid Extractor Result: '{c_hyb}' ({conf_hyb*100:.1f}%)")
