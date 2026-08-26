import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import DynamicGestureTracker

extractor = ASLLandmarkFeatureExtractor()

# Base L pose
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.5, 0.7, 0.0]
pts[1] = [0.45, 0.65, 0.0]; pts[2] = [0.40, 0.63, 0.0]; pts[3] = [0.35, 0.62, 0.0]; pts[4] = [0.30, 0.62, 0.0] # Thumb
pts[5] = [0.48, 0.55, 0.0]; pts[6] = [0.48, 0.45, 0.0]; pts[7] = [0.48, 0.35, 0.0]; pts[8] = [0.48, 0.25, 0.0] # Index
pts[9] = [0.52, 0.55, 0.0]; pts[10] = [0.52, 0.50, 0.0]; pts[11] = [0.52, 0.53, 0.0]; pts[12] = [0.52, 0.56, 0.0]
pts[13] = [0.55, 0.56, 0.0]; pts[14] = [0.55, 0.52, 0.0]; pts[15] = [0.55, 0.54, 0.0]; pts[16] = [0.55, 0.57, 0.0]
pts[17] = [0.58, 0.58, 0.0]; pts[18] = [0.58, 0.55, 0.0]; pts[19] = [0.58, 0.57, 0.0]; pts[20] = [0.58, 0.60, 0.0]

print("=== TESTING 30 FRAMES OF L POSE WITH NATURAL HAND JITTER & MICRO-MOVEMENTS ===")
detected_chars = []
for frame in range(30):
    # Add random tremor/jitter (+- 0.015)
    jitter = np.random.uniform(-0.015, 0.015, size=pts.shape)
    jitter_pts = pts + jitter
    c, conf = extractor.extract_features(jitter_pts)
    detected_chars.append(c)

print(f"Detected Characters: {set(detected_chars)}")
if 'Z' not in detected_chars and 'J' not in detected_chars:
    print("[+] SUCCESS: Zero false positive Z/J triggers during hand tremor/micro-movement!")
else:
    print("[!] ERROR: False positive detected!")
