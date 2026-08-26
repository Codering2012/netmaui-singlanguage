import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import classify_asl_geometry, extract_per_finger_vectors, TEMPLATES_PER_FINGER, DynamicGestureTracker
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor

# Build an L pose landmark from MediaPipe standard coordinates:
# Thumb: extended to the side (+X)
# Index: pointing straight up (+Y)
# Middle, Ring, Pinky: curled into fist
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.5, 0.7, 0.0] # Wrist

# Thumb: extended horizontally left in mirrored image (-X)
pts[1] = [0.45, 0.65, 0.0]
pts[2] = [0.40, 0.63, 0.0]
pts[3] = [0.35, 0.62, 0.0]
pts[4] = [0.30, 0.62, 0.0] # Thumb tip

# Index: extended straight UP (-Y in image coords)
pts[5] = [0.48, 0.55, 0.0]
pts[6] = [0.48, 0.45, 0.0]
pts[7] = [0.48, 0.35, 0.0]
pts[8] = [0.48, 0.25, 0.0] # Index tip

# Middle: curled
pts[9] = [0.52, 0.55, 0.0]
pts[10] = [0.52, 0.50, 0.0]
pts[11] = [0.52, 0.53, 0.0]
pts[12] = [0.52, 0.56, 0.0]

# Ring: curled
pts[13] = [0.55, 0.56, 0.0]
pts[14] = [0.55, 0.52, 0.0]
pts[15] = [0.55, 0.54, 0.0]
pts[16] = [0.55, 0.57, 0.0]

# Pinky: curled
pts[17] = [0.58, 0.58, 0.0]
pts[18] = [0.58, 0.55, 0.0]
pts[19] = [0.58, 0.57, 0.0]
pts[20] = [0.58, 0.60, 0.0]

char_geo, conf_geo = classify_asl_geometry(pts)
print(f"Geometric Only: Char = {char_geo}, Conf = {conf_geo:.2f}")

extractor = ASLLandmarkFeatureExtractor()
char_hybrid, conf_hybrid = extractor.extract_features(pts)
print(f"Hybrid Extractor: Char = {char_hybrid}, Conf = {conf_hybrid:.2f}")
