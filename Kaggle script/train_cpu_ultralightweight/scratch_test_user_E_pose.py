import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import classify_asl_geometry, compute_local_hand_frame, extract_per_finger_vectors, TEMPLATES_PER_FINGER

# Build real E pose from user's screenshot:
# Wrist: [0.5, 0.8, 0.0]
# 4 fingers: tightly curled claw (fingertips curled down into palm)
# Thumb: tucked horizontally across under the 4 fingertips
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.50, 0.80, 0.00] # Wrist

# MCPs
pts[1] = [0.44, 0.70, -0.01]
pts[2] = [0.42, 0.64, -0.02]
pts[3] = [0.46, 0.63, 0.01]
pts[4] = [0.50, 0.64, 0.02] # Thumb tip across under fingers

# Index (5-8) curled down
pts[5] = [0.44, 0.52, 0.00]
pts[6] = [0.44, 0.46, 0.02]
pts[7] = [0.44, 0.50, 0.04]
pts[8] = [0.44, 0.56, 0.03] # Index tip

# Middle (9-12) curled down
pts[9] = [0.50, 0.50, 0.00]
pts[10] = [0.50, 0.44, 0.02]
pts[11] = [0.50, 0.48, 0.04]
pts[12] = [0.50, 0.55, 0.03] # Middle tip

# Ring (13-16) curled down
pts[13] = [0.56, 0.52, 0.00]
pts[14] = [0.56, 0.46, 0.02]
pts[15] = [0.56, 0.50, 0.04]
pts[16] = [0.56, 0.57, 0.03] # Ring tip

# Pinky (17-20) curled down
pts[17] = [0.62, 0.56, 0.00]
pts[18] = [0.62, 0.50, 0.02]
pts[19] = [0.62, 0.54, 0.04]
pts[20] = [0.62, 0.60, 0.03] # Pinky tip

extractor = ASLLandmarkFeatureExtractor()
char_geo, conf_geo = classify_asl_geometry(pts)
print(f"Geometric Only on E pose: Char = {char_geo}, Conf = {conf_geo*100:.1f}%")

char_hyb, conf_hyb = extractor.extract_features(pts)
print(f"Hybrid Extractor on E pose: Char = {char_hyb}, Conf = {conf_hyb*100:.1f}%")
