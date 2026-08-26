import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import extract_per_finger_vectors, TEMPLATES_PER_FINGER, compute_local_hand_frame, classify_asl_geometry
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor

pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.58, 0.84, 0.0] # Wrist

pts[1] = [0.51, 0.82, 0.0]
pts[2] = [0.44, 0.74, 0.0]
pts[3] = [0.39, 0.72, 0.0]
pts[4] = [0.34, 0.72, 0.0] # Thumb tip

pts[5] = [0.48, 0.56, 0.0]
pts[6] = [0.47, 0.44, 0.0]
pts[7] = [0.46, 0.36, 0.0]
pts[8] = [0.46, 0.28, 0.0] # Index tip

pts[9] = [0.52, 0.56, 0.0]
pts[10] = [0.48, 0.60, 0.0]
pts[11] = [0.48, 0.67, 0.0]
pts[12] = [0.50, 0.72, 0.0] # Middle tip

pts[13] = [0.55, 0.58, 0.0]
pts[14] = [0.52, 0.62, 0.0]
pts[15] = [0.52, 0.69, 0.0]
pts[16] = [0.53, 0.72, 0.0] # Ring tip

pts[17] = [0.60, 0.60, 0.0]
pts[18] = [0.59, 0.70, 0.0]
pts[19] = [0.55, 0.65, 0.0]
pts[20] = [0.55, 0.69, 0.0] # Pinky tip

c_geo, conf_geo = classify_asl_geometry(pts)
print(f"Geometric matching: Char = '{c_geo}', Conf = {conf_geo*100:.1f}%")

extractor = ASLLandmarkFeatureExtractor()
c_hyb, conf_hyb = extractor.extract_features(pts)
print(f"Hybrid extractor: Char = '{c_hyb}', Conf = {conf_hyb*100:.1f}%")
