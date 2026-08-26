import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor

extractor = ASLLandmarkFeatureExtractor()

# User's actual L pose from screenshot:
pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.55, 0.82, 0.0]
pts[1] = [0.49, 0.77, 0.0]; pts[2] = [0.43, 0.72, 0.0]; pts[3] = [0.38, 0.70, 0.0]; pts[4] = [0.32, 0.68, 0.0] # Thumb extended
pts[5] = [0.50, 0.58, 0.0]; pts[6] = [0.50, 0.48, 0.0]; pts[7] = [0.50, 0.38, 0.0]; pts[8] = [0.50, 0.28, 0.0] # Index extended up
pts[9] = [0.55, 0.58, 0.0]; pts[10] = [0.55, 0.52, 0.0]; pts[11] = [0.55, 0.56, 0.0]; pts[12] = [0.55, 0.60, 0.0] # Middle curled
pts[13] = [0.58, 0.60, 0.0]; pts[14] = [0.58, 0.54, 0.0]; pts[15] = [0.58, 0.58, 0.0]; pts[16] = [0.58, 0.62, 0.0] # Ring curled
pts[17] = [0.61, 0.62, 0.0]; pts[18] = [0.61, 0.56, 0.0]; pts[19] = [0.61, 0.60, 0.0]; pts[20] = [0.61, 0.64, 0.0] # Pinky curled

char_out, conf_out = extractor.extract_features(pts)
print(f"User Screenshot L Pose Extractor Output: '{char_out}' (Confidence: {conf_out*100:.1f}%)")
