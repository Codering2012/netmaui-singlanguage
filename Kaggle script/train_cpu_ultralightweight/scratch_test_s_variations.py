import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import extract_per_finger_vectors, TEMPLATES_PER_FINGER, classify_asl_geometry
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor

extractor = ASLLandmarkFeatureExtractor()

# Test various natural camera variations of S:
# Variation 1: Thumb slightly looser or higher
# Variation 2: Fist slightly rotated or flat
# Variation 3: Thumb tip touching index PIP vs middle PIP
for thumb_y_offset in [-0.04, -0.02, 0.0, 0.02, 0.04]:
    for thumb_z_offset in [0.00, 0.02, 0.04, 0.06]:
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = [0.50, 0.80, 0.00]
        pts[1] = [0.46, 0.72, 0.01]; pts[2] = [0.44, 0.65, 0.02]
        pts[3] = [0.48, 0.63 + thumb_y_offset, 0.03 + thumb_z_offset]
        pts[4] = [0.52, 0.62 + thumb_y_offset, 0.03 + thumb_z_offset] # Thumb tip across
        pts[5] = [0.44, 0.54, 0.00]; pts[6] = [0.44, 0.60, 0.02]; pts[7] = [0.44, 0.66, 0.02]; pts[8] = [0.44, 0.70, 0.01]
        pts[9] = [0.50, 0.52, 0.00]; pts[10] = [0.50, 0.58, 0.02]; pts[11] = [0.50, 0.65, 0.02]; pts[12] = [0.50, 0.69, 0.01]
        pts[13] = [0.56, 0.54, 0.00]; pts[14] = [0.56, 0.60, 0.02]; pts[15] = [0.56, 0.66, 0.02]; pts[16] = [0.56, 0.70, 0.01]
        pts[17] = [0.62, 0.58, 0.00]; pts[18] = [0.62, 0.64, 0.02]; pts[19] = [0.62, 0.68, 0.02]; pts[20] = [0.62, 0.72, 0.01]

        char_out, conf_out = extractor.extract_features(pts)
        c_geo, conf_geo = classify_asl_geometry(pts)
        if char_out != 'S':
            print(f"FAILED for y_off={thumb_y_offset:+.2f}, z_off={thumb_z_offset:+.2f} -> Geo: '{c_geo}' ({conf_geo*100:.1f}%), Extractor: '{char_out}' ({conf_out*100:.1f}%)")
