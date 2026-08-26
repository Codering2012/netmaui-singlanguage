import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import get_accurate_asl_template

extractor = ASLLandmarkFeatureExtractor()

print("=== VERIFYING FIST CLUSTER (A, E, S, T, M, N) ===")
for c in ['A', 'E', 'S', 'T', 'M', 'N']:
    templ = get_accurate_asl_template(c)
    pred_char, conf = extractor.extract_features(templ)
    print(f"Letter '{c}' -> Recognized as '{pred_char}' (Confidence: {conf*100:.1f}%) [{'PASS' if pred_char == c else 'FAIL'}]")
