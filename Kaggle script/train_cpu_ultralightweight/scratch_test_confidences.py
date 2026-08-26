import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import classify_asl_geometry, compute_local_hand_frame, extract_per_finger_vectors, TEMPLATES_PER_FINGER, get_accurate_asl_template

extractor = ASLLandmarkFeatureExtractor()

for letter in ['A', 'B', 'C', 'D', 'E', 'L', 'P', 'Y']:
    templ = get_accurate_asl_template(letter)
    geo_char, geo_conf = classify_asl_geometry(templ)
    hyb_char, hyb_conf = extractor.extract_features(templ)
    print(f"Template {letter}: Geo -> ({geo_char}, {geo_conf:.2f}), Hybrid -> ({hyb_char}, {hyb_conf:.2f})")
