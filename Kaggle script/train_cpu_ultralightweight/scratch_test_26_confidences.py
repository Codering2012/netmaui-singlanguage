import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import get_accurate_asl_template, ALPHABET

print("=== CONFIDENCE TEST ACROSS ALL 26 LETTERS A-Z ===")
low_conf_count = 0
for letter in ALPHABET:
    extractor = ASLLandmarkFeatureExtractor()
    templ = get_accurate_asl_template(letter)
    char_out, conf_out = extractor.extract_features(templ)
    status = "OK" if conf_out >= 0.90 else "LOW"
    if conf_out < 0.90:
        low_conf_count += 1
    print(f"Letter '{letter}': Detected '{char_out}' | Confidence: {conf_out*100:5.1f}% [{status}]")

print(f"\nSummary: {26 - low_conf_count}/26 letters have high confidence (>= 90%)!")
