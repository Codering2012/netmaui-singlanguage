import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Test static alphabet vs dynamic
from asl_geometric import ALPHABET, TEMPLATES_PER_FINGER, DynamicGestureTracker, classify_asl_geometry, get_accurate_asl_template

STATIC_ALPHABET = [c for c in ALPHABET if c not in ['J', 'Z']]
print(f"Static Alphabet (24 letters): {STATIC_ALPHABET}")

for c in ['X', 'D', 'L', 'P', 'Y']:
    templ = get_accurate_asl_template(c)
    char_out, conf = classify_asl_geometry(templ)
    print(f"Template {c} -> Detected: {char_out} (Conf: {conf*100:.1f}%)")
