import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import classify_asl_geometry, get_accurate_asl_template

templ_l = get_accurate_asl_template('L')
templ_b = get_accurate_asl_template('B')

cL, confL = classify_asl_geometry(templ_l)
cB, confB = classify_asl_geometry(templ_b)

print(f"L Template -> Detected: '{cL}' (Conf: {confL*100:.1f}%)")
print(f"B Template -> Detected: '{cB}' (Conf: {confB*100:.1f}%)")
