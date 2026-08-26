import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import get_accurate_asl_template, classify_asl_geometry

templ_x = get_accurate_asl_template('X')
c_geo, conf_geo = classify_asl_geometry(templ_x)
print(f"Geometric matching for X template: Char = {c_geo}, Conf = {conf_geo*100:.1f}%")
