import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template

for char in ['A', 'E', 'S', 'T', 'M', 'N']:
    t = get_accurate_asl_template(char)
    lp = compute_local_hand_frame(t)
    print(f"--- TEMPLATE '{char}' ---")
    print(f"  Thumb CMC (1): X={lp[1,0]:.3f}, Y={lp[1,1]:.3f}, Z={lp[1,2]:.3f}")
    print(f"  Thumb MCP (2): X={lp[2,0]:.3f}, Y={lp[2,1]:.3f}, Z={lp[2,2]:.3f}")
    print(f"  Thumb IP  (3): X={lp[3,0]:.3f}, Y={lp[3,1]:.3f}, Z={lp[3,2]:.3f}")
    print(f"  Thumb Tip (4): X={lp[4,0]:.3f}, Y={lp[4,1]:.3f}, Z={lp[4,2]:.3f}")
    print(f"  Index Tip (8): X={lp[8,0]:.3f}, Y={lp[8,1]:.3f}, Z={lp[8,2]:.3f}")
    print(f"  Middle Tip(12): X={lp[12,0]:.3f}, Y={lp[12,1]:.3f}, Z={lp[12,2]:.3f}")
