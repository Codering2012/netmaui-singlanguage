import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template, disambiguate_closed_fist_cluster

# Realistic S pose from camera:
# 4 fingers curled into a fist
# Thumb folded across the front of index & middle fingers:
# Joint 0: Wrist [0.50, 0.80, 0.0]
# Joint 1: Thumb CMC [0.46, 0.72, 0.02]
# Joint 2: Thumb MCP [0.44, 0.65, 0.04]
# Joint 3: Thumb IP  [0.48, 0.63, 0.06] # Crossing over index
# Joint 4: Thumb Tip [0.53, 0.64, 0.06] # Crossing over middle

pts_s = np.zeros((21, 3), dtype=np.float32)
pts_s[0] = [0.50, 0.80, 0.00]
pts_s[1] = [0.46, 0.72, 0.02]; pts_s[2] = [0.44, 0.65, 0.04]; pts_s[3] = [0.48, 0.63, 0.06]; pts_s[4] = [0.53, 0.64, 0.06]
pts_s[5] = [0.44, 0.54, 0.00]; pts_s[6] = [0.44, 0.60, 0.02]; pts_s[7] = [0.44, 0.66, 0.02]; pts_s[8] = [0.44, 0.70, 0.01]
pts_s[9] = [0.50, 0.52, 0.00]; pts_s[10] = [0.50, 0.58, 0.02]; pts_s[11] = [0.50, 0.65, 0.02]; pts_s[12] = [0.50, 0.69, 0.01]
pts_s[13] = [0.56, 0.54, 0.00]; pts_s[14] = [0.56, 0.60, 0.02]; pts_s[15] = [0.56, 0.66, 0.02]; pts_s[16] = [0.56, 0.70, 0.01]
pts_s[17] = [0.62, 0.58, 0.00]; pts_s[18] = [0.62, 0.64, 0.02]; pts_s[19] = [0.62, 0.68, 0.02]; pts_s[20] = [0.62, 0.72, 0.01]

local_s = compute_local_hand_frame(pts_s)
print("Local S coordinates:")
print(f"  Thumb Tip  : X={local_s[4,0]:.3f}, Y={local_s[4,1]:.3f}, Z={local_s[4,2]:.3f}")
print(f"  Thumb MCP  : X={local_s[2,0]:.3f}, Y={local_s[2,1]:.3f}, Z={local_s[2,2]:.3f}")
print(f"  Index PIP  : X={local_s[6,0]:.3f}, Y={local_s[6,1]:.3f}, Z={local_s[6,2]:.3f}")
print(f"  Middle PIP : X={local_s[10,0]:.3f}, Y={local_s[10,1]:.3f}, Z={local_s[10,2]:.3f}")
print(f"  Index Tip  : X={local_s[8,0]:.3f}, Y={local_s[8,1]:.3f}, Z={local_s[8,2]:.3f}")

res_char, res_conf = disambiguate_closed_fist_cluster(local_s)
print(f"\nCurrent disambiguator output on S: '{res_char}' ({res_conf*100:.1f}%)")
