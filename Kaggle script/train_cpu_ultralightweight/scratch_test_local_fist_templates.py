import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import get_accurate_asl_template, compute_local_hand_frame

for c in ['A', 'E', 'S', 'T', 'M', 'N']:
    templ = get_accurate_asl_template(c)
    local_pts = compute_local_hand_frame(templ)
    thumb_tip = local_pts[4]
    index_tip = local_pts[8]
    middle_tip = local_pts[12]
    print(f"Letter '{c}':")
    print(f"  Thumb Tip  : X={thumb_tip[0]:.3f}, Y={thumb_tip[1]:.3f}, Z={thumb_tip[2]:.3f}")
    print(f"  Index Tip  : X={index_tip[0]:.3f}, Y={index_tip[1]:.3f}, Z={index_tip[2]:.3f}")
    print(f"  Middle Tip : X={middle_tip[0]:.3f}, Y={middle_tip[1]:.3f}, Z={middle_tip[2]:.3f}")
