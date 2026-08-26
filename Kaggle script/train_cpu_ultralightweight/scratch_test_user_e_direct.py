import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template

def disambiguate_closed_fist_cluster(local_pts):
    thumb_tip = local_pts[4]
    index_tip = local_pts[8]
    middle_tip = local_pts[12]
    ring_tip = local_pts[16]
    pinky_tip = local_pts[20]
    
    avg_tips_y = (index_tip[1] + middle_tip[1] + ring_tip[1] + pinky_tip[1]) / 4.0
    avg_tips_z = (index_tip[2] + middle_tip[2] + ring_tip[2] + pinky_tip[2]) / 4.0

    if thumb_tip[0] > 0.26 and thumb_tip[1] > 0.90:
        return 'A', 0.99

    if middle_tip[0] < thumb_tip[0] < index_tip[0] and thumb_tip[1] > avg_tips_y + 0.10:
        return 'T', 0.99

    if thumb_tip[2] > avg_tips_z + 0.06 and thumb_tip[1] >= avg_tips_y - 0.05:
        return 'S', 0.99

    if thumb_tip[0] < ring_tip[0]:
        return 'M', 0.99
    elif thumb_tip[0] < middle_tip[0] - 0.04:
        return 'N', 0.99
    else:
        return 'E', 0.99

# Real user E pose
pts_user_e = np.zeros((21, 3), dtype=np.float32)
pts_user_e[0] = [0.50, 0.80, 0.00]
pts_user_e[1] = [0.44, 0.70, -0.01]; pts_user_e[2] = [0.42, 0.64, -0.02]; pts_user_e[3] = [0.46, 0.63, 0.01]; pts_user_e[4] = [0.50, 0.64, 0.02]
pts_user_e[5] = [0.44, 0.52, 0.00]; pts_user_e[6] = [0.44, 0.46, 0.02]; pts_user_e[7] = [0.44, 0.50, 0.04]; pts_user_e[8] = [0.44, 0.56, 0.03]
pts_user_e[9] = [0.50, 0.50, 0.00]; pts_user_e[10] = [0.50, 0.44, 0.02]; pts_user_e[11] = [0.50, 0.48, 0.04]; pts_user_e[12] = [0.50, 0.55, 0.03]
pts_user_e[13] = [0.56, 0.52, 0.00]; pts_user_e[14] = [0.56, 0.46, 0.02]; pts_user_e[15] = [0.56, 0.50, 0.04]; pts_user_e[16] = [0.56, 0.57, 0.03]
pts_user_e[17] = [0.62, 0.56, 0.00]; pts_user_e[18] = [0.62, 0.50, 0.02]; pts_user_e[19] = [0.62, 0.54, 0.04]; pts_user_e[20] = [0.62, 0.60, 0.03]

local_user_e = compute_local_hand_frame(pts_user_e)
char_e, conf_e = disambiguate_closed_fist_cluster(local_user_e)
print(f"User E Pose -> Detected '{char_e}' (Confidence: {conf_e*100:.1f}%)")
