import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import compute_local_hand_frame, get_accurate_asl_template, ALPHABET, STATIC_ALPHABET

# Test on user real L pose
pts_user_l = np.zeros((21, 3), dtype=np.float32)
pts_user_l[0] = [0.58, 0.84, 0.0]
pts_user_l[1] = [0.51, 0.82, 0.0]; pts_user_l[2] = [0.44, 0.74, 0.0]; pts_user_l[3] = [0.39, 0.72, 0.0]; pts_user_l[4] = [0.34, 0.72, 0.0]
pts_user_l[5] = [0.48, 0.56, 0.0]; pts_user_l[6] = [0.47, 0.44, 0.0]; pts_user_l[7] = [0.46, 0.36, 0.0]; pts_user_l[8] = [0.46, 0.28, 0.0]
pts_user_l[9] = [0.52, 0.56, 0.0]; pts_user_l[10] = [0.48, 0.60, 0.0]; pts_user_l[11] = [0.48, 0.67, 0.0]; pts_user_l[12] = [0.50, 0.72, 0.0]
pts_user_l[13] = [0.55, 0.58, 0.0]; pts_user_l[14] = [0.52, 0.62, 0.0]; pts_user_l[15] = [0.52, 0.69, 0.0]; pts_user_l[16] = [0.53, 0.72, 0.0]
pts_user_l[17] = [0.60, 0.60, 0.0]; pts_user_l[18] = [0.59, 0.70, 0.0]; pts_user_l[19] = [0.55, 0.65, 0.0]; pts_user_l[20] = [0.55, 0.69, 0.0]

print("Test complete!")
