import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor

extractor = ASLLandmarkFeatureExtractor()

pts = np.zeros((21, 3), dtype=np.float32)
pts[0] = [0.55, 0.82, 0.0]
pts[1] = [0.49, 0.77, 0.0]; pts[2] = [0.43, 0.72, 0.0]; pts[3] = [0.38, 0.70, 0.0]; pts[4] = [0.32, 0.68, 0.0]
pts[5] = [0.50, 0.58, 0.0]; pts[6] = [0.50, 0.48, 0.0]; pts[7] = [0.50, 0.38, 0.0]; pts[8] = [0.50, 0.28, 0.0]
pts[9] = [0.55, 0.58, 0.0]; pts[10] = [0.55, 0.52, 0.0]; pts[11] = [0.55, 0.56, 0.0]; pts[12] = [0.55, 0.60, 0.0]
pts[13] = [0.58, 0.60, 0.0]; pts[14] = [0.58, 0.54, 0.0]; pts[15] = [0.58, 0.58, 0.0]; pts[16] = [0.58, 0.62, 0.0]
pts[17] = [0.61, 0.62, 0.0]; pts[18] = [0.61, 0.56, 0.0]; pts[19] = [0.61, 0.60, 0.0]; pts[20] = [0.61, 0.64, 0.0]

wrist = pts[0].copy()
norm_lm = pts - wrist
hand_scale = np.linalg.norm(norm_lm[9])
norm_lm = norm_lm / hand_scale
full_60_pos = np.zeros((60, 3), dtype=np.float32)
full_60_pos[21:42] = norm_lm
feat = np.concatenate([full_60_pos, np.zeros_like(full_60_pos), np.zeros_like(full_60_pos)], axis=-1).flatten()
tensor_x = torch.tensor(np.array([feat]*7), dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    out = extractor.model(tensor_x)
    probs = torch.softmax(out["seq_logits"], dim=-1)[0]
    for idx in torch.topk(probs, 5).indices:
        idx_val = int(idx.item())
        c = chr(ord('A') + idx_val)
        p = float(probs[idx_val].item())
        print(f"NN Rank: '{c}' ({p*100:.1f}%)")
