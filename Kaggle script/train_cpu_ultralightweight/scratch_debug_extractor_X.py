import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from asl_mediapipe_simulator import ASLLandmarkFeatureExtractor
from asl_geometric import get_accurate_asl_template, classify_asl_geometry

extractor = ASLLandmarkFeatureExtractor()
templ_x = get_accurate_asl_template('X')

geo_char, geo_conf = classify_asl_geometry(templ_x, tracker=extractor.tracker)
print(f"Geo: {geo_char}, Conf: {geo_conf}")

# Neural model inference:
wrist = templ_x[0].copy()
norm_lm = templ_x - wrist
hand_scale = np.linalg.norm(norm_lm[9])
norm_lm = norm_lm / hand_scale
full_60_pos = np.zeros((60, 3), dtype=np.float32)
full_60_pos[21:42] = norm_lm
feat = np.concatenate([full_60_pos, np.zeros_like(full_60_pos), np.zeros_like(full_60_pos)], axis=-1).flatten()
tensor_x = torch.tensor(np.array([feat]*7), dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    out = extractor.model(tensor_x)
    probs = torch.softmax(out["seq_logits"], dim=-1)[0]
    nn_prob, nn_idx = torch.max(probs, dim=-1)
    nn_char = chr(ord('A') + int(nn_idx.item()))
    print(f"NN: {nn_char}, Conf: {float(nn_prob.item())}")

char_hyb, conf_hyb = extractor.extract_features(templ_x)
print(f"Hybrid: {char_hyb}, Conf: {conf_hyb}")
