import sys
import os
sys.path.insert(0, r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train")

import torch
import torch.nn.functional as F
from model import UltraLightSignModel
import json

# 1. Load the first shard to get a batch instantly
print("[*] Loading 1 batch of 32 sequences directly from shard 0...")
data = torch.load(r"E:\datasets\results\asl_preprocessed_phase1\train\shard_0000.pt", weights_only=False, map_location="cpu")

batch_size = 32
max_len = 256
batch_features = []
batch_targets = []
batch_masks = []

for i in range(batch_size):
    record = data[i]
    feat = torch.tensor(record["features"], dtype=torch.float32)
    seq_len = feat.size(0)
    
    # Pad to max_len
    if seq_len < max_len:
        pad_size = max_len - seq_len
        pad = torch.zeros((pad_size, 60, 9), dtype=torch.float32)
        padded_feat = torch.cat([feat, pad], dim=0)
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_size)], dim=0).bool()
    else:
        padded_feat = feat[:max_len]
        mask = torch.ones(max_len).bool()
        
    batch_features.append(padded_feat)
    batch_masks.append(mask)
    batch_targets.append(record["label_idx"])

features = torch.stack(batch_features)
masks = torch.stack(batch_masks)
targets = torch.tensor(batch_targets, dtype=torch.long)

# 2. Instantiate Model
print(f"[*] Instantiating UltraLightSignModel...")
model = UltraLightSignModel(
    num_classes=6152,
    num_keypoints=60,
    channels_per_kp=9,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.1,
    max_len=256
)
model.train()

# 3. Forward Pass
print("[*] Running Forward Pass...")
outputs = model(features, mask=masks, return_aux=True)
logits, lex_logits, conf_pred = outputs[0], outputs[1], outputs[2]

# 4. Compute Loss
print("[*] Computing Loss...")
loss_sign = F.cross_entropy(logits, targets, label_smoothing=0.1)

print("="*50)
print(f"BATCH SIZE : {batch_size}")
print(f"FEATURES   : {features.shape}")
print(f"TARGETS    : {targets.shape}")
print("-" * 50)
print(f"MODEL CLASSIFICATION LOSS : {loss_sign.item():.4f}")
print("="*50)
