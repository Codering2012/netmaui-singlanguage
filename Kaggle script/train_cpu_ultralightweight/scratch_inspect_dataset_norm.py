import torch
import numpy as np

shard = torch.load("E:/datasets/asl_dataset/asl_preprocessed_phase1/train/shard_0000.pt", map_location="cpu", weights_only=False)
print(f"Loaded {len(shard)} items from shard_0000.pt")

for i, item in enumerate(shard[:10]):
    feats = item["features"]
    lbl = item.get("label", item.get("label_idx"))
    print(f"Item {i}: label={lbl}, feats shape={feats.shape}")
    # Inspect hand slice (21:42)
    if feats.dim() == 2:
        f_60_9 = feats.view(feats.shape[0], 60, 9)
    else:
        f_60_9 = feats.view(feats.shape[0], 60, 9)
    hand_pos = f_60_9[:, 21:42, 0:3]
    print(f"  Hand pos min={hand_pos.min():.3f}, max={hand_pos.max():.3f}, mean={hand_pos.mean():.3f}, std={hand_pos.std():.3f}")
    # Check wrist (point 21)
    wrist = hand_pos[0, 0]
    print(f"  Wrist frame 0: {wrist.numpy()}")
    # Check middle MCP (point 30 -> 21+9)
    middle_mcp = hand_pos[0, 9]
    print(f"  Middle MCP frame 0: {middle_mcp.numpy()}")
    break
