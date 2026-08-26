import sys
import time
import torch
import numpy as np

def old_collate(batch, max_len=384, eng_pad_id=0):
    bsz = len(batch)
    input_padded = torch.full((bsz, max_len), eng_pad_id, dtype=torch.int32)
    target_padded = torch.full((bsz, max_len), eng_pad_id, dtype=torch.int32)
    is_dae_mask = torch.zeros(bsz, dtype=torch.bool)
    for i, x in enumerate(batch):
        in_seq = x["input_ids"]
        tgt_seq = x["target_ids"]
        in_len = min(max_len, len(in_seq))
        tgt_len = min(max_len, len(tgt_seq))
        input_padded[i, :in_len] = in_seq[:in_len] if isinstance(in_seq, torch.Tensor) else torch.tensor(in_seq[:in_len], dtype=torch.int32)
        target_padded[i, :tgt_len] = tgt_seq[:tgt_len] if isinstance(tgt_seq, torch.Tensor) else torch.tensor(tgt_seq[:tgt_len], dtype=torch.int32)
        is_dae_mask[i] = bool(x.get("is_dae", False))
    return {"input_ids": input_padded, "target_ids": target_padded, "is_dae": is_dae_mask}

def fast_collate(batch, max_len=384, eng_pad_id=0):
    bsz = len(batch)
    input_padded = torch.full((bsz, max_len), eng_pad_id, dtype=torch.int32)
    target_padded = torch.full((bsz, max_len), eng_pad_id, dtype=torch.int32)
    is_dae_mask = torch.zeros(bsz, dtype=torch.bool)
    for i, x in enumerate(batch):
        in_seq = x["input_ids"]
        tgt_seq = x["target_ids"]
        in_len = min(max_len, len(in_seq))
        tgt_len = min(max_len, len(tgt_seq))
        if in_len > 0:
            input_padded[i, :in_len] = torch.as_tensor(in_seq[:in_len], dtype=torch.int32)
        if tgt_len > 0:
            target_padded[i, :tgt_len] = torch.as_tensor(tgt_seq[:tgt_len], dtype=torch.int32)
        if x.get("is_dae", False):
            is_dae_mask[i] = True
    return {"input_ids": input_padded, "target_ids": target_padded, "is_dae": is_dae_mask}

# Giả lập batch 256 mẫu
dummy_batch = [{"input_ids": list(range(50)), "target_ids": list(range(50)), "is_dae": (i%2==0)} for i in range(256)]

t0 = time.time()
for _ in range(100):
    _ = old_collate(dummy_batch)
t_old = time.time() - t0

t0 = time.time()
for _ in range(100):
    _ = fast_collate(dummy_batch)
t_fast = time.time() - t0

print(f"Old collate 100 batches: {t_old:.4f}s")
print(f"Fast collate 100 batches: {t_fast:.4f}s (Speedup: {t_old/t_fast:.2f}x)")
