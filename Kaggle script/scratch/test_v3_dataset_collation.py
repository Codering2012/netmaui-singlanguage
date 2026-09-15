#!/usr/bin/env python3
"""
Test hypothesis script: Robust multimodal collation for 1D compact tokens and 4D images.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

def collate_visual_stream(batch, actual_len, key, default_img_shape):
    has_stream = any(b.get(key) is not None for b in batch)
    if not has_stream:
        return None

    ref_item = next(b[key] for b in batch if b.get(key) is not None)
    if not isinstance(ref_item, torch.Tensor):
        ref_item = torch.tensor(ref_item)
    is_4d_raw = (ref_item.dim() == 4 and ref_item.shape[-1] == 3) or (ref_item.dim() == 4 and ref_item.shape[1] == 3)
    trailing_shape = default_img_shape if is_4d_raw else (ref_item.shape[-1],)

    out_list = []
    for item in batch:
        v = item.get(key)
        if v is None:
            out_list.append(torch.zeros((actual_len, *trailing_shape), dtype=torch.float32))
            continue
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32)
        if v.dim() == 4 and v.shape[-1] == 3:
            v = v.permute(0, 3, 1, 2)
        cur_len = v.shape[0]
        if cur_len < actual_len:
            pad_shape = [0] * (2 * (v.dim() - 1)) + [0, actual_len - cur_len]
            v = F.pad(v, pad_shape)
        elif cur_len > actual_len:
            v = v[:actual_len]
        out_list.append(v)
    return torch.stack(out_list, dim=0)

def test_collation():
    actual_len = 32
    # Case 1: Compact 1D visual tokens [T, 128], heterogeneous (one sample has None)
    batch_1 = [
        {"roi_visual": torch.randn(20, 128)},
        {"roi_visual": None},
        {"roi_visual": torch.randn(35, 128)},
    ]
    res_1 = collate_visual_stream(batch_1, actual_len, "roi_visual", (3, 256, 256))
    print("Case 1 (Compact tokens) shape:", res_1.shape)
    assert res_1.shape == (3, actual_len, 128)
    assert (res_1[1] == 0).all() # None sample is zero-padded

    # Case 2: Raw 4D video frames [T, 256, 256, 3]
    batch_2 = [
        {"roi_visual": torch.randn(15, 256, 256, 3)},
        {"roi_visual": None},
    ]
    res_2 = collate_visual_stream(batch_2, actual_len, "roi_visual", (3, 256, 256))
    print("Case 2 (Raw frames) shape:", res_2.shape)
    assert res_2.shape == (2, actual_len, 3, 256, 256)
    assert (res_2[1] == 0).all()

    # Case 3: All None
    batch_3 = [{"roi_visual": None}, {"roi_visual": None}]
    res_3 = collate_visual_stream(batch_3, actual_len, "roi_visual", (3, 256, 256))
    assert res_3 is None
    print("Case 3 (All None) result:", res_3)

    print("[SUCCESS] Multimodal visual collation logic empirically verified!")

if __name__ == "__main__":
    test_collation()
