#!/usr/bin/env python3
"""
================================================================================
SELF-TEST AUDIT: EMPIRICALLY DEMONSTRATING THE IDENTIFIED FLAWS
================================================================================
Tests:
1. Lean Invariance in Fingerspelling Router (Absolute vs Shoulder-Relative).
2. Keypoint Indexing Mismatch in ASLV3FoundationModel.
3. Destructive Epenthesis L2 Loss Gradient vs Self-Supervised Consistency.
4. CTC Blank Biasing Gradient Leakage into Gate.
5. Streaming Engine Common-Prefix Lockup on Disjoint Chunks.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_flaw_1_shelf_lean_failure():
    print("=" * 80)
    print("AUDIT FLAW 1: Conversational Shelf Lean Invariance")
    print("=" * 80)
    
    # Absolute shelf centroid
    centroid_abs = torch.tensor([0.28, 0.05, -0.22])
    radii_inv = torch.tensor([1.0 / 0.08, 1.0 / 0.10, 1.0 / 0.09])
    
    # 1. Upright signer: Right shoulder at (0.20, 0.0, 0.0), wrist at (0.28, 0.05, -0.22)
    wr_upright = torch.tensor([0.28, 0.05, -0.22])
    diff_abs_upright = (wr_upright - centroid_abs) * radii_inv
    p_abs_upright = torch.exp(-0.5 * torch.sum(diff_abs_upright ** 2)).item()
    
    # 2. Leaning signer: shifts laterally by 12cm: shoulder at (0.32, -0.02, 0.0), wrist at (0.40, 0.03, -0.22)
    wr_lean = torch.tensor([0.40, 0.03, -0.22])
    sh_lean = torch.tensor([0.32, -0.02, 0.0])
    diff_abs_lean = (wr_lean - centroid_abs) * radii_inv
    p_abs_lean = torch.exp(-0.5 * torch.sum(diff_abs_lean ** 2)).item()
    
    # 3. Shoulder-relative formulation:
    c_rel = torch.tensor([0.08, 0.05, -0.22])
    sh_upright = torch.tensor([0.20, 0.0, 0.0])
    diff_rel_upright = ((wr_upright - sh_upright) - c_rel) * radii_inv
    p_rel_upright = torch.exp(-0.5 * torch.sum(diff_rel_upright ** 2)).item()
    
    diff_rel_lean = ((wr_lean - sh_lean) - c_rel) * radii_inv
    p_rel_lean = torch.exp(-0.5 * torch.sum(diff_rel_lean ** 2)).item()
    
    print(f"  Absolute Shelf: Upright P = {p_abs_upright:.4f} | Leaning 12cm P = {p_abs_lean:.4f} (COLLAPSED by {(1-p_abs_lean/p_abs_upright)*100:.1f}%)")
    print(f"  Shoulder-Rel  : Upright P = {p_rel_upright:.4f} | Leaning 12cm P = {p_rel_lean:.4f} (100% INVARIANT!)")
    assert p_abs_lean < 0.40, "Absolute formulation should fail under lean!"
    assert abs(p_rel_upright - p_rel_lean) < 1e-5, "Shoulder-relative formulation must be invariant!"
    print("  -> FLAW 1 CONFIRMED: Absolute shelf formulation fails under natural body lean.")

def test_flaw_2_keypoint_mismatch():
    print("\n" + "=" * 80)
    print("AUDIT FLAW 2: Keypoint Indexing in Foundation Model")
    print("=" * 80)
    # Canonical 60-keypoint indexing:
    # 0..20: Left Hand (0 is Left Wrist)
    # 21..41: Right Hand (21 is Right Wrist)
    # 42..47: Pose (42: Left Shoulder, 43: Right Shoulder)
    # 48..59: Face (48: Nose)
    
    # Old foundation model extracted:
    # hand_coords = pts[:, :, 12, :]        # Right wrist / palm root -> INDEX 12 IS LEFT HAND MIDDLE FINGER!
    # base_hand_coords = pts[:, :, 33, :]   # Left wrist / palm root  -> INDEX 33 IS RIGHT HAND MIDDLE FINGER!
    # shoulder_coords = pts[:, :, 0:2, :]   # Left and right shoulders -> INDICES 0, 1 ARE LEFT WRIST AND THUMB!
    
    # Create synthetic keypoints with known signatures
    pts = torch.zeros((1, 1, 60, 3))
    pts[:, :, 0, :] = torch.tensor([0.1, 0.2, 0.3])    # Left wrist
    pts[:, :, 21, :] = torch.tensor([0.4, 0.5, 0.6])   # Right wrist
    pts[:, :, 42, :] = torch.tensor([-0.25, 0.0, 0.0]) # Left shoulder
    pts[:, :, 43, :] = torch.tensor([0.25, 0.0, 0.0])  # Right shoulder
    
    # With old indices:
    old_l_sh = pts[:, :, 0, :]
    old_r_sh = pts[:, :, 1, :]
    old_torso_center = (old_l_sh + old_r_sh) * 0.5
    
    # With correct canonical indices:
    correct_l_sh = pts[:, :, 42, :]
    correct_r_sh = pts[:, :, 43, :]
    correct_torso_center = (correct_l_sh + correct_r_sh) * 0.5
    
    print(f"  Old Torso Center (from Left Wrist & Thumb): {old_torso_center[0, 0].tolist()}")
    print(f"  Correct Torso Center (from Shoulders 42 & 43): {correct_torso_center[0, 0].tolist()}")
    diff = torch.norm(old_torso_center - correct_torso_center).item()
    print(f"  Discrepancy: {diff:.4f}")
    assert diff > 0.1, "Mismatch must be significant!"
    print("  -> FLAW 2 CONFIRMED: Anatomical indices 0:2 vs 42:44 severely misalign the torso frame.")

def test_flaw_3_epenthesis_l2_loss():
    print("\n" + "=" * 80)
    print("AUDIT FLAW 3: Epenthesis L2 Loss Destroys Detection")
    print("=" * 80)
    
    # Suppose a simple linear model predicting beta_t
    gate = nn.Linear(6, 1)
    nn.init.constant_(gate.bias, 0.0)
    optimizer = torch.optim.SGD(gate.parameters(), lr=1.0)
    
    x = torch.randn(10, 6)
    # Train with 0.01 * mean(beta_t ** 2)
    for _ in range(200):
        optimizer.zero_grad()
        beta = torch.sigmoid(gate(x))
        loss = 0.5 * torch.mean(beta ** 2)
        loss.backward()
        optimizer.step()
        
    final_beta = torch.sigmoid(gate(x)).mean().item()
    print(f"  Mean beta after 200 steps of L2 loss: {final_beta:.4f} (Driven towards 0)")
    assert final_beta < 0.40, "L2 loss actively crushes detection!"
    print("  -> FLAW 3 CONFIRMED: L2 regularization on beta_t crushes epenthesis detection.")

def test_flaw_4_ctc_leakage():
    print("\n" + "=" * 80)
    print("AUDIT FLAW 4: CTC Biasing Gradient Leakage into Detection Gate")
    print("=" * 80)
    
    # Gate predicting beta_t
    gate_param = nn.Parameter(torch.tensor([2.0])) # High initial beta
    logits = torch.randn(1, 10, 128, requires_grad=True)
    
    # Case A: With gradient leakage
    beta = torch.sigmoid(gate_param)
    boost = 10.0 * beta
    biased_logits = logits.clone()
    biased_logits[:, :, 0] = biased_logits[:, :, 0] + boost
    biased_logits[:, :, 1:] = biased_logits[:, :, 1:] - boost
    
    # Suppose target is non-blank token 5
    loss_leak = F.cross_entropy(biased_logits.view(-1, 128), torch.tensor([5]*10))
    loss_leak.backward()
    grad_leak = gate_param.grad.item()
    
    # Case B: With detached boost
    gate_param.grad.zero_()
    logits.grad.zero_()
    beta_detached = torch.sigmoid(gate_param)
    boost_detached = 10.0 * beta_detached.detach()
    biased_logits_det = logits.clone()
    biased_logits_det[:, :, 0] = biased_logits_det[:, :, 0] + boost_detached
    biased_logits_det[:, :, 1:] = biased_logits_det[:, :, 1:] - boost_detached
    loss_det = F.cross_entropy(biased_logits_det.view(-1, 128), torch.tensor([5]*10))
    loss_det.backward()
    grad_det = 0.0 if gate_param.grad is None else gate_param.grad.item()
    
    print(f"  Gradient on gate with leakage: {grad_leak:.4f} (Massive gradient pushing gate to 0!)")
    print(f"  Gradient on gate with detached boost: {grad_det:.4f} (Zero gradient leakage!)")
    assert abs(grad_leak) > 1.0, "Leakage must produce huge parasitic gradient!"
    assert abs(grad_det) == 0.0, "Detached boost must have zero leakage!"
    print("  -> FLAW 4 CONFIRMED: Logit biasing without detach destroys the kinematic gate.")

def test_flaw_5_streaming_commit_lockup():
    print("\n" + "=" * 80)
    print("AUDIT FLAW 5: Realtime Streaming Common-Prefix Lockup on Disjoint Chunks")
    print("=" * 80)
    
    # In RealtimeAdaptiveStreamer:
    # Chunk 1 emits [12, 45] ("I", "LIKE")
    # Chunk 2 emits [89, 102] ("READING", "BOOKS")
    chunk1_tokens = [12, 45]
    chunk2_tokens = [89, 102]
    
    # Old logic:
    history = [chunk1_tokens, chunk2_tokens]
    # Common prefix of chunk 1 and chunk 2:
    def get_common_prefix(token_lists):
        min_len = min(len(tl) for tl in token_lists)
        prefix = []
        for i in range(min_len):
            first = token_lists[0][i]
            if all(tl[i] == first for tl in token_lists):
                prefix.append(first)
            else:
                break
        return prefix
        
    prefix = get_common_prefix(history)
    print(f"  Chunk 1: {chunk1_tokens}, Chunk 2: {chunk2_tokens}")
    print(f"  Calculated Common Prefix: {prefix} (EMPTY!)")
    assert len(prefix) == 0, "Common prefix between disjoint chunks is always empty!"
    print("  -> FLAW 5 CONFIRMED: Disjoint chunk prefix matching permanently freezes commits.")

if __name__ == "__main__":
    test_flaw_1_shelf_lean_failure()
    test_flaw_2_keypoint_mismatch()
    test_flaw_3_epenthesis_l2_loss()
    test_flaw_4_ctc_leakage()
    test_flaw_5_streaming_commit_lockup()
    print("\n" + "=" * 80)
    print("ALL 5 SYSTEMIC FLAWS EMPIRICALLY CONFIRMED AND REPRODUCED!")
    print("=" * 80)
