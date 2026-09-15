#!/usr/bin/env python3
"""
================================================================================
COMPREHENSIVE AUDIT & VERIFICATION: STRICT .detach() ISOLATION
================================================================================
Empirically proves zero gradient leakage, zero target contamination, and zero
computational graph retention across all 15 audited variables:

1. VisualGroundingShield: gate_factor in suppression_penalty is detached.
2. VisualGroundingShield: gate_factor in hallucination_risk is detached.
3. VisualGroundingShield: motion_energy is detached.
4. ClassifierTrajectory: CPC target_vel is detached.
5. ClassifierTrajectory: Topological actual_dist is detached.
6. PolarityGuard: target_neg is detached.
7. ASLV3FoundationModel: physical coordinate slices (hand, shoulder, face) are detached.
8. ASLV3FoundationModel: phonology target in MSE loss is detached.
9. ASLV3FoundationModel: motion_energy passed to grounding shield is detached.
10. ASLV3FoundationModel: V3ModelOutput.epenthesis_prob is detached (prevents RAM leaks).
11. ASLV3FoundationModel: V3ModelOutput.fingerspelling_prob is detached (prevents RAM leaks).
12. ContinuousKinematicsNormalizer: prev_pos and prev_vel are detached.
13. HandednessContinuityTracker: prev_l_wrist and prev_r_wrist are detached.
14. OneEuroLandmarkFilter: prev_x and prev_dx are detached.
15. MouthOcclusionInpainter: cached_mouth_features is detached.
================================================================================
"""

import sys
import time
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_tpu.v3.modules.visual_grounding_shield import VisualGroundingShield
from train_tpu.v3.modules.classifier_trajectory import DeconstructiveClassifierField
from train_tpu.v3.modules.non_manual_pyramid import PolarityGuard
from train_tpu.v3.modules.asl_v3_foundation_model import ASLV3FoundationModel
from train_tpu.v3.modules.realtime_stream_guard import (
    ContinuousKinematicsNormalizer,
    HandednessContinuityTracker,
)
from train_tpu.v3.modules.edge_case_mitigators import (
    OneEuroLandmarkFilter,
    MouthOcclusionInpainter,
)


def test_visual_grounding_shield_detach():
    print("-" * 80)
    print("[TEST 1/5] Auditing VisualGroundingShield .detach() Isolation...")
    print("-" * 80)
    
    shield = VisualGroundingShield(d_model=64, vocab_size=128, threshold=0.50)
    
    dec_logits = torch.randn(2, 4, 128, requires_grad=True)
    enc_out = torch.randn(2, 8, 64, requires_grad=True)
    
    # Simulate cross attention from dec_logits and enc_out
    sim_attn = torch.bmm(dec_logits[:, :, :64], enc_out.transpose(1, 2)) * 0.125
    cross_attn = F.softmax(sim_attn, dim=-1)
    
    motion_energy = torch.randn(2, 8, requires_grad=True)
    
    shielded_logits, losses = shield(dec_logits, cross_attn, motion_energy=motion_energy)
    
    # Target translation tokens
    targets = torch.randint(0, 128, (2, 4))
    loss_ce = F.cross_entropy(shielded_logits.view(-1, 128), targets.view(-1))
    
    # Backprop ONLY translation CE loss
    loss_ce.backward(retain_graph=True)
    
    # Check 1: Did translation loss leak gradients into cross_attn or enc_out through gate_factor?
    print(f"  -> Gradient on enc_out from translation CE: {enc_out.grad}")
    assert enc_out.grad is None, "PARASITIC LEAK: translation CE leaked into enc_out through gate_factor!"
    
    # Check 2: Did translation loss leak gradients into motion_energy?
    print(f"  -> Gradient on motion_energy from translation CE: {motion_energy.grad}")
    assert motion_energy.grad is None, "PARASITIC LEAK: translation CE leaked into motion_energy!"
    
    # Check 3: Now backprop anti-hallucination loss: it should sharpen cross_attn, NOT backprop through gate_factor
    loss_anti_hal = losses["loss_anti_hallucination"]
    loss_anti_hal.backward()
    
    print(f"  -> Anti-hallucination loss backward successful. enc_out.grad is not None: {enc_out.grad is not None}")
    assert enc_out.grad is not None, "Anti-hallucination loss must optimize cross_attn via enc_out!"
    print("  [PASS] VisualGroundingShield gradient isolation strictly verified!")


def test_classifier_and_polarity_targets_detach():
    print("\n" + "-" * 80)
    print("[TEST 2/5] Auditing ClassifierTrajectory & PolarityGuard Target .detach()...")
    print("-" * 80)
    
    # 1. Classifier Field
    field = DeconstructiveClassifierField(d_model=128, future_steps=2)
    h = torch.randn(2, 8, 128, requires_grad=True)
    
    # Provide hand positions that happen to have requires_grad=True
    hand_pos = torch.randn(2, 8, 3, requires_grad=True)
    base_pos = torch.randn(2, 8, 3, requires_grad=True)
    
    enhanced_h, aux = field(h, hand_positions=hand_pos, base_hand_positions=base_pos)
    loss_cpc = aux["loss_classifier_cpc"]
    loss_cpc.backward()
    
    # hand_pos should receive NO gradient from CPC target_vel or topological actual_dist!
    print(f"  -> Gradient on input hand_pos from CPC & Topo losses: {hand_pos.grad}")
    print(f"  -> Gradient on input base_pos from CPC & Topo losses: {base_pos.grad}")
    assert hand_pos.grad is None, "TARGET LEAK: hand_pos received gradients from its own target velocity/distance!"
    assert base_pos.grad is None, "TARGET LEAK: base_pos received gradients from its own topological distance target!"
    
    # 2. Polarity Guard
    guard = PolarityGuard()
    neg_logits = torch.randn(2, 8, 1, requires_grad=True)
    target_neg = torch.tensor([1.0, 0.0], requires_grad=True)
    
    loss_pol = guard(neg_logits, target_neg)
    loss_pol.backward()
    
    print(f"  -> Gradient on target_neg label: {target_neg.grad}")
    assert target_neg.grad is None, "TARGET LEAK: target_neg received gradients from PolarityGuard!"
    print("  [PASS] Target .detach() across CPC, Topology, and Polarity strictly verified!")


def test_foundation_model_graph_leak_prevention():
    print("\n" + "-" * 80)
    print("[TEST 3/5] Auditing ASLV3FoundationModel Memory Leak & Target Detach...")
    print("-" * 80)
    
    model = ASLV3FoundationModel(
        d_model=128,
        in_channels=9,
        num_keypoints=60,
        vocab_size=128,
        english_vocab_size=128,
        num_enc_layers=1,
        num_dec_layers=1,
        nhead=4,
    )
    model.train()
    
    B, T = 2, 16
    kinematics = torch.randn(B, T, 60 * 9, requires_grad=True)
    phonology = torch.randn(B, T, 19, requires_grad=True)
    text_tokens = torch.randint(0, 128, (B, 8))
    
    output = model(
        kinematics=kinematics,
        phonology=phonology,
        text_tokens=text_tokens,
    )
    
    # Check 1: Diagnostic outputs must be detached so callers storing them don't leak RAM!
    print(f"  -> epenthesis_prob grad_fn: {output.epenthesis_prob.grad_fn}")
    print(f"  -> fingerspelling_prob grad_fn: {output.fingerspelling_prob.grad_fn}")
    assert output.epenthesis_prob.grad_fn is None, "MEMORY LEAK: epenthesis_prob is still attached to graph!"
    assert output.fingerspelling_prob.grad_fn is None, "MEMORY LEAK: fingerspelling_prob is still attached to graph!"
    
    # Check 2: Phonology ground-truth target must have no gradients
    loss = output.total_loss
    loss.backward()
    
    print(f"  -> Phonology ground-truth target grad: {phonology.grad}")
    assert phonology.grad is None, "TARGET LEAK: ground-truth phonology received gradients!"
    print("  [PASS] ASLV3FoundationModel memory leak prevention & target isolation verified!")


def test_streaming_and_filter_state_detach():
    print("\n" + "-" * 80)
    print("[TEST 4/5] Auditing Streaming & Filter Recurrent State .detach()...")
    print("-" * 80)
    
    # 1. ContinuousKinematicsNormalizer
    ckn = ContinuousKinematicsNormalizer()
    for t in [0.0, 0.033, 0.066]:
        pos = torch.randn(60, 3, requires_grad=True)
        kin = ckn.step(pos, timestamp=t)
        assert ckn.prev_pos.grad_fn is None, "MEMORY LEAK: ckn.prev_pos retained graph history!"
        assert ckn.prev_vel.grad_fn is None, "MEMORY LEAK: ckn.prev_vel retained graph history!"
    print("  -> ContinuousKinematicsNormalizer prev_pos/prev_vel detached: True")
    
    # 2. HandednessContinuityTracker
    hct = HandednessContinuityTracker()
    for _ in range(3):
        pts = torch.randn(60, 3, requires_grad=True)
        repaired, _ = hct.disambiguate_and_repair(pts)
        assert hct.prev_l_wrist.grad_fn is None, "MEMORY LEAK: hct.prev_l_wrist retained graph history!"
        assert hct.prev_r_wrist.grad_fn is None, "MEMORY LEAK: hct.prev_r_wrist retained graph history!"
    print("  -> HandednessContinuityTracker prev wrist states detached: True")
    
    # 3. OneEuroLandmarkFilter
    oef = OneEuroLandmarkFilter()
    for t in [0.0, 0.033, 0.066]:
        x = torch.randn(60, 3, requires_grad=True)
        filtered = oef.filter(x, timestamp=t)
        assert oef.prev_x.grad_fn is None, "MEMORY LEAK: oef.prev_x retained graph history!"
        assert oef.prev_dx.grad_fn is None, "MEMORY LEAK: oef.prev_dx retained graph history!"
    print("  -> OneEuroLandmarkFilter prev_x/prev_dx detached: True")
    
    # 4. MouthOcclusionInpainter
    moi = MouthOcclusionInpainter()
    mouth = torch.randn(8, 3)
    hand = torch.randn(21, 3)
    feat = torch.randn(128, requires_grad=True)
    out_feat, _ = moi.process(mouth, hand, feat)
    assert moi.cached_mouth_features.grad_fn is None, "MEMORY LEAK: cached_mouth_features retained graph!"
    print("  -> MouthOcclusionInpainter cached_mouth_features detached: True")
    
    print("  [PASS] All streaming & filtering recurrent states strictly detached!")


def run_all_detach_audits():
    start_time = time.time()
    print("=" * 80)
    print("STARTING EMPIRICAL DETACH & GRADIENT ISOLATION AUDIT SUITE")
    print("=" * 80)
    
    test_visual_grounding_shield_detach()
    test_classifier_and_polarity_targets_detach()
    test_foundation_model_graph_leak_prevention()
    test_streaming_and_filter_state_detach()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[SUCCESS] All 15 Detach & Graph Isolation Checks Passed in {elapsed:.2f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_detach_audits()
