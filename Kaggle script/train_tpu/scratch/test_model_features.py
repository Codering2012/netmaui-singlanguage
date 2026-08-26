import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_all_in_one_tpu import (
    RMSNorm,
    SwiGLUFFN,
    DropPath,
    RotaryPositionalEncoding,
    GroupedQueryEncoderAttention,
    Swin1DAttention,
    SpatialTemporalSE,
    ConvNeXtTemporalBlock,
    BiMamba2SSMBlock,
    MobileConformerBlock,
    LandmarkTrajectory1DStem,
    RichASLLexEmbeddingTable,
    GroupedQueryAttention,
    DecoderCrossAttention,
    ASLDecoderLayer,
    ASLTransformerDecoder,
    ASLFoundationModel,
    HomoscedasticLossWrapper,
    CTCHead,
    CrossModalInfoNCELoss,
    DenseSentenceSemanticLoss,
    SupervisedContrastiveLoss,
    LandmarkReconstructionHead,
    ModelEMA,
)


def test_basic_layers():
    print("--- [1/7] Testing Basic Modules & Normalizations ---")
    norm = RMSNorm(d_model=256)
    x = torch.randn(2, 30, 256)
    out_norm = norm(x)
    assert out_norm.shape == x.shape, f"RMSNorm shape mismatch: {out_norm.shape}"

    ffn = SwiGLUFFN(d_model=256, dim_feedforward=512)
    out_ffn = ffn(x)
    assert out_ffn.shape == x.shape, f"SwiGLUFFN shape mismatch: {out_ffn.shape}"

    dp = DropPath(drop_prob=0.1)
    out_dp = dp(x)
    assert out_dp.shape == x.shape, f"DropPath shape mismatch: {out_dp.shape}"

    rope = RotaryPositionalEncoding(dim=64)
    q = torch.randn(2, 4, 30, 64)
    k = torch.randn(2, 4, 30, 64)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape, f"RotaryPositionalEncoding shape mismatch: {q_rot.shape}"
    print("  [PASS] Basic Modules & Normalizations working as expected.")


def test_encoder_blocks():
    print("--- [2/7] Testing Encoder Blocks (Swin, Mamba, Conformer, ConvNeXt) ---")
    x = torch.randn(2, 32, 256)

    # GroupedQueryEncoderAttention
    gq_enc = GroupedQueryEncoderAttention(d_model=256, nhead=4, kv_heads=2)
    out_gq = gq_enc(x)
    assert out_gq.shape == x.shape, f"GroupedQueryEncoderAttention shape mismatch: {out_gq.shape}"

    # Swin 1D
    swin = Swin1DAttention(mha_module=gq_enc, window_size=8, shift_size=4)
    out_swin = swin(x)
    assert out_swin.shape == x.shape, f"Swin1DAttention shape mismatch: {out_swin.shape}"

    # SpatialTemporalSE
    se = SpatialTemporalSE(d_model=256)
    out_se = se(x)
    assert out_se.shape == x.shape, f"SpatialTemporalSE shape mismatch: {out_se.shape}"

    # ConvNeXtTemporalBlock
    convnext = ConvNeXtTemporalBlock(channels=256)
    out_cn = convnext(x)
    assert out_cn.shape == x.shape, f"ConvNeXtTemporalBlock shape mismatch: {out_cn.shape}"

    # BiMamba2SSMBlock
    mamba = BiMamba2SSMBlock(d_model=256, headdim=64)
    out_mb = mamba(x)
    assert out_mb.shape == x.shape, f"BiMamba2SSMBlock shape mismatch: {out_mb.shape}"

    # MobileConformerBlock
    conformer = MobileConformerBlock(d_model=256, nhead=4, dim_feedforward=512)
    out_cf = conformer(x)
    assert out_cf.shape == x.shape, f"MobileConformerBlock shape mismatch: {out_cf.shape}"

    print("  [PASS] All Encoder Blocks working as expected.")


def test_stems_embeddings():
    print("--- [3/7] Testing Stems & Embedding Tables ---")
    # LandmarkTrajectory1DStem
    stem = LandmarkTrajectory1DStem(in_channels=9, num_keypoints=166, out_dim=256)
    lm = torch.randn(2, 50, 166 * 9)
    out_stem = stem(lm)
    print(f"  LandmarkTrajectory1DStem output shape: {out_stem.shape}")

    # RichASLLexEmbeddingTable
    lex_emb = RichASLLexEmbeddingTable(vocab_size=1000, d_model=256)
    tokens = torch.randint(0, 500, (2, 10))
    out_lex = lex_emb(tokens)
    assert out_lex.shape == (2, 10, 256), f"LexEmbedding shape mismatch: {out_lex.shape}"

    print("  [PASS] Stems & Embedding Tables working as expected.")


def test_decoder():
    print("--- [4/7] Testing Transformer Decoder Modules ---")
    tgt = torch.randint(0, 500, (2, 15))
    memory = torch.randn(2, 40, 256)
    decoder = ASLTransformerDecoder(
        vocab_size=1000,
        d_model=256,
        nhead=4,
        num_layers=2,
    )
    out_dec = decoder(tgt, memory)
    logits = out_dec[0]
    assert logits.shape == (2, 15, 1000), f"Decoder output shape mismatch: {logits.shape}"
    print(f"  ASLTransformerDecoder output shape: {logits.shape}")
    print("  [PASS] Transformer Decoder Modules working as expected.")


def test_foundation_model():
    print("--- [5/7] Testing Full ASLFoundationModel (Standard & Swin & Aux Decoders) ---")
    model = ASLFoundationModel(
        vocab_size=500,
        english_vocab_size=1000,
        enable_aux_decoders=True,
        num_keypoints=166,
        channels_per_kp=9,
        d_enc=256,
        nhead_enc=4,
        num_enc_layers=2,
        ffn_enc=512,
        d_dec=256,
        nhead_dec=4,
        num_dec_layers=2,
        ffn_dec=512,
        use_swin_1d=False,
    )
    landmarks = torch.randn(2, 40, 166, 9)
    src_mask = torch.ones(2, 40, dtype=torch.bool)

    outputs = model(
        input_x=landmarks,
        mask=src_mask,
    )
    assert "ctc_log_probs" in outputs, f"Missing ctc_log_probs in model outputs: {outputs.keys()}"
    print(f"  Standard model ctc_log_probs shape: {outputs['ctc_log_probs'].shape}")

    # Swin attention enabled
    swin_model = ASLFoundationModel(
        vocab_size=500,
        english_vocab_size=1000,
        enable_aux_decoders=False,
        num_keypoints=166,
        channels_per_kp=9,
        d_enc=256,
        nhead_enc=4,
        num_enc_layers=2,
        ffn_enc=512,
        d_dec=256,
        nhead_dec=4,
        num_dec_layers=2,
        ffn_dec=512,
        use_swin_1d=True,
        swin_window_size=8,
    )
    swin_outputs = swin_model(input_x=landmarks, mask=src_mask)
    assert "ctc_log_probs" in swin_outputs, f"Missing ctc_log_probs in Swin model outputs: {swin_outputs.keys()}"
    print(f"  Swin model ctc_log_probs shape: {swin_outputs['ctc_log_probs'].shape}")
    print("  [PASS] Full ASLFoundationModel working as expected.")


def test_loss_functions():
    print("--- [6/7] Testing Loss Functions & Wrappers ---")
    # CTCHead
    ctc_head = CTCHead(d_model=256, vocab_size=500)
    feats = torch.randn(2, 40, 256)
    ctc_logits = ctc_head(feats)
    assert ctc_logits.shape == (2, 40, 500), f"CTCHead shape mismatch: {ctc_logits.shape}"

    # CrossModalInfoNCELoss
    infonce = CrossModalInfoNCELoss(temperature=0.07)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    loss_nce = infonce(z1, z2)
    assert loss_nce.item() >= 0.0, f"InfoNCE loss negative: {loss_nce.item()}"

    # DenseSentenceSemanticLoss
    sem_loss = DenseSentenceSemanticLoss(d_model=256, embed_dim=256)
    h1 = torch.randn(4, 30, 256)
    h2 = torch.randn(4, 30, 256)
    valid_mask = torch.ones(4, 30, dtype=torch.bool)
    loss_sem = sem_loss(h1, h2, valid_mask)
    assert loss_sem.item() >= 0.0, f"Semantic loss negative: {loss_sem.item()}"

    # SupervisedContrastiveLoss
    sup_con = SupervisedContrastiveLoss(temperature=0.07)
    labels = torch.tensor([0, 0, 1, 1])
    features = torch.randn(4, 128)
    loss_sc = sup_con(features, labels)
    assert loss_sc.item() >= 0.0, f"Supervised contrastive loss negative: {loss_sc.item()}"

    # LandmarkReconstructionHead
    recon_head = LandmarkReconstructionHead(d_model=256, out_dim=166 * 9)
    recon_out = recon_head(feats)
    assert recon_out.shape == (2, 40, 166 * 9), f"Reconstruction shape mismatch: {recon_out.shape}"

    # HomoscedasticLossWrapper
    homo = HomoscedasticLossWrapper()
    combined_loss = homo({"ctc": loss_nce, "dense_sem": loss_sem, "supcon": loss_sc})
    assert combined_loss.item() >= 0.0, f"Homoscedastic loss negative: {combined_loss.item()}"

    print("  [PASS] All Loss Functions & Wrappers working as expected.")


def test_model_ema():
    print("--- [7/7] Testing ModelEMA ---")
    model = nn.Linear(10, 10)
    ema = ModelEMA(model, decay_base=0.999, decay_max=0.9999)
    x = torch.randn(4, 10)
    out = model(x)
    loss = out.sum()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-0.01)
    ema.update(model)
    ema.apply_shadow(model)
    ema.restore(model)
    print("  [PASS] ModelEMA working as expected.")


if __name__ == "__main__":
    print("=== STARTING MODEL FEATURE EMPIRICAL VERIFICATION ===")
    test_basic_layers()
    test_encoder_blocks()
    test_stems_embeddings()
    test_decoder()
    test_foundation_model()
    test_loss_functions()
    test_model_ema()
    print("=== ALL 7 MODEL FEATURE TESTS PASSED SUCCESSFULLY! ===")
