#!/usr/bin/env python3
"""
================================================================================
ASL V3 FLAGSHIP FOUNDATION MODEL ARCHITECTURE
================================================================================
Unified Multi-Tier Spatial-Topographic & Non-Manual Foundation Architecture.
Directly addresses core linguistic and physical failure modes of ASL translation:

1. Dynamic 3D Spatial Locus Memory Bank (Discourse entity referencing).
2. Non-Manual Feature Pyramid & Polarity Guard (Eyebrows, Mouth Morphemes, Cranial IMU).
3. Deconstructive Classifier Predicates Field (Continuous 3D trajectory & topology).
4. Monotonic Chunk Permutation Transducer (OSV -> SVO syntactic reordering).
5. Cross-Attention Visual Grounding Shield (Exposure bias anti-hallucination gate).
6. Native Dual Visual Stems: Upper-body (256x256) + Dual Hand Crops (128x128).
7. Strict Cloud TPU v5e 128x128 MXU tile alignment & single-graph execution.
================================================================================
"""

from typing import Tuple, Optional, Dict, Any, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_locus_memory import Dynamic3DLocusMemoryBank
from .non_manual_pyramid import NonManualFeaturePyramid, PolarityGuard
from .classifier_trajectory import DeconstructiveClassifierField
from .chunk_permutation_transducer import ChunkPermutationTransducer
from .visual_grounding_shield import VisualGroundingShield
from .movement_epenthesis_suppressor import MovementEpenthesisSuppressor
from .fingerspelling_hybrid_transducer import (
    ContinuousFingerspellingRouter,
    CharacterLevelCTCDecoder,
    FingerspellingWordHybridWeaver,
)
from .gpt2_translation_decoder import GPT2CrossModalTranslationDecoder


class VisualROI256Stem(nn.Module):
    """
    Multimodal upper-body visual stem supporting:
    1. Raw video crops: [B, T, 3, 256, 256] -> Depthwise separable CNN downsampling -> [B, T, d_model]
    2. Pre-extracted compact embeddings: [B, T, D_vis] -> Linear projection -> [B, T, d_model]
    """

    def __init__(self, d_model: int = 128, vis_feat_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32, bias=False)
        self.conv2_pw = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.conv3_pw = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.conv4_pw = nn.Conv2d(256, d_model, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(d_model)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact embedding projection for lightweight token streaming
        self.linear_stem = nn.Linear(vis_feat_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # Compact 1D visual features [B, T, D_vis]
            if x.shape[-1] <= self.linear_stem.in_features:
                return F.linear(x, self.linear_stem.weight[:, :x.shape[-1]], self.linear_stem.bias)
            return self.linear_stem(x[:, :, :self.linear_stem.in_features])
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2_pw(self.conv2(x))))
        x = F.gelu(self.bn3(self.conv3_pw(self.conv3(x))))
        x = F.gelu(self.bn4(self.conv4_pw(self.conv4(x))))
        x = self.pool(x).view(B, T, -1)
        return x


class VisualHandCrop128Stem(nn.Module):
    """
    Multimodal hand visual stem supporting:
    1. Raw hand crops: [B, T, 3, 128, 128] -> Depthwise separable CNN downsampling -> [B, T, d_model]
    2. Pre-extracted compact embeddings: [B, T, D_vis] -> Linear projection -> [B, T, d_model]
    """

    def __init__(self, d_model: int = 128, vis_feat_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32, bias=False)
        self.conv2_pw = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.conv3_pw = nn.Conv2d(128, d_model, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact embedding projection for lightweight token streaming
        self.linear_stem = nn.Linear(vis_feat_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # Compact 1D hand features [B, T, D_vis]
            if x.shape[-1] <= self.linear_stem.in_features:
                return F.linear(x, self.linear_stem.weight[:, :x.shape[-1]], self.linear_stem.bias)
            return self.linear_stem(x[:, :, :self.linear_stem.in_features])
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2_pw(self.conv2(x))))
        x = F.gelu(self.bn3(self.conv3_pw(self.conv3(x))))
        x = self.pool(x).view(B, T, -1)
        return x


class V3ModelOutput(NamedTuple):
    ctc_logits: torch.Tensor
    english_ctc_logits: torch.Tensor
    english_inter_ctc_logits: torch.Tensor
    decoder_logits: Optional[torch.Tensor]
    encoded_features: torch.Tensor
    multi_task_losses: Dict[str, torch.Tensor]
    total_loss: Optional[torch.Tensor]
    char_ctc_logits: Optional[torch.Tensor] = None
    epenthesis_prob: Optional[torch.Tensor] = None
    fingerspelling_prob: Optional[torch.Tensor] = None
    raw_ctc_logits: Optional[torch.Tensor] = None
    raw_english_ctc_logits: Optional[torch.Tensor] = None


class SignerAdaIN(nn.Module):
    """
    Signer-Adaptive Instance Normalization.
    Decouples individual signer body proportions and personal style from semantic linguistic features.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.style_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2 * d_model),
        )
        nn.init.zeros_(self.style_mlp[-1].weight)
        nn.init.zeros_(self.style_mlp[-1].bias)

    def forward(self, x: torch.Tensor, style_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        if style_vec is None:
            style_vec = x.mean(dim=1).detach()
        style_params = self.style_mlp(style_vec)
        gamma, beta = style_params.chunk(2, dim=-1)
        gamma = 1.0 + gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mean) / std
        return gamma * x_norm + beta


class ASLV3FoundationModel(nn.Module):
    r"""
    ASL V3 Flagship Foundation Architecture.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        vocab_size: int = 256,
        english_vocab_size: int = 512,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        nhead: int = 4,
        max_seq_len: int = 256,
        chunk_size: int = 16,
        use_gpt2_decoder: bool = True,
    ):
        super().__init__()
        # Ensure vocab sizes are exact multiples of 128 for TPU v5e MXU tiling
        self.d_model = (d_model + 127) // 128 * 128
        self.vocab_size = (vocab_size + 127) // 128 * 128
        self.english_vocab_size = (english_vocab_size + 127) // 128 * 128
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels
        self.use_gpt2_decoder = use_gpt2_decoder

        # 1. Stems
        self.kinematics_stem = nn.Sequential(
            nn.Linear(num_keypoints * in_channels, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.visual_stem = VisualROI256Stem(d_model=self.d_model)
        self.hand_stem = VisualHandCrop128Stem(d_model=self.d_model)
        self.phonology_stem = nn.Linear(19, self.d_model)

        # Multimodal Gated Cross-Attention Fusion
        self.modality_gate = nn.Linear(self.d_model * 4, 4)
        self.fusion_norm = nn.LayerNorm(self.d_model)

        # 2. Specialized V3 Architectural Engines
        self.locus_memory = Dynamic3DLocusMemoryBank(d_model=self.d_model, num_slots=8)
        self.nmm_pyramid = NonManualFeaturePyramid(d_model=self.d_model, num_mouth_classes=10)
        self.polarity_guard = PolarityGuard()
        self.classifier_field = DeconstructiveClassifierField(d_model=self.d_model, num_classifier_types=16)
        self.chunk_transducer = ChunkPermutationTransducer(d_model=self.d_model, chunk_size=chunk_size, num_heads=nhead)

        # 3. Contextual Conformer Encoder Layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # 4. Phonology Reconstruction Head
        self.phonology_head = nn.Linear(self.d_model, 19)

        # 5. CTC Heads (Tile-aligned to 128)
        self.ctc_head = nn.Linear(self.d_model, self.vocab_size)
        self.english_ctc_head = nn.Linear(self.d_model, self.english_vocab_size)
        self.english_inter_ctc_head = nn.Linear(self.d_model, self.english_vocab_size)
        self.english_early_ctc_head = nn.Linear(self.d_model, self.english_vocab_size)

        # 6. Autoregressive Translation Decoder with Visual Grounding Shield
        self.text_embedding = nn.Embedding(self.english_vocab_size, self.d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.decoder_head = nn.Linear(self.d_model, self.english_vocab_size)

        # GPT-2 Cross-Modal Translation Decoder
        if self.use_gpt2_decoder:
            self.gpt2_decoder = GPT2CrossModalTranslationDecoder(
                vocab_size=self.english_vocab_size,
                max_position_embeddings=max_seq_len,
                d_model=self.d_model,
                d_encoder=self.d_model,
                num_layers=num_dec_layers,
                num_heads=nhead,
            )
        else:
            self.gpt2_decoder = None

        self.grounding_shield = VisualGroundingShield(d_model=self.d_model, vocab_size=self.english_vocab_size)

        # 7. Frontier Linguistic Engines: Epenthesis Suppressor & Fingerspelling Hybrid Transducer
        self.epenthesis_suppressor = MovementEpenthesisSuppressor(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            d_model=self.d_model,
        )
        self.fs_router = ContinuousFingerspellingRouter(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            d_model=self.d_model,
        )
        self.char_decoder = CharacterLevelCTCDecoder(d_model=self.d_model)
        self.fs_weaver = FingerspellingWordHybridWeaver(d_model=self.d_model)
        self.signer_adain = SignerAdaIN(d_model=self.d_model)

    # Canonical 60-keypoint structural constants
    LEFT_WRIST_IDX: int = 0
    RIGHT_WRIST_IDX: int = 21
    LEFT_SHOULDER_IDX: int = 42
    RIGHT_SHOULDER_IDX: int = 43
    NOSE_TIP_IDX: int = 48
    FACE_START_IDX: int = 48
    FACE_END_IDX: int = 60

    def forward(
        self,
        kinematics: torch.Tensor,                               # [B, T, num_kp * in_ch] or [B, T, num_kp, in_ch]
        roi_visual: Optional[torch.Tensor] = None,              # [B, T, 3, 256, 256] or [B, T, D_vis]
        hand_visual: Optional[torch.Tensor] = None,             # [B, T, 3, 128, 128] or [B, T, D_vis]
        phonology: Optional[torch.Tensor] = None,               # [B, T, 19]
        face_landmarks: Optional[torch.Tensor] = None,          # [B, T, 12, 3]
        cranial_imu: Optional[torch.Tensor] = None,             # [B, T, 3]
        text_tokens: Optional[torch.Tensor] = None,             # [B, L]
        text_is_negative: Optional[torch.Tensor] = None,        # [B]
        hand_mask: Optional[torch.Tensor] = None,               # [B, T, 2]
        streaming_mode: bool = False,
    ) -> V3ModelOutput:
        B, T = kinematics.shape[:2]
        multi_task_losses: Dict[str, torch.Tensor] = {}

        # Detach auxiliary targets/inputs to isolate from dataset tensor autograd paths
        if phonology is not None:
            phonology = phonology.detach()
        if text_is_negative is not None:
            text_is_negative = text_is_negative.detach()
        if text_tokens is not None:
            text_tokens = text_tokens.detach()
        if cranial_imu is not None:
            cranial_imu = cranial_imu.detach()
        if face_landmarks is not None:
            face_landmarks = face_landmarks.detach()

        # 1. Flatten kinematics if passed in 4D [B, T, K, C]
        if kinematics.dim() == 4:
            kinematics = kinematics.view(B, T, -1)
        kin_feat = self.kinematics_stem(kinematics)  # [B, T, d_model]

        # 2. Visual Stems
        vis_feat = self.visual_stem(roi_visual) if roi_visual is not None else torch.zeros_like(kin_feat)
        hand_feat = self.hand_stem(hand_visual) if hand_visual is not None else torch.zeros_like(kin_feat)
        phon_feat = self.phonology_stem(phonology) if phonology is not None else torch.zeros_like(kin_feat)

        # 3. Multimodal Gated Cross-Attention Fusion
        concat_modalities = torch.cat([kin_feat, vis_feat, hand_feat, phon_feat], dim=-1)
        gates = F.softmax(self.modality_gate(concat_modalities), dim=-1)  # [B, T, 4]
        fused = (
            gates[:, :, 0:1] * kin_feat +
            gates[:, :, 1:2] * vis_feat +
            gates[:, :, 2:3] * hand_feat +
            gates[:, :, 3:4] * phon_feat
        )
        h = self.fusion_norm(fused)
        # Signer-Adaptive Instance Normalization (AdaIN) decoupling individual signer style
        h = self.signer_adain(h)

        # Extract coordinates for specialized geometric engines
        # Canonical 60 keypoints: 0-20 Left Hand (0 wrist), 21-41 Right Hand (21 wrist), 42-47 Pose (42 Left Sh, 43 Right Sh), 48-59 Face
        hand_coords = None
        base_hand_coords = None
        shoulder_coords = None
        if kinematics.shape[-1] >= 60 * 3:
            pts = kinematics.detach().view(B, T, self.num_keypoints, -1)[:, :, :, :3]
            hand_coords = pts[:, :, self.RIGHT_WRIST_IDX, :]        # Right wrist / palm root (canonical kp 21)
            base_hand_coords = pts[:, :, self.LEFT_WRIST_IDX, :]    # Left wrist / palm root (canonical kp 0)
            shoulder_coords = pts[:, :, self.LEFT_SHOULDER_IDX:self.RIGHT_SHOULDER_IDX + 1, :] # Shoulders (42 & 43)
            if face_landmarks is None and self.num_keypoints >= self.FACE_END_IDX:
                face_landmarks = pts[:, :, self.FACE_START_IDX:self.FACE_END_IDX, :]

        # 4. Engine 1: 3D Spatial Locus Memory Bank
        h, locus_losses = self.locus_memory(h, hand_coords=hand_coords, shoulder_coords=shoulder_coords)
        multi_task_losses.update(locus_losses)

        # 5. Engine 2: Non-Manual Feature Pyramid & Polarity Guard
        h, nmm_preds = self.nmm_pyramid(h, face_landmarks=face_landmarks, cranial_imu=cranial_imu)
        multi_task_losses["loss_nmm"] = nmm_preds["loss_nmm"]
        if text_is_negative is not None:
            loss_polarity = self.polarity_guard(nmm_preds["negation_logits"], text_is_negative)
            multi_task_losses["loss_polarity"] = loss_polarity * 0.1

        # 6. Engine 3: Deconstructive Classifier Predicates Field
        h, class_losses = self.classifier_field(h, hand_positions=hand_coords, base_hand_positions=base_hand_coords)
        multi_task_losses["loss_classifier"] = class_losses["loss_classifier_cpc"]

        # 7. Contextual Conformer Encoder Backbone
        early_h = h
        h = self.encoder(h)

        # 8. Engine 4: Monotonic Chunk Permutation Transducer (OSV -> SVO reordering)
        h_reordered, perm_losses = self.chunk_transducer(h)
        multi_task_losses["loss_permutation"] = perm_losses["loss_permutation"]

        # 9. Phonology Auxiliary Reconstruction Loss
        pred_phon = self.phonology_head(h)
        if phonology is not None:
            loss_phon = F.mse_loss(pred_phon, phonology.detach())
            multi_task_losses["loss_phonology"] = loss_phon * 0.1

        # 10. Multi-Tier CTC Heads (Tile-aligned)
        ctc_logits = self.ctc_head(h)
        english_ctc_logits = self.english_ctc_head(h_reordered)
        english_inter_ctc_logits = self.english_inter_ctc_head(h)
        english_early_ctc_logits = self.english_early_ctc_head(early_h)

        # 10b. Frontier Breakthroughs: Movement Epenthesis Suppression & Fingerspelling Decoupling
        biased_ctc_logits, beta_t = self.epenthesis_suppressor.apply_ctc_blank_bias(ctc_logits, kinematics)
        gamma_t, is_fingerspelling = self.fs_router(kinematics)
        # Protect English CTC from both fingerspelling overlap and movement epenthesis hallucinations
        biased_english_ctc = self.fs_weaver.suppress_word_logits_on_fingerspelling(english_ctc_logits, gamma_t)
        biased_english_ctc, _ = self.epenthesis_suppressor.apply_ctc_blank_bias(biased_english_ctc, kinematics)

        # Character-level CTC logits from hand features (for fingerspelled proper nouns)
        char_input = hand_feat if hand_visual is not None else h
        char_ctc_logits = self.char_decoder(char_input)

        # Self-supervised physical consistency losses for frontier breakthrough engines
        multi_task_losses["loss_epenthesis_consistency"] = self.epenthesis_suppressor.compute_consistency_loss(kinematics)
        multi_task_losses["loss_fs_consistency"] = self.fs_router.compute_consistency_loss(kinematics)

        # 11. Autoregressive Translation Decoder with Engine 5 (Visual Grounding Shield)
        decoder_logits = None
        if text_tokens is not None:
            if self.use_gpt2_decoder and self.gpt2_decoder is not None:
                raw_dec_logits, cross_attn = self.gpt2_decoder(input_ids=text_tokens, memory=h_reordered)
            else:
                L = text_tokens.shape[1]
                tgt_embed = self.text_embedding(text_tokens)
                # Causal mask for decoder
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=text_tokens.device)
                dec_out = self.decoder(tgt=tgt_embed, memory=h_reordered, tgt_mask=tgt_mask)
                raw_dec_logits = self.decoder_head(dec_out)

                # Simulated cross-attention weights for grounding evaluation
                # [B, L, T]
                sim_attn = torch.bmm(dec_out, h_reordered.transpose(1, 2)) * (1.0 / math.sqrt(self.d_model))
                cross_attn = F.softmax(sim_attn, dim=-1)

            # Visual Grounding Shield (detach motion_energy to avoid parasitic graph expansion)
            motion_energy = torch.norm(torch.diff(h_reordered.detach(), dim=1, prepend=h_reordered.detach()[:, :1, :]), dim=-1)
            shielded_logits, shield_losses = self.grounding_shield(
                raw_dec_logits, cross_attn, motion_energy=motion_energy
            )
            decoder_logits = shielded_logits
            multi_task_losses["loss_anti_hallucination"] = shield_losses["loss_anti_hallucination"]

        loss_tensors = [v for k, v in multi_task_losses.items() if k.startswith("loss_")]
        total_loss = sum(loss_tensors) if loss_tensors else None

        return V3ModelOutput(
            ctc_logits=biased_ctc_logits,
            english_ctc_logits=biased_english_ctc,
            english_inter_ctc_logits=english_inter_ctc_logits,
            decoder_logits=decoder_logits,
            encoded_features=h_reordered,
            multi_task_losses=multi_task_losses,
            total_loss=total_loss,
            char_ctc_logits=char_ctc_logits,
            epenthesis_prob=beta_t.detach() if beta_t is not None else None,
            fingerspelling_prob=gamma_t.detach() if gamma_t is not None else None,
            raw_ctc_logits=ctc_logits,
            raw_english_ctc_logits=english_ctc_logits,
        )
