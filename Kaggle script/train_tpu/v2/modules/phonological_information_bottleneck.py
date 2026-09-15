#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — PHONOLOGICAL INFORMATION BOTTLENECK (INFOBOTTLENECK-SLT)
================================================================================
Implements Variational Information Bottleneck (VIB) & Multi-Positive InfoNCE:
1. Variational Information Bottleneck (VIB):
     Minimizes I(Z; X) while maximizing I(Z; Y):
     L_IB = L_task(Z, Y) + beta * max(tau_free, D_KL(q(z|x) || p(z)))
     Compromises out signer jitter, sensor noise, and clothing artifacts while
     retaining core linguistic morphology.
2. Reparameterized Stochastic Latents:
     mu(x), logvar(x) in R^{d_latent}; z = mu + sigma * epsilon, epsilon ~ N(0, I)
3. Multi-Positive Phonological Mutual Information Maximization:
     Simultaneously aligns latent z with handshape, trajectory, and gloss semantic anchors:
     L_multipos = - (1/|P|) * sum_{p in P} log ( exp(<z, z_p> / tau) / sum exp(<z, z_k> / tau) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoBottleneckOutput(NamedTuple):
    latent_samples: torch.Tensor        # [B, d_latent] Stochastic compressed latent z
    latent_mu: torch.Tensor             # [B, d_latent] Mean vector
    latent_logvar: torch.Tensor         # [B, d_latent] Log variance vector
    kl_divergence: torch.Tensor         # Scalar KL divergence loss (nats)
    multi_positive_loss: torch.Tensor   # Scalar multi-positive InfoNCE loss
    total_loss: torch.Tensor            # Combined VIB loss


class ASLPhonologicalInfoBottleneckEngine(nn.Module):
    """
    Phonological Variational Information Bottleneck & Multi-Positive InfoNCE Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_latent: int = 64,
        beta_kl: float = 1e-3,
        free_bits_threshold: float = 0.50,
        tau: float = 0.07,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.beta_kl = beta_kl
        self.free_bits = free_bits_threshold
        self.tau = tau

        # VIB Parameter Heads
        self.mu_head = nn.Linear(d_model, d_latent)
        self.logvar_head = nn.Linear(d_model, d_latent)

        # Multi-positive anchor projection heads
        self.proj_z = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.proj_hand = nn.Linear(d_model, d_model)
        self.proj_traj = nn.Linear(d_model, d_model)
        self.proj_gloss = nn.Linear(d_model, d_model)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = mu + sigma * eps, eps ~ N(0, I).
        """
        if self.training:
            std = torch.exp(0.50 * logvar.clamp(-10.0, 10.0))
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def compute_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        D_KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).
        """
        kl_per_dim = -0.50 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar))  # [B, d_latent]
        kl_nats = kl_per_dim.sum(dim=-1).mean()  # Scalar mean over batch
        # Free-bits clamping to prevent posterior collapse
        kl_clamped = torch.clamp(kl_nats, min=self.free_bits)
        return kl_clamped

    def compute_multi_positive_loss(
        self,
        z_emb: torch.Tensor,                 # [B, d_model] Projected latent representation
        hand_anchor: torch.Tensor,           # [B, d_model] Handshape feature anchor
        traj_anchor: torch.Tensor,           # [B, d_model] Trajectory dynamics anchor
        gloss_anchor: Optional[torch.Tensor] = None, # [B, d_model] Gloss text semantic anchor
    ) -> torch.Tensor:
        """
        Computes Multi-Positive InfoNCE loss aligning z with multiple phonological anchors.
        """
        B, D = z_emb.shape
        device = z_emb.device

        z_norm = F.normalize(z_emb, p=2, dim=-1)
        h_norm = F.normalize(self.proj_hand(hand_anchor), p=2, dim=-1)
        t_norm = F.normalize(self.proj_traj(traj_anchor), p=2, dim=-1)

        pos_list = [h_norm, t_norm]
        if gloss_anchor is not None:
            g_norm = F.normalize(self.proj_gloss(gloss_anchor), p=2, dim=-1)
            pos_list.append(g_norm)

        num_pos = len(pos_list)
        total_loss = torch.tensor(0.0, device=device)

        # Negative candidates: all other batch items across all positive modalities
        all_anchors = torch.cat(pos_list, dim=0)  # [B * num_pos, D]

        for p_idx, pos_anchor in enumerate(pos_list):
            # Cosine similarity matrix: [B, B * num_pos]
            sim_mat = torch.matmul(z_norm, all_anchors.t()) / self.tau
            # Target positive index for sample i is: p_idx * B + i
            targets = torch.arange(B, device=device) + p_idx * B
            loss_p = F.cross_entropy(sim_mat, targets)
            total_loss = total_loss + loss_p

        return total_loss / float(num_pos)

    def forward(
        self,
        h_seq: torch.Tensor,                         # [B, T, d_model] Encoded visual sequence
        hand_features: torch.Tensor,                 # [B, d_model] or [B, T, d_model]
        traj_features: torch.Tensor,                 # [B, d_model] or [B, T, d_model]
        gloss_features: Optional[torch.Tensor] = None, # [B, d_model] optional text semantics
    ) -> InfoBottleneckOutput:
        """
        Executes Variational Information Bottleneck sampling and multi-positive alignment.
        """
        B, T, D = h_seq.shape
        h_pooled = h_seq.mean(dim=1)  # [B, D]

        # Pool anchors if 3D
        if hand_features.dim() == 3:
            hand_features = hand_features.mean(dim=1)
        if traj_features.dim() == 3:
            traj_features = traj_features.mean(dim=1)
        if gloss_features is not None and gloss_features.dim() == 3:
            gloss_features = gloss_features.mean(dim=1)

        # 1. Variational Parameter Estimation
        mu = self.mu_head(h_pooled)          # [B, d_latent]
        logvar = self.logvar_head(h_pooled)  # [B, d_latent]

        # 2. Stochastic Latent Sampling
        z_sample = self.reparameterize(mu, logvar)  # [B, d_latent]
        z_projected = self.proj_z(z_sample)         # [B, d_model]

        # 3. KL Divergence Input Compression Loss
        kl_loss = self.compute_kl_divergence(mu, logvar)

        # 4. Multi-Positive Phonological Mutual Information Loss
        loss_multipos = self.compute_multi_positive_loss(
            z_emb=z_projected,
            hand_anchor=hand_features,
            traj_anchor=traj_features,
            gloss_anchor=gloss_features,
        )

        total_loss = loss_multipos + self.beta_kl * kl_loss

        return InfoBottleneckOutput(
            latent_samples=z_sample,
            latent_mu=mu,
            latent_logvar=logvar,
            kl_divergence=kl_loss,
            multi_positive_loss=loss_multipos,
            total_loss=total_loss,
        )
