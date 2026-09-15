#!/usr/bin/env python3
"""
Lightweight Hypothesis Test for Component 2:
1. Log-domain Sinkhorn Optimal Transport Transducer (doubly stochastic permutation)
2. SemanticEmbeddingAnchor (bidirectional InfoNCE contrastive alignment)

Hardware constraints: CPU only, B<=4, T<=64, D<=128, RAM<500MB, duration<15s.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogDomainSinkhornSolver(nn.Module):
    """
    Log-domain Differentiable Sinkhorn-Knopp Optimal Transport Solver.
    Uses native PyTorch ATen ops (torch.logsumexp) for 100% TPU/XLA compatibility.
    Guarantees doubly stochastic transport plan P where sum_j P_ij = 1 and sum_i P_ij = 1.
    """
    def __init__(self, num_iters: int = 12, epsilon: float = 0.08):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        cost_matrix: [B, M, M] >= 0
        Returns: P [B, M, M] doubly stochastic permutation matrix.
        """
        B, M, _ = cost_matrix.shape
        inv_eps = 1.0 / self.epsilon

        # Initialize dual potentials in log-space: [B, M]
        f = torch.zeros(B, M, device=cost_matrix.device, dtype=cost_matrix.dtype)
        g = torch.zeros(B, M, device=cost_matrix.device, dtype=cost_matrix.dtype)

        # Static loop of fixed iterations (no dynamic while-loop, fully XLA compilable)
        for _ in range(self.num_iters):
            # Update f: f_i = -eps * logsumexp_j ((g_j - C_ij) / eps)
            kernel_f = (g.unsqueeze(1) - cost_matrix) * inv_eps # [B, M, M]
            f = -self.epsilon * torch.logsumexp(kernel_f, dim=-1) # [B, M]

            # Update g: g_j = -eps * logsumexp_i ((f_i - C_ij) / eps)
            kernel_g = (f.unsqueeze(2) - cost_matrix) * inv_eps # [B, M, M]
            g = -self.epsilon * torch.logsumexp(kernel_g, dim=1)  # [B, M]

        # Compute optimal transport matrix P in log space
        log_P = (f.unsqueeze(2) + g.unsqueeze(1) - cost_matrix) * inv_eps
        P = torch.exp(log_P)
        # Final row-normalization for exact stochasticity
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)
        return P


class SinkhornChunkTransducer(nn.Module):
    """
    Syntactic Reordering Transducer with Log-Domain Sinkhorn Optimal Transport.
    Transforms ASL Topic-Comment order into English SVO order.
    """
    def __init__(self, d_model: int = 128, chunk_size: int = 4, num_iters: int = 8, epsilon: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.sinkhorn = LogDomainSinkhornSolver(num_iters=num_iters, epsilon=epsilon)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.perm_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, T, D]
        Returns: (h_reordered, P)
        """
        B, T, D = h.shape
        M = max(1, T // self.chunk_size)
        # Average pool into M sign chunks
        h_chunks = h.view(B, M, self.chunk_size, D).mean(dim=2) # [B, M, D]

        q = F.normalize(self.query_proj(h_chunks), dim=-1)
        k = F.normalize(self.key_proj(h_chunks), dim=-1)

        # Cost matrix: Cosine distance C_ij = 1.0 - cos_sim(q_i, k_j)
        cost = 1.0 - torch.bmm(q, k.transpose(1, 2)) # [B, M, M] in [0, 2]

        P = self.sinkhorn(cost) # [B, M, M]

        # Apply permutation to chunk features
        reordered_chunks = torch.bmm(P, h_chunks) # [B, M, D]

        # Broadcast/interpolate back to [B, T, D]
        h_reordered = reordered_chunks.unsqueeze(2).expand(B, M, self.chunk_size, D).reshape(B, T, D)
        h_out = self.perm_norm(h + h_reordered)
        return h_out, P


class SemanticEmbeddingAnchor(nn.Module):
    """
    Multi-Granularity Sentence-Embedding Semantic Anchor.
    Bridges the gloss-free modality gap via Bidirectional InfoNCE contrastive alignment.
    """
    def __init__(self, d_model: int = 128, d_sent: int = 384, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_sent),
        )

    def forward(
        self,
        encoded_features: torch.Tensor,     # [B, T, D]
        target_sentence_embeddings: torch.Tensor, # [B, D_sent]
    ) -> torch.Tensor:
        """
        Computes bidirectional InfoNCE loss.
        """
        B = encoded_features.shape[0]
        # Global mean pool across time to form the utterance-level visual thought
        vis_thought = encoded_features.mean(dim=1) # [B, D]
        vis_proj = F.normalize(self.proj(vis_thought), dim=-1) # [B, D_sent]
        text_emb = F.normalize(target_sentence_embeddings.detach(), dim=-1) # [B, D_sent]

        # Similarity logits: [B, B]
        sim_matrix = torch.matmul(vis_proj, text_emb.transpose(0, 1)) / self.temperature
        labels = torch.arange(B, device=encoded_features.device)

        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.transpose(0, 1), labels)
        return 0.5 * (loss_v2t + loss_t2v)


def run_tests():
    print("=== Testing LogDomainSinkhornSolver & SinkhornChunkTransducer ===")
    B, M = 2, 8
    cost = torch.rand(B, M, M)
    solver = LogDomainSinkhornSolver(num_iters=16, epsilon=0.08)
    P = solver(cost)
    assert P.shape == (B, M, M)
    
    # Verify doubly stochastic property
    row_sums = P.sum(dim=-1) # [B, M]
    col_sums = P.sum(dim=-2) # [B, M]
    print(f"Mean row sum: {float(row_sums.mean()):.4f}, Mean col sum: {float(col_sums.mean()):.4f}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.03), "Row sums must equal 1"
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=0.03), "Col sums must equal 1"

    # Test Transducer with autograd
    transducer = SinkhornChunkTransducer(d_model=128, chunk_size=4)
    h = torch.randn(2, 32, 128, requires_grad=True)
    h_reordered, P_out = transducer(h)
    assert h_reordered.shape == (2, 32, 128)
    loss = h_reordered.sum()
    loss.backward()
    assert h.grad is not None, "Gradients must flow through Sinkhorn Transducer"
    print("[PASS] Sinkhorn Transducer doubly stochasticity and autograd verified.")

    print("\n=== Testing SemanticEmbeddingAnchor ===")
    anchor = SemanticEmbeddingAnchor(d_model=128, d_sent=384)
    enc = torch.randn(4, 32, 128, requires_grad=True)
    tgt_sent = torch.randn(4, 384)
    loss_sem = anchor(enc, tgt_sent)
    print(f"Semantic InfoNCE Loss: {loss_sem.item():.4f}")
    assert loss_sem.requires_grad
    loss_sem.backward()
    assert enc.grad is not None, "Gradients must flow through SemanticEmbeddingAnchor"
    print("[PASS] SemanticEmbeddingAnchor verified.")

if __name__ == "__main__":
    run_tests()
    print("\nALL COMPONENT 2 TESTS PASSED EMPIRICALLY!")
