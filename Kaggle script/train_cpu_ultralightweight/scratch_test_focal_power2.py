import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class PowerOfTwoFocalAndClusterMarginLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, cluster_margin: float = 1.5, cluster_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.cluster_margin = cluster_margin
        self.cluster_weight = cluster_weight
        # 'A'->0, 'E'->4, 'M'->12, 'N'->13, 'S'->18, 'T'->19
        self.confused_indices = torch.tensor([0, 4, 12, 13, 18, 19], dtype=torch.long)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Hệ số phạt lũy thừa 2: (1 - p_t)^2
        focal_weight = torch.pow(1.0 - target_p, self.gamma)
        focal_loss = -focal_weight * target_log_p
        
        # 2. Quadratic Hard Margin Penalty cho cụm {A, E, S, T, M, N}
        device = logits.device
        confused_mask = torch.isin(targets, self.confused_indices.to(device))
        
        margin_loss = torch.tensor(0.0, device=device)
        if confused_mask.any():
            conf_logits = logits[confused_mask]
            conf_targets = targets[confused_mask]
            
            target_scores = conf_logits.gather(dim=-1, index=conf_targets.unsqueeze(-1))
            cluster_logits = conf_logits[:, self.confused_indices.to(device)]
            diff = cluster_logits - target_scores + self.cluster_margin
            
            hard_penalty = torch.clamp(diff, min=0.0)
            margin_loss = torch.mean(torch.pow(hard_penalty, 2))
            
        total_loss = torch.mean(focal_loss) + self.cluster_weight * margin_loss
        return total_loss

print("[*] Kiểm thử hàm mất mát phạt lũy thừa 2 (Power of 2 Penalty Loss)...")
criterion = PowerOfTwoFocalAndClusterMarginLoss(gamma=2.0, cluster_margin=1.5)

# Giả lập logits 26 lớp
dummy_logits = torch.randn(8, 26, requires_grad=True)
dummy_targets = torch.tensor([0, 4, 18, 19, 1, 2, 12, 13]) # A, E, S, T, B, C, M, N

loss = criterion(dummy_logits, dummy_targets)
loss.backward()

print(f"  -> Loss Value: {loss.item():.4f}")
print(f"  -> Gradient Norm: {dummy_logits.grad.norm().item():.4f} [PASS]")
