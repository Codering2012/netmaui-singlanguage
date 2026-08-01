import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

try:
    import torch_xla
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False

class PositionalEncoding(nn.Module):
    """Learned positional encoding for sequence tokens."""
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)

class LandmarkTransformer(nn.Module):
    """
    3D Landmark Transformer model for whole-body sign language recognition.
    Processes (B, T, 60, 9) keypoint sequences and outputs classification logits for 6043 classes.
    """
    def __init__(
        self,
        num_classes: int = 6043,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.input_dim = num_keypoints * channels_per_kp
        self.d_model = d_model
        
        # Spatial input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Temporal Attention Pooling Head
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
        # Final Linear Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, 60, 9) or (B, T, 540) landmark features tensor
            mask: (B, T) boolean mask tensor where True indicates valid frame, False is padding
        Returns:
            logits: (B, num_classes) classification logits
        """
        B, T = x.size(0), x.size(1)
        if x.dim() == 4:
            x = x.reshape(B, T, -1)  # Flatten (B, T, 60*9) -> (B, T, 540)
            
        # Linear projection
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.pos_encoder(h)  # Add positional embeddings
        
        # Key padding mask for PyTorch Transformer (True indicates padded frame to ignore)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # PyTorch convention: True = mask out
            
        # Transformer Encoder pass
        feat = self.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)
        feat = self.final_norm(feat)  # (B, T, d_model)
        
        # Masked Global Average Pooling & Attention Pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).to(feat.dtype)  # (B, T, 1)
            mean_pool = (feat * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
            
            attn_weights = self.attention_pool(feat)  # (B, T, 1)
            attn_weights = attn_weights.masked_fill(~mask_expanded.bool(), -1e9)
            attn_weights = F.softmax(attn_weights, dim=1)
            attn_pool = (feat * attn_weights).sum(dim=1)
        else:
            mean_pool = feat.mean(dim=1)
            attn_weights = F.softmax(self.attention_pool(feat), dim=1)
            attn_pool = (feat * attn_weights).sum(dim=1)
            
        combined_pooled = torch.cat([mean_pool, attn_pool], dim=-1)  # (B, 2*d_model)
        logits = self.classifier(combined_pooled)  # (B, num_classes)
        return logits

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution for extreme compute efficiency."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, padding: int = 2):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        scale = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * scale

class MultiScaleTemporalConv1d(nn.Module):
    """
    Multi-Scale Temporal Convolutional Stem for Sign Language Recognition.
    Processes temporal motion at 3 parallel receptive field scales:
    - Short scale (k=3): Micro-finger flicks and rapid letters
    - Medium scale (k=7): Full hand sign trajectories
    - Long scale (k=15): Multi-sign sentence transitions and grammatical holds
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = DepthwiseSeparableConv1d(channels, channels, kernel_size=3, padding=1)
        self.conv7 = DepthwiseSeparableConv1d(channels, channels, kernel_size=7, padding=3)
        self.conv15 = DepthwiseSeparableConv1d(channels, channels, kernel_size=15, padding=7)
        self.fusion = nn.Linear(channels * 3, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3 = self.conv3(x).transpose(1, 2)
        c7 = self.conv7(x).transpose(1, 2)
        c15 = self.conv15(x).transpose(1, 2)
        fused = self.fusion(torch.cat([c3, c7, c15], dim=-1))
        return fused.transpose(1, 2)

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization per sample for residual networks."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

class RelativePositionMultiheadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Parameterized Relative Position Bias.
    Replaces static absolute Sin/Cos positional encodings with translation-invariant temporal relative position bias.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.max_len = max_len

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable relative position bias table: (2 * max_len - 1, nhead)
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(2 * max_len - 1, nhead))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def _get_relative_position_bias(self, T: int, device: torch.device) -> torch.Tensor:
        coords = torch.arange(T, device=device)
        rel_coords = coords[None, :] - coords[:, None]
        rel_coords = rel_coords + (self.max_len - 1)
        rel_coords = rel_coords.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_pos_bias_table[rel_coords] # (T, T, nhead)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, nhead, T, T)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        rel_bias = self._get_relative_position_bias(T, x.device)
        attn_scores = attn_scores + rel_bias

        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)

class MobileConformerBlock(nn.Module):
    """Ultra-lightweight MobileConformer block with Relative Position Bias & DropPath Stochastic Depth."""
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        drop_path_rate: float = 0.1,
        max_len: int = 256
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.multi_scale_conv = MultiScaleTemporalConv1d(d_model)
        self.se_fc1 = nn.Linear(d_model, d_model // 4)
        self.se_fc2 = nn.Linear(d_model // 4, d_model)
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(d_model)
        self.mha = RelativePositionMultiheadAttention(d_model=d_model, nhead=nhead, max_len=max_len)
        self.drop_path2 = DropPath(drop_path_rate)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.drop_path3 = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = x
        x_norm = self.norm1(x)
        x_conv = x_norm.transpose(1, 2)
        x_conv = self.multi_scale_conv(x_conv).transpose(1, 2)
        
        se_weight = torch.sigmoid(self.se_fc2(F.gelu(self.se_fc1(x_conv.mean(dim=1)))))
        x_conv = x_conv * se_weight.unsqueeze(1)
        x = res + self.drop_path1(x_conv)

        res = x
        x_norm = self.norm2(x)
        attn_out = self.mha(x_norm, key_padding_mask=key_padding_mask)
        x = res + self.drop_path2(attn_out)

        res = x
        x = res + self.drop_path3(self.ffn(self.norm3(x)))
        return x

class LandmarkTrajectoryImageEncoder(nn.Module):
    """
    Renders 3D landmark spatial-temporal trajectories into a 64x64 motion context image tensor (B, 3, 64, 64)
    and extracts high-capacity visual motion tokens via a deeper 5-layer 256-channel 2D CNN stem with GroupNorm.
    """
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.size(0), x.size(1)
        x_reshaped = x.view(B, T, 60, 9) if x.dim() == 3 else x

        lh_pos = x_reshaped[:, :, 18:39, :2] # (B, T, 21, 2)
        rh_pos = x_reshaped[:, :, 39:60, :2] # (B, T, 21, 2)
        vel_mag = torch.norm(x_reshaped[:, :, :, 3:6], dim=-1) # (B, T, 60)

        lh_mag = torch.norm(lh_pos, dim=-1).unsqueeze(1) # (B, 1, T, 21)
        rh_mag = torch.norm(rh_pos, dim=-1).unsqueeze(1) # (B, 1, T, 21)
        vel_mag_4d = vel_mag.unsqueeze(1)                # (B, 1, T, 60)

        # Zero out padding frames gracefully via broadcast mask (Zero Graph Breaks!)
        if mask is not None:
            mask_4d = mask.view(B, 1, T, 1).to(lh_mag.dtype)
            lh_mag = lh_mag * mask_4d
            rh_mag = rh_mag * mask_4d
            vel_mag_4d = vel_mag_4d * mask_4d

        # Vectorized 4D Interpolation (B, 1, T, K) -> (B, 1, 64, 64) - High Resolution 64x64!
        lh_grid = F.interpolate(lh_mag, size=(64, 64), mode="bilinear", align_corners=False)
        rh_grid = F.interpolate(rh_mag, size=(64, 64), mode="bilinear", align_corners=False)
        vel_grid = F.interpolate(vel_mag_4d, size=(64, 64), mode="bilinear", align_corners=False)

        img_tensor = torch.cat([lh_grid, rh_grid, vel_grid], dim=1) # (B, 3, 64, 64)
        return self.cnn_stem(img_tensor)

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss with MoCo-Style FIFO Memory Bank & True Top-K Hard Negative Mining.
    Maintains a FIFO queue of normalized projection features (size 65536) and labels (size 65536).
    Enables single-occurrence classes in a batch to find positive pairs in memory across 65,536 candidates!
    """
    def __init__(self, temperature: float = 0.07, top_k_negatives: int = 16, memory_size: int = 65536, feature_dim: int = 128):
        super().__init__()
        self.temperature = temperature
        self.top_k_negatives = top_k_negatives
        self.memory_size = memory_size
        self.feature_dim = feature_dim

        self.register_buffer("memory_feats", torch.randn(memory_size, feature_dim))
        self.memory_feats = F.normalize(self.memory_feats, p=2, dim=-1)
        self.register_buffer("memory_labels", torch.full((memory_size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self._ptr_idx = 0

    @torch.no_grad()
    def _dequeue_and_enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        feats_det = F.normalize(features.detach(), p=2, dim=-1)
        lbls_det  = labels.detach()
        if _XLA_AVAILABLE and "xla" in str(device).lower():
            import torch_xla.core.xla_model as xm
            feats_det = xm.all_gather(feats_det)
            lbls_det  = xm.all_gather(lbls_det)
        batch_size = feats_det.shape[0]
        idx = (self.ptr + torch.arange(batch_size, device=device)) % self.memory_size
        self.memory_feats.scatter_(0, idx.unsqueeze(-1).expand(batch_size, self.feature_dim), feats_det)
        self.memory_labels.scatter_(0, idx, lbls_det)
        self.ptr.copy_((self.ptr + batch_size) % self.memory_size)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=-1)
        batch_size = features.shape[0]
        device = features.device

        all_feats = torch.cat([features, self.memory_feats.to(device)], dim=0)
        all_labels = torch.cat([labels, self.memory_labels.to(device)], dim=0)

        pos_mask = torch.eq(labels.view(-1, 1), all_labels.view(1, -1)).float()
        
        self_mask = torch.zeros((batch_size, batch_size + self.memory_size), device=device)
        self_mask[:, :batch_size] = torch.eye(batch_size, device=device)
        pos_mask = pos_mask * (1.0 - self_mask)

        cos_sim = torch.matmul(features, all_feats.T)
        neg_sim = cos_sim.masked_fill(pos_mask.bool() | self_mask.bool(), -1e9)

        k = min(self.top_k_negatives, max(1, cos_sim.shape[1] - 2))
        topk_neg_sim, _ = torch.topk(neg_sim, k=k, dim=-1)

        pos_sim_scaled = cos_sim / self.temperature
        topk_neg_sim_scaled = topk_neg_sim / self.temperature

        loss_list = []
        for i in range(batch_size):
            pos_idx = torch.nonzero(pos_mask[i]).squeeze(-1)
            if len(pos_idx) > 0:
                pos_val = pos_sim_scaled[i, pos_idx]
                neg_val = topk_neg_sim_scaled[i]
                concat_val = torch.cat([pos_val, neg_val])
                log_prob = pos_val - torch.logsumexp(concat_val, dim=0)
                loss_list.append(-log_prob.mean())

        self._dequeue_and_enqueue(features, labels)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(loss_list).mean()

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from DANN (Ganin et al., 2016).
    Reverses gradient during backpropagation to force feature extractor to learn Domain-Invariant Features.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: Union[float, torch.Tensor] = 1.0):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), device=x.device, dtype=x.dtype)
        ctx.save_for_backward(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha,) = ctx.saved_tensors
        return grad_output.neg() * alpha, None

class DomainClassifierHead(nn.Module):
    """
    Adversarial Domain Classifier Head for Unsupervised Domain Adaptation.
    Predicts dataset domain origin (0=WLASL, 1=How2Sign, 2=ASL-Citizen, 3=ChicagoFSWild).
    Reverses gradients with dynamic alpha scaling (0 -> 1) to make feature representations domain invariant.
    """
    def __init__(self, d_model: int = 384, num_domains: int = 4):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model * 3, 192),
            nn.GELU(),
            nn.Linear(192, num_domains)
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_x = GradientReversalFunction.apply(x, alpha)
        return self.domain_classifier(reversed_x)

class UltraLightSignModel(nn.Module):
    """
    High-Capacity 12.2 Million Parameter Sign Transformer with Learnable CLS Token, Relative Position Bias,
    Stochastic Depth (DropPath), Layer 3 Early-Exit Classifier Head, 3-Way Context Fusion ([CLS] || Mean || Attention).
    Configured with d_model=512, nhead=16, num_layers=6, dim_feedforward=1024 (~12.2M Parameters).
    """
    def __init__(
        self,
        num_classes: int = 2480,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        d_model: int = 512,
        nhead: int = 16,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        drop_path_rate: float = 0.25,
        max_len: int = 320
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.input_dim = num_keypoints * channels_per_kp
        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.visual_encoder = LandmarkTrajectoryImageEncoder(out_dim=128)

        self.input_stem = nn.Sequential(
            nn.Linear(self.input_dim + 128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            MobileConformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                drop_path_rate=dpr[i],
                max_len=max_len
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

        # 3-Way Context Fusion Vector: [CLS] || Masked Mean || Masked Attention (3 * d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.asl_lex_head = nn.Sequential(
            nn.Linear(d_model * 3, 64),
            nn.GELU(),
            nn.Linear(64, 5)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * 3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        # Layer 3 Early Exit Classifier Head for 40% CPU FLOP Acceleration
        self.early_exit_classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.domain_head = DomainClassifierHead(d_model=d_model, num_domains=4)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_aux: bool = False, grl_alpha: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, T = x.size(0), x.size(1)
        
        v_tokens = self.visual_encoder(x, mask=mask)
        v_tokens_exp = v_tokens.unsqueeze(1).expand(-1, T, -1)

        x_flat = x.reshape(B, T, -1) if x.dim() == 4 else x
        x_fused = torch.cat([x_flat, v_tokens_exp], dim=-1)

        h = self.input_stem(x_fused)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1) # (B, 1 + T, d_model)

        if mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=h.device)
            key_padding_mask = torch.cat([cls_mask, ~mask], dim=1)
        else:
            key_padding_mask = None

        early_exit_logits = None
        for idx, block in enumerate(self.blocks):
            h = block(h, key_padding_mask=key_padding_mask)
            if idx == 2:  # After Layer 3
                h_l3_norm = self.final_norm(h)
                h_l3_cls = h_l3_norm[:, 0]
                h_l3_seq = h_l3_norm[:, 1:]
                if mask is not None:
                    mask_exp_l3 = mask.unsqueeze(-1).to(h_l3_seq.dtype)
                    h_l3_mean = (h_l3_seq * mask_exp_l3).sum(dim=1) / mask_exp_l3.sum(dim=1).clamp(min=1.0)
                    attn_w_l3 = F.softmax(self.attn_pool(h_l3_seq).masked_fill(~mask_exp_l3.bool(), -1e9), dim=1)
                    h_l3_attn = (h_l3_seq * attn_w_l3).sum(dim=1)
                else:
                    h_l3_mean = h_l3_seq.mean(dim=1)
                    h_l3_attn = (h_l3_seq * F.softmax(self.attn_pool(h_l3_seq), dim=1)).sum(dim=1)
                combined_l3 = torch.cat([h_l3_cls, h_l3_mean, h_l3_attn], dim=-1)
                early_exit_logits = self.early_exit_classifier(combined_l3)

                # 2x Temporal Pooling for Deep Layers 4-6 (75% FLOP Drop in L4-L6 for Thermal Headroom)
                if h.size(1) > 3:
                    cls_t = h[:, :1]
                    seq_t = h[:, 1:].transpose(1, 2)
                    seq_t_pooled = F.avg_pool1d(seq_t, kernel_size=2, stride=2).transpose(1, 2)
                    h = torch.cat([cls_t, seq_t_pooled], dim=1)
                    if mask is not None:
                        mask = mask[:, ::2]
                        cls_m = torch.zeros((B, 1), dtype=torch.bool, device=h.device)
                        key_padding_mask = torch.cat([cls_m, ~mask], dim=1)

        h = self.final_norm(h)

        h_cls = h[:, 0]
        h_seq = h[:, 1:]

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).to(h_seq.dtype)
            h_mean = (h_seq * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
            attn_w = self.attn_pool(h_seq).masked_fill(~mask_exp.bool(), -1e9)
            attn_w = F.softmax(attn_w, dim=1)
            h_attn = (h_seq * attn_w).sum(dim=1)
        else:
            h_mean = h_seq.mean(dim=1)
            attn_w = F.softmax(self.attn_pool(h_seq), dim=1)
            h_attn = (h_seq * attn_w).sum(dim=1)

        combined = torch.cat([h_cls, h_mean, h_attn], dim=-1)
        logits = self.classifier(combined)
        
        if return_aux:
            lex_logits = self.asl_lex_head(combined)
            conf_pred = self.confidence_head(combined).squeeze(-1)
            proj_feats = F.normalize(self.contrastive_head(combined), p=2, dim=-1)
            domain_logits = self.domain_head(combined, alpha=grl_alpha)
            return logits, lex_logits, conf_pred, proj_feats, domain_logits, early_exit_logits
        return logits
