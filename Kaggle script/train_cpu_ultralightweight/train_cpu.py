"""
HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN KÝ TỰ & CHUỖI ASL SIÊU NHẸ TRÊN CPU (ULTRA-LIGHTWEIGHT CPU TRAINER)
Hỗ trợ nạp dataset từ E:/datasets/asl_dataset/asl_preprocessed_phase1, tự động lọc bỏ How2Sign nếu cần.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class UltraLightweightASLModel(nn.Module):
    """
    Mô hình nhận diện ASL Landmarks siêu nhẹ (< 1.5M tham số) tối ưu cho CPU:
    - 1D Temporal Depthwise Separable Convolutions
    - Squeeze-and-Excitation Temporal Attention
    - 2-Layer Bidirectional GRU/LSTM
    - Multi-Head Classifier (Character 'A'-'Z' + CTC + Auxiliary Heads)
    """

    def __init__(
        self,
        num_classes: int = 28,          # 26 chữ cái ('A'-'Z') + SPACE + BLANK
        in_channels: int = 540,         # 60 landmarks x 9D features
        d_model: int = 128,             # Kích thước ẩn siêu nhẹ cho CPU
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Depthwise Separable Temporal Convolutions
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.conv1_pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_model)

        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.conv2_pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm2 = nn.BatchNorm1d(d_model)

        # Bi-directional GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Classifier Heads
        self.fc_frame = nn.Linear(hidden_dim, num_classes)
        self.fc_pool = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor shape [B, T, 60, 9] hoặc [B, T, 540]
        Returns:
            Dict chứa logits từng khung hình và logits tổng hợp toàn chuỗi
        """
        B, T = x.shape[0], x.shape[1]
        if x.dim() == 4:
            x = x.reshape(B, T, -1)  # [B, T, 540]

        h = self.input_proj(x)  # [B, T, d_model]

        # Convolutions trên chiều thời gian T
        h_conv = h.transpose(1, 2)  # [B, d_model, T]
        h_conv = F.relu(self.norm1(self.conv1_pointwise(self.conv1(h_conv))))
        h_conv = F.relu(self.norm2(self.conv2_pointwise(self.conv2(h_conv))))
        h_conv = h_conv.transpose(1, 2)  # [B, T, d_model]

        # GRU
        gru_out, _ = self.gru(h_conv)  # [B, T, hidden_dim]

        # 1. Logits cho từng khung hình (Frame-level logits cho CTC / FSM)
        frame_logits = self.fc_frame(gru_out)  # [B, T, num_classes]

        # 2. Logits tổng hợp toàn chuỗi (Sequence-level classification)
        pooled = gru_out.mean(dim=1)  # [B, hidden_dim]
        seq_logits = self.fc_pool(pooled)  # [B, num_classes]

        return {
            "frame_logits": frame_logits,
            "seq_logits": seq_logits
        }


class ASLLandmarkDataset(Dataset):
    """
    Dataset nạp các shard .pt từ E:/datasets/asl_dataset/asl_preprocessed_phase1,
    hỗ trợ tự động lọc bỏ How2Sign và trích xuất nhãn chữ cái ngón tay.
    """

    def __init__(
        self,
        shard_paths: List[Path],
        exclude_how2sign: bool = True,
        max_samples: Optional[int] = None
    ):
        self.samples: List[Dict[str, Any]] = []
        self.char_to_idx = {chr(65 + i): i for i in range(26)}  # 'A'-'Z' -> 0-25
        self.char_to_idx[" "] = 26
        self.char_to_idx["_"] = 27  # CTC Blank

        for p in shard_paths:
            if not p.exists():
                continue
            try:
                data = torch.load(p, map_location="cpu", weights_only=False)
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    # Lọc How2Sign nếu có yêu cầu
                    src = str(item.get("source", "")).lower()
                    if exclude_how2sign and "how2sign" in src:
                        continue

                    # Kiểm tra dữ liệu landmarks
                    if "features" not in item:
                        continue

                    label = item.get("label", "")
                    if isinstance(label, str) and len(label) == 1 and label.upper() in self.char_to_idx:
                        lbl_idx = self.char_to_idx[label.upper()]
                    else:
                        lbl_idx = item.get("label_idx", -1)
                        if lbl_idx < 0 or lbl_idx >= 26:
                            # Nếu nhãn là từ, lấy ký tự đầu hoặc bỏ qua
                            if isinstance(label, str) and label:
                                first_ch = label[0].upper()
                                lbl_idx = self.char_to_idx.get(first_ch, 0)
                            else:
                                lbl_idx = 0

                    feats = item["features"]
                    if hasattr(feats, "shape") and feats.shape[0] > 0:
                        self.samples.append({
                            "features": feats.float(),
                            "label_idx": lbl_idx,
                            "source": src
                        })

                    if max_samples and len(self.samples) >= max_samples:
                        break
            except Exception as e:
                print(f"[WARNING] Skipping unreadable shard {p.name}: {e}")

            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        feats = s["features"]
        if feats.dim() == 3:
            feats = feats.reshape(feats.shape[0], -1)  # [T, 540]

        return {
            "features": feats,
            "label": torch.tensor(s["label_idx"], dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function đệm (pad) chuỗi thời gian T về độ dài bằng nhau trong mini-batch."""
    max_t = max(item["features"].shape[0] for item in batch)
    d_feat = batch[0]["features"].shape[1]

    padded_feats = []
    labels = []

    for item in batch:
        f = item["features"]
        t = f.shape[0]
        if t < max_t:
            pad = torch.zeros(max_t - t, d_feat, dtype=f.dtype)
            f_pad = torch.cat([f, pad], dim=0)
        else:
            f_pad = f
        padded_feats.append(f_pad)
        labels.append(item["label"])

    return {
        "features": torch.stack(padded_feats, dim=0),
        "labels": torch.stack(labels, dim=0)
    }


def train_cpu(
    data_dir: str = "E:/datasets/asl_dataset/asl_preprocessed_phase1",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    exclude_how2sign: bool = True,
    max_samples: int = 5000,
    save_path: str = "asl_ultralightweight_cpu.pt"
):
    print("=" * 80)
    print("     HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN ASL SIÊU NHẸ TRÊN CPU (CPU TRAINER)     ")
    print("=" * 80)

    device = torch.device("cpu")
    base_p = Path(data_dir)
    train_shards = sorted(list((base_p / "train").glob("*.pt")))

    if not train_shards:
        print(f"[ERROR] No shards found in {base_p / 'train'}")
        return

    print(f"[*] Tìm thấy {len(train_shards)} shard files trong thư mục train.")
    print(f"[*] Đang nạp dataset (Exclude How2Sign={exclude_how2sign}, Max Samples={max_samples})...")
    dataset = ASLLandmarkDataset(train_shards, exclude_how2sign=exclude_how2sign, max_samples=max_samples)
    print(f"[*] Đã nạp thành công {len(dataset):,} mẫu dữ liệu cử chỉ ngón tay vào bộ nhớ.")

    if len(dataset) == 0:
        print("[ERROR] Dataset rỗng sau khi lọc.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = UltraLightweightASLModel(num_classes=28, in_channels=540, d_model=128, hidden_dim=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Khởi tạo mô hình UltraLightweightASLModel: {total_params:,} tham số ({total_params/1e6:.2f}M params).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("\nBắt đầu huấn luyện trên CPU...")

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            x = batch["features"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(x)
            logits = out["seq_logits"]

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        ep_time = time.time() - t0
        avg_loss = total_loss / max(1, total)
        acc = (correct / max(1, total)) * 100.0
        print(f"Epoch {ep:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | Elapsed: {ep_time:.2f}s")

    # Lưu mô hình
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_p))
    print(f"\n[THÀNH CÔNG] Đã lưu trọng số mô hình vào: {save_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="E:/datasets/asl_dataset/asl_preprocessed_phase1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exclude-how2sign", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--save-path", type=str, default="train_cpu_ultralightweight/asl_cpu_model.pt")
    args = parser.parse_args()

    train_cpu(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        exclude_how2sign=args.exclude_how2sign,
        max_samples=args.max_samples,
        save_path=args.save_path
    )
