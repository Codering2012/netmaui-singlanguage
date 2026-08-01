#!/usr/bin/env python3
"""
==============================================================================
KAGGLE 2x T4 GPU DEPLOYMENT SCRIPT
Trains the ASLFoundationModel from train_all_in_one_tpu.py using 2x T4 GPUs.
Handles the conversion of legacy dataset tuples to the new seq2seq dict format.
==============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add current directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the foundation model and training loop from the monolithic script
from train_all_in_one_tpu import (
    ASLFoundationModel,
    train_epoch_tpu,
    HomoscedasticLossWrapper,
    SupervisedContrastiveLoss,
    GlossVocabulary
)

# Import dataset utilities from the older dataset script
from dataset import create_dataloader

class Seq2SeqDataLoaderWrapper:
    """
    Wraps the older ASLShardedDataset (which returns tuples for classification)
    to yield dictionaries expected by the ASLFoundationModel seq2seq training loop.
    """
    def __init__(self, loader, vocab, device):
        self.loader = loader
        self.vocab = vocab
        self.device = device
        
    def __len__(self):
        return len(self.loader)
        
    def __iter__(self):
        for batch_data in self.loader:
            if len(batch_data) == 5:
                features, mask, targets, sample_weights, lex_targets = batch_data
            else:
                features, mask, targets, sample_weights = batch_data
            
            B = features.size(0)
            
            # Construct seq2seq dummy targets from the integer classification label
            # [BOS, target_token, EOS]
            gloss_seq = torch.zeros((B, 3), dtype=torch.long, device=self.device)
            gloss_seq[:, 0] = self.vocab.BOS_ID
            gloss_seq[:, 1] = targets.to(self.device) + self.vocab.OFFSET
            gloss_seq[:, 2] = self.vocab.EOS_ID
            
            gloss_len = torch.full((B,), 3, dtype=torch.long, device=self.device)
            
            yield {
                "feature": features.to(self.device),
                "mask": mask.to(self.device),
                "label": targets.to(self.device),
                "domain_label": torch.zeros_like(targets, device=self.device),
                "has_domain_label": torch.zeros_like(targets, dtype=torch.bool, device=self.device),
                "gloss_seq": gloss_seq,
                "gloss_len": gloss_len,
                "has_valid_gloss": torch.ones(B, dtype=torch.bool, device=self.device),
                "mlm_mask": None
            }

def get_dataset_dir(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        candidates = [
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/asl-preprocessed-phase1"),
            Path("./asl_preprocessed_phase1"),
            Path(r"E:\datasets\results\asl_preprocessed_phase1"),
        ]
        data_dir = next((c for c in candidates if c.exists()), data_dir)
    return data_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/asl-preprocessed-phase1", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--save-dir", type=str, default="/kaggle/working/checkpoints", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    data_dir = get_dataset_dir(args)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found. Please verify Kaggle dataset paths.")

    print(f"[*] Loading training dataset from '{data_dir}'...")
    train_loader_base = create_dataloader(
        dataset_dir=data_dir,
        split="train",
        batch_size=args.batch_size,
        max_len=args.max_len,
        worker_idx=0,
        num_workers=1,
        num_dataloader_workers=4,
        shuffle=True,
        augment=True
    )
    
    num_classes = train_loader_base.dataset.num_classes
    print(f"[*] Dataset loaded: {num_classes} classes detected.")

    # Create vocabulary mapping 0..num_classes to token IDs
    label_to_idx = {str(i): i for i in range(num_classes)}
    vocab = GlossVocabulary(label_to_idx=label_to_idx)

    print("[*] Initializing ASLFoundationModel (MobileConformer + Transformer Decoder)...")
    model = ASLFoundationModel(
        vocab_size=vocab.vocab_size,
        d_enc=320,
        num_enc_layers=8,
        nhead_enc=8,
        ffn_enc=1280,
        d_dec=320,
        num_dec_layers=8,
        nhead_dec=8,
        ffn_dec=1280
    ).to(device)

    # Wrap for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"[*] Wrapping model in nn.DataParallel for {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    loss_wrapper = HomoscedasticLossWrapper(num_losses=6).to(device)
    supcon_fn = SupervisedContrastiveLoss(temperature=0.07).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_wrapper.parameters()), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Generator wrapper to feed dicts to train_epoch_tpu
        wrapped_loader = Seq2SeqDataLoaderWrapper(train_loader_base, vocab, device)
        
        # Use existing monolithic training loop
        avg_loss, token_acc = train_epoch_tpu(
            model=model,
            loader=wrapped_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_wrapper=loss_wrapper,
            ema=None,
            supcon_fn=supcon_fn,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            prec_dtype=torch.float16,  # Use float16 on T4s
            is_master=True,
            accum_steps=2
        )
        
        print(f"Epoch {epoch} complete | Loss: {avg_loss:.4f} | Acc: {token_acc:.2f}%")
        
        # Save checkpoints
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "acc": token_acc
        }
        torch.save(ckpt_data, Path(args.save_dir) / "latest_model.pt")

if __name__ == "__main__":
    main()
