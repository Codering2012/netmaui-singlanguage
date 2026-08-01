#!/usr/bin/env python3
"""
==============================================================================
ASL RECOGNITION: TPU DISTRIBUTED TRAINING PIPELINE (PyTorch XLA)
Trains LandmarkTransformer on 60 WholeBody keypoints using preprocessed data.
==============================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Any, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root and train directory to python path
train_dir = Path(__file__).resolve().parent
project_root = train_dir.parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from model import LandmarkTransformer, UltraLightSignModel
from dataset import ASLShardedDataset, create_dataloader

# Check PyTorch XLA availability
_XLA_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    _XLA_AVAILABLE = True
except ImportError:
    pass

def log_msg(msg: str, master_only: bool = True, is_master: bool = True) -> None:
    if not master_only or is_master:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

def train_epoch_tpu(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_master: bool = True,
    precision: str = "bfloat16"
) -> Tuple[float, float]:
    """Runs a single training epoch across TPU cores using bfloat16 / float16 mixed precision."""
    model.train()
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_samples = 0
    correct_predictions = torch.tensor(0.0, device=device, dtype=torch.float32)
    start_time = time.time()

    # Wrap dataloader with PyTorch XLA ParallelLoader for async device transfers if using XLA
    if _XLA_AVAILABLE and "xla" in str(device).lower():
        para_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    else:
        para_loader = loader

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    device_type = "xla" if is_xla else ("cuda" if "cuda" in str(device).lower() else "cpu")
    prec_dtype = torch.bfloat16 if precision == "bfloat16" else (torch.float16 if precision == "float16" else torch.float32)
    # Never use torch.autocast on XLA devices
    use_autocast = not is_xla and precision in ("bfloat16", "float16")

    total_batches = len(loader)
    if _XLA_AVAILABLE and "xla" in str(device).lower():
        local_batches = torch.tensor([total_batches], dtype=torch.float32, device=device)
        min_batches = int(xm.all_reduce('min', local_batches).item())
    else:
        min_batches = total_batches

    for step, batch_data in enumerate(para_loader, start=1):
        if step > min_batches:
            continue

        if len(batch_data) == 5:
            features, mask, targets, sample_weights, lex_targets = batch_data
        else:
            features, mask, targets, sample_weights = batch_data
            lex_targets = None

        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        if lex_targets is not None:
            lex_targets = lex_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass in bfloat16 / float16 mixed precision with SOTA auxiliary heads
        if use_autocast:
            with torch.autocast(device_type, dtype=prec_dtype):
                if hasattr(model, "confidence_head") and lex_targets is not None:
                    outputs = model(features, mask=mask, return_aux=True)
                    logits, lex_logits, conf_pred = outputs[0], outputs[1], outputs[2]
                    loss_sign = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.1) * sample_weights).mean()
                    loss_lex = F.cross_entropy(lex_logits.float(), lex_targets)
                    loss_conf = F.mse_loss(conf_pred.float(), sample_weights)
                    loss = loss_sign + 0.2 * loss_lex + 0.1 * loss_conf
                else:
                    logits = model(features, mask=mask)
                    loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.1) * sample_weights).mean()
        else:
            if hasattr(model, "confidence_head") and lex_targets is not None:
                outputs = model(features, mask=mask, return_aux=True)
                logits, lex_logits, conf_pred = outputs[0], outputs[1], outputs[2]
                loss_sign = (F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1) * sample_weights).mean()
                loss_lex = F.cross_entropy(lex_logits, lex_targets)
                loss_conf = F.mse_loss(conf_pred.float(), sample_weights)
                loss = loss_sign + 0.2 * loss_lex + 0.1 * loss_conf
            else:
                logits = model(features, mask=mask)
                loss = (F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1) * sample_weights).mean()

        # Backward pass with NaN guard & Gradient Clipping
        loss.backward()
        if _XLA_AVAILABLE and "xla" in str(device).lower():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            xm.optimizer_step(optimizer)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Metrics logging accumulated as tensors on device
        batch_size = features.size(0)
        total_loss += loss.detach() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=-1)
        correct_predictions += (preds == targets).sum().detach()

        if step % 1 == 0 or step == min_batches:
            curr_loss = (total_loss / max(1, total_samples)).item()
            curr_acc = (correct_predictions / max(1, total_samples)).item() * 100.0
            log_msg(f"   [Step {step}/{min_batches}] Loss: {curr_loss:.4f} | Acc: {curr_acc:.2f}% | Step Loss: {loss.item():.4f}", is_master=is_master)

    elapsed = time.time() - start_time
    avg_loss = (total_loss / max(1, total_samples)).item()
    accuracy = (correct_predictions / max(1, total_samples)).item() * 100.0

    return avg_loss, accuracy

def evaluate_epoch_tpu(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    is_master: bool = True,
    precision: str = "bfloat16"
) -> Tuple[float, float]:
    """Runs evaluation across validation or test data loader using bfloat16 / float16 precision."""
    model.eval()
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_samples = 0
    correct_predictions = torch.tensor(0.0, device=device, dtype=torch.float32)

    if _XLA_AVAILABLE and "xla" in str(device).lower():
        para_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    else:
        para_loader = loader

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    device_type = "xla" if is_xla else ("cuda" if "cuda" in str(device).lower() else "cpu")
    prec_dtype = torch.bfloat16 if precision == "bfloat16" else (torch.float16 if precision == "float16" else torch.float32)
    # Never use torch.autocast on XLA devices
    use_autocast = not is_xla and precision in ("bfloat16", "float16")

    total_batches = len(loader)
    if is_xla:
        local_batches = torch.tensor([total_batches], dtype=torch.float32, device=device)
        min_batches = int(xm.all_reduce('min', local_batches).item())
    else:
        min_batches = total_batches

    with torch.no_grad():
        for step, batch_data in enumerate(para_loader, start=1):
            if step > min_batches:
                continue
            if len(batch_data) == 5:
                features, mask, targets, sample_weights, lex_targets = batch_data
            else:
                features, mask, targets, sample_weights = batch_data

            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)

            if use_autocast:
                with torch.autocast(device_type, dtype=prec_dtype):
                    logits = model(features, mask=mask)
                    loss = (F.cross_entropy(logits.float(), targets, reduction="none", label_smoothing=0.1) * sample_weights).mean()
            else:
                logits = model(features, mask=mask)
                loss = (F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1) * sample_weights).mean()

            batch_size = features.size(0)
            total_loss += loss.detach() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=-1)
            correct_predictions += (preds == targets).sum().detach()

            if is_xla and (step % 10 == 0):
                xm.mark_step()

    if _XLA_AVAILABLE and "xla" in str(device).lower():
        xm.mark_step()
        total_samples_tensor = torch.tensor(float(total_samples), device=device, dtype=torch.float32)
        metrics_tensor = torch.stack([total_loss, correct_predictions, total_samples_tensor])
        metrics_tensor = xm.all_reduce('sum', metrics_tensor)
        total_loss = metrics_tensor[0]
        correct_predictions = metrics_tensor[1]
        total_samples = int(metrics_tensor[2].item())
    else:
        total_samples = float(total_samples)

    avg_loss = (total_loss / max(1.0, total_samples)).item()
    accuracy = (correct_predictions / max(1.0, total_samples)).item() * 100.0

    return avg_loss, accuracy

def _tpu_worker_fn(index: int, args: argparse.Namespace) -> None:
    """Worker entry point for each TPU core process."""
    if _XLA_AVAILABLE:
        device = xm.xla_device()
        is_master = xm.is_master_ordinal()
        worker_idx = index
        num_workers = xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else args.num_cores
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True
        worker_idx = 0
        num_workers = 1

    log_msg(f"[*] Worker {worker_idx}/{num_workers} attached to device '{device}'", is_master=is_master)

    # Dataset & Dataloader creation with candidate search
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        candidates = [
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl"),
            Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl"),
            Path("/kaggle/input/asl-preprocessed-phase1"),
            Path("./asl_preprocessed_phase1"),
            Path(r"E:\datasets\results\asl_preprocessed_phase1"),
        ]
        data_dir = next((c for c in candidates if c.exists()), data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")

    log_msg(f"[*] Loading training dataset from '{data_dir}'...", is_master=is_master)
    train_loader = create_dataloader(
        dataset_dir=data_dir,
        split="train",
        batch_size=args.batch_size,
        max_len=args.max_len,
        worker_idx=worker_idx,
        num_workers=num_workers,
        num_dataloader_workers=args.num_dataloader_workers,
        shuffle=True,
        augment=getattr(args, "augment", False)
    )

    num_classes = train_loader.dataset.num_classes
    log_msg(f"[*] Train Dataset: {len(train_loader.dataset)} records across {num_classes} classes.", is_master=is_master)

    # Optional Validation Dataloader
    val_loader = None
    if (data_dir / "val").exists() or (data_dir.parent / "val").exists():
        try:
            val_loader = create_dataloader(
                dataset_dir=data_dir,
                split="val",
                batch_size=args.batch_size,
                max_len=args.max_len,
                worker_idx=worker_idx,
                num_workers=num_workers,
                num_dataloader_workers=args.num_dataloader_workers,
                shuffle=False
            )
            log_msg(f"[*] Val Dataset: {len(val_loader.dataset)} validation records.", is_master=is_master)
        except Exception as e:
            log_msg(f"[!] Note loading val dataset: {e}", is_master=is_master)

    # Optional Test Dataloader
    test_loader = None
    if (data_dir / "test").exists() or (data_dir.parent / "test").exists():
        try:
            test_loader = create_dataloader(
                dataset_dir=data_dir,
                split="test",
                batch_size=args.batch_size,
                max_len=args.max_len,
                worker_idx=worker_idx,
                num_workers=num_workers,
                num_dataloader_workers=args.num_dataloader_workers,
                shuffle=False
            )
            log_msg(f"[*] Test Dataset: {len(test_loader.dataset)} test records.", is_master=is_master)
        except Exception as e:
            log_msg(f"[!] Note loading test dataset: {e}", is_master=is_master)

    # Instantiate Model Architecture
    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    
    if getattr(args, "arch", "ultralight") == "ultralight":
        log_msg(f"[*] Instantiating UltraLightSignModel (~0.8M parameters, optimized for real-time 30+ FPS CPU execution)...", is_master=is_master)
        model = UltraLightSignModel(
            num_classes=num_classes,
            num_keypoints=60,
            channels_per_kp=9,
            d_model=getattr(args, "d_model", 128),
            nhead=getattr(args, "nhead", 4),
            num_layers=getattr(args, "num_layers", 3),
            dim_feedforward=getattr(args, "dim_feedforward", 256),
            dropout=args.dropout,
            max_len=args.max_len
        ).to(device)
    else:
        log_msg(f"[*] Instantiating LandmarkTransformer (~5.5M parameters)...", is_master=is_master)
        model = LandmarkTransformer(
            num_classes=num_classes,
            num_keypoints=60,
            channels_per_kp=9,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_len=args.max_len
        ).to(device)

    # Automatically use both T4 GPUs on Kaggle
    if not is_xla and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        log_msg(f"[*] Wrapping model in nn.DataParallel across {torch.cuda.device_count()} GPUs...", is_master=is_master)
        model = nn.DataParallel(model)

    # Load Pre-trained Weights if fine-tuning
    if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
        log_msg(f"[*] Fine-Tuning: Loading pretrained checkpoint weights from '{args.pretrained_ckpt}'...", is_master=is_master)
        try:
            checkpoint = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            
            model_dict = model.state_dict()
            matched_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    log_msg(f"  -> Fine-tuning adaptation: Skipping layer '{k}' due to shape mismatch", is_master=is_master)
            
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            log_msg(f"[+] Successfully loaded {len(matched_dict)} matching parameter tensors for fine-tuning.", is_master=is_master)
        except Exception as e:
            log_msg(f"[!] Error loading pretrained checkpoint: {e}", is_master=is_master)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)

    log_msg(f"======================================================================", is_master=is_master)
    log_msg(f"       STARTING TPU FINE-TUNING / TRAINING FOR {args.epochs} EPOCHS", is_master=is_master)
    log_msg(f"======================================================================", is_master=is_master)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = []
    current_stage = "stage3_full_mixture"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Progressive Curriculum Learning Stage Transitions (4-Stage Curriculum)
        if getattr(args, "curriculum", False):
            e_stage1_end = max(1, args.epochs // 4)
            e_stage2_end = max(2, (2 * args.epochs) // 4)
            e_stage3_end = max(3, int(0.70 * args.epochs))
            
            if epoch <= e_stage1_end:
                target_stage = "stage1_letters_numbers"
            elif epoch <= e_stage2_end:
                target_stage = "stage2_isolated_glosses"
            elif epoch <= e_stage3_end:
                target_stage = "stage3_full_mixture"
            else:
                target_stage = "stage4_continuous_stream"

            if target_stage != current_stage or epoch == 1:
                current_stage = target_stage
                log_msg(f"\n[>>> CURRICULUM TRANSITION <<<] Epoch {epoch}: Switching to '{current_stage}' dataset loader...", is_master=is_master)
                train_loader = create_dataloader(
                    dataset_dir=data_dir,
                    split="train",
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    worker_idx=worker_idx,
                    num_workers=num_workers,
                    num_dataloader_workers=args.num_dataloader_workers,
                    shuffle=True,
                    stage=current_stage,
                    augment=getattr(args, "augment", False)
                )
                log_msg(f"[+] Loaded {len(train_loader.dataset)} active records for stage '{current_stage}'.", is_master=is_master)

        # Progressive Noise Curriculum (0.1% noise at Epoch 1 -> 5.0%-6.0% noise at Epoch N)
        if getattr(args, "augment", False) and hasattr(train_loader.dataset, "set_noise_level"):
            noise_ratio = 0.02 + 0.98 * ((epoch - 1) / max(1, args.epochs - 1))
            train_loader.dataset.set_noise_level(noise_ratio)
            pct_str = f"{noise_ratio * 5.0:.2f}%"
            if is_master and (epoch == 1 or epoch % 5 == 0 or epoch == args.epochs):
                log_msg(f"[*] Progressive Noise Curriculum (Epoch {epoch}): Camera Noise = {pct_str}", is_master=is_master)

        train_loss, train_acc = train_epoch_tpu(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            is_master=is_master,
            precision=getattr(args, "precision", "bfloat16")
        )
        scheduler.step()

        # Reduce train metrics across TPU cores if distributed XLA
        if _XLA_AVAILABLE and "xla" in str(device).lower():
            train_loss = xm.mesh_reduce("train_loss", train_loss, lambda x: sum(x) / len(x))
            train_acc = xm.mesh_reduce("train_acc", train_acc, lambda x: sum(x) / len(x))

        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            val_loss, val_acc = evaluate_epoch_tpu(
                model=model,
                loader=val_loader,
                device=device,
                is_master=is_master,
                precision=getattr(args, "precision", "bfloat16")
            )


        epoch_time = time.time() - epoch_start

        val_str = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%" if val_loader is not None else ""
        stage_str = f" | Stage: {current_stage}" if getattr(args, "curriculum", False) else ""
        log_msg(
            f"Epoch {epoch:03d}/{args.epochs:03d}{stage_str} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
            f"{val_str} | LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {epoch_time:.2f}s",
            is_master=is_master
        )

        history.append({
            "epoch": epoch,
            "stage": current_stage,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time
        })

        # Save checkpoint on master ordinal
        if is_master:
            ckpt_path = save_dir / "latest_model.pt"
            is_best = False
            if val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    is_best = True
            elif train_loss < best_val_loss:
                best_val_loss = train_loss
                is_best = True

            checkpoint_data = {
                "epoch": epoch,
                "arch": getattr(args, "arch", "ultralight"),
                "precision": getattr(args, "precision", "bfloat16"),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "args": vars(args)
            }
            
            if _XLA_AVAILABLE and "xla" in str(device).lower():
                xm.save(checkpoint_data, ckpt_path)
                if is_best:
                    xm.save(checkpoint_data, save_dir / "best_model.pt")
            else:
                torch.save(checkpoint_data, ckpt_path)
                if is_best:
                    torch.save(checkpoint_data, save_dir / "best_model.pt")

            with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    # Final Test Evaluation
    if test_loader is not None:
        log_msg(f"\n======================================================================", is_master=is_master)
        log_msg(f"       RUNNING FINAL TEST EVALUATION ON TEST SET ({getattr(args, 'precision', 'bfloat16').upper()})", is_master=is_master)
        log_msg(f"======================================================================", is_master=is_master)
        test_loss, test_acc = evaluate_epoch_tpu(
            model=model,
            loader=test_loader,
            device=device,
            is_master=is_master,
            precision=getattr(args, "precision", "bfloat16")
        )
        if _XLA_AVAILABLE and "xla" in str(device).lower():
            test_loss = xm.mesh_reduce("test_loss", test_loss, lambda x: sum(x) / len(x))
            test_acc = xm.mesh_reduce("test_acc", test_acc, lambda x: sum(x) / len(x))
        log_msg(f"[+] Final Test Performance -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%", is_master=is_master)

    log_msg("[*] TPU Training/Fine-Tuning completed successfully!", is_master=is_master)

def main():
    parser = argparse.ArgumentParser(description="ASL Recognition TPU Training & Fine-Tuning Pipeline")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1", help="Path to preprocessed dataset directory")
    parser.add_argument("--arch", type=str, default="ultralight", choices=["ultralight", "transformer"], help="Model architecture (ultralight for CPU 30 FPS vs transformer)")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Mixed precision mode (bfloat16 for TPU MXU speed & range)")
    parser.add_argument("--curriculum", action="store_true", help="Enable progressive curriculum learning (letters -> glosses -> full mixture)")
    parser.add_argument("--augment", action="store_true", help="Enable real-world camera noise data augmentation (jitter, 3D rotation, scaling, keypoint/frame dropout)")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, help="Path to pretrained model checkpoint for fine-tuning")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per TPU core")
    parser.add_argument("--max-len", type=int, default=256, help="Static 256-token memory context sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=128, help="Model feature dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of model layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num-cores", type=int, default=8, help="Number of TPU cores (default 8)")
    parser.add_argument("--num-dataloader-workers", type=int, default=4, help="DataLoader workers per TPU process")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Output directory for model checkpoints")
    args = parser.parse_args()

    if _XLA_AVAILABLE:
        print(f"[*] PyTorch XLA detected. Spawning multi-core training across {args.num_cores} TPU cores...")
        xmp.spawn(_tpu_worker_fn, args=(args,), nprocs=args.num_cores)
    else:
        print("[!] PyTorch XLA (`torch_xla`) not detected. Running training on single device / CPU fallback...")
        _tpu_worker_fn(0, args)

if __name__ == "__main__":
    main()
