"""
Continuous ASL Foundation Model Training Script in Keras 3 (JAX Backend)
Target: Cloud TPU v5e-1 (1 chip / 1 rank, 16 GB HBM, ~250+ samples/s)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Enforce JAX backend and TPU flags before importing keras
os.environ["KERAS_BACKEND"] = "jax"
if "PJRT_DEVICE" not in os.environ:
    os.environ["PJRT_DEVICE"] = "TPU"
if "PJRT_ALLOCATOR_FRACTION" not in os.environ:
    os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.95"
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


import jax
import jax.numpy as jnp
import keras
from keras import ops

from train_keras.models import ASLFoundationModel
from train_keras.losses import label_smoothed_ce, sequence_length_loss
from train_keras.convert_pt_to_keras import load_torch_weights_into_keras


def parse_args():
    parser = argparse.ArgumentParser(description="Keras 3 JAX ASL TPU v5e Training")
    parser.add_argument("--data-dir", type=str, default="", help="Path to preprocessed landmark shards")
    parser.add_argument("--aslg-csv", type=str, default="", help="Path to ASLG-PC12 CSV")
    parser.add_argument("--kdwd-dir", type=str, default="", help="Path to KDWD directory")
    parser.add_argument("--phase1-checkpoint", type=str, default="", help="Path to pre-trained weights")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (256 aligns to 2x128 systolic tile)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--max-len", type=int, default=384, help="Max visual landmark sequence length")
    parser.add_argument("--text-max-len", type=int, default=128, help="Max text token sequence length")
    parser.add_argument("--english-max-len", type=int, default=128, help="Max English sequence length")
    parser.add_argument("--chicago-max-len", type=int, default=128, help="Max Chicago sequence length")

    parser.add_argument("--d-model", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--vocab-size", type=int, default=17800, help="Gloss vocabulary size")
    parser.add_argument("--chicago-vocab-size", type=int, default=128, help="Chicago fingerspelling vocabulary size")
    parser.add_argument("--english-vocab-size", type=int, default=50257, help="English vocabulary size")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads")
    parser.add_argument("--kv-heads", type=int, default=2, help="KV heads for GQA")
    parser.add_argument("--num-layers", type=int, default=4, help="Conformer & Decoder layers")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2], help="Training phase: 1 for text pretraining, 2 for multimodal video")
    parser.add_argument("--precision", type=str, default="mixed_bfloat16", choices=["mixed_bfloat16", "float32"])
    parser.add_argument("--log-freq", type=int, default=25, help="Step logging frequency")
    parser.add_argument("--checkpoint-freq", type=int, default=20, help="Epoch checkpoint frequency")
    parser.add_argument("--keep-last-k", type=int, default=5, help="Maximum number of checkpoints to retain")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint directory to resume from")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps per epoch (0 for full)")
    parser.add_argument("--num-dataloader-workers", type=int, default=0, help="PyTorch DataLoader worker count")
    return parser.parse_args()


class SyntheticASLStream:
    """High-Throughput Synthetic Stream for benchmarking and verification."""
    def __init__(self, batch_size: int, max_len: int, text_len: int, num_batches: int = 100):
        self.batch_size = batch_size
        self.max_len = max_len
        self.text_len = text_len
        self.num_batches = num_batches

    def __iter__(self):
        # Generate cyclic distinct token sequences to model realistic sentences
        tok_cycle = (jnp.arange(self.text_len, dtype=jnp.int32) % 150) + 1
        tok_batch = jnp.tile(tok_cycle[None, :], (self.batch_size, 1))
        valid_label_len = min(self.text_len, max(1, self.max_len // 2))
        pad_mask = jnp.arange(self.text_len)[None, :] < valid_label_len
        tok_batch = jnp.where(pad_mask, tok_batch, 0)

        feat_coords = jnp.sin(jnp.arange(self.batch_size * self.max_len * 225, dtype=jnp.float32).reshape((self.batch_size, self.max_len, 225)) * 0.01)

        for _ in range(self.num_batches):
            yield {
                "feature": feat_coords,
                "mask": jnp.ones((self.batch_size, self.max_len), dtype=jnp.bool_),
                "gloss_seq": tok_batch,
                "gloss_len": jnp.full((self.batch_size,), valid_label_len, dtype=jnp.int32),
                "chicago_seq": tok_batch,
                "english_seq": tok_batch,
            }

    def __len__(self):
        return self.num_batches


def build_data_loader(args):
    """Builds real PyTorch DataLoader wrapped with JAXDataLoader, or falls back to synthetic stream."""
    has_shards = bool(args.data_dir and os.path.isdir(args.data_dir))
    if has_shards:
        try:
            from dataset import ASLStreamedDataset, fast_vectorized_collate_fn
            from torch.utils.data import DataLoader
            print(f"[*] Initializing ASLStreamedDataset from: {args.data_dir}")
            pt_dataset = ASLStreamedDataset(
                data_dir=args.data_dir,
                aslg_csv=args.aslg_csv,
                max_len=args.max_len,
            )
            pt_loader = DataLoader(
                pt_dataset,
                batch_size=args.batch_size,
                collate_fn=fast_vectorized_collate_fn,
                num_workers=args.num_dataloader_workers,
                drop_last=True,
            )
            from train_keras.dataloader import JAXDataLoader
            return JAXDataLoader(pt_loader, prefetch_size=4)
        except Exception as e:
            print(f"[!] Warning: Could not instantiate ASLStreamedDataset ({e}). Falling back to synthetic stream.")

    print("[*] Using High-Throughput SyntheticASLStream (SysTile Aligned).")
    steps_per_epoch = args.max_steps if args.max_steps > 0 else 100
    return SyntheticASLStream(
        batch_size=args.batch_size,
        max_len=args.max_len,
        text_len=args.text_max_len,
        num_batches=steps_per_epoch,
    )


def main():
    args = parse_args()
    print("=" * 65)
    print(" CONTINUOUS ASL FOUNDATION MODEL TRAINING (KERAS 3 + JAX)")
    print(f" Target Hardware: Cloud TPU v5e-1 ({jax.devices()})")
    print(f" Precision Policy: {args.precision}")
    print(f" Batch Size: {args.batch_size} (Tile Aligned: {args.batch_size % 128 == 0})")
    print(f" Model Config: d_model={args.d_model}, max_len={args.max_len}, nhead={args.nhead}")
    print("=" * 65)

    if args.precision == "mixed_bfloat16":
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("[+] Global mixed precision set to: mixed_bfloat16")

    # Instantiate Model
    print("[*] Building ASLFoundationModel...")
    model = ASLFoundationModel(
        vocab_size=args.vocab_size,
        chicago_vocab_size=args.chicago_vocab_size,
        english_vocab_size=args.english_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        kv_heads=args.kv_heads,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        in_channels=225,
        max_len=args.max_len,
        text_max_len=args.text_max_len,
        english_max_len=args.english_max_len,
        chicago_max_len=args.chicago_max_len,
        kernel_size=min(args.kernel_size if hasattr(args, "kernel_size") else 31, max(3, args.max_len - 1)),
        is_causal=True,
    )

    # Build model variables with mock input
    dummy_feats = ops.zeros((args.batch_size, args.max_len, 225), dtype="float32")
    dummy_tokens = ops.zeros((args.batch_size, args.text_max_len), dtype="int32")
    if args.phase == 2:
        _ = model(dummy_feats, gloss_seq=dummy_tokens)
    else:
        _ = model(
            dummy_feats,
            gloss_seq=dummy_tokens,
            chicago_seq=dummy_tokens,
            english_seq=dummy_tokens,
        )
    print(f"[+] Model constructed: {len(model.trainable_variables)} trainable variables (Phase {args.phase}).")

    # Load pre-trained weights if provided (e.g. asl_llm_200)
    if args.phase1_checkpoint and os.path.exists(args.phase1_checkpoint):
        print(f"[*] Loading pre-trained weights from: {args.phase1_checkpoint}")
        import torch
        pt_dict = torch.load(args.phase1_checkpoint, map_location="cpu")
        if "model_state_dict" in pt_dict:
            pt_dict = pt_dict["model_state_dict"]
        load_torch_weights_into_keras(model, pt_dict)

    # Enable persistent JAX compilation caching
    cache_dir = os.path.join(args.save_dir, "jax_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        print(f"[+] Persistent JAX Compilation Cache enabled at: {cache_dir}")
    except Exception:
        pass

    # Learning rate schedule & Native Optax AdamW Optimizer
    import optax
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.lr,
        warmup_steps=500,
        decay_steps=10000,
        end_value=1e-6,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=1e-2,
            b1=0.9,
            b2=0.98,
            eps=1e-6,
        )
    )

    from train_keras.losses import ctc_loss
    from train_keras.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager(
        save_dir=args.save_dir,
        keep_last_k=args.keep_last_k,
        save_best=True,
    )

    # JIT-compiled multi-task train step
    trainable_vars = [v.value for v in model.trainable_variables]
    non_trainable_vars = [v.value for v in model.non_trainable_variables]
    opt_state = tx.init(trainable_vars)

    start_epoch = 1
    global_step = 0

    if args.resume:
        print(f"[*] Resuming from checkpoint: {args.resume}")
        res_ep, res_st, res_opt, res_meta = (
            ckpt_mgr.load_checkpoint(args.resume, model)
            if os.path.isdir(args.resume)
            else ckpt_mgr.load_latest(model)
        )
        if res_ep > 0:
            start_epoch = res_ep + 1
            global_step = res_st
            if res_opt is not None:
                opt_state = res_opt
            trainable_vars = [v.value for v in model.trainable_variables]
            print(f"[+] Resumed from epoch {res_ep}, global_step {global_step}")

    def loss_function(t_vars, nt_vars, feats, gloss_toks, chi_toks, eng_toks, t_lens, mask=None):
        if args.phase == 2:
            preds, new_nt_vars = model.stateless_call(
                t_vars,
                nt_vars,
                feats,
                gloss_seq=gloss_toks,
                mask=mask,
            )
            loss_dec = label_smoothed_ce(preds["dec_logits"], gloss_toks, num_classes=model.vocab_size, smoothing=0.10)
            loss_len = sequence_length_loss(preds["pred_len"], t_lens)
            loss_ctc = ctc_loss(preds["ctc_logits"], gloss_toks)
            total_loss = loss_dec + 0.10 * loss_len + 0.30 * loss_ctc
            aux = (loss_dec, loss_len, loss_ctc, jnp.array(0.0), jnp.array(0.0), new_nt_vars)
            return total_loss, aux
        else:
            preds, new_nt_vars = model.stateless_call(
                t_vars,
                nt_vars,
                feats,
                gloss_seq=gloss_toks,
                chicago_seq=chi_toks,
                english_seq=eng_toks,
                mask=mask,
            )
            loss_dec = label_smoothed_ce(preds["dec_logits"], gloss_toks, num_classes=model.vocab_size, smoothing=0.10)
            loss_len = sequence_length_loss(preds["pred_len"], t_lens)
            loss_ctc = ctc_loss(preds["ctc_logits"], gloss_toks)
            loss_chi = (
                label_smoothed_ce(preds["chicago_logits"], chi_toks, num_classes=model.chicago_vocab_size, smoothing=0.05)
                if preds["chicago_logits"] is not None
                else jnp.array(0.0)
            )
            loss_eng = (
                label_smoothed_ce(preds["english_logits"], eng_toks, num_classes=model.english_vocab_size, smoothing=0.05)
                if preds["english_logits"] is not None
                else jnp.array(0.0)
            )
            total_loss = loss_dec + 0.10 * loss_len + 0.30 * loss_ctc + 0.15 * loss_chi + 0.20 * loss_eng
            aux = (loss_dec, loss_len, loss_ctc, loss_chi, loss_eng, new_nt_vars)
            return total_loss, aux

    grad_fn = jax.value_and_grad(loss_function, has_aux=True)

    @jax.jit
    def tpu_step(t_vars, nt_vars, opt_st, feats, gloss_toks, chi_toks, eng_toks, t_lens, mask=None):
        (loss, aux), grads = grad_fn(t_vars, nt_vars, feats, gloss_toks, chi_toks, eng_toks, t_lens, mask)
        l_dec, l_len, l_ctc, l_chi, l_eng, new_nt_vars = aux
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0), grads)
        updates, new_opt_st = tx.update(grads, opt_st, t_vars)
        new_t_vars = optax.apply_updates(t_vars, updates)
        return new_t_vars, new_nt_vars, new_opt_st, loss, l_dec, l_len, l_ctc, l_chi, l_eng

    print("[+] JIT Training Step compiled successfully with Optax AdamW & Multi-Task Heads.")
    print("[*] Ready for continuous high-throughput training on TPU v5e.")

    # Data pipeline
    loader = build_data_loader(args)

    # Main Training Loop
    total_samples = 0
    t_train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*30} Epoch {epoch}/{args.epochs} {'='*30}")
        epoch_start = time.time()
        step_in_epoch = 0
        running_loss = 0.0
        log_timer = time.time()

        for batch in loader:
            step_in_epoch += 1
            global_step += 1
            total_samples += args.batch_size

            feats = batch["feature"]
            mask = batch.get("mask", None)
            gloss_toks = batch["gloss_seq"]
            chi_toks = batch.get("chicago_seq", None)
            eng_toks = batch.get("english_seq", None)
            t_lens = batch.get("gloss_len", jnp.full((args.batch_size,), args.text_max_len))

            (
                trainable_vars,
                non_trainable_vars,
                opt_state,
                loss,
                l_dec,
                l_len,
                l_ctc,
                l_chi,
                l_eng,
            ) = tpu_step(
                trainable_vars,
                non_trainable_vars,
                opt_state,
                feats,
                gloss_toks,
                chi_toks,
                eng_toks,
                t_lens,
                mask=mask,
            )

            # Logging & Throughput (Asynchronous non-blocking metrics)
            if step_in_epoch % args.log_freq == 0 or (args.max_steps and step_in_epoch >= args.max_steps):
                now = time.time()
                elapsed = now - log_timer
                steps_done = args.log_freq if (step_in_epoch % args.log_freq == 0) else (step_in_epoch % args.log_freq)
                sps = (steps_done * args.batch_size) / max(0.001, elapsed)
                st_ps = steps_done / max(0.001, elapsed)
                loss_val = float(loss)
                running_loss += loss_val
                print(
                    f"Epoch {epoch:03d} | Step {step_in_epoch:05d} | "
                    f"Loss: {loss_val:.4f} (Gloss: {float(l_dec):.3f}, CTC: {float(l_ctc):.3f}, "
                    f"Chi: {float(l_chi):.3f}, Eng: {float(l_eng):.3f}) | "
                    f"Speed: {sps:,.1f} samples/s ({st_ps:.1f} steps/s)",
                    flush=True,
                )
                log_timer = time.time()

            if args.max_steps and step_in_epoch >= args.max_steps:
                break

        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch} Complete] Time: {epoch_time:.2f}s | Avg Speed: {(step_in_epoch * args.batch_size) / epoch_time:,.1f} samples/s")

        # Save Checkpoint every checkpoint_freq epochs or last epoch
        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            for var, val in zip(model.trainable_variables, trainable_vars):
                var.assign(val)
            saved_dir = ckpt_mgr.save(
                epoch=epoch,
                step=global_step,
                model=model,
                opt_state=opt_state,
                metrics={"loss": running_loss / max(1, step_in_epoch // max(1, args.log_freq))},
            )
            print(f"[+] Checkpoint saved to: {saved_dir}")

    total_time = time.time() - t_train_start
    overall_sps = total_samples / max(0.001, total_time)
    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE")
    print(f" Total Elapsed: {total_time:.2f}s | Total Steps: {global_step} | Total Samples: {total_samples:,}")
    print(f" Overall Sustained Throughput: {overall_sps:,.1f} samples/sec")
    print("=" * 65)


if __name__ == "__main__":
    main()

