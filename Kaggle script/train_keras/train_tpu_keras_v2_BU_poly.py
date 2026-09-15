"""
Next-Generation Continuous ASL Foundation Model V2 Training Script in Keras 3 (JAX Backend)
Target: Cloud TPU v5e-1 (1 chip / 1 rank, 16 GB HBM, ~250-500+ samples/s)
Supports:
  - Full V2 Dual-Stream Multimodal (Kinematics 540 + Phonology 19 + Visual ROI 256x256)
  - Hybrid Conformer + BiMamba-2 SSM with TemporalStridedPool
  - Hardware-fused label-smoothed cross entropy, sequence length loss, and CTC
  - Multi-task homoscedastic uncertainty weighting
  - Zero-copy DLPack data streaming with JAXDataLoader
  - JAX in-place buffer donation (donate_argnums=(0, 2)) for maximum TPU memory efficiency
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Enforce JAX backend and TPU memory flags before importing keras
os.environ["KERAS_BACKEND"] = "jax"
if "PJRT_DEVICE" not in os.environ:
    os.environ["PJRT_DEVICE"] = "TPU"
if "PJRT_ALLOCATOR_FRACTION" not in os.environ:
    os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.95"
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
import keras
from keras import ops

# Enable persistent JAX compilation caching
cache_dir = (
    "/kaggle/working/jax_cache"
    if os.path.exists("/kaggle")
    else ("/content/jax_cache" if os.path.exists("/content") else os.path.join(".", "scratch", "jax_cache"))
)
try:
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    print(f"[+] Persistent JAX Compilation Cache enabled at: {cache_dir}")
except Exception:
    pass

from train_keras.models import ASLFoundationModelV2
from train_keras.losses import label_smoothed_ce, sequence_length_loss, ctc_loss, HomoscedasticLossWrapper
from train_keras.dataloader import V4ShardedDataset, JAXDataLoader, fast_v4_collate_fn, SyntheticMultimodalStream
from train_keras.checkpoint import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Keras 3 JAX ASL TPU v5e V2 Training")
    parser.add_argument("--data-dir", type=str, default="", help="Path to preprocessor_v4 shards")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_v2", help="Save directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (must be multiple of 128)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--max-len", type=int, default=256, help="Max visual landmark sequence length (multiple of 128)")
    parser.add_argument("--text-max-len", type=int, default=128, help="Max text token sequence length (multiple of 128)")
    parser.add_argument("--d-model", type=int, default=512, help="Encoder/Decoder hidden dimension")
    parser.add_argument("--vocab-size", type=int, default=17800, help="Gloss vocabulary size")
    parser.add_argument("--chicago-vocab-size", type=int, default=128, help="Chicago fingerspelling vocabulary size")
    parser.add_argument("--english-vocab-size", type=int, default=50257, help="English vocabulary size")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads")
    parser.add_argument("--kv-heads", type=int, default=2, help="KV heads for GQA")
    parser.add_argument("--num-enc-layers", type=int, default=6, help="Conformer + BiMamba layers")
    parser.add_argument("--num-dec-layers", type=int, default=4, help="Transformer decoder layers")
    parser.add_argument("--use-mamba", action="store_true", default=True, help="Enable BiMamba-2 SSM block")
    parser.add_argument("--include-roi", action="store_true", default=False, help="Include 256x256 upper-body ROI crops")
    parser.add_argument("--tie-embeddings", action="store_true", default=False, help="Tie decoder word embeddings to lm_head projection to save 35M+ parameters")
    parser.add_argument("--enable-aux-decoders", action="store_true", default=False, help="Enable Chicago and English decoders (default False for 3x faster Phase 1 pretraining)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps via optax.MultiSteps (default 1)")
    parser.add_argument("--precision", type=str, default="mixed_bfloat16", choices=["mixed_bfloat16", "float32"])
    parser.add_argument("--log-freq", type=int, default=25, help="Step logging frequency")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--keep-last-k", type=int, default=5, help="Max checkpoints to retain")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps per epoch (0 for full)")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint directory to resume from")
    return parser.parse_args()


def build_data_loader(args):
    """Builds real V4ShardedDataset loader or falls back to synthetic multimodal stream."""
    has_shards = bool(args.data_dir and os.path.isdir(args.data_dir))
    if has_shards:
        try:
            from torch.utils.data import DataLoader
            print(f"[*] Initializing V4ShardedDataset from: {args.data_dir}")
            pt_dataset = V4ShardedDataset(
                shard_dir=args.data_dir,
                max_len=args.max_len,
                text_max_len=args.text_max_len,
                shuffle=True,
                split="train",
            )
            pt_loader = DataLoader(
                pt_dataset,
                batch_size=args.batch_size,
                collate_fn=fast_v4_collate_fn,
                num_workers=0,
                drop_last=True,
            )
            return JAXDataLoader(pt_loader, prefetch_size=4)
        except Exception as e:
            print(f"[!] Warning: Could not instantiate V4ShardedDataset ({e}). Falling back to synthetic stream.")

    print(f"[*] Using High-Throughput SyntheticMultimodalStream (SysTile Aligned B={args.batch_size}, T={args.max_len}).")
    steps = args.max_steps if args.max_steps > 0 else 100
    return SyntheticMultimodalStream(
        batch_size=args.batch_size,
        max_len=args.max_len,
        text_len=args.text_max_len,
        num_batches=steps,
        include_roi=args.include_roi,
    )


def main():
    args = parse_args()
    print("=" * 70)
    print(" CONTINUOUS ASL FOUNDATION MODEL V2 TRAINING (KERAS 3 + JAX)")
    print(f" Target Hardware: Cloud TPU v5e-1 ({jax.devices()})")
    print(f" Precision: {args.precision} | Batch Size: {args.batch_size} (128-Aligned: {args.batch_size % 128 == 0})")
    print(f" Architecture: d_model={args.d_model}, EncLayers={args.num_enc_layers}, DecLayers={args.num_dec_layers}")
    print(f" SOTA Modules: BiMamba={args.use_mamba}, TemporalStridedPool=True, Multimodal ROI={args.include_roi}")
    print("=" * 70)

    if args.precision == "mixed_bfloat16":
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("[+] Global mixed precision policy set to: mixed_bfloat16")

    if hasattr(keras.config, "enable_flash_attention"):
        try:
            keras.config.enable_flash_attention()
            print("[+] Native Keras 3 FlashAttention enabled.")
        except Exception:
            pass

    # Instantiate V2 Model
    print("[*] Constructing ASLFoundationModelV2...")
    model = ASLFoundationModelV2(
        vocab_size=args.vocab_size,
        chicago_vocab_size=args.chicago_vocab_size,
        english_vocab_size=args.english_vocab_size,
        num_keypoints=60,
        channels_per_kp=9,
        d_enc=args.d_model,
        nhead_enc=args.nhead,
        num_enc_layers=args.num_enc_layers,
        ffn_enc=args.d_model * 4,
        d_dec=args.d_model,
        nhead_dec=args.nhead,
        kv_heads_dec=args.kv_heads,
        num_dec_layers=args.num_dec_layers,
        ffn_dec=args.d_model * 4,
        max_enc_len=args.max_len,
        text_max_len=args.text_max_len,
        english_max_len=args.text_max_len,
        chicago_max_len=args.text_max_len,
        use_mamba=args.use_mamba,
        enable_aux_decoders=args.enable_aux_decoders,
        tie_word_embeddings=args.tie_embeddings,
    )

    # Multi-task loss wrapper
    loss_wrapper = HomoscedasticLossWrapper()

    # Build model variables
    dummy_feats = ops.zeros((args.batch_size, args.max_len, 60, 9), dtype="float32")
    dummy_phon = ops.zeros((args.batch_size, args.max_len, 19), dtype="float32")
    dummy_tokens = ops.zeros((args.batch_size, args.text_max_len), dtype="int32")
    dummy_roi = ops.zeros((args.batch_size, args.max_len, 256, 256, 3), dtype="uint8") if args.include_roi else None

    dummy_c = dummy_tokens if args.enable_aux_decoders else None
    dummy_e = dummy_tokens if args.enable_aux_decoders else None
    _ = model(
        dummy_feats,
        phonology=dummy_phon,
        roi_visual=dummy_roi,
        gloss_seq=dummy_tokens,
        chicago_seq=dummy_c,
        english_seq=dummy_e,
    )
    _ = loss_wrapper({"dec": ops.array(1.0), "ctc": ops.array(1.0)})
    print(f"[+] Model constructed: {len(model.trainable_variables)} trainable variables in V2 Architecture (Aux Decoders: {args.enable_aux_decoders}).")

    # Optimizer & Learning Rate Schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.lr,
        warmup_steps=500,
        decay_steps=10000,
        end_value=1e-6,
    )
    tx = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=1e-2,
            b1=0.9,
            b2=0.98,
            eps=1e-6,
        ),
    )
    if args.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=args.grad_accum_steps)
        print(f"[+] Gradient accumulation enabled via optax.MultiSteps (every {args.grad_accum_steps} steps).")


    # Checkpoint Manager
    ckpt_mgr = CheckpointManager(
        save_dir=args.save_dir,
        keep_last_k=args.keep_last_k,
        save_best=True,
    )

    trainable_vars = [v.value for v in model.trainable_variables]
    non_trainable_vars = [v.value for v in model.non_trainable_variables]
    loss_wrapper_vars = [v.value for v in loss_wrapper.trainable_variables]
    opt_state = tx.init(trainable_vars + loss_wrapper_vars)

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

    # Loss computation function
    def loss_fn(t_vars_all, nt_vars, feats, phon, roi, g_toks, c_toks, e_toks, t_lens, mask):
        n_m_vars = len(model.trainable_variables)
        m_t_vars = t_vars_all[:n_m_vars]
        lw_t_vars = t_vars_all[n_m_vars:]

        preds, new_nt_vars = model.stateless_call(
            m_t_vars,
            nt_vars,
            feats,
            phonology=phon,
            roi_visual=roi,
            gloss_seq=g_toks,
            chicago_seq=c_toks,
            english_seq=e_toks,
            mask=mask,
        )

        l_dec = label_smoothed_ce(preds["dec_logits"], g_toks, num_classes=model.vocab_size, smoothing=0.10)
        l_len = sequence_length_loss(preds["pred_len"], t_lens)
        l_ctc = ctc_loss(preds["ctc_logits"], g_toks, target_lengths=t_lens)
        l_chi = (
            label_smoothed_ce(preds["chicago_logits"], c_toks, num_classes=model.chicago_vocab_size, smoothing=0.05)
            if "chicago_logits" in preds
            else ops.array(0.0)
        )
        l_eng = (
            label_smoothed_ce(preds["english_logits"], e_toks, num_classes=model.english_vocab_size, smoothing=0.05)
            if "english_logits" in preds
            else ops.array(0.0)
        )

        # Multi-task homoscedastic weighting (only active heads)
        lw_losses = {
            "dec": l_dec,
            "ctc": l_ctc,
            "len": l_len,
        }
        if "chicago_logits" in preds:
            lw_losses["chi"] = l_chi
        if "english_logits" in preds:
            lw_losses["eng"] = l_eng

        total_loss, _ = loss_wrapper.stateless_call(lw_t_vars, [], lw_losses)
        aux = (total_loss, l_dec, l_ctc, l_len, l_chi, l_eng, new_nt_vars)
        return total_loss, aux


    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    from functools import partial

    # TPU JIT step with buffer donation for zero HBM reallocation
    @partial(jax.jit, donate_argnums=(0, 2))
    def tpu_v2_step(t_vars_all, nt_vars, opt_st, feats, phon, roi, g_toks, c_toks, e_toks, t_lens, mask):
        (loss, aux), grads = grad_fn(t_vars_all, nt_vars, feats, phon, roi, g_toks, c_toks, e_toks, t_lens, mask)
        total_loss, l_dec, l_ctc, l_len, l_chi, l_eng, new_nt_vars = aux
        updates, new_opt_st = tx.update(grads, opt_st, t_vars_all)
        new_t_vars_all = optax.apply_updates(t_vars_all, updates)
        return new_t_vars_all, new_nt_vars, new_opt_st, total_loss, l_dec, l_ctc, l_len, l_chi, l_eng

    print("[+] JIT Training Step compiled with Buffer Donation (donate_argnums=(0, 2)).")
    print("[*] Ready for continuous high-throughput V2 training on Cloud TPU v5e.")

    # Data Loader
    loader = build_data_loader(args)

    # Training Loop
    total_samples = 0
    t_train_start = time.time()
    csv_path = os.path.join(args.save_dir, "training_metrics_v2.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,step,total_loss,loss_dec,loss_ctc,loss_len,loss_chi,loss_eng,samples_per_sec,elapsed_sec\n")

    combined_trainable = trainable_vars + loss_wrapper_vars

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
            phon = batch.get("phonology", None)
            roi = batch.get("roi_visual", None)
            mask = batch.get("mask", None)
            g_toks = batch["gloss_seq"]
            c_toks = batch.get("chicago_seq", None)
            e_toks = batch.get("english_seq", None)
            t_lens = batch.get("gloss_len", jnp.full((args.batch_size,), args.text_max_len))

            (
                combined_trainable,
                non_trainable_vars,
                opt_state,
                tot_l,
                l_dec,
                l_ctc,
                l_len,
                l_chi,
                l_eng,
            ) = tpu_v2_step(
                combined_trainable,
                non_trainable_vars,
                opt_state,
                feats,
                phon,
                roi,
                g_toks,
                c_toks,
                e_toks,
                t_lens,
                mask,
            )

            running_loss += float(tot_l)

            if step_in_epoch % args.log_freq == 0:
                elapsed = time.time() - log_timer
                speed = (args.batch_size * args.log_freq) / max(1e-4, elapsed)
                avg_l = running_loss / args.log_freq
                print(
                    f"  [Epoch {epoch:03d} | Step {step_in_epoch:04d}] "
                    f"Loss: {avg_l:.4f} (Dec: {float(l_dec):.3f}, CTC: {float(l_ctc):.3f}, "
                    f"Chi: {float(l_chi):.3f}, Eng: {float(l_eng):.3f}) | "
                    f"Throughput: {speed:.1f} samples/s"
                )
                running_loss = 0.0
                log_timer = time.time()

                with open(csv_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch},{global_step},{avg_l:.4f},{float(l_dec):.4f},{float(l_ctc):.4f},"
                        f"{float(l_len):.4f},{float(l_chi):.4f},{float(l_eng):.4f},{speed:.1f},"
                        f"{time.time() - t_train_start:.2f}\n"
                    )

            if args.max_steps > 0 and step_in_epoch >= args.max_steps:
                break

        ep_duration = time.time() - epoch_start
        print(f"[*] Epoch {epoch} complete in {ep_duration:.2f}s.")

        # Checkpointing
        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            n_m_vars = len(model.trainable_variables)
            model_t_vars = combined_trainable[:n_m_vars]
            for var, val in zip(model.trainable_variables, model_t_vars):
                var.assign(val)
            for var, val in zip(model.non_trainable_variables, non_trainable_vars):
                var.assign(val)
            saved_dir = ckpt_mgr.save_checkpoint(
                model=model,
                epoch=epoch,
                step=global_step,
                loss=float(tot_l),
                optimizer_state=opt_state,
                metadata={"epoch": epoch, "step": global_step, "samples": total_samples},
            )
            print(f"[+] Successfully saved checkpoint: {saved_dir}")

    print(f"\n[***] Training finished! Total samples processed: {total_samples}")


if __name__ == "__main__":
    main()
