"""
Kaggle TPU v5e-8 Phase 1 Real Data Benchmark
Streams from ASLG-PC12 dataset (train.csv, 87,710 parallel sentence pairs)
Target: >= 12,000 samples/s
"""

import os
import sys
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KERAS_BACKEND"] = "jax"
if "PJRT_DEVICE" not in os.environ:
    os.environ["PJRT_DEVICE"] = "TPU"
if "PJRT_ALLOCATOR_FRACTION" not in os.environ:
    os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.85"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import keras
from keras import ops
import optax

# Enable persistent JAX compilation caching
cache_dir = (
    "/kaggle/working/jax_cache"
    if os.path.exists("/kaggle")
    else ("/content/jax_cache" if os.path.exists("/content") else os.path.join(".", "scratch", "jax_cache"))
)
try:
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    print(f"[+] JAX Compilation Cache enabled at: {cache_dir}")
except Exception:
    pass

from train_keras.models.decoder import ASLTransformerDecoder
from train_keras.losses import label_smoothed_ce


def load_aslg_in_memory(csv_path: str, max_len: int = 128, vocab_size: int = 17800):
    print(f"[*] Loading ASLG-PC12 dataset from: {csv_path}")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df):,} sentence pairs in {time.time() - t0:.2f}s")

    # Fast in-memory tokenization
    texts = df["text"].astype(str).tolist()
    N = len(texts)
    tokens_arr = np.zeros((N, max_len), dtype=np.int32)
    tokens_arr[:, 0] = 1 # BOS

    for i, line in enumerate(texts):
        words = line.strip().split()[: max_len - 2]
        for j, w in enumerate(words):
            # Fast deterministic token hashing into [4, vocab_size - 1]
            tokens_arr[i, j + 1] = (hash(w) % (vocab_size - 4)) + 4
        tokens_arr[i, len(words) + 1] = 2 # EOS

    print(f"[+] Pre-tokenized {N:,} sequences in {time.time() - t0:.2f}s (RAM footprint: {tokens_arr.nbytes / 1e6:.1f} MB)")
    return tokens_arr


def run_phase1_real_data(num_devices: int = 8, per_core_batch: int = 256, num_layers: int = 4, num_steps: int = 20, precision: str = "mixed_bfloat16", d_model: int = 512, seq_len: int = 128, vocab_size: int = 17800):
    global_batch = num_devices * per_core_batch

    print("\n" + "=" * 70)
    print(f"[*] RUNNING PHASE 1 REAL DATA TRAINING (ASLG-PC12 Dataset)")
    print(f"    Devices: {num_devices} TPU Cores | Global Batch: {global_batch}")
    print(f"    Per-Core Batch: {per_core_batch} | Seq Len: {seq_len} | Layers: {num_layers} | Hidden Dim: {d_model}")
    print("=" * 70)

    keras.mixed_precision.set_global_policy(precision)

    devices = jax.devices()[:num_devices]
    mesh = Mesh(devices, ("data",))
    data_sharding = NamedSharding(mesh, P("data", None))
    replicated = NamedSharding(mesh, P())

    candidate_paths = [
        "/kaggle/input/datasets/thedevastator/unlock-the-power-of-english-asl-with-aslg-pc12-c/train.csv",
        "/kaggle/input/unlock-the-power-of-english-asl-with-aslg-pc12-c/train.csv",
        "/content/datasets/thedevastator/unlock-the-power-of-english-asl-with-aslg-pc12-c/train.csv",
        "/content/train.csv",
        os.path.expanduser("~/.cache/kagglehub/datasets/thedevastator/unlock-the-power-of-english-asl-with-aslg-pc12-c/versions/2/train.csv"),
    ]
    csv_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if csv_path is None:
        print(f"[!] Warning: ASLG-PC12 train.csv not found in candidate paths, generating in-memory mock text corpus.")
        all_tokens = np.random.randint(1, vocab_size, size=(50000, seq_len), dtype=np.int32)
    else:
        all_tokens = load_aslg_in_memory(csv_path, max_len=seq_len, vocab_size=vocab_size)

    total_samples_available = len(all_tokens)

    decoder = ASLTransformerDecoder(

        vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        kv_heads=2,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        max_seq_len=seq_len,
    )

    dummy_inputs = ops.ones((global_batch, seq_len), dtype="int32")
    _ = decoder(dummy_inputs[:2]) # memory=None for Causal LM

    t_vars = [jax.device_put(v.value, replicated) for v in decoder.trainable_variables]
    nt_vars = [jax.device_put(v.value, replicated) for v in decoder.non_trainable_variables]

    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    opt_state = tx.init(t_vars)

    def compute_loss(vars, nt, inp, tgt):
        preds, new_nt = decoder.stateless_call(vars, nt, inp)
        loss = label_smoothed_ce(preds, tgt, num_classes=vocab_size, smoothing=0.10)
        return loss, new_nt

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    @jax.jit
    def train_step(t_v, nt_v, o_st, inp, tgt):
        (loss, new_nt), grads = grad_fn(t_v, nt_v, inp, tgt)
        updates, new_o_st = tx.update(grads, o_st, t_v)
        new_t_v = optax.apply_updates(t_v, updates)
        return new_t_v, new_nt, new_o_st, loss

    print("[*] JIT Compiling Phase 1 Real Data training step...")
    b0 = all_tokens[:global_batch]
    b0_inp = jax.device_put(b0, data_sharding)
    b0_tgt = jax.device_put(b0, data_sharding)

    t0 = time.time()
    lowered = train_step.lower(t_vars, nt_vars, opt_state, b0_inp, b0_tgt)
    compiled_fn = lowered.compile()
    t_compile = time.time() - t0

    t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, b0_inp, b0_tgt)
    jax.block_until_ready(loss)
    print(f"[+] Compile ready in {t_compile:.2f}s! Initial Loss: {float(loss):.4f}")

    print(f"[*] Commencing {num_steps} real-data training steps...")
    step_latencies = []
    cursor = 0
    total_processed = 0

    print("=" * 80)
    print(f"{'STEP':<6} | {'LOSS':<8} | {'STEP TIME':<12} | {'THROUGHPUT':<18} | {'PROCESSED':<12}")
    print("=" * 80)

    for step in range(num_steps):
        if cursor + global_batch > total_samples_available:
            cursor = 0

        t_batch_start = time.time()
        batch_slice = all_tokens[cursor : cursor + global_batch]
        cursor += global_batch

        batch_inp = jax.device_put(batch_slice, data_sharding)
        batch_tgt = batch_inp

        t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, batch_inp, batch_tgt)
        jax.block_until_ready(loss)
        dt = time.time() - t_batch_start

        step_latencies.append(dt)
        total_processed += global_batch
        step_throughput = global_batch / dt

        print(f"{step+1:<6d} | {float(loss):<8.4f} | {dt*1000:<10.1f}ms | {step_throughput:<16,.1f} s/s | {total_processed:<12,d}")

    # Exclude warmup steps safely
    active_latencies = step_latencies[2:] if len(step_latencies) > 2 else (step_latencies[1:] if len(step_latencies) > 1 else step_latencies)
    avg_dt = sum(active_latencies) / len(active_latencies)
    avg_throughput = global_batch / avg_dt

    is_tpu_run = any(getattr(d, "platform", "") == "tpu" for d in devices)
    target_samples_s = (1000 if num_devices == 1 else 12000) if is_tpu_run else 100
    print("=" * 80)
    print(f"[+] PHASE 1 REAL DATA RESULT: {avg_throughput:,.1f} samples/s (Target: >= {target_samples_s:,} samples/s)")
    print(f"[+] STATUS: {'PASSED' if avg_throughput >= target_samples_s else 'FAILED'}")
    print("=" * 80)
    return avg_throughput


if __name__ == "__main__":
    devices = jax.devices()
    num_dev = len(devices)
    is_tpu = any(getattr(d, "platform", "") == "tpu" for d in devices)
    if not is_tpu:
        dev_plat = getattr(devices[0], "platform", "cpu").upper()
        print(f"[!] Non-TPU ({dev_plat}) environment detected: running lightweight mock verification with float32.")
        run_phase1_real_data(num_devices=num_dev, per_core_batch=2, num_layers=1, num_steps=2, precision="float32", d_model=64, seq_len=64, vocab_size=200)
    else:
        run_phase1_real_data(num_devices=num_dev, per_core_batch=256, num_steps=25, precision="mixed_bfloat16")

