"""
Kaggle TPU v5e-8 Speed & Compile Time Benchmark for Phase 2 and Phase 3
Target Goals:
  - Compile Time: 10-20 seconds
  - Throughput: >= 5,000 seq/s on Phase 2 & Phase 3
"""

import os
import sys
import time

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
import numpy as np

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
except Exception as e:
    print(f"[!] Warning: Could not initialize compilation cache at {cache_dir}: {e}")

from train_keras.models.conformer import MobileConformerEncoder
from train_keras.models.decoder import ASLTransformerDecoder
from train_keras.losses import label_smoothed_ce, sequence_length_loss


def benchmark_phase1(num_devices: int, per_core_batch: int = 256, num_layers: int = 4, warmup_steps: int = 2, active_steps: int = 10, d_model: int = 512, seq_len: int = 128, vocab_size: int = 17800, precision: str = "mixed_bfloat16"):
    global_batch = num_devices * per_core_batch

    print("\n" + "=" * 65)
    print(f"[*] BENCHMARKING PHASE 1 (Causal Text Pretraining / CLM)")
    print(f"    Devices: {num_devices} TPU Cores | Global Batch: {global_batch}")
    print(f"    Per-Core Batch: {per_core_batch} | Seq Len: {seq_len} | Layers: {num_layers} | Hidden Dim: {d_model}")
    print("=" * 65)

    keras.mixed_precision.set_global_policy(precision)

    devices = jax.devices()[:num_devices]
    mesh = Mesh(devices, ("data",))
    data_sharding = NamedSharding(mesh, P("data", None))
    replicated = NamedSharding(mesh, P())

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
    dummy_targets = ops.ones((global_batch, seq_len), dtype="int32")

    _ = decoder(dummy_inputs[:2])
    t_vars = [jax.device_put(v.value, replicated) for v in decoder.trainable_variables]
    nt_vars = [jax.device_put(v.value, replicated) for v in decoder.non_trainable_variables]

    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    opt_state = tx.init(t_vars)

    inputs_sharded = jax.device_put(dummy_inputs, data_sharding)
    targets_sharded = jax.device_put(dummy_targets, data_sharding)

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

    print(f"[*] JIT Compiling Phase 1 ({num_layers} Layers) SPMD graph across TPU cores...")
    t0 = time.time()
    lowered = train_step.lower(t_vars, nt_vars, opt_state, inputs_sharded, targets_sharded)
    t_lower = time.time() - t0

    t1 = time.time()
    compiled_fn = lowered.compile()
    t_comp = time.time() - t1

    t2 = time.time()
    t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, inputs_sharded, targets_sharded)
    jax.block_until_ready(loss)
    t_first = time.time() - t2
    total_compile = t_lower + t_comp

    print(f"[+] Phase 1 Pure Compile: {total_compile:.2f}s (Lowering: {t_lower:.2f}s, XLA: {t_comp:.2f}s)")
    print(f"[+] Phase 1 First Step Execution: {t_first:.2f}s")

    print(f"[*] Running {active_steps} active steps to measure steady-state throughput...")
    latencies = []
    for step in range(active_steps):
        t_s = time.time()
        t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, inputs_sharded, targets_sharded)
        jax.block_until_ready(loss)
        dt = time.time() - t_s
        latencies.append(dt)
        step_samples_s = global_batch / dt
        print(f"    Step {step+1:02d}: Loss = {float(loss):.4f} | Latency = {dt*1000:.1f}ms | Throughput = {step_samples_s:.1f} samples/s")

    avg_dt = sum(latencies[warmup_steps:]) / len(latencies[warmup_steps:])
    avg_throughput = global_batch / avg_dt
    print("-" * 65)
    print(f"[+] PHASE 1 RESULT: {avg_throughput:.1f} samples/s (Target: >= 12,000 samples/s)")
    print(f"[+] PHASE 1 COMPILE: {total_compile:.2f}s (Target: 10-20s)")
    print("-" * 65)
    return total_compile, avg_throughput


def benchmark_phase2(num_devices: int, per_core_batch: int = 256, num_layers: int = 4, warmup_steps: int = 2, active_steps: int = 10, d_model: int = 512, seq_len: int = 128, vocab_size: int = 17800, precision: str = "mixed_bfloat16"):
    global_batch = num_devices * per_core_batch

    print("\n" + "=" * 65)
    print(f"[*] BENCHMARKING PHASE 2 (Gloss Translation Decoder)")
    print(f"    Devices: {num_devices} TPU Cores | Global Batch: {global_batch}")
    print(f"    Per-Core Batch: {per_core_batch} | Seq Len: {seq_len} | Hidden Dim: {d_model}")
    print("=" * 65)

    keras.mixed_precision.set_global_policy(precision)

    devices = jax.devices()[:num_devices]
    mesh = Mesh(devices, ("data",))
    data_sharding = NamedSharding(mesh, P("data", None))
    mem_sharding = NamedSharding(mesh, P("data", None, None))
    replicated = NamedSharding(mesh, P())

    decoder = ASLTransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        kv_heads=2,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        max_seq_len=seq_len,
    )

    dummy_toks = ops.ones((global_batch, seq_len), dtype="int32")
    dummy_mem = ops.ones((global_batch, seq_len, d_model), dtype="float32")

    _ = decoder(dummy_toks[:2], memory=dummy_mem[:2])
    t_vars = [jax.device_put(v.value, replicated) for v in decoder.trainable_variables]
    nt_vars = [jax.device_put(v.value, replicated) for v in decoder.non_trainable_variables]

    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    opt_state = tx.init(t_vars)

    toks_sharded = jax.device_put(dummy_toks, data_sharding)
    mem_sharded = jax.device_put(dummy_mem, mem_sharding)

    def compute_loss(vars, nt, toks, mem):
        preds, new_nt = decoder.stateless_call(vars, nt, toks, memory=mem)
        loss = label_smoothed_ce(preds, toks, num_classes=vocab_size, smoothing=0.10)
        return loss, new_nt

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    @jax.jit
    def train_step(t_v, nt_v, o_st, toks, mem):
        (loss, new_nt), grads = grad_fn(t_v, nt_v, toks, mem)
        updates, new_o_st = tx.update(grads, o_st, t_v)
        new_t_v = optax.apply_updates(t_v, updates)
        return new_t_v, new_nt, new_o_st, loss

    print("[*] JIT Compiling Phase 2 SPMD graph across TPU cores...")
    t0 = time.time()
    lowered = train_step.lower(t_vars, nt_vars, opt_state, toks_sharded, mem_sharded)
    t_lower = time.time() - t0

    t1 = time.time()
    compiled_fn = lowered.compile()
    t_comp = time.time() - t1

    t2 = time.time()
    t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, toks_sharded, mem_sharded)
    jax.block_until_ready(loss)
    t_first = time.time() - t2
    total_compile = t_lower + t_comp

    print(f"[+] Phase 2 Pure Compile: {total_compile:.2f}s (Lowering: {t_lower:.2f}s, XLA: {t_comp:.2f}s)")
    print(f"[+] Phase 2 First Step Execution: {t_first:.2f}s")

    print(f"[*] Running {active_steps} active steps to measure steady-state throughput...")
    latencies = []
    for step in range(active_steps):
        t_s = time.time()
        t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, toks_sharded, mem_sharded)
        jax.block_until_ready(loss)
        dt = time.time() - t_s
        latencies.append(dt)
        step_seq_s = global_batch / dt
        print(f"    Step {step+1:02d}: Loss = {float(loss):.4f} | Latency = {dt*1000:.1f}ms | Throughput = {step_seq_s:.1f} seq/s")

    avg_dt = sum(latencies[warmup_steps:]) / len(latencies[warmup_steps:])
    avg_throughput = global_batch / avg_dt
    print("-" * 65)
    print(f"[+] PHASE 2 RESULT: {avg_throughput:.1f} seq/s (Target: >= 5,000 seq/s)")
    print(f"[+] PHASE 2 COMPILE: {total_compile:.2f}s (Target: 10-20s)")
    print("-" * 65)
    return total_compile, avg_throughput


def benchmark_phase3(num_devices: int, per_core_batch: int = 128, num_layers: int = 2, warmup_steps: int = 2, active_steps: int = 10, d_model: int = 512, l_land: int = 384, l_tok: int = 128, vocab_size: int = 17800, precision: str = "mixed_bfloat16"):
    global_batch = num_devices * per_core_batch

    print("\n" + "=" * 65)
    print(f"[*] BENCHMARKING PHASE 3 (Continuous ASL Foundation Model, {num_layers} Layers)")
    print(f"    Devices: {num_devices} TPU Cores | Global Batch: {global_batch}")
    print(f"    Per-Core Batch: {per_core_batch} | Landmark Len: {l_land} | Token Len: {l_tok}")
    print("=" * 65)

    keras.mixed_precision.set_global_policy(precision)

    devices = jax.devices()[:num_devices]
    mesh = Mesh(devices, ("data",))
    feat_sharding = NamedSharding(mesh, P("data", None, None))
    tok_sharding = NamedSharding(mesh, P("data", None))
    len_sharding = NamedSharding(mesh, P("data",))
    replicated = NamedSharding(mesh, P())

    class Phase3Model(keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = MobileConformerEncoder(
                num_layers=num_layers,
                d_model=d_model,
                nhead=4,
                kv_heads=2,
                dim_feedforward=d_model * 4,
                in_channels=225,
                max_len=l_land,
                kernel_size=min(31, max(3, l_land - 1)),
                is_causal=True,
            )
            self.len_head = keras.layers.Dense(1, use_bias=True)
            self.decoder = ASLTransformerDecoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=4,
                kv_heads=2,
                num_layers=num_layers,
                dim_feedforward=d_model * 4,
                max_seq_len=l_land,
            )

        def call(self, feats, toks):
            enc_out = self.encoder(feats)
            enc_mean = ops.mean(enc_out, axis=1)
            pred_len = ops.squeeze(self.len_head(enc_mean), axis=-1)
            dec_logits = self.decoder(toks, memory=enc_out)
            return {"dec_logits": dec_logits, "pred_len": pred_len}

    model = Phase3Model()
    dummy_feats = ops.ones((global_batch, l_land, 225), dtype="float32")
    dummy_toks = ops.ones((global_batch, l_tok), dtype="int32")
    dummy_lens = ops.ones((global_batch,), dtype="float32") * 16.0

    _ = model(dummy_feats[:2], dummy_toks[:2])
    t_vars = [jax.device_put(v.value, replicated) for v in model.trainable_variables]
    nt_vars = [jax.device_put(v.value, replicated) for v in model.non_trainable_variables]

    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    opt_state = tx.init(t_vars)

    feats_sharded = jax.device_put(dummy_feats, feat_sharding)
    toks_sharded = jax.device_put(dummy_toks, tok_sharding)
    lens_sharded = jax.device_put(dummy_lens, len_sharding)

    def compute_loss(vars, nt, feats, toks, lens):
        preds, new_nt = model.stateless_call(vars, nt, feats, toks)
        l_dec = label_smoothed_ce(preds["dec_logits"], toks, num_classes=vocab_size, smoothing=0.10)
        l_len = sequence_length_loss(preds["pred_len"], lens)
        return l_dec + 0.10 * l_len, new_nt

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    @jax.jit
    def train_step(t_v, nt_v, o_st, feats, toks, lens):
        (loss, new_nt), grads = grad_fn(t_v, nt_v, feats, toks, lens)
        updates, new_o_st = tx.update(grads, o_st, t_v)
        new_t_v = optax.apply_updates(t_v, updates)
        return new_t_v, new_nt, new_o_st, loss

    print(f"[*] JIT Compiling Phase 3 ({num_layers} Layers) SPMD graph across TPU cores...")
    t0 = time.time()
    lowered = train_step.lower(t_vars, nt_vars, opt_state, feats_sharded, toks_sharded, lens_sharded)
    t_lower = time.time() - t0

    t1 = time.time()
    compiled_fn = lowered.compile()
    t_comp = time.time() - t1

    t2 = time.time()
    t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, feats_sharded, toks_sharded, lens_sharded)
    jax.block_until_ready(loss)
    t_first = time.time() - t2
    total_compile = t_lower + t_comp

    print(f"[+] Phase 3 Pure Compile: {total_compile:.2f}s (Lowering: {t_lower:.2f}s, XLA: {t_comp:.2f}s)")
    print(f"[+] Phase 3 First Step Execution: {t_first:.2f}s")

    print(f"[*] Running {active_steps} active steps to measure steady-state throughput...")
    latencies = []
    for step in range(active_steps):
        t_s = time.time()
        t_vars, nt_vars, opt_state, loss = compiled_fn(t_vars, nt_vars, opt_state, feats_sharded, toks_sharded, lens_sharded)
        jax.block_until_ready(loss)
        dt = time.time() - t_s
        latencies.append(dt)
        step_seq_s = global_batch / dt
        print(f"    Step {step+1:02d}: Loss = {float(loss):.4f} | Latency = {dt*1000:.1f}ms | Throughput = {step_seq_s:.1f} seq/s")

    avg_dt = sum(latencies[warmup_steps:]) / len(latencies[warmup_steps:])
    avg_throughput = global_batch / avg_dt
    print("-" * 65)
    print(f"[+] PHASE 3 ({num_layers}L) RESULT: {avg_throughput:.1f} seq/s (Target: >= 5,000 seq/s)")
    print(f"[+] PHASE 3 ({num_layers}L) COMPILE: {total_compile:.2f}s (Target: 10-20s)")
    print("-" * 65)
    return total_compile, avg_throughput


if __name__ == "__main__":
    devices = jax.devices()
    num_devices = len(devices)
    is_tpu = any(getattr(d, "platform", "") == "tpu" for d in devices)
    print(f"[+] Detected {num_devices} JAX device(s): {devices} (is_tpu={is_tpu})")

    if not is_tpu:
        dev_plat = getattr(devices[0], "platform", "cpu").upper()
        print(f"[!] Non-TPU ({dev_plat}) environment detected: running lightweight mock benchmark with float32.")
        benchmark_phase1(num_devices=num_devices, per_core_batch=2, num_layers=2, warmup_steps=1, active_steps=2, d_model=64, seq_len=64, vocab_size=200, precision="float32")
        benchmark_phase2(num_devices=num_devices, per_core_batch=2, num_layers=2, warmup_steps=1, active_steps=2, d_model=64, seq_len=64, vocab_size=200, precision="float32")
        benchmark_phase3(num_devices=num_devices, per_core_batch=2, num_layers=2, warmup_steps=1, active_steps=2, d_model=64, l_land=64, l_tok=32, vocab_size=200, precision="float32")
    elif num_devices == 1:
        # Cloud TPU v5e-1 Benchmark (User Goal: Phase 1: 1,000-1,200 samples/s, Phase 2 & 3: >= 300 seq/s)
        print("\n" + "=" * 70)
        print("          BENCHMARK SUITE FOR CLOUD TPU v5e-1 (Single TensorCore)")
        print("=" * 70)
        c1, s1 = benchmark_phase1(num_devices=1, per_core_batch=256, num_layers=4, warmup_steps=2, active_steps=10)
        c2, s2 = benchmark_phase2(num_devices=1, per_core_batch=128, warmup_steps=2, active_steps=10)
        c3, s3 = benchmark_phase3(num_devices=1, per_core_batch=128, num_layers=2, warmup_steps=2, active_steps=10)

        print("\n" + "=" * 70)
        print("        FINAL BENCHMARK SUMMARY FOR CLOUD TPU v5e-1")
        print("=" * 70)
        print(f" Phase 1 (CLM Pretraining): Compile = {c1:.2f}s | Throughput = {s1:,.1f} samples/s (Target: 1,000-1,200 | Pass: {s1 >= 1000})")
        print(f" Phase 2 (Gloss Decoder):   Compile = {c2:.2f}s | Throughput = {s2:,.1f} seq/s     (Target: >= 300     | Pass: {s2 >= 300})")
        print(f" Phase 3 (Conformer+Dec):   Compile = {c3:.2f}s | Throughput = {s3:,.1f} seq/s     (Target: >= 300     | Pass: {s3 >= 300})")
        print("=" * 70)
    else:
        # Cloud TPU v5e-8 Benchmark
        print("\n" + "=" * 70)
        print("          BENCHMARK SUITE FOR CLOUD TPU v5e-8 (Pod Slice 8 Cores)")
        print("=" * 70)
        c1, s1 = benchmark_phase1(num_devices=8, per_core_batch=256, num_layers=4, warmup_steps=2, active_steps=10)
        c2, s2 = benchmark_phase2(num_devices=8, per_core_batch=256, warmup_steps=2, active_steps=10)
        c3, s3 = benchmark_phase3(num_devices=8, per_core_batch=128, num_layers=2, warmup_steps=2, active_steps=10)

        print("\n" + "=" * 70)
        print("        FINAL BENCHMARK SUMMARY FOR KAGGLE CLOUD TPU v5e-8")
        print("=" * 70)
        print(f" Phase 1 (CLM Pretraining): Compile = {c1:.2f}s | Throughput = {s1:,.1f} samples/s (Target: >= 12,000 | Pass: {s1 >= 12000})")
        print(f" Phase 2 (Gloss Decoder):   Compile = {c2:.2f}s | Throughput = {s2:,.1f} seq/s     (Target: >=  6,000 | Pass: {s2 >= 6000})")
        print(f" Phase 3 (Conformer+Dec):   Compile = {c3:.2f}s | Throughput = {s3:,.1f} seq/s     (Target: >=  2,400 | Pass: {s3 >= 2400})")
        print("=" * 70)

