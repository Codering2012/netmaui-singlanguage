import sys
import os

# Critical: MUST be set on line 1 before ANY C libraries (numpy, mkl, openmp, torch, torch_xla) are loaded
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["GOMP_SPINCOUNT"] = "0"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "65536"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["XLA_USE_BF16"] = "1"
os.environ.pop("XLA_DOWNCAST_BF16", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_TRANSFER_STREAM_LIMIT"] = "4"

# Route XLA Persistent Compilation Cache to /tmp (prevents exhausting Kaggle 20GB /kaggle/working quota)
cache_path = os.environ.get("XLA_PERSISTENT_CACHE_PATH", "/tmp/xla_cache" if os.name != "nt" else "./xla_cache")
if cache_path.startswith("/kaggle/working"):
    cache_path = "/tmp/xla_cache"
os.environ["XLA_PERSISTENT_CACHE_PATH"] = cache_path

# LibTPU & XLA Fast Compilation and Hardware Acceleration Flags (instant systolic execution on TPU v5e)
if "LIBTPU_INIT_ARGS" not in os.environ:
    os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_enable_flash_attention=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_rwb_fusion=true"
else:
    os.environ["LIBTPU_INIT_ARGS"] = os.environ["LIBTPU_INIT_ARGS"].replace(
        "xla_tpu_enable_async_collective_fusion_multiple_bars",
        "xla_tpu_enable_async_collective_fusion_multiple_steps",
    )
    if "xla_tpu_rwb_fusion" not in os.environ["LIBTPU_INIT_ARGS"]:
        os.environ["LIBTPU_INIT_ARGS"] += " --xla_tpu_rwb_fusion=true"

_existing_xla = os.environ.get("XLA_FLAGS", "")
# Sanitize any legacy or unknown fast_math flags from earlier runs in the same notebook kernel
_existing_xla = _existing_xla.replace("--xla_tpu_fast_math=true", "").replace("--xla_tpu_fast_math", "").strip()
_fast_flags = [
    "--xla_cpu_multi_thread_eigen=true",
]
for _ff in _fast_flags:
    if _ff not in _existing_xla:
        _existing_xla = (_existing_xla + " " + _ff).strip()
os.environ["XLA_FLAGS"] = _existing_xla

# Force Local PJRT mode to avoid gRPC proxy concurrency limit and fork deadlocks
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)

# Ensure repo root and train_tpu/v1 are in sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
v1_dir = os.path.join(repo_root, "train_tpu", "v1")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if v1_dir not in sys.path:
    sys.path.insert(0, v1_dir)

from train_tpu.v1.train_all_in_one_tpu import main

if __name__ == "__main__":
    main()
