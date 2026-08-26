import sys

def simulate_step_count(batch_size, world_size=8, dataset_len=117450):
    raw_core_batch = max(1, batch_size // world_size)
    per_core_batch = min(64, raw_core_batch)
    cluster_batch = max(1, per_core_batch * world_size)
    steps_limit = dataset_len // cluster_batch
    return {
        "batch_size_arg": batch_size,
        "raw_core_batch": raw_core_batch,
        "per_core_batch_clamped": per_core_batch,
        "cluster_batch": cluster_batch,
        "steps_limit": steps_limit
    }

print("Run 1 (batch_size=1024, accum=4):", simulate_step_count(1024))
print("Run 2 (batch_size=2048, accum=8):", simulate_step_count(2048))
