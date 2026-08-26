def calc_speed(batch_size, steps_per_sec, world_size=8):
    per_core_batch = batch_size // world_size
    samples_per_sec = steps_per_sec * per_core_batch * world_size
    steps_per_epoch = 117450 // batch_size
    epoch_time_sec = steps_per_epoch / steps_per_sec
    print(f"=== Batch Size: {batch_size} (Per-Core: {per_core_batch}) ===")
    print(f"  -> Cluster Batch Size : {batch_size}")
    print(f"  -> Step Execution Rate : {steps_per_sec:.2f} steps/s")
    print(f"  -> Throughput Speed    : {samples_per_sec:.1f} samples/s")
    print(f"  -> Steps per Epoch     : {steps_per_epoch} steps")
    print(f"  -> Time per Epoch      : {epoch_time_sec:.1f} seconds ({epoch_time_sec/60:.1f} min)")

print("CURRENT RUN (What you ran):")
calc_speed(batch_size=512, steps_per_sec=2.62)

print("\nRECOMMENDED RUN (To achieve 4,000+ samp/s):")
calc_speed(batch_size=2048, steps_per_sec=2.40)
