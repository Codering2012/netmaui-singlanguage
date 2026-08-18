
with open('dataset.py', 'r') as f:
    text = f.read()

# Fix in load_records_metadata
old_cache_load = """        import random

        records_by_shard = defaultdict(list)
        for r in temp_metadata:
            records_by_shard[r[0]].append(r)

        shard_keys = list(records_by_shard.keys())
        if self.shuffle_shards:
            random.shuffle(shard_keys)

        grouped_active = []
        for sk in shard_keys:
            s_recs = records_by_shard[sk]
            if self.shuffle_shards:
                random.shuffle(s_recs)
            grouped_active.extend(s_recs)

        self.class_counts = class_counts
        self.valid_label_ids = set(int(v) for v in self.label_to_idx.values())
        _GLOBAL_RECORDS_CACHE[cache_key] = temp_metadata
        _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = grouped_active"""
new_cache_load = """        import random
        import numpy as np

        records_by_shard = defaultdict(list)
        for i, r in enumerate(temp_metadata):
            records_by_shard[r[0]].append(i)

        shard_keys = list(records_by_shard.keys())
        if self.shuffle_shards:
            random.shuffle(shard_keys)

        grouped_active = []
        for sk in shard_keys:
            s_recs = records_by_shard[sk]
            if self.shuffle_shards:
                random.shuffle(s_recs)
            grouped_active.extend(s_recs)

        self.class_counts = class_counts
        self.valid_label_ids = set(int(v) for v in self.label_to_idx.values())
        _GLOBAL_RECORDS_CACHE[cache_key] = temp_metadata
        _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = np.array(grouped_active, dtype=np.int32)"""

text = text.replace(old_cache_load, new_cache_load)


# Fix in set_epoch
old_set_epoch = """            temp_metadata = _GLOBAL_RECORDS_CACHE[self.dataset_id]
            records_by_shard = defaultdict(list)
            for r in temp_metadata:
                records_by_shard[r[0]].append(r)

            shard_keys = list(records_by_shard.keys())
            rng.shuffle(shard_keys)

            grouped_active = []
            for sk in shard_keys:
                s_recs = records_by_shard[sk]
                rng.shuffle(s_recs)
                grouped_active.extend(s_recs)

            _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id] = grouped_active"""
new_set_epoch = """            import numpy as np
            temp_metadata = _GLOBAL_RECORDS_CACHE[self.dataset_id]
            records_by_shard = defaultdict(list)
            for i, r in enumerate(temp_metadata):
                records_by_shard[r[0]].append(i)

            shard_keys = list(records_by_shard.keys())
            rng.shuffle(shard_keys)

            grouped_active = []
            for sk in shard_keys:
                s_recs = records_by_shard[sk]
                rng.shuffle(s_recs)
                grouped_active.extend(s_recs)

            _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id] = np.array(grouped_active, dtype=np.int32)"""

text = text.replace(old_set_epoch, new_set_epoch)


# Fix in __getitem__
old_getitem = """        full_records = _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id]

        # We rely on set_epoch() dynamically reshuffling the _GLOBAL_ACTIVE_RECORDS_CACHE
        # block-by-block to guarantee shard locality and prevent RAM OOM, so we DO NOT permute globally here.

        # Implement cyclic wrapping in case this dataset is artificially padded to match max_len
        idx = idx % len(full_records)

        meta = full_records[idx]
        if len(meta) == 9:"""
new_getitem = """        full_records = _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id]

        # We rely on set_epoch() dynamically reshuffling the _GLOBAL_ACTIVE_RECORDS_CACHE
        # block-by-block to guarantee shard locality and prevent RAM OOM, so we DO NOT permute globally here.

        # Implement cyclic wrapping in case this dataset is artificially padded to match max_len
        idx = idx % len(full_records)

        ptr = full_records[idx]
        meta = _GLOBAL_RECORDS_CACHE[self.dataset_id][ptr]
        if len(meta) == 9:"""

text = text.replace(old_getitem, new_getitem)


with open('dataset.py', 'w') as f:
    f.write(text)
    
print("dataset.py patched")


with open('train_all_in_one_tpu.py', 'r') as f:
    text = f.read()
    
# Resume checkpoint retention
old_resume = """        if "rng_state_random" in ckpt:
            import random

            random.setstate(ckpt["rng_state_random"])

        start_epoch = ckpt.get("epoch", 0) + 1"""

new_resume = """        if "rng_state_random" in ckpt:
            import random

            random.setstate(ckpt["rng_state_random"])

        start_epoch = ckpt.get("epoch", 0) + 1
        del ckpt
        import gc
        gc.collect()"""

text = text.replace(old_resume, new_resume)

with open('train_all_in_one_tpu.py', 'w') as f:
    f.write(text)
    
print("train_all_in_one_tpu.py patched")

