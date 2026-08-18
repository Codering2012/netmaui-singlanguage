
append_str = """
class ShardPreservingSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import random
        dataset_id = self.dataset.dataset_id
        if dataset_id not in _GLOBAL_SHARD_GROUPS_CACHE:
            self.dataset._load_records_metadata()

        shard_groups = _GLOBAL_SHARD_GROUPS_CACHE[dataset_id]
        
        blocks = []
        for sg in shard_groups:
            blocks.append(sg.tolist())
            
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            np_rng = np.random.default_rng(self.seed + self.epoch)
            
            rng.shuffle(blocks)
            
            active_indices = []
            for b in blocks:
                b_copy = np.array(b, dtype=np.int32)
                np_rng.shuffle(b_copy)
                active_indices.append(b_copy)
            
            if active_indices:
                indices = np.concatenate(active_indices)
            else:
                indices = np.array([], dtype=np.int32)
        else:
            indices = np.concatenate([np.array(b, dtype=np.int32) for b in blocks]) if blocks else np.array([], dtype=np.int32)
                
        dataset_len = len(self.dataset)
        if len(indices) == 0:
            return iter([])
            
        def gen():
            count = 0
            while count < dataset_len:
                for idx in indices:
                    yield int(idx)
                    count += 1
                    if count >= dataset_len:
                        return
                        
        return gen()

    def __len__(self):
        return len(self.dataset)
"""

with open('c:\\\\Users\\\\Windows 10 21H1\\\\source\\\\repos\\\\Kaggle script\\\\train\\\\dataset.py', 'r') as f:
    content = f.read()

idx = content.find('class ASLStreamedDataset(IterableDataset):')
if idx != -1:
    with open('c:\\\\Users\\\\Windows 10 21H1\\\\source\\\\repos\\\\Kaggle script\\\\train\\\\dataset.py', 'w') as f:
        f.write(content[:idx] + append_str + '\\n' + content[idx:])
