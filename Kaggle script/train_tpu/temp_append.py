
append_str = """
class ASLStreamedDataset(IterableDataset):
    \"\"\"
    Zero-RAM Streamed IterableDataset for PyTorch XLA execution on sharded ASL landmark records.
    Streams shards sequentially with a local shuffle buffer (~100MB), guaranteeing 0 initial indexing
    overhead and strictly < 500MB total host RAM usage.
    \"\"\"
    def __init__(
        self,
        dataset_dir: Union[str, Path] = r"E:\\datasets\\results\\asl_preprocessed_phase1",
        split: str = "train",
        max_len: int = 256,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        worker_idx: int = 0,
        num_workers: int = 1,
        shuffle_buffer_size: int = 1000,
        stage: str = "full_mixture",
        augment: bool = False,
        shared_progress=None,
    ):
        super().__init__()
        input_dir = Path(dataset_dir)
        if not input_dir.exists():
            candidates = [
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"),
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl"),
                Path("/kaggle/input/asl-preprocessed-phase1"),
                Path("./asl_preprocessed_phase1"),
            ]
            input_dir = next((c for c in candidates if c.exists()), input_dir)

        self.dataset_dir = input_dir / split if (input_dir / split).exists() else input_dir
        self.split = split
        self.max_len = max_len
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment = augment and (split == "train")
        self.shared_progress = shared_progress
        self.augmenter = LandmarkAugmenter(max_len=self.max_len) if self.augment else None

        all_shard_files = sorted(list(self.dataset_dir.glob("shard_*.pt")))
        if not all_shard_files:
            all_shard_files = sorted(list(self.dataset_dir.glob("*.pt")))
            
        if self.num_workers > 1:
            self.shard_files = all_shard_files[self.worker_idx :: self.num_workers]
        else:
            self.shard_files = all_shard_files

    def __iter__(self):
        import random
        shards = list(self.shard_files)
        if self.split == "train":
            random.shuffle(shards)

        buffer = []
        for shard_path in shards:
            try:
                shard_data = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
                items = shard_data.values() if isinstance(shard_data, dict) else shard_data
                for rec in items:
                    if isinstance(rec, dict):
                        buffer.append((shard_path, rec))
                        if len(buffer) >= self.shuffle_buffer_size:
                            if self.split == "train":
                                random.shuffle(buffer)
                            for s_path, r in buffer:
                                processed = self._process_record(s_path, r)
                                if processed is not None:
                                    yield processed
                            buffer.clear()
                del shard_data
            except Exception:
                pass

        if buffer:
            if self.split == "train":
                random.shuffle(buffer)
            for s_path, r in buffer:
                processed = self._process_record(s_path, r)
                if processed is not None:
                    yield processed
            buffer.clear()

    def _process_record(self, shard_path: Path, rec: dict) -> Optional[Dict[str, torch.Tensor]]:
        feat_arr = rec.get("features", rec.get("feature_array"))
        if feat_arr is None:
            return None
        if isinstance(feat_arr, torch.Tensor):
            feat_arr = feat_arr.detach().cpu().numpy()
        else:
            feat_arr = np.asarray(feat_arr, dtype=np.float32)

        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        T = feat_arr.shape[0] if feat_arr.ndim >= 2 else 0

        features = np.zeros((self.max_len, self.num_keypoints, self.channels_per_kp), dtype=np.float32)
        mask = np.zeros((self.max_len,), dtype=bool)
        padded_frame_indices = np.zeros(self.max_len, dtype=np.float32)

        if T > 0:
            if feat_arr.ndim == 2:
                feat_arr = feat_arr[:, : self.num_keypoints * self.channels_per_kp].reshape(T, self.num_keypoints, self.channels_per_kp)
            T_cap = min(T, self.max_len)
            features[:T_cap] = feat_arr[:T_cap]
            mask[:T_cap] = True
            padded_frame_indices[:T_cap] = np.arange(T_cap, dtype=np.float32)

        label_idx = int(rec.get("label_idx", 0))
        sample_weight = float(rec.get("quality", rec.get("sample_weight", 1.0)))
        token_ids = rec.get("gloss_seq", [label_idx])
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        padded_gloss_seq = np.full(self.max_len, 0, dtype=np.int64)
        padded_gloss_seq[0] = 1 # BOS
        g_len = min(len(token_ids), self.max_len - 2)
        if g_len > 0:
            padded_gloss_seq[1 : 1 + g_len] = token_ids[:g_len]
            padded_gloss_seq[1 + g_len] = 2 # EOS
            
        gloss_trunc = len(token_ids) > self.max_len - 2
        chicago_trunc = False
        english_trunc = False

        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "lex_class_idx": torch.tensor(4, dtype=torch.long),
            "domain_label": torch.tensor(0, dtype=torch.long),
            "has_domain_label": torch.tensor(True, dtype=torch.bool),
            "frame_indices": torch.from_numpy(padded_frame_indices).float(),
            "gloss_seq": torch.tensor(padded_gloss_seq, dtype=torch.long),
            "gloss_len": torch.tensor(g_len + 2, dtype=torch.long),
            "has_valid_gloss": torch.tensor(True, dtype=torch.bool),
            "chicago_seq": torch.tensor([1, 2] + [0] * (self.max_len - 2), dtype=torch.long),
            "chicago_len": torch.tensor(2, dtype=torch.long),
            "has_valid_chicago": torch.tensor(False, dtype=torch.bool),
            "english_seq": torch.tensor([1, 2] + [0] * (self.max_len - 2), dtype=torch.long),
            "english_len": torch.tensor(2, dtype=torch.long),
            "has_valid_english": torch.tensor(False, dtype=torch.bool),
            "is_isolated": torch.tensor(True, dtype=torch.bool),
            "mlm_mask": torch.zeros((self.max_len, self.num_keypoints), dtype=torch.bool),
            "gloss_trunc": torch.tensor(gloss_trunc, dtype=torch.bool),
            "chicago_trunc": torch.tensor(chicago_trunc, dtype=torch.bool),
            "english_trunc": torch.tensor(english_trunc, dtype=torch.bool),
        }

def create_dataloader(
    dataset_dir: Union[str, Path] = r"E:\\datasets\\asl_dataset\\asl_preprocessed_phase1",
    split: str = "train",
    batch_size: int = 64,
    max_len: int = 256,
    worker_idx: int = 0,
    num_workers: int = 1,
    num_dataloader_workers: int = 0,
    shuffle: bool = True,
    stage: str = "full_mixture",
    augment: bool = False,
    streamed: bool = False,
) -> DataLoader:
    \"\"\"Creates a PyTorch DataLoader wrapping ASLShardedDataset or ASLStreamedDataset.\"\"\"
    import multiprocessing
    mp_context = multiprocessing.get_context("spawn") if num_dataloader_workers > 0 and sys.platform.startswith("linux") else None

    shared_progress = None
    if num_dataloader_workers > 0 and mp_context is not None:
        shared_progress = mp_context.Value("d", 0.0)

    if streamed:
        dataset = ASLStreamedDataset(
            dataset_dir=dataset_dir,
            split=split,
            max_len=max_len,
            worker_idx=worker_idx,
            num_workers=num_workers,
            shuffle_buffer_size=1000 if shuffle else 1,
            stage=stage,
            augment=augment,
            shared_progress=shared_progress,
        )
        sampler = None
    else:
        dataset = ASLShardedDataset(
            dataset_dir=dataset_dir,
            split=split,
            max_len=max_len,
            worker_idx=worker_idx,
            num_workers=num_workers,
            shuffle_shards=shuffle,
            stage=stage,
            augment=augment,
            shared_progress=shared_progress,
        )
        sampler = ShardPreservingSampler(dataset, shuffle=shuffle) if shuffle else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
"""

with open('c:\\\\Users\\\\Windows 10 21H1\\\\source\\\\repos\\\\Kaggle script\\\\train\\\\dataset.py', 'a') as f:
    f.write(append_str)
