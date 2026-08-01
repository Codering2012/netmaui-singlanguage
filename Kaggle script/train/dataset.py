import os
import glob
import json
import random
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader

class LandmarkAugmenter:
    """
    Progressive Noise Curriculum Data Augmentation for 3D WholeBody landmark sequences (T, 60, 9).
    Ramps up noise intensity smoothly from 0.1% (clean baseline) at Epoch 1 to 5.0% - 6.0% (hard real-camera noise) at Epoch N.
    """
    def __init__(
        self,
        base_jitter_std: float = 0.025,
        max_scale_range: Tuple[float, float] = (0.925, 1.075),
        max_shift_range: float = 0.025,
        max_rotation_range: float = 7.5,
        max_kp_drop_prob: float = 0.04,
        max_frame_drop_prob: float = 0.025,
        noise_level: float = 0.02
    ):
        self.base_jitter_std = base_jitter_std
        self.max_scale_range = max_scale_range
        self.max_shift_range = max_shift_range
        self.max_rotation_range = max_rotation_range
        self.max_kp_drop_prob = max_kp_drop_prob
        self.max_frame_drop_prob = max_frame_drop_prob
        self.noise_level = max(0.001, min(1.0, noise_level))

    def set_noise_level(self, level: float) -> None:
        """Sets progressive noise level ratio between 0.02 (0.1% noise) and 1.0 (5%-6% noise)."""
        self.noise_level = max(0.001, min(1.0, level))

    def __call__(self, feat_arr: np.ndarray) -> np.ndarray:
        T, K, C = feat_arr.shape
        if T == 0:
            return feat_arr

        aug = feat_arr.copy()
        pos = aug[:, :, :3]

        # Scaled noise parameters based on progressive noise level
        jitter_std = self.base_jitter_std * self.noise_level
        rot_range = self.max_rotation_range * self.noise_level
        shift_range = self.max_shift_range * self.noise_level
        kp_drop_prob = self.max_kp_drop_prob * self.noise_level
        frame_drop_prob = self.max_frame_drop_prob * self.noise_level
        scale_delta = (self.max_scale_range[1] - 1.0) * self.noise_level

        # Dynamic curriculum probabilities derived from noise_level ratio (50% Reduced)
        choke_prob = 0.005 + 0.01 * self.noise_level
        finger_drop_prob = 0.01 + 0.025 * self.noise_level
        timestretch_prob = 0.01 + 0.04 * self.noise_level
        warping_prob = 0.01 + 0.05 * self.noise_level
        hand_occ_prob = 0.005 + 0.01 * self.noise_level

        # 0. Ambidextrous Horizontal Mirroring (50% probability left/right hand flip)
        if np.random.rand() > 0.5:
            pos[:, :, 0] = -pos[:, :, 0]
            pos_lh = pos[:, 18:39, :].copy()
            pos_rh = pos[:, 39:60, :].copy()
            pos[:, 18:39, :] = pos_rh
            pos[:, 39:60, :] = pos_lh

        # 1. Progressive 3D Scaling
        scale = np.random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
        pos = pos * scale

        # 2. Progressive 2D Shift / Translation
        if shift_range > 0:
            shift_x = np.random.uniform(-shift_range, shift_range)
            shift_y = np.random.uniform(-shift_range, shift_range)
            pos[:, :, 0] += shift_x
            pos[:, :, 1] += shift_y

        # 3. Progressive 3D Rotation (Roll Z, Pitch X, Yaw Y up to 30 degrees)
        if rot_range > 0:
            pitch_deg = np.random.uniform(-rot_range * 2, rot_range * 2)
            yaw_deg = np.random.uniform(-rot_range * 2, rot_range * 2)
            roll_deg = np.random.uniform(-rot_range, rot_range)

            rad_p, rad_y, rad_r = np.radians(pitch_deg), np.radians(yaw_deg), np.radians(roll_deg)

            rx = np.array([[1, 0, 0], [0, np.cos(rad_p), -np.sin(rad_p)], [0, np.sin(rad_p), np.cos(rad_p)]], dtype=np.float32)
            ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]], dtype=np.float32)
            rz = np.array([[np.cos(rad_r), -np.sin(rad_r), 0], [np.sin(rad_r), np.cos(rad_r), 0], [0, 0, 1]], dtype=np.float32)

            rot_mat = np.dot(rz, np.dot(ry, rx))
            center = pos.mean(axis=(0, 1), keepdims=True)
            pos = pos - center
            pos = np.dot(pos.reshape(-1, 3), rot_mat).reshape(T, K, 3)
            pos = pos + center

        # 4. Progressive Gaussian Sensor Noise (from 0.1% to 5.0%)
        if jitter_std > 0:
            jitter = np.random.normal(0, jitter_std, size=pos.shape).astype(np.float32)
            pos += jitter

        # 5. Progressive Keypoint Occlusion Dropout
        if kp_drop_prob > 0:
            kp_mask = (np.random.rand(T, K, 1) > kp_drop_prob).astype(np.float32)
            pos *= kp_mask

        # 5a. Progressive Random Frame Dropout (Temporal Frame Dropping)
        if frame_drop_prob > 0 and T > 8:
            keep_mask = np.random.rand(T) > frame_drop_prob
            if np.sum(keep_mask) >= 4:
                pos = pos[keep_mask]
                T = pos.shape[0]

        # 5b. Structured FingerDropout (erases whole finger clusters)
        if np.random.rand() < finger_drop_prob:
            lh_fingers = [list(range(18, 23)), list(range(23, 27)), list(range(27, 31)), list(range(31, 35)), list(range(35, 39))]
            rh_fingers = [list(range(39, 44)), list(range(44, 48)), list(range(48, 52)), list(range(52, 56)), list(range(56, 60))]
            all_fingers = lh_fingers + rh_fingers
            drop_fingers = random.sample(all_fingers, k=random.randint(1, 2))
            for f_indices in drop_fingers:
                pos[:, f_indices, :] = 0.0

        # 5c. Entire Hand Occlusion / Disappearance (erases entire hand for 30%-50% of frames)
        if np.random.rand() < hand_occ_prob and T > 10:
            occ_start = np.random.randint(0, max(1, T - 8))
            occ_len = np.random.randint(4, max(5, T // 2))
            hand_idx = range(18, 39) if np.random.rand() > 0.5 else range(39, 60)
            pos[occ_start : occ_start + occ_len, hand_idx, :] = 0.0

        # 5d. TimeStretch Temporal Resampling (0.75x - 1.25x speed variation)
        if T > 10 and np.random.rand() < timestretch_prob:
            stretch_factor = np.random.uniform(0.75, 1.25)
            new_T = int(round(T * stretch_factor))
            if new_T > 4:
                idx = np.linspace(0, T - 1, num=new_T, dtype=int)
                pos = pos[idx]; T = new_T
                aug = np.zeros((T, K, C), dtype=np.float32)

        # 5e. Dynamic In-Place Physiological Stalling (1% to 20%: Palm 3D Drift & Finger Contraction in-place)
        # Stalls in-place without expanding sequence length T, avoiding uniform downsampling distortion!
        if T > 20 and np.random.rand() < choke_prob:
            freeze_len = min(int(T * 0.25), np.random.randint(10, 25))
            if freeze_len > 1:
                freeze_idx = np.random.randint(2, max(3, T - freeze_len))
                base_frame = pos[freeze_idx].copy()
                t_steps = np.linspace(0, 2 * np.pi, num=freeze_len, dtype=np.float32)
                drift_x = 0.015 * np.sin(t_steps) + np.random.normal(0, 0.002, size=(freeze_len,)).astype(np.float32)
                drift_y = 0.010 * np.cos(t_steps * 1.5) + np.random.normal(0, 0.002, size=(freeze_len,)).astype(np.float32)
                drift_z = 0.010 * np.sin(t_steps * 0.7) + np.random.normal(0, 0.002, size=(freeze_len,)).astype(np.float32)
                
                for step_i in range(freeze_len):
                    f_frame = base_frame.copy()
                    f_frame[:, 0] += drift_x[step_i]
                    f_frame[:, 1] += drift_y[step_i]
                    f_frame[:, 2] += drift_z[step_i]
                    f_scale = 1.0 + 0.02 * np.sin(t_steps[step_i] * 3.0)
                    f_frame[18:60, :] *= f_scale
                    pos[freeze_idx + step_i] = f_frame

        # 5f. 0.5s - 1.5s Interval Speed Warping (In-place non-linear time warping)
        if T > 25 and np.random.rand() < warping_prob:
            warp_factor = np.random.uniform(0.5, 1.5)
            t_orig = np.linspace(0, 1, num=T)
            t_warped = np.power(t_orig, warp_factor)
            warp_indices = np.clip((t_warped * (T - 1)).astype(int), 0, T - 1)
            pos = pos[warp_indices]

        # 5g. Hand Anatomy Sudden Displacement & Return (3 to 8 frames, 2px to 30px offset shift and return)
        disp_prob = 0.01 + 0.04 * self.noise_level
        if T > 15 and np.random.rand() < disp_prob:
            d_idx = np.random.randint(3, max(4, T - 10))
            d_len = np.random.randint(3, 8)
            h_indices = range(18, 39) if np.random.rand() > 0.5 else range(39, 60)
            
            offset_dist = np.random.uniform(0.01, 0.15)  # 2px to 30px offset in normalized scale
            offset_dir = np.random.normal(0, 1, size=(3,)).astype(np.float32)
            offset_dir = (offset_dir / max(1e-6, np.linalg.norm(offset_dir))) * offset_dist
            
            arc_t = np.linspace(0, np.pi, num=d_len, dtype=np.float32)
            for step_i in range(d_len):
                if d_idx + step_i < T:
                    shift_weight = np.sin(arc_t[step_i])
                    pos[d_idx + step_i, h_indices, :3] += shift_weight * offset_dir

        # 5h. Anatomical Coordinate Blackout with Kinematic Preservation (2% Chance SOTA Extension)
        # Zeroes out 3D position coordinates (x, y, z -> 0) while keeping velocity & acceleration intact
        blackout_mask = np.ones_like(pos, dtype=np.float32)
        if np.random.rand() < 0.02 and T > 5:
            b_idx = np.random.randint(0, T)
            target_kp = range(18, 39) if np.random.rand() > 0.5 else range(39, 60)
            blackout_mask[b_idx, target_kp, :] = 0.0

        pos = pos * blackout_mask

        # 6. Progressive Temporal Frame Dropout
        if frame_drop_prob > 0 and T > 5:
            frame_mask = (np.random.rand(T, 1, 1) > frame_drop_prob).astype(np.float32)
            pos *= frame_mask

        # Recompute Kinematic Velocity & Acceleration Derivatives for Speed Invariance
        vel = np.zeros_like(pos)
        if T > 2:
            vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0
            vel[0] = pos[1] - pos[0]
            vel[-1] = pos[-1] - pos[-2]
        elif T == 2:
            vel[:] = pos[1] - pos[0]
        
        acc = np.zeros_like(pos)
        if T > 2:
            acc[1:-1] = pos[2:] - 2*pos[1:-1] + pos[:-2]
            acc[0] = 2*pos[0] - 5*pos[1] + 4*pos[2] - pos[3] if T > 3 else 0
            acc[-1] = 2*pos[-1] - 5*pos[-2] + 4*pos[-3] - pos[-4] if T > 3 else 0

        return np.concatenate([pos, vel, acc], axis=-1)

def motion_aware_sample_indices(feat_arr: np.ndarray, max_len: int) -> np.ndarray:
    """
    Downsamples a sequence of length T > max_len to max_len frames.
    Uses motion velocity energy to preserve high-movement frames while maintaining temporal ordering.
    """
    T = feat_arr.shape[0]
    if T <= max_len:
        return np.arange(T)

    # 1. Compute frame-to-frame motion magnitude
    if feat_arr.ndim >= 2:
        flat_feat = feat_arr.reshape(T, -1)
        motion_energy = np.linalg.norm(flat_feat[1:] - flat_feat[:-1], axis=-1)
        motion_energy = np.pad(motion_energy, (0, 1), mode='edge')
    else:
        motion_energy = np.ones(T, dtype=np.float32)

    # 2. Hybrid sampling: 70% uniform grid to preserve global timeline, 30% top motion frames
    uniform_count = int(max_len * 0.70)
    motion_count = max_len - uniform_count

    uniform_idx = np.linspace(0, T - 1, num=uniform_count, dtype=int)
    
    mask = np.ones(T, dtype=bool)
    mask[uniform_idx] = False
    remaining_idx = np.where(mask)[0]

    if len(remaining_idx) > 0 and motion_count > 0:
        top_motion_local = np.argsort(motion_energy[remaining_idx])[-motion_count:]
        motion_idx = remaining_idx[top_motion_local]
        selected_idx = np.concatenate([uniform_idx, motion_idx])
    else:
        selected_idx = np.linspace(0, T - 1, num=max_len, dtype=int)

    selected_idx = np.sort(selected_idx)
    return selected_idx

# Global metadata & shard data caches to avoid massive IPC transfer and repeated disk reads
_GLOBAL_RECORDS_CACHE = {}
_GLOBAL_ACTIVE_RECORDS_CACHE = {}
_GLOBAL_SHARD_DATA_CACHE = {}

class ASLShardedDataset(Dataset):
    """
    PyTorch Dataset for reading sharded ASL landmark records from preprocessed phase 1 directory.
    Enforces strict static sequence padding and static batch shapes for PyTorch XLA TPU execution.
    Integrates ASL-LEX lexical features and real-world camera noise augmentation.
    """
    def __init__(
        self,
        dataset_dir: Union[str, Path] = r"E:\datasets\results\asl_preprocessed_phase1",
        split: str = "train",
        max_len: int = 256,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        worker_idx: int = 0,
        num_workers: int = 1,
        shuffle_shards: bool = True,
        stage: str = "full_mixture",
        augment: bool = False
    ):
        super().__init__()
        
        # Auto-discover candidate directories if specified directory doesn't exist
        input_dir = Path(dataset_dir)
        if not input_dir.exists():
            candidates = [
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"),
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl"),
                Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl"),
                Path("/kaggle/input/asl-preprocessed-phase1"),
                Path("./asl_preprocessed_phase1"),
                Path(r"E:\datasets\results\asl_preprocessed_phase1"),
            ]
            input_dir = next((c for c in candidates if c.exists()), input_dir)

        if (input_dir / split).exists():
            self.dataset_dir = input_dir / split
        else:
            self.dataset_dir = input_dir

        self.split = split
        self.max_len = max_len
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.feature_dim = num_keypoints * channels_per_kp
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.shuffle_shards = shuffle_shards
        self.stage = stage
        self.augment = augment and (split == "train")
        self.augmenter = LandmarkAugmenter() if self.augment else None

        # Resolve Master Vocabulary Mapping
        self.label_to_idx = {}
        vocab_candidates = [
            self.dataset_dir / "vocabulary_mapping_global.json",
            input_dir / "vocabulary_mapping_global.json",
            self.dataset_dir / f"vocabulary_mapping_{split}.json",
            self.dataset_dir / "vocabulary_mapping_train.json",
            input_dir / "vocabulary_mapping_train.json",
            input_dir / f"vocabulary_mapping_{split}.json",
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1/vocabulary_mapping_global.json"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1/vocabulary_mapping_train.json"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1/vocabulary_mapping_train.json"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/vocabulary_mapping_train.json"),
        ]

        for vc in vocab_candidates:
            if vc.exists():
                try:
                    with open(vc, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.label_to_idx = data.get("label_to_idx", {})
                        if self.label_to_idx:
                            break
                except Exception:
                    pass

        metadata_path = self.dataset_dir / "metadata.json"
        if metadata_path.exists() and not self.label_to_idx:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.label_to_idx = meta.get("label_to_idx", {})
            except Exception:
                pass

        if not self.label_to_idx:
            raise FileNotFoundError(f"Could not load vocabulary mapping from '{self.dataset_dir}' or candidate paths.")

        self.num_classes = len(self.label_to_idx)

        # ASL-LEX Lexical Grammatical Map Initialization
        self.asl_lex_map = {}
        grammar_candidates = [
            Path(__file__).resolve().parent.parent / "preprocessing" / "grammar_logic.json",
            input_dir / "grammar_logic.json",
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/grammar_logic.json"),
        ]
        csv_candidates = [
            Path("/kaggle/input/datasets/tranquocbao2012/asl-lex/signdata.csv"),
            Path("/kaggle/input/asl-lex/signdata.csv"),
            Path(__file__).resolve().parent.parent / "preprocessing" / "signdata.csv",
        ]

        pos_categories = {"Noun": 0, "Verb": 1, "Adjective": 2, "Adverb": 3}
        for gc in grammar_candidates:
            if gc.exists():
                try:
                    with open(gc, "r", encoding="utf-8") as f:
                        g_data = json.load(f)
                        for word_key, val_dict in g_data.items():
                            cls_str = val_dict.get("class", "Other") if isinstance(val_dict, dict) else str(val_dict)
                            word_clean = word_key.strip().lower()
                            self.asl_lex_map[word_clean] = pos_categories.get(cls_str, 4)
                        if self.asl_lex_map:
                            break
                except Exception:
                    pass

        if not self.asl_lex_map:
            for cc in csv_candidates:
                if cc.exists():
                    try:
                        import csv
                        with open(cc, "r", encoding="utf-8", errors="ignore") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                word_clean = (row.get("LemmaID") or row.get("EntryID") or "").strip().lower()
                                cls_str = (row.get("LexicalClass") or "Other").strip()
                                if word_clean:
                                    self.asl_lex_map[word_clean] = pos_categories.get(cls_str, 4)
                        if self.asl_lex_map:
                            break
                    except Exception:
                        pass

        # Collect shard files and partition among workers if distributed
        all_shard_files = sorted(list(self.dataset_dir.glob("shard_*.pt")))
        if not all_shard_files:
            all_shard_files = sorted(list(self.dataset_dir.glob("*.pt")))
        if not all_shard_files:
            raise FileNotFoundError(f"No shard_*.pt or .pt files found in '{self.dataset_dir}'")

        # Partition shards across TPU workers
        if num_workers > 1:
            self.shard_files = all_shard_files[worker_idx::num_workers]
        else:
            self.shard_files = all_shard_files

        if self.shuffle_shards:
            random.shuffle(self.shard_files)

        # Load records metadata from allocated shards
        self.dataset_id = id(self)
        self.cached_shard_path: Optional[Path] = None
        self.cached_shard_data: Optional[List] = None

        if self.dataset_id not in _GLOBAL_RECORDS_CACHE:
            self._load_records_metadata()

    def _load_records_metadata(self) -> None:
        """Loads metadata from all allocated shards in parallel across CPU cores, utilizing a cache file to prevent 15+ minute startup times on Windows/HDD."""
        global _GLOBAL_RECORDS_CACHE, _GLOBAL_ACTIVE_RECORDS_CACHE
        
        # Determine the cache key based on the split to support multiple dataset instances (train/val/test)
        cache_key = self.dataset_id
        
        cache_file = self.dataset_dir / "records_metadata_cache.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, "r") as f:
                    records = json.load(f)
                    
                # Convert string paths back to Path objects
                for r in records:
                    if "shard_path" in r:
                        r["shard_path"] = Path(r["shard_path"])
                        
                _GLOBAL_RECORDS_CACHE[cache_key] = records
                _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = list(records)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}")

        def _index_shard(shard_path: Path):
            local_metas = []
            local_counts = defaultdict(int)
            try:
                shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
                for item_idx, rec in enumerate(shard_data):
                    lbl_idx = rec.get("label_idx", -1)
                    if lbl_idx is None or lbl_idx < 0:
                        lbl_str = rec.get("label", "")
                        lbl_idx = self.label_to_idx.get(lbl_str, 0)
                    lbl_clean = max(0, int(lbl_idx if lbl_idx is not None else 0))
                    local_counts[lbl_clean] += 1
                    q_score = float(rec.get("sample_weight", rec.get("quality", rec.get("quality_score", 1.0))))
                    m_comp = float(rec.get("motion_complexity", 0.5))

                    local_metas.append({
                        "shard_path": shard_path,
                        "item_idx": item_idx,
                        "label_idx": lbl_clean,
                        "label": str(rec.get("label", "")),
                        "quality_score": q_score,
                        "motion_complexity": m_comp
                    })
            except Exception:
                pass
            return local_metas, local_counts

        class_counts = defaultdict(int)
        temp_metadata = []

        from concurrent.futures import ThreadPoolExecutor
        import os
        with ThreadPoolExecutor(max_workers=min(16, max(1, os.cpu_count() or 4))) as executor:
            results = executor.map(_index_shard, self.shard_files)
            for local_metas, local_counts in results:
                temp_metadata.extend(local_metas)
                for k, v in local_counts.items():
                    class_counts[k] += v

        max_c = max(1, max(class_counts.values())) if class_counts else 1
        for meta in temp_metadata:
            lbl_c = class_counts.get(meta["label_idx"], 1)
            rarity_score = float(np.log(1.0 + max_c / (lbl_c + 1))) / float(np.log(1.0 + max_c))
            meta["difficulty"] = 0.40 * rarity_score + 0.30 * meta["motion_complexity"] + 0.30 * (1.0 - max(0.0, min(1.0, meta["quality_score"])))
        
        temp_metadata.sort(key=lambda x: x.get("difficulty", 0.5))
        
        # Group active records by shard_path so DataLoader reads contiguous records from 1 shard at a time
        # Local defaultdict import removed to avoid UnboundLocalError
        import random
        records_by_shard = defaultdict(list)
        for r in temp_metadata:
            records_by_shard[str(r["shard_path"])].append(r)

        shard_keys = list(records_by_shard.keys())
        if self.shuffle_shards:
            random.shuffle(shard_keys)

        grouped_active = []
        for sk in shard_keys:
            s_recs = records_by_shard[sk]
            if self.shuffle_shards:
                random.shuffle(s_recs)
            grouped_active.extend(s_recs)

        _GLOBAL_RECORDS_CACHE[cache_key] = temp_metadata
        _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = grouped_active
        
        # Save cache
        try:
            import json
            cache_data = []
            for r in temp_metadata:
                r_copy = dict(r)
                if "shard_path" in r_copy:
                    r_copy["shard_path"] = str(r_copy["shard_path"])
                cache_data.append(r_copy)
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def set_noise_level(self, level: float) -> None:
        """Dynamically adjusts augmentation noise level and active Curriculum by Difficulty subset."""
        if self.augmenter:
            self.augmenter.noise_level = level
        
        cache_key = self.dataset_id
        if cache_key not in _GLOBAL_RECORDS_CACHE:
            return
            
        full_records = _GLOBAL_RECORDS_CACHE[cache_key]

        if self.stage == "full_mixture":
            _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = list(full_records)
        elif self.stage == "curriculum_easy":
            _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = full_records[:max(10, int(len(full_records) * 0.33))]
        elif self.stage == "curriculum_medium":
            _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = full_records[:max(10, int(len(full_records) * 0.66))]

    def __len__(self) -> int:
        if self.dataset_id not in _GLOBAL_ACTIVE_RECORDS_CACHE:
            self._load_records_metadata()
        return len(_GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id])

    def _get_record_feature(self, shard_path: Path, item_idx: int) -> np.ndarray:
        global _GLOBAL_SHARD_DATA_CACHE
        str_path = str(shard_path)
        if str_path not in _GLOBAL_SHARD_DATA_CACHE:
            _GLOBAL_SHARD_DATA_CACHE[str_path] = torch.load(shard_path, map_location="cpu", weights_only=False)

        rec = _GLOBAL_SHARD_DATA_CACHE[str_path][item_idx]
        raw_feat = rec.get("features", rec.get("feature_array", None))
        if isinstance(raw_feat, torch.Tensor):
            feat_arr = raw_feat.detach().cpu().numpy()
        else:
            feat_arr = np.asarray(raw_feat)
        return feat_arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dataset_id not in _GLOBAL_ACTIVE_RECORDS_CACHE:
            self._load_records_metadata()
        meta = _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id][idx]
        feat_arr = self._get_record_feature(meta["shard_path"], meta["item_idx"])
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        if feat_arr.ndim == 2:
            T = feat_arr.shape[0]
            feat_arr = feat_arr.reshape(T, self.num_keypoints, self.channels_per_kp)
        elif feat_arr.ndim == 3:
            T = feat_arr.shape[0]
        else:
            T = 0
            feat_arr = np.zeros((0, self.num_keypoints, self.channels_per_kp), dtype=np.float32)

        # Real-World Camera Noise Data Augmentation during training
        if self.augment and self.augmenter is not None and T > 0:
            feat_arr = self.augmenter(feat_arr)
            T = feat_arr.shape[0]

        # Enforce static sequence length (max_len) with Motion-Aware Priority Sampling
        features = np.zeros((self.max_len, self.num_keypoints, self.channels_per_kp), dtype=np.float32)
        mask = np.zeros((self.max_len,), dtype=bool)

        if T > 0:
            if T <= self.max_len:
                features[:T] = feat_arr[:T]
                mask[:T] = True
            else:
                indices = motion_aware_sample_indices(feat_arr, self.max_len)
                features = feat_arr[indices]
                mask[:] = True

        label_idx = int(meta.get("label_idx", 0))
        sample_weight = float(meta.get("quality_score", 1.0))
        
        # Resolve ASL-LEX Grammatical Class
        lbl_str = str(meta.get("label", "")).strip().lower()
        lex_class_idx = self.asl_lex_map.get(lbl_str, 4)

        return (
            torch.from_numpy(features),
            torch.from_numpy(mask),
            torch.tensor(label_idx, dtype=torch.long),
            torch.tensor(sample_weight, dtype=torch.float32),
            torch.tensor(lex_class_idx, dtype=torch.long)
        )

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(
    dataset_dir: Union[str, Path] = r"E:\datasets\results\asl_preprocessed_phase1",
    split: str = "train",
    batch_size: int = 64,
    max_len: int = 256,
    worker_idx: int = 0,
    num_workers: int = 1,
    num_dataloader_workers: int = 4,
    shuffle: bool = True,
    stage: str = "full_mixture",
    augment: bool = False
) -> DataLoader:
    """Creates a PyTorch DataLoader wrapping ASLShardedDataset with curriculum learning & real-life camera augmentation."""
    dataset = ASLShardedDataset(
        dataset_dir=dataset_dir,
        split=split,
        max_len=max_len,
        worker_idx=worker_idx,
        num_workers=num_workers,
        shuffle_shards=shuffle,
        stage=stage,
        augment=augment
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sharding is pre-shuffled at the shard & block level in ASLShardedDataset
        num_workers=num_dataloader_workers,
        pin_memory=True,
        persistent_workers=True if num_dataloader_workers > 0 else False,
        prefetch_factor=4 if num_dataloader_workers > 0 else None,
        drop_last=True,  # Ensure static batch size for TPU XLA recompilation protection
        worker_init_fn=_seed_worker
    )
