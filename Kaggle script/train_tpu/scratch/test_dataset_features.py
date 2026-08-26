import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (
    GlossVocabulary,
    EnglishVocabulary,
    LandmarkAugmenter,
    motion_aware_sample_indices,
    ASLShardedDataset,
    ASLStreamedDataset,
    ShardPreservingSampler,
    create_dataloader,
    apply_dae_corruptions,
    ASLGPC12Dataset,
    KDWDDataset,
    Phase1MixedIterable,
    phase1_collate_fn,
    phase2_collate_fn,
    clear_global_dataset_caches,
)

DATA_DIR = r"E:\datasets\asl_dataset\asl_preprocessed_phase1"
ASLG_CSV = r"E:\datasets\asl_dataset\ASLG-PC12\train.csv"
KDWD_DIR = r"E:\datasets\asl_dataset\wikitext"


def test_vocabularies():
    print("--- [1/13] Testing GlossVocabulary & EnglishVocabulary ---")
    raw_map = {"hello": 4, "world": 5, "sign": 6}
    gloss_vocab = GlossVocabulary(label_to_idx=raw_map)
    assert gloss_vocab.encode("hello world") == [4, 5], f"Unexpected gloss encode: {gloss_vocab.encode('hello world')}"
    assert gloss_vocab.token_to_gloss(4) == "hello"

    eng_vocab_path = os.path.join(DATA_DIR, "english_vocab.json")
    eng_vocab = EnglishVocabulary(vocab_path=eng_vocab_path)
    encoded = eng_vocab.encode("hello world sign language")
    print(f"  EnglishVocab encoded 'hello world sign language': {encoded[:5]}...")
    assert len(encoded) > 0, "EnglishVocab failed to encode text"
    print("  [PASS] GlossVocabulary & EnglishVocabulary working as expected.")


def test_landmark_augmenter():
    print("--- [2/13] Testing LandmarkAugmenter ---")
    augmenter = LandmarkAugmenter(
        base_jitter_std=0.035,
        max_scale_range=(0.85, 1.15),
        max_shift_range=0.035,
        max_rotation_range=10.0,
        max_kp_drop_prob=0.05,
        max_frame_drop_prob=0.035,
        noise_level=0.5,
    )
    dummy_feat = np.random.randn(50, 166, 3).astype(np.float32)
    aug_out = augmenter(dummy_feat)
    if isinstance(aug_out, tuple):
        aug_feat, frame_idx = aug_out
    else:
        aug_feat = aug_out
    assert isinstance(aug_feat, np.ndarray), "Augmenter output is not ndarray"
    assert aug_feat.shape[1] == 166 and aug_feat.shape[2] in (3, 9), f"Unexpected shape after augmentation: {aug_feat.shape}"
    print(f"  Augmented dummy features shape: {aug_feat.shape}")
    print("  [PASS] LandmarkAugmenter working as expected.")


def test_motion_aware_sample():
    print("--- [3/13] Testing motion_aware_sample_indices ---")
    dummy_feat = np.random.randn(100, 166 * 3).astype(np.float32)
    indices = motion_aware_sample_indices(dummy_feat, max_len=30)
    assert len(indices) == 30, f"Expected 30 indices, got {len(indices)}"
    assert np.all(indices[:-1] <= indices[1:]), "Indices are not sorted"
    print("  [PASS] motion_aware_sample_indices working as expected.")


def test_asl_sharded_dataset():
    print("--- [4/13] Testing ASLShardedDataset ---")
    dataset = ASLShardedDataset(
        dataset_dir=DATA_DIR,
        split="val",
        max_len=128,
        stage="full_mixture",
        augment=True,
    )
    print(f"  ASLShardedDataset val size: {len(dataset)}")
    assert len(dataset) > 0, "ASLShardedDataset is empty"

    sample = dataset[0]
    assert "feature" in sample or "landmarks" in sample or "feat" in sample, f"Sample keys: {sample.keys()}"
    landmarks = sample.get("feature", sample.get("landmarks", sample.get("feat")))
    print(f"  Sample 0 landmark shape: {landmarks.shape}")

    # Test stage parameter
    stg_ds = ASLShardedDataset(dataset_dir=DATA_DIR, split="val", max_len=128, stage="full_mixture")
    print(f"  Stage 'full_mixture' sample count: {len(stg_ds)}")
    print("  [PASS] ASLShardedDataset working as expected.")


def test_shard_preserving_sampler():
    print("--- [5/13] Testing ShardPreservingSampler ---")
    dataset = ASLShardedDataset(dataset_dir=DATA_DIR, split="val", max_len=128)
    sampler = ShardPreservingSampler(dataset, shuffle=True)
    indices = list(iter(sampler))
    assert len(indices) == len(dataset), f"Sampler count mismatch: {len(indices)} vs {len(dataset)}"
    print(f"  Sampled {len(indices)} indices with ShardPreservingSampler.")
    print("  [PASS] ShardPreservingSampler working as expected.")


def test_asl_streamed_dataset():
    print("--- [6/13] Testing ASLStreamedDataset ---")
    streamed_ds = ASLStreamedDataset(
        dataset_dir=DATA_DIR,
        split="val",
        max_len=128,
        worker_idx=0,
        num_workers=1,
        shuffle_buffer_size=10,
    )
    items = []
    for i, sample in enumerate(streamed_ds):
        items.append(sample)
        if i >= 5:
            break
    assert len(items) == 6, f"Expected 6 streamed items, got {len(items)}"
    print(f"  Streamed {len(items)} samples successfully.")
    print("  [PASS] ASLStreamedDataset working as expected.")


def test_create_dataloader():
    print("--- [7/13] Testing create_dataloader (with spawn multiprocessing) ---")
    loader = create_dataloader(
        dataset_dir=DATA_DIR,
        split="val",
        batch_size=4,
        max_len=128,
        num_dataloader_workers=1,  # Uses spawn context on Windows!
        shuffle=True,
        streamed=True,
    )
    batch = next(iter(loader))
    landmarks = batch.get("feature", batch.get("landmarks", batch.get("feat")))
    print(f"  DataLoader batch landmarks shape: {landmarks.shape}")
    assert landmarks.shape[0] == 4, f"Expected batch size 4, got {landmarks.shape[0]}"
    print("  [PASS] create_dataloader with spawn context working as expected.")


def test_apply_dae_corruptions():
    print("--- [8/13] Testing apply_dae_corruptions ---")
    tokens = [1, 10, 20, 30, 40, 50, 2]
    corrupted = apply_dae_corruptions(tokens, unk_id=3, mask_prob=0.15, drop_prob=0.10, shuffle_prob=0.10)
    assert isinstance(corrupted, list), "DAE corrupted output is not a list"
    print(f"  Original tokens len: {len(tokens)}, corrupted len: {len(corrupted)}")
    print("  [PASS] apply_dae_corruptions working as expected.")


def test_aslg_pc12_dataset():
    print("--- [9/13] Testing ASLGPC12Dataset ---")
    eng_vocab = EnglishVocabulary(vocab_path=os.path.join(DATA_DIR, "english_vocab.json"))
    gloss_vocab = GlossVocabulary(label_to_idx={})
    aslg_ds = ASLGPC12Dataset(csv_path=ASLG_CSV, eng_vocab=eng_vocab, gloss_vocab=gloss_vocab, max_len=128)
    print(f"  ASLGPC12Dataset length: {len(aslg_ds)}")
    assert len(aslg_ds) > 0, "ASLGPC12Dataset is empty"
    item = aslg_ds[0]
    assert "input_ids" in item and "target_ids" in item, f"Missing keys in ASLG item: {item.keys()}"
    print(f"  ASLG sample input_ids len: {len(item['input_ids'])}, target_ids len: {len(item['target_ids'])}")
    print("  [PASS] ASLGPC12Dataset working as expected.")


def test_kdwd_dataset():
    print("--- [10/13] Testing KDWDDataset ---")
    eng_vocab = EnglishVocabulary(vocab_path=os.path.join(DATA_DIR, "english_vocab.json"))
    kdwd_ds = KDWDDataset(kdwd_dir=KDWD_DIR, eng_vocab=eng_vocab, max_len=128, views_threshold=0)
    print("  KDWDDataset initialized successfully.")
    print("  [PASS] KDWDDataset working as expected.")


def test_phase1_mixed_iterable():
    print("--- [11/13] Testing Phase1MixedIterable ---")
    eng_vocab = EnglishVocabulary(vocab_path=os.path.join(DATA_DIR, "english_vocab.json"))
    gloss_vocab = GlossVocabulary(label_to_idx={})
    mixed_stream = Phase1MixedIterable(
        kdwd_dir="",
        aslg_csv=ASLG_CSV,
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=128,
    )
    batch = []
    for i, sample in enumerate(mixed_stream):
        batch.append(sample)
        if i >= 3:
            break
    assert len(batch) == 4, f"Expected 4 items from Phase1MixedIterable, got {len(batch)}"
    print("  [PASS] Phase1MixedIterable working as expected.")


def test_collate_functions():
    print("--- [12/13] Testing phase1_collate_fn & phase2_collate_fn ---")
    dummy_phase1_batch = [
        {"input_ids": [1, 10, 20, 2], "target_ids": [1, 5, 2], "is_dae": False},
        {"input_ids": [1, 15, 25, 35, 2], "target_ids": [1, 8, 9, 2], "is_dae": False},
    ]
    p1_collated = phase1_collate_fn(dummy_phase1_batch, max_len=30, eng_pad_id=0)
    assert "input_ids" in p1_collated and "target_ids" in p1_collated, f"Keys: {p1_collated.keys()}"
    print(f"  phase1_collate_fn input_ids shape: {p1_collated['input_ids'].shape}")

    dummy_phase2_batch = [
        {"input_ids": [1, 5, 10, 2], "target_ids": [1, 8, 2]},
        {"input_ids": [1, 15, 25, 2], "target_ids": [1, 9, 2]},
    ]
    p2_collated = phase2_collate_fn(dummy_phase2_batch, max_len=30, eng_pad_id=0)
    assert "input_ids" in p2_collated, f"Keys: {p2_collated.keys()}"
    print(f"  phase2_collate_fn input_ids shape: {p2_collated['input_ids'].shape}")
    print("  [PASS] phase1_collate_fn & phase2_collate_fn working as expected.")


def test_cache_clearing():
    print("--- [13/13] Testing clear_global_dataset_caches ---")
    clear_global_dataset_caches()
    print("  [PASS] clear_global_dataset_caches working as expected.")


if __name__ == "__main__":
    print("=== STARTING DATASET FEATURE EMPIRICAL VERIFICATION ===")
    test_vocabularies()
    test_landmark_augmenter()
    test_motion_aware_sample()
    test_asl_sharded_dataset()
    test_shard_preserving_sampler()
    test_asl_streamed_dataset()
    test_create_dataloader()
    test_apply_dae_corruptions()
    test_aslg_pc12_dataset()
    test_kdwd_dataset()
    test_phase1_mixed_iterable()
    test_collate_functions()
    test_cache_clearing()
    print("=== ALL 13 DATASET FEATURE TESTS PASSED SUCCESSFULLY! ===")
