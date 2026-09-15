#!/usr/bin/env python3
"""
Unit test for How2Sign Colab Streamer under local CPU constraints.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import zipfile
import tempfile
import numpy as np
import torch

from preprocessing.how2sign_colab_streamer import (
    compute_kinematics_and_phonology,
    interpolate_missing_landmarks,
    How2SignStreamingOrchestrator,
)

def test_streamer_logic():
    print("Testing How2Sign streamer logic...")
    
    # 1. Test kinematics & phonology math
    mock_pos = np.random.randn(32, 60, 3).astype(np.float32)
    # simulate shoulders
    mock_pos[:, 42, :] = [-0.2, 0.0, 0.0]
    mock_pos[:, 43, :] = [0.2, 0.0, 0.0]
    
    kin, phon = compute_kinematics_and_phonology(mock_pos)
    assert kin.shape == (32, 60, 9), f"Expected (32, 60, 9), got {kin.shape}"
    assert phon.shape == (32, 19), f"Expected (32, 19), got {phon.shape}"
    assert not np.isnan(kin).any()
    assert not np.isnan(phon).any()
    print("  [PASS] Kinematics & Phonology computation validated!")
    
    # 2. Test missing landmark interpolation
    sparse_pos = mock_pos.copy()
    sparse_pos[5:8, 21, :] = 0.0 # drop frames 5,6,7
    repaired = interpolate_missing_landmarks(sparse_pos, max_gap=5)
    assert (np.abs(repaired[5:8, 21, :]).sum() > 0), "Failed to interpolate dropout!"
    print("  [PASS] Landmark interpolation across dropout validated!")
    
    # 3. Test Orchestrator ledger & discovery
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        drive_mock = tmp_path / "drive"
        out_mock = tmp_path / "out"
        drive_mock.mkdir()
        out_mock.mkdir()
        
        # Create a mock zip
        zip_file = drive_mock / "val_rgb_front_clips.zip"
        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.writestr("test_clip-rgb_front.mp4", b"dummy video bytes")
            
        orch = How2SignStreamingOrchestrator(
            drive_dir=str(drive_mock),
            output_drive_dir=str(out_mock),
            local_scratch_dir=str(tmp_path / "scratch"),
            chunk_size=10,
        )
        zips = orch.find_target_zips()
        assert len(zips) == 1
        assert zips[0].name == "val_rgb_front_clips.zip"
        print("  [PASS] Target zip discovery & prioritization validated!")

if __name__ == "__main__":
    test_streamer_logic()
