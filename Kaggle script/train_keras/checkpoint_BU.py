"""
Production-Grade Atomic Checkpoint Manager for Keras 3 (JAX Backend)
Features:
  - Atomic directory writes (Save-to-temp -> Atomic rename) to avoid corruption
  - Manages model weights (.weights.h5), Optax optimizer state (.pkl), and metadata (.json)
  - Automatic keep_last_k pruning to avoid disk exhaustion
  - Tracking and preservation of best checkpoint based on loss
  - Full resumption support for seamless training restarts
"""

import os
import json
import shutil
import pickle
import glob
from typing import Optional, Dict, Any, Tuple


class CheckpointManager:
    """
    Manages periodic checkpointing and resumption for Keras 3 models with JAX backend.
    """
    def __init__(
        self,
        save_dir: str,
        keep_last_k: int = 5,
        save_best: bool = True,
    ):
        self.save_dir = os.path.abspath(save_dir)
        self.keep_last_k = max(1, keep_last_k)
        self.save_best = save_best
        self.best_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

    def save(
        self,
        epoch: int,
        step: int,
        model,
        opt_state: Any,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Saves model weights, optax optimizer state, and training metadata atomically.
        """
        ckpt_name = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}"
        temp_dir = os.path.join(self.save_dir, f".temp_{ckpt_name}")
        final_dir = os.path.join(self.save_dir, ckpt_name)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)

        # 1. Save Model Weights
        weights_path = os.path.join(temp_dir, "model.weights.h5")
        model.save_weights(weights_path)

        # 2. Save Optimizer State
        opt_path = os.path.join(temp_dir, "opt_state.pkl")
        with open(opt_path, "wb") as f:
            pickle.dump(opt_state, f)

        # 3. Save Metadata JSON
        meta = {
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }
        meta_path = os.path.join(temp_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 4. Atomic directory commit
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir, ignore_errors=True)
        os.rename(temp_dir, final_dir)

        # 5. Update latest marker file
        latest_file = os.path.join(self.save_dir, "latest_checkpoint.txt")
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(ckpt_name)

        # 6. Track and save best checkpoint if applicable
        current_loss = metrics.get("loss", None) if metrics else None
        if current_loss is not None and self.save_best and current_loss < self.best_loss:
            self.best_loss = current_loss
            best_dir = os.path.join(self.save_dir, "checkpoint_best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir, ignore_errors=True)
            shutil.copytree(final_dir, best_dir)

        # 7. Prune older checkpoints to keep at most keep_last_k
        self._prune_checkpoints()

        return final_dir

    def _prune_checkpoints(self):
        """Removes older checkpoints exceeding keep_last_k."""
        pattern = os.path.join(self.save_dir, "checkpoint_epoch_*")
        ckpt_dirs = sorted(glob.glob(pattern))
        while len(ckpt_dirs) > self.keep_last_k:
            oldest = ckpt_dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    def load_latest(
        self,
        model,
    ) -> Tuple[int, int, Optional[Any], Dict[str, Any]]:
        """
        Loads the most recent checkpoint if available.
        Returns: (epoch, step, opt_state, meta)
        """
        latest_file = os.path.join(self.save_dir, "latest_checkpoint.txt")
        if not os.path.exists(latest_file):
            return 0, 0, None, {}

        with open(latest_file, "r", encoding="utf-8") as f:
            latest_name = f.read().strip()

        ckpt_dir = os.path.join(self.save_dir, latest_name)
        return self.load_checkpoint(ckpt_dir, model)

    def load_checkpoint(
        self,
        ckpt_dir: str,
        model,
    ) -> Tuple[int, int, Optional[Any], Dict[str, Any]]:
        """
        Loads a specific checkpoint directory into model.
        Returns: (epoch, step, opt_state, meta)
        """
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found!")

        weights_path = os.path.join(ckpt_dir, "model.weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)

        opt_path = os.path.join(ckpt_dir, "opt_state.pkl")
        opt_state = None
        if os.path.exists(opt_path):
            with open(opt_path, "rb") as f:
                opt_state = pickle.load(f)

        meta_path = os.path.join(ckpt_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        epoch = meta.get("epoch", 0)
        step = meta.get("step", 0)
        return epoch, step, opt_state, meta
