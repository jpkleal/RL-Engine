"""
SaveSystem
==========

The "every n epochs, save the current networks both to files and to an
index" piece. Owns a directory laid out as:

    root/
      checkpoints/
        epoch_{n}.pt   -- algorithm.save_checkpoint() output (full
                           training state: all models, all optimizers,
                           any algorithm-specific scalars) at epoch n
        latest.pt       -- same, always overwritten with the most
                           recent save, for fast resume without
                           scanning the index
      current/
        {model_name}.pt -- algorithm.save_for_playing() output: the
                           playing-service-compatible checkpoint for
                           each model, always pointing at the latest
                           policy. This is the path set that should be
                           referenced in a TestAbstraction ModelUpdate
                           event (see model.build_model_update_event).
      replay_memory/
        {role_id}.pt    -- ReplayMemory.save() output, per role, where
                           applicable (off-policy only -- on-policy
                           RolloutBuffers are intentionally not
                           persisted; they're meant to be discarded
                           after each update, and stale on-policy data
                           is invalid to resume training from anyway)
      index.jsonl        -- one line per save: epoch, timestamp, paths,
                           metrics at that point
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("rl_engine.save_system")

PathLike = Union[str, Path]


class SaveSystem:
    def __init__(self, root_dir: PathLike):
        self.root = Path(root_dir)
        self.checkpoints_dir = self.root / "checkpoints"
        self.current_dir = self.root / "current"
        self.index_path = self.root / "index.jsonl"

        for d in (self.checkpoints_dir, self.current_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch: int,
        algorithm: "Algorithm",  # noqa: F821
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Saves full training state (networks + buffers in one checkpoint),
        playing-compatible model files, and appends to index.jsonl."""
        versioned_path = self.checkpoints_dir / f"epoch_{epoch}.pt"
        latest_path = self.checkpoints_dir / "latest.pt"
        algorithm.save_checkpoint(versioned_path)
        algorithm.save_checkpoint(latest_path)

        playing_paths = algorithm.save_for_playing(self.current_dir)

        entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            "training_checkpoint": str(versioned_path),
            "playing_checkpoints": playing_paths,
            "metrics": metrics or {},
        }
        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info("SaveSystem: saved epoch %d (%d model files)", epoch, len(playing_paths))
        return entry

    def latest_training_checkpoint(self) -> Optional[Path]:
        p = self.checkpoints_dir / "latest.pt"
        return p if p.exists() else None

    def load_index(self) -> List[dict]:
        if not self.index_path.exists():
            return []
        with open(self.index_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def latest_epoch(self) -> Optional[int]:
        index = self.load_index()
        return index[-1]["epoch"] if index else None