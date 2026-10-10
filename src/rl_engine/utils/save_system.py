from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("rl_engine.save_system")

PathLike = Union[str, Path]


class SaveSystem:
    def __init__(self, root_dir: PathLike, keep_last: int = 5):
        self.root = Path(root_dir)
        self.checkpoints_dir = self.root / "checkpoints"
        self.current_dir = self.root / "current"
        self.index_path = self.root / "index.jsonl"
        # Each SAC checkpoint embeds the whole replay buffer, so they run
        # to tens of megabytes. Keep only the most recent few.
        self.keep_last = keep_last

        for d in (self.checkpoints_dir, self.current_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _prune_old_checkpoints(self) -> None:
        """Deletes epoch_*.pt beyond the newest `keep_last`.

        Sorted by epoch number, not by name -- epoch_10 sorts before
        epoch_9 alphabetically. latest.pt is never touched.
        """
        if self.keep_last is None or self.keep_last < 0:
            return

        versioned = []
        for path in self.checkpoints_dir.glob("epoch_*.pt"):
            try:
                epoch = int(path.stem.split("_", 1)[1])
            except ValueError:
                continue
            versioned.append((epoch, path))

        for _, path in sorted(versioned, reverse=True)[self.keep_last:]:
            logger.info("SaveSystem: pruning old checkpoint %s", path.name)
            path.unlink(missing_ok=True)

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

        self._prune_old_checkpoints()

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
