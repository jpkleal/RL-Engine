from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from torch.utils.tensorboard import SummaryWriter

PathLike = Union[str, Path]


class MetricsLogger:
    def __init__(self, log_dir: PathLike):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step=step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
