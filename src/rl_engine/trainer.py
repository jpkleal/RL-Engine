"""
Trainer
=======

The training loop. One epoch:

    1. Snapshot current policies to disk (playing-service format).
    2. Call the test runner via TestAbstraction, get a result_file.
    3. Feed the result_file to algorithm.ingest() -- each role's
       algorithm pulls its own transitions out internally.
    4. If algorithm.ready(), call algorithm.update() and log metrics.
    5. Every save_every_n_epochs, persist via SaveSystem.
    6. Repeat.

The algorithm owns its buffers. Trainer has no knowledge of replay
memories, rollout buffers, batch sizes, or buffer kinds -- all of that
is encapsulated inside the algorithm.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from .algorithms.base import Algorithm
from .metrics_logger import MetricsLogger
from .model import build_model_update_event
from .save_system import SaveSystem
from .test_abstraction import TestAbstraction, TestCase, TestResult

logger = logging.getLogger("rl_engine.trainer")

PathLike = Union[str, Path]
TransitionsPathResolver = Callable[[TestResult], Optional[PathLike]]


class Trainer:
    def __init__(
        self,
        algorithm: Algorithm,
        test_abstraction: TestAbstraction,
        save_system: SaveSystem,
        metrics_logger: MetricsLogger,
        test_case: "TestCase | str",
        input_module: str,
        rollout_batch_size: int,
        save_every_n_epochs: int = 10,
        verbose_out: bool = True,
        transitions_path: Optional[TransitionsPathResolver] = None,
    ):
        self.algorithm = algorithm
        self.test_abstraction = test_abstraction
        self.save_system = save_system
        self.metrics_logger = metrics_logger
        self.test_case = test_case
        self.input_module = input_module
        self.rollout_batch_size = rollout_batch_size
        self.save_every_n_epochs = save_every_n_epochs
        self.verbose_out = verbose_out
        self._transitions_path = transitions_path or (lambda result: result.result_file)

        self._epoch = 0
        self._resume_if_possible()

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    def _resume_if_possible(self) -> None:
        ckpt = self.save_system.latest_training_checkpoint()
        if ckpt is None:
            logger.info("No checkpoint found; starting fresh")
            return
        self.algorithm.load_checkpoint(ckpt)
        latest = self.save_system.latest_epoch()
        self._epoch = (latest + 1) if latest is not None else 0
        logger.info("Resumed from %s at epoch %d", ckpt, self._epoch)

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #

    def run(self, num_epochs: Optional[int] = None) -> None:
        """Run until num_epochs is reached, or indefinitely if None."""
        while True:
            if num_epochs is not None and self._epoch >= num_epochs:
                logger.info("Reached %d epochs", num_epochs)
                return
            self._run_epoch()
            self._epoch += 1

    def _run_epoch(self) -> None:
        epoch = self._epoch
        t0 = time.monotonic()

        # 1. snapshot policies for the playing service
        playing_paths = self.algorithm.save_for_playing(self.save_system.current_dir)
        event_data = build_model_update_event(playing_paths)

        # 2. run the test
        result = self.test_abstraction.start_test(
            test_case=self.test_case,
            batch_size=self.rollout_batch_size,
            input_module=self.input_module,
            module_config={"event_type": "ModelUpdate", "event_data": event_data},
            verbose_out=self.verbose_out,
        )
        self.metrics_logger.log_scalar("rollout/success", float(result.success), step=epoch)

        if not result.success:
            logger.warning("epoch %d: test failed (%s)", epoch, result.error)
            self.metrics_logger.log_scalar("epoch/duration_seconds", time.monotonic() - t0, step=epoch)
            return

        # 3. ingest transitions
        path = self._transitions_path(result)
        if path is None:
            logger.warning("epoch %d: no transitions path", epoch)
            return

        try:
            counts = self.algorithm.ingest(path)
        except FileNotFoundError:
            logger.warning("epoch %d: transitions file not found: %s", epoch, path)
            return

        for role_id, n in counts.items():
            self.metrics_logger.log_scalar(f"{role_id}/transitions_ingested", n, step=epoch)

        # 4. update
        update_metrics: Dict[str, float] = {}
        if self.algorithm.ready():
            update_metrics = self.algorithm.update()
            self.metrics_logger.log_scalars(update_metrics, step=epoch)
        else:
            logger.debug("epoch %d: not ready yet, skipping update", epoch)

        # 5. maybe save
        if epoch % self.save_every_n_epochs == 0:
            self.save_system.save(epoch=epoch, algorithm=self.algorithm, metrics=update_metrics)

        self.metrics_logger.log_scalar("epoch/duration_seconds", time.monotonic() - t0, step=epoch)