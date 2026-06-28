"""
main.py
=======

Entry point for RL-Engine training.

Usage:
    python main.py                        # uses config.toml in same directory
    python main.py --config other.toml
    python main.py --epochs 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_engine.main")


def build_algorithm(cfg):
    t = cfg.training
    role_ids = t.role_ids

    if t.algorithm == "sac":
        from rl_engine.algorithms.sac import SharedCriticSAC
        return SharedCriticSAC(role_ids, t.state_dim, t.action_dim, cfg.sac)

    elif t.algorithm == "ppo":
        from rl_engine.algorithms.ppo import SharedCriticPPO
        return SharedCriticPPO(role_ids, t.state_dim, t.action_dim, cfg.ppo)

    else:
        raise ValueError(f"unknown algorithm {t.algorithm!r}, expected 'sac' or 'ppo'")


def build_trainer(cfg):
    from rl_engine.metrics_logger import MetricsLogger
    from rl_engine.save_system import SaveSystem
    from rl_engine.test_abstraction import TestAbstraction, TestCase
    from rl_engine.trainer import Trainer

    t = cfg.training
    return Trainer(
        algorithm=build_algorithm(cfg),
        test_abstraction=TestAbstraction(
            host=cfg.runner.host,
            port=cfg.runner.port,
            connect_timeout=cfg.runner.connect_timeout,
            ack_timeout=cfg.runner.ack_timeout,
        ),
        save_system=SaveSystem(cfg.paths.save_dir),
        metrics_logger=MetricsLogger(cfg.paths.log_dir),
        test_case=TestCase(t.test_case),
        input_module=t.input_module,
        rollout_batch_size=t.rollout_batch_size,
        save_every_n_epochs=t.save_every_n_epochs,
    )


def main():
    parser = argparse.ArgumentParser(description="RL-Engine training")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config not found: %s", config_path)
        sys.exit(1)

    from rl_engine.config import Config
    cfg = Config.from_toml(config_path)

    logger.info(
        "algorithm=%s  roles=%s  state_dim=%d  action_dim=%d",
        cfg.training.algorithm,
        cfg.training.role_ids,
        cfg.training.state_dim,
        cfg.training.action_dim,
    )

    trainer = build_trainer(cfg)
    logger.info(
        "Running %s",
        f"for {args.epochs} epochs" if args.epochs else "indefinitely (Ctrl+C to stop)",
    )

    try:
        trainer.run(num_epochs=args.epochs)
    except KeyboardInterrupt:
        logger.info("Interrupted — saving final checkpoint")
        trainer.save_system.save(epoch=trainer._epoch, algorithm=trainer.algorithm)
        trainer.metrics_logger.close()
        logger.info("Done")


if __name__ == "__main__":
    main()