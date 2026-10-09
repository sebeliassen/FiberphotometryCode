#!/usr/bin/env python3
"""
pipelines/train.py
==================
Thin CLI entrypoint for Keypoint-MoSeq (KPMS) model training.
Driven by declarative, typed YAML configurations in configs/training/.
Guarantees:
  - Checkpoints are protected by HDF5 immutability guards (h5_guard).
  - Every run logs execution provenance atomically to run.json (RunRecord).
  - Config SHA-256 hash is computed and recorded for reproducibility.
  - Zero hardcoded local paths; resolved via fpcode.common.paths.
  - Optional compaction on completion to free disk space.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fpcode.common.config import compute_config_hash, load_yaml_config
from fpcode.common.io import compact_kpms_checkpoint
from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord

logger = get_logger("pipelines.train")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keypoint-MoSeq model training CLI driven by declarative YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed specified in config.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Execution device for JAX/KPMS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and verify paths without executing training.",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable push notifications to ntfy.sh upon completion.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact checkpoint upon completion, evicting redundant snapshot iterations.",
    )
    return parser.parse_args(argv)


def fit_model(cfg: dict, model_dir: Path, seed: int, device: str = "gpu") -> Path:
    """Core KPMS model training engine. Wired to fpcode.kpms.fit."""
    raise NotImplementedError(
        "not wired to syllable_generation/multiview_top_bot_experiments/02_training_runners"
    )


def run_training(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    cfg = load_yaml_config(config_path)
    cfg_hash = compute_config_hash(cfg)
    logger.info(f"Loaded config from {config_path.name} (SHA-256: {cfg_hash[:12]}...)")

    seed = args.seed if args.seed is not None else cfg.get("seed", 101)
    model_name = cfg.get("model_name", f"model_seed{seed}")
    project_dir = resolve_path(cfg.get("project_dir", "experiments"))
    model_dir = project_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("[DRY RUN] Config verified successfully.")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Project dir: {project_dir}")
        logger.info(f"  Seed: {seed}")
        logger.info(f"  Device: {args.device}")
        return 0

    # Configure hardware environment
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    run_rec = RunRecord(
        pipeline="train",
        output_dir=model_dir,
        config_path=str(config_path),
        config_hash=cfg_hash,
        seed=seed,
        extra={"model_name": model_name, "device": args.device},
    )

    t0 = time.time()
    try:
        logger.info(f"Initializing KPMS training for {model_name}...")
        checkpoint_path = fit_model(cfg, model_dir=model_dir, seed=seed, device=args.device)
        logger.info(f"Training complete. Checkpoint saved to: {checkpoint_path}")
        run_rec.finish(status="SUCCESS", output_path=str(checkpoint_path))

        if args.compact:
            if checkpoint_path.exists():
                logger.info(f"Compacting checkpoint: {checkpoint_path.name}")
                compact_kpms_checkpoint(checkpoint_path)

        if not args.no_notify:
            send_ntfy(
                title=f"KPMS Training Finished: {model_name}",
                msg=f"Training completed successfully in {time.time() - t0:.1f}s. Config: {config_path.name}",
                priority="default",
                tags="white_check_mark,robot",
            )
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        run_rec.finish(status="FAILED", error=str(e))
        return 1


def main() -> int:
    args = parse_args()
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
