#!/usr/bin/env python3
"""
pipelines/fuse.py
=================
Thin CLI entrypoint for multi-camera posture tracking fusion.
Synchronizes TopCam and BotCam video tracking using crystal clock calibration
and spatial affine transforms, writing 21kp / 23kp tracking files with sidecar manifests.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from fpcode.common.config import compute_config_hash, load_yaml_config
from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_data_path, resolve_path
from fpcode.common.run_record import RunRecord
from fpcode.tracking.fusion import fuse_single_pair
from fpcode.tracking.pair_matching import find_matched_pairs
from fpcode.tracking.static_fusion import fuse_single_pair_static

logger = get_logger("pipelines.fuse")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-camera posture tracking fusion CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to YAML fusion configuration file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enforce strict Basler crystal clock calibration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Match video pairs and verify inputs without executing fusion.",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable ntfy push notifications.",
    )
    return parser.parse_args(argv)


def run_fusion(args: argparse.Namespace) -> int:
    config_path = resolve_path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    cfg = load_yaml_config(config_path)
    cfg_hash = compute_config_hash(cfg)
    logger.info(f"Loaded fusion config from {config_path.name} (SHA-256: {cfg_hash[:12]}...)")

    mode = cfg.get("mode", "static_21kp")
    top_dir = resolve_data_path(cfg.get("top_dir", "karen_project/h5_files"))
    bot_dir = resolve_data_path(cfg.get("bot_dir", "karen_project/botcam_h5_files"))
    out_dir = resolve_path(cfg.get("output_dir", "data/fused"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = (
        find_matched_pairs(top_h5_dir=top_dir, bot_prod_dir=bot_dir) if top_dir.exists() and bot_dir.exists() else []
    )
    logger.info(f"Identified {len(pairs)} matched session pairs (Mode: {mode})")

    if args.dry_run:
        logger.info("[DRY RUN] Fusion configuration and pair matching validated.")
        for p in pairs[:5]:
            logger.info(f"  Session: {p.get('sid')} | Top: {Path(p['top_h5']).name} | Bot: {Path(p['bot_h5']).name}")
        return 0

    run_rec = RunRecord(
        pipeline="fuse",
        output_dir=out_dir,
        config_path=str(config_path),
        config_hash=cfg_hash,
        extra={"mode": mode, "n_pairs": len(pairs)},
    )

    t0 = time.time()
    fuse_fn = fuse_single_pair_static if "static" in mode else fuse_single_pair
    n_success = 0
    n_failed = 0

    for pair in pairs:
        try:
            res = fuse_fn(pair, out_dir=out_dir)
            if res.get("status") in ("FUSED", "CACHED_VALID"):
                n_success += 1
            else:
                n_failed += 1
        except Exception as e:
            logger.error(f"Failed fusing {pair.get('sid')}: {e}")
            n_failed += 1

    dur = time.time() - t0
    status = "SUCCESS" if n_failed == 0 else "PARTIAL_SUCCESS"
    run_rec.finish(status=status, output_path=str(out_dir), extra={"success": n_success, "failed": n_failed})

    if not args.no_notify:
        send_ntfy(
            title=f"Fusion Completed: {config_path.name}",
            msg=f"Fused {n_success} sessions ({n_failed} failed) in {dur:.1f}s.",
            priority="default" if n_failed == 0 else "high",
            tags="white_check_mark" if n_failed == 0 else "warning",
        )
    return 0 if n_failed == 0 else 1


def main() -> int:
    args = parse_args()
    return run_fusion(args)


if __name__ == "__main__":
    sys.exit(main())
