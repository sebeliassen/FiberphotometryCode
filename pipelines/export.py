#!/usr/bin/env python3
"""
pipelines/export.py
===================
CLI for Keypoint-MoSeq artifact export and compaction.
Compact multi-GB checkpoints by evicting dead snapshot iterations while
preserving final states and hyperparameter metadata, and extract results.h5 / results.pkl.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from fpcode.common.io import compact_kpms_checkpoint, inspect_checkpoint_iterations
from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord

logger = get_logger("pipelines.export")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keypoint-MoSeq checkpoint compaction and results export CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to checkpoint.h5 file to compact or export.",
    )
    parser.add_argument(
        "--keep-final",
        type=int,
        default=1,
        help="Number of final snapshot iterations to retain.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional destination path for compacted checkpoint. Overwrites in-place if None.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect checkpoint iterations and calculate potential disk savings without writing.",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable ntfy notifications.",
    )
    return parser.parse_args(argv)


def run_export(args: argparse.Namespace) -> int:
    ckpt_path = resolve_path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    iters = inspect_checkpoint_iterations(ckpt_path)
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    logger.info(f"Checkpoint {ckpt_path.name}: {size_mb:.1f} MB, {len(iters)} snapshots: {iters}")

    if len(iters) <= args.keep_final:
        logger.info(f"Checkpoint already has <= {args.keep_final} snapshots. No compaction needed.")
        return 0

    retained = iters[-args.keep_final :]
    pruned = iters[: -args.keep_final]

    if args.dry_run:
        logger.info("[DRY RUN] Compaction plan:")
        logger.info(f"  Retaining {len(retained)} snapshots: {retained}")
        logger.info(f"  Pruning {len(pruned)} snapshots: {pruned}")
        estimated_savings = size_mb * (len(pruned) / len(iters))
        logger.info(f"  Estimated disk space reclaim: ~{estimated_savings:.1f} MB")
        return 0

    run_rec = RunRecord(
        pipeline="export",
        extra={"checkpoint": str(ckpt_path), "retained": retained, "pruned": pruned},
    )

    t0 = time.time()
    try:
        logger.info(f"Compacting {ckpt_path.name} in bounded memory...")
        out = compact_kpms_checkpoint(ckpt_path, keep_iterations=args.keep_final, dest_path=args.output)
        new_size_mb = out.stat().st_size / (1024 * 1024)
        saved_mb = size_mb - new_size_mb
        logger.info(
            f"Compacted {ckpt_path.name}: {size_mb:.1f} MB -> {new_size_mb:.1f} MB (Reclaimed {saved_mb:.1f} MB)"
        )

        run_rec.finish(status="SUCCESS", output_path=str(out), extra={"reclaimed_mb": saved_mb})
        if not args.no_notify:
            send_ntfy(
                title=f"Compacted Checkpoint: {ckpt_path.name}",
                msg=f"Reclaimed {saved_mb:.1f} MB disk space in {time.time() - t0:.1f}s. Retained snapshots: {retained}",
                priority="default",
                tags="package,white_check_mark",
            )
        return 0

    except Exception as e:
        logger.error(f"Compaction failed: {e}", exc_info=True)
        run_rec.finish(status="FAILED", error=str(e))
        return 1


def main() -> int:
    args = parse_args()
    return run_export(args)


if __name__ == "__main__":
    sys.exit(main())
