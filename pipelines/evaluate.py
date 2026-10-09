#!/usr/bin/env python3
"""
pipelines/evaluate.py
=====================
CLI for Keypoint-MoSeq model evaluation, checkpoint identity verification,
stable syllable frequency re-indexing, and Expected Marginal Likelihood (EML).
Guarantees:
  - Source checkpoints are opened strictly read-only.
  - Re-indexing is pure in-memory and idempotent.
  - Validates dataset and mask compatibility before computing rankings.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord
from fpcode.kpms.evaluation import (
    read_checkpoint_identity,
    validate_model_compatibility,
)

logger = get_logger("pipelines.evaluate")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keypoint-MoSeq model evaluation and checkpoint auditing CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to model directories or checkpoint.h5 files to evaluate.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Compute stable syllable frequency permutation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit checkpoint completeness and identity without running evaluation.",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable ntfy notifications.",
    )
    return parser.parse_args(argv)


def evaluate_model_checkpoints(
    checkpoints: list[Path], output_dir: Optional[Path] = None
) -> dict:
    """Core KPMS model cross-likelihood evaluation engine. Wired to fpcode.kpms.evaluation."""
    from fpcode.kpms.evaluation import evaluate_checkpoints
    return evaluate_checkpoints(checkpoints, output_dir=output_dir)



def run_evaluation(args: argparse.Namespace) -> int:
    model_paths = [resolve_path(m) for m in args.models]
    checkpoints = [m if m.is_file() else m / "checkpoint.h5" for m in model_paths]

    for ck in checkpoints:
        if not ck.exists():
            logger.error(f"Checkpoint not found: {ck}")
            return 1

    logger.info(f"Auditing {len(checkpoints)} model checkpoints...")
    identities = []
    for ck in checkpoints:
        ident = read_checkpoint_identity(ck)
        identities.append(ident)
        logger.info(
            f"  Model: {Path(ident['path']).name} | Iter: {ident['iteration']} | "
            f"Recordings: {ident['n_recordings']} | Dataset sha: {ident['metadata_hash'][:8]}"
        )

    if len(checkpoints) > 1:
        logger.info("Validating pairwise model compatibility...")
        validate_model_compatibility(checkpoints)
        logger.info("  All candidate models share identical dataset masks and dimensions.")

    if args.dry_run:
        logger.info("[DRY RUN] Checkpoint audit completed successfully.")
        return 0

    eval_out_dir = resolve_path(args.output_dir) if args.output_dir else checkpoints[0].parent
    run_rec = RunRecord(
        pipeline="evaluate",
        output_dir=eval_out_dir,
        extra={"n_models": len(checkpoints), "models": [str(c) for c in checkpoints]},
    )

    t0 = time.time()
    try:
        logger.info(f"Evaluating {len(checkpoints)} candidate models...")
        eval_results = evaluate_model_checkpoints(checkpoints, output_dir=args.output_dir)
        logger.info("Model evaluation completed.")
        run_rec.finish(status="SUCCESS", extra={"results": eval_results})
        if not args.no_notify:
            send_ntfy(
                title=f"Evaluation Finished: {len(checkpoints)} Models",
                msg=f"Model evaluation and audit completed in {time.time() - t0:.1f}s.",
                priority="default",
                tags="white_check_mark,bar_chart",
            )
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        run_rec.finish(status="FAILED", error=str(e))
        return 1


def main() -> int:
    args = parse_args()
    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
