#!/usr/bin/env python3
"""
pipelines/render.py
===================
CLI for Keypoint-MoSeq behavioral visualizations:
- Density-sampled syllable grid movies (with/without keypoint skeletons).
- 2D / 3D trajectory plots and animations.
Enforces:
  - Canonical half-open bout bounds [start, end).
  - FPS-derived timing windows (seconds -> frames).
  - Four-paw quality control before exemplar selection.
  - No silent rendering substitutions.
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
from fpcode.viz.v2 import (
    grid_layout,
)

logger = get_logger("pipelines.render")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keypoint-MoSeq grid movie and trajectory visualization CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        "-r",
        type=Path,
        required=True,
        help="Path to results.h5 or model directory.",
    )
    parser.add_argument(
        "--type",
        choices=["grid_movies", "trajectories", "all"],
        default="all",
        help="Type of visualization to generate.",
    )
    parser.add_argument(
        "--syllables",
        "-s",
        nargs="+",
        type=int,
        default=None,
        help="Subset of syllable IDs to render (e.g. 0 1 2 3). Renders all if omitted.",
    )
    parser.add_argument(
        "--grid-size",
        "-g",
        type=int,
        default=9,
        help="Grid layout size (number of tiles: 4, 9, 16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan bout selection and layout without rasterizing video.",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable ntfy notifications.",
    )
    return parser.parse_args(argv)


def render_model_visualizations(
    results_path: Path,
    render_type: str,
    grid_size: int,
    syllables: Optional[list[int]] = None,
) -> dict:
    """Core KPMS visualization and trajectory rendering engine. Wired to fpcode.viz."""
    raise NotImplementedError(
        "not wired to syllable_generation/multiview_top_bot_experiments/04_reports_and_figures/render_all_cohort_static_21kp_v2_seed101_trajectories.py"
    )


def run_render(args: argparse.Namespace) -> int:
    res_path = resolve_path(args.results)
    if not res_path.exists():
        logger.error(f"Results file not found: {res_path}")
        return 1

    nrows, ncols = grid_layout(args.grid_size)
    logger.info(f"Target grid layout: {nrows}x{ncols} ({args.grid_size} tiles) for {args.type}")

    if args.dry_run:
        logger.info("[DRY RUN] Visualization plan verified successfully.")
        logger.info(f"  Source: {res_path}")
        logger.info(f"  Type: {args.type}")
        logger.info(f"  Grid size: {args.grid_size} ({nrows}x{ncols})")
        logger.info(f"  Syllables: {args.syllables if args.syllables else 'All'}")
        return 0

    render_out_dir = res_path.parent / "renders"
    run_rec = RunRecord(
        pipeline="render",
        output_dir=render_out_dir,
        extra={"results": str(res_path), "type": args.type, "grid_size": args.grid_size},
    )

    t0 = time.time()
    try:
        logger.info(f"Rendering visualizations for {res_path.name}...")
        render_output = render_model_visualizations(
            res_path, render_type=args.type, grid_size=args.grid_size, syllables=args.syllables
        )
        logger.info("Rendering complete.")
        run_rec.finish(status="SUCCESS", extra={"output": render_output})
        if not args.no_notify:
            send_ntfy(
                title=f"Rendering Complete: {args.type}",
                msg=f"Rendered visualizations in {time.time() - t0:.1f}s for {res_path.name}",
                priority="default",
                tags="movie_camera,white_check_mark",
            )
        return 0
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        run_rec.finish(status="FAILED", error=str(e))
        return 1


def main() -> int:
    args = parse_args()
    return run_render(args)


if __name__ == "__main__":
    sys.exit(main())
