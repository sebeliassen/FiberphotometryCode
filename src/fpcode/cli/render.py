"""Visualization CLI that honors requested artifact types and syllable subsets."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fpcode.common.config import compute_config_hash, load_yaml_config
from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord
from fpcode.viz.grid_movies import exact_grid_layout as grid_layout
from fpcode.workflows.rendering import load_render_inputs, plan_rendering, render_model_visualizations

logger = get_logger("fpcode.cli.render")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render KPMS grid movies or trajectory figures.")
    parser.add_argument("--results", "-r", type=Path, required=True)
    parser.add_argument("--type", choices=["grid_movies", "trajectories", "all"], default="all")
    parser.add_argument("--syllables", "-s", nargs="+", type=int)
    parser.add_argument("--grid-size", "-g", type=int, default=9)
    parser.add_argument("--manifest", type=Path, help="YAML associating recordings with coordinates/video paths and FPS.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    return parser.parse_args(argv)


def run_render(args):
    record = None
    try:
        results_path = resolve_path(args.results)
        if results_path.is_dir():
            results_path = results_path / "results.h5"
        manifest = resolve_path(args.manifest) if getattr(args, "manifest", None) else None
        grid_layout(args.grid_size)
        output_dir = resolve_path(args.output_dir) if getattr(args, "output_dir", None) else (
            results_path.parent / "renders" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        if args.dry_run:
            plan_rendering(load_render_inputs(results_path, manifest), args.type, args.grid_size, args.syllables)
            logger.info(f"Validated rendering inputs: {results_path}")
            return 0
        manifest_cfg = load_yaml_config(manifest) if manifest is not None else {}
        record = RunRecord(pipeline="render", output_dir=output_dir,
                           config={"results": str(results_path), "type": args.type, "grid_size": args.grid_size,
                                   "syllables": args.syllables, "output_dir": str(output_dir),
                                   "render_manifest": {"fps": 40, "sampling_mode": "density", **manifest_cfg}},
                           config_path=str(manifest) if manifest is not None else None,
                           config_hash=compute_config_hash(manifest_cfg) if manifest is not None else None,
                           extra={"results": str(results_path), "type": args.type, "grid_size": args.grid_size})
        kwargs = {"manifest": manifest, "output_dir": output_dir}
        output = render_model_visualizations(results_path, render_type=args.type, grid_size=args.grid_size,
                                            syllables=args.syllables, **kwargs)
        record.finish(status="SUCCESS", extra={"output": output})
    except Exception as error:
        logger.error(f"Rendering failed: {error}")
        if record is not None:
            record.finish(status="FAILED", error=str(error))
        return 1
    if not args.no_notify:
        try:
            send_ntfy(title=f"Rendering Complete: {args.type}", msg=f"Artifacts: {output_dir}",
                      priority="default", tags="movie_camera,white_check_mark")
        except Exception as error:
            logger.warning(f"Completion notification failed: {error}")
    return 0


def main(argv=None):
    return run_render(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
