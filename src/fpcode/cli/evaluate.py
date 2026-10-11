"""Read-only checkpoint evaluation CLI."""

import argparse
from pathlib import Path

from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord
from fpcode.kpms.evaluation import read_checkpoint_identity, validate_model_compatibility
from fpcode.workflows.evaluation import evaluate_model_checkpoints

logger = get_logger("fpcode.cli.evaluate")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit KPMS checkpoints and evaluate syllable frequencies.")
    parser.add_argument("--models", "-m", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reindex", action="store_true", help="Record stable frequency label mappings; source files stay unchanged.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    return parser.parse_args(argv)


def run_evaluation(args):
    record = None
    try:
        paths = [resolve_path(path) for path in args.models]
        checkpoints = [path / "checkpoint.h5" if path.is_dir() else path for path in paths]
        if not checkpoints:
            raise ValueError("At least one checkpoint is required")
        output_dir = resolve_path(args.output_dir) if getattr(args, "output_dir", None) else checkpoints[0].parent
        if not args.dry_run:
            record = RunRecord(pipeline="evaluate", output_dir=output_dir,
                               config={"checkpoints": [str(p) for p in checkpoints], "output_dir": str(output_dir),
                                       "reindex": getattr(args, "reindex", False)},
                               extra={"n_models": len(checkpoints), "models": [str(p) for p in checkpoints],
                                      "reindex": getattr(args, "reindex", False)})
        identities = {str(path): read_checkpoint_identity(path) for path in checkpoints}
        if len(identities) > 1:
            validate_model_compatibility(identities)
        if args.dry_run:
            logger.info(f"Validated {len(checkpoints)} checkpoint(s)")
            return 0
        results = evaluate_model_checkpoints(checkpoints, output_dir=output_dir)
        record.finish(status="SUCCESS", extra={"results": results})
    except Exception as error:
        logger.error(f"Evaluation failed: {error}")
        if record is not None:
            record.finish(status="FAILED", error=str(error))
        return 1
    if not args.no_notify:
        try:
            send_ntfy(title=f"Evaluation Finished: {len(checkpoints)} Models", msg="Checkpoint audit and frequency evaluation completed.",
                      priority="default", tags="white_check_mark,bar_chart")
        except Exception as error:
            logger.warning(f"Completion notification failed: {error}")
    return 0


def main(argv=None):
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
