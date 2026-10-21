"""Checkpoint compaction CLI; inspection and importing are backend-free."""

import argparse
import time
from pathlib import Path

from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord
from fpcode.kpms.checkpoint_io import compact_kpms_checkpoint, inspect_checkpoint_iterations

logger = get_logger("fpcode.cli.export")


def _positive_integer(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compact a KPMS checkpoint while retaining selected final snapshots.")
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--keep-final", type=_positive_integer, default=1,
                        help="Number of final snapshot iterations to retain.")
    parser.add_argument("--output", "-o", type=Path,
                        help="Destination checkpoint; omission preserves the historical in-place behavior.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect the retention plan without writing.")
    parser.add_argument("--no-notify", action="store_true")
    return parser.parse_args(argv)


def run_export(args):
    record = None
    started = time.monotonic()
    try:
        checkpoint = resolve_path(args.checkpoint)
        if isinstance(args.keep_final, bool) or not isinstance(args.keep_final, int) or args.keep_final < 1:
            raise ValueError("keep_final must be a positive integer")
        iterations = inspect_checkpoint_iterations(checkpoint)
        if not iterations:
            raise ValueError("Checkpoint has no numeric model snapshots")
        retained = iterations[-args.keep_final:]
        pruned = iterations[:-args.keep_final]
        destination = resolve_path(args.output) if args.output is not None else checkpoint
        logger.info("Checkpoint %s: retaining %s; pruning %s", checkpoint, retained, pruned)
        if args.dry_run:
            return 0
        config = {"checkpoint": str(checkpoint), "output": str(destination), "keep_iterations": retained}
        record = RunRecord(pipeline="export", output_dir=destination.parent, config=config,
                           extra={"retained": retained, "pruned": pruned})
        before = checkpoint.stat().st_size
        output = compact_kpms_checkpoint(checkpoint, keep_iterations=retained, dest_path=destination)
        reclaimed = before - output.stat().st_size
        record.finish(status="SUCCESS", output_path=str(output), extra={"reclaimed_bytes": reclaimed})
    except Exception as error:
        logger.error("Checkpoint compaction failed: %s", error)
        if record is not None:
            record.finish(status="FAILED", error=str(error))
        return 1
    if not args.no_notify:
        try:
            send_ntfy(title=f"Compacted Checkpoint: {checkpoint.name}",
                      msg=f"Retained snapshots {retained}; reclaimed {reclaimed / 1024**2:.1f} MB "
                          f"in {time.monotonic() - started:.1f}s.",
                      priority="default", tags="package,white_check_mark")
        except Exception as error:
            logger.warning("Completion notification failed: %s", error)
    return 0


def main(argv=None):
    return run_export(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
