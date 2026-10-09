"""Multi-camera fusion command-line interface."""

import argparse
from pathlib import Path

from fpcode.common.config import compute_config_hash, load_yaml_config
from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path, resolve_data_path
from fpcode.common.run_record import RunRecord
from fpcode.workflows.fusion import (
    execute_fusion, find_matched_pairs, fuse_single_pair, fuse_single_pair_static, fusion_options,
)

logger = get_logger("fpcode.cli.fuse")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fuse matched camera tracking using YAML settings.")
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    return parser.parse_args(argv)


def run_fusion(args):
    record = None
    try:
        config_path = resolve_path(args.config)
        source = load_yaml_config(config_path)
        if not isinstance(source, dict):
            raise ValueError("Fusion configuration must be a mapping")
        cfg = dict(source)
        if getattr(args, "strict", None) is not None:
            cfg["strict_calibration"] = args.strict
        _, options = fusion_options(cfg)
        top = resolve_data_path(cfg.get("top_dir", "karen_project/h5_files"))
        bottom = resolve_data_path(cfg.get("bot_dir", "karen_project/botcam_h5_files"))
        output_dir = resolve_path(cfg.get("output_dir", "data/fused"))
        if not args.dry_run:
            record = RunRecord(pipeline="fuse", output_dir=output_dir,
                               config={"mode": cfg.get("mode", "static_21kp"), "top_dir": str(top),
                                       "bot_dir": str(bottom), "output_dir": str(output_dir), **options},
                               config_path=str(config_path), config_hash=compute_config_hash(source),
                               extra={"mode": cfg.get("mode", "static_21kp"), "resolved_options": options})
        if not top.is_dir() or not bottom.is_dir():
            raise FileNotFoundError(f"Fusion input directories must exist: {top}, {bottom}")
        pairs = find_matched_pairs(top_h5_dir=top, bot_prod_dir=bottom)
        if not pairs:
            raise ValueError("No matched input pairs; fusion would produce no outputs")
        if args.dry_run:
            logger.info(f"Validated {len(pairs)} fusion pair(s): {options}")
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        result = execute_fusion(pairs, cfg, output_dir, static_backend=fuse_single_pair_static,
                                dynamic_backend=fuse_single_pair)
        record.finish(status=result["status"], output_path=str(output_dir), extra=result)
    except Exception as error:
        logger.error(f"Fusion failed: {error}")
        if record is not None:
            record.finish(status="FAILED", error=str(error))
        return 1
    if not args.no_notify:
        try:
            send_ntfy(title=f"Fusion Completed: {config_path.name}",
                      msg=f"Fused {result['success']} sessions ({result['failed']} failed).",
                      priority="default" if not result["failed"] else "high",
                      tags="white_check_mark" if not result["failed"] else "warning")
        except Exception as error:
            logger.warning(f"Completion notification failed: {error}")
    return 0 if not result["failed"] else 1


def main(argv=None):
    return run_fusion(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
