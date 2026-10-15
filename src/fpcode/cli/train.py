"""Command-line training interface; importing it does not initialize JAX."""

import argparse
import os
from pathlib import Path

from fpcode.common.config import compute_config_hash, load_yaml_config
from fpcode.common.logging import get_logger
from fpcode.common.notify import send_ntfy
from fpcode.common.paths import resolve_path
from fpcode.common.run_record import RunRecord
from fpcode.kpms.checkpoint_io import compact_kpms_checkpoint
from fpcode.kpms.config import resolve_training_config
from fpcode.workflows.training import fit_model, training_input_path, validate_training_input

logger = get_logger("fpcode.cli.train")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a KPMS model from validated YAML settings.")
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args(argv)


def run_training(args):
    try:
        config_path = resolve_path(args.config)
        source = load_yaml_config(config_path)
        cfg = resolve_training_config(source)
        seed = args.seed if args.seed is not None else cfg.get("seed", 101)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        name = cfg.get("model_name", f"model_seed{seed}")
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("model_name must be a single directory name")
        model_dir = resolve_path(cfg.get("project_dir", "experiments")) / name
        if args.dry_run:
            validate_training_input(cfg, model_dir)
            logger.info(f"Validated {name}: {cfg['num_ar_iters']} AR + {cfg['num_slds_iters']} SLDS iterations")
            return 0
    except Exception as error:
        logger.error(f"Invalid training request: {error}")
        return 1

    cfg.update(seed=seed, device=args.device, model_name=name, project_dir=str(model_dir.parent),
               data_path=str(training_input_path(cfg, model_dir)),
               pca_path=str(resolve_path(cfg["pca_path"])) if cfg.get("pca_path") else str(model_dir.parent / "pca.p"))
    record = RunRecord(pipeline="train", output_dir=model_dir, config=cfg,
                       config_path=str(config_path), config_hash=compute_config_hash(source), seed=seed,
                       resource_estimates=cfg.get("resource_estimates"),
                       extra={"model_name": name, "device": args.device, "resolved_config": cfg})
    try:
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["JAX_PLATFORMS"] = "cpu"
        else:
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        checkpoint = Path(fit_model(cfg, model_dir=model_dir, seed=seed, device=args.device))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training returned a missing checkpoint: {checkpoint}")
        if args.compact:
            compact_kpms_checkpoint(checkpoint)
        record.finish(status="SUCCESS", output_path=str(checkpoint))
    except Exception as error:
        logger.error(f"Training failed: {error}")
        record.finish(status="FAILED", error=str(error))
        return 1
    if not args.no_notify:
        try:
            send_ntfy(title=f"KPMS Training Finished: {name}", msg=f"Checkpoint: {checkpoint}",
                      priority="default", tags="white_check_mark,robot")
        except Exception as error:
            logger.warning(f"Completion notification failed: {error}")
    return 0


def main(argv=None):
    return run_training(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
