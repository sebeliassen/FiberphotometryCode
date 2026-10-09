"""Read-only checkpoint auditing and frequency evaluation."""

from pathlib import Path

from fpcode.kpms.evaluation import read_checkpoint_identity, validate_model_compatibility


def audit_checkpoints(checkpoints: list[Path]) -> dict:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    identities = {str(path): read_checkpoint_identity(path) for path in checkpoints}
    if len(identities) > 1:
        validate_model_compatibility(identities)
    return identities


def evaluate_model_checkpoints(checkpoints: list[Path], output_dir: Path | None = None) -> dict:
    from fpcode.kpms.evaluation import evaluate_checkpoints
    return evaluate_checkpoints(checkpoints, output_dir=output_dir)
