"""KPMS result extraction and backend-free reads; cohort export remains legacy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from fpcode.adapters.kpms_results import read_kpms_results as read_results


def extract_and_export_results(
    model: dict,
    metadata: tuple,
    project_dir: Union[Path, str],
    model_name: str,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Extract syllable and latent state results using keypoint_moseq.fitting.extract_results.
    """
    import keypoint_moseq as kpms

    project_dir = str(project_dir)
    return kpms.extract_results(model, metadata, project_dir, model_name, save_results=save_results)
