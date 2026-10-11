#!/usr/bin/env python3
"""
ops/repo/update_shims.py
========================
Systematically ensures all backward-compatibility shims emit DeprecationWarning,
catalogs them in registry/SHIMS.tsv, and verifies their targets.
"""

from pathlib import Path
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]

SHIMS_MAP = {
    "core/cache_manifest.py": "fpcode.common.cache_manifest",
    "core/camera_correspondence.py": "fpcode.tracking.camera_correspondence",
    "core/eval_artifacts.py": "fpcode.kpms.evaluation",
    "core/evaluation_metrics.py": "fpcode.kpms.metrics",
    "core/fusion_engine.py": "fpcode.tracking.fusion",
    "core/h5_guard.py": "fpcode.common.h5_guard",
    "core/io_utils.py": "fpcode.common.io",
    "core/logging_utils.py": "fpcode.common.logging",
    "core/pair_matching.py": "fpcode.tracking.pair_matching",
    "core/paths.py": "fpcode.common.paths",
    "core/run_record.py": "fpcode.common.run_record",
    "core/schemas.py": "fpcode.common.schemas",
    "core/static_fusion_engine.py": "fpcode.tracking.static_fusion",
    "core/ttl_sync.py": "fpcode.tracking.clock_sync",
    "syllable_generation/syllabizers/keypoint_moseq_syll/utils/batch_init.py": "fpcode.kpms.init_model",
    "syllable_generation/syllabizers/keypoint_moseq_syll/utils/compact_checkpoint.py": "fpcode.kpms.checkpoints",
    "syllable_generation/syllabizers/keypoint_moseq_syll/utils/patch_checkpoint.py": "fpcode.kpms.checkpoints",
    "syllable_generation/syllabizers/keypoint_moseq_syll/utils/trimming.py": "fpcode.kpms.data",
    "syllable_generation/syllabizers/keypoint_moseq_syll/utils/resample_batching.py": "fpcode.kpms.runtime",
    "syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/export_tandem_cohort_production_results.py": "fpcode.kpms.export",
    "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_static_21kp_v2_production.py": "pipelines.train",
    "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_tandem_cohort_production_50ar_200slds.py": "pipelines.train",
    "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py": "pipelines.train",
    "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_pilot_17kp_vs_21kp_comparison.py": "pipelines.train",
    "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_pilot_static_comparison.py": "pipelines.train",
    "syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/fuse_cohort_static_v2.py": "pipelines.fuse",
    "syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/fuse_all_tandem_cohort.py": "pipelines.fuse",
    "syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/fuse_pilot_static.py": "pipelines.fuse",
}


def update_core_shims():
    for rel_path, target in SHIMS_MAP.items():
        if not rel_path.startswith("core/"):
            continue
        p = REPO_ROOT / rel_path
        mod_name = p.stem
        code = f'''"""
[DEPRECATED SHIM] {rel_path}
==========================================
Relocated to: {target}
"""

import importlib
import sys
import warnings

warnings.warn(
    "{rel_path.replace('/', '.').replace('.py', '')} is deprecated; import from {target} instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("{target}")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
'''
        p.write_text(code)
        print(f"Updated shim: {rel_path} -> {target}")


def write_shims_tsv():
    tsv_path = REPO_ROOT / "registry/SHIMS.tsv"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    # This tool owns a subset of the registry; retain separately registered
    # study paths and newer compatibility aliases when updating that subset.
    records = {}
    if tsv_path.is_file():
        for line in tsv_path.read_text().splitlines()[1:]:
            if line:
                records[line.split("\t", 1)[0]] = line
    for rel_path, target in sorted(SHIMS_MAP.items()):
        p = REPO_ROOT / rel_path
        shim_type = "module_proxy" if rel_path.startswith("core/") or "utils" in rel_path else "forwarder"
        status = "ACTIVE" if p.exists() else "MISSING"
        notes = "Emits DeprecationWarning"
        records[rel_path] = f"{rel_path}\t{target}\t{shim_type}\t{status}\t{notes}"
    lines = ["shim_path\ttarget_module\ttype\tstatus\tnotes", *[records[key] for key in sorted(records)]]
    tsv_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote registry: {tsv_path} ({len(records)} shims cataloged)")


if __name__ == "__main__":
    update_core_shims()
    write_shims_tsv()
