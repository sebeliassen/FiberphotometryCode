"""
tests.test_config_equivalence
=============================
Proves mathematical and parameter equivalence between centralized YAML configs
and historical production training runners.
"""

import importlib.util
from pathlib import Path

import yaml

from core.paths import REPO_ROOT


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tandem_cohort_config_equivalence():
    config_file = REPO_ROOT / "configs" / "tandem_cohort_50ar_200slds.yml"
    assert config_file.exists(), f"Config file missing: {config_file}"

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    runner_path = (
        REPO_ROOT
        / "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_tandem_cohort_production_50ar_200slds.py"
    )
    tandem_runner = load_module_from_path("tandem_runner", runner_path)

    # Validate model section
    assert cfg["model"]["name"] == tandem_runner.MODEL_NAME
    assert cfg["model"]["target_fps"] == tandem_runner.TARGET_FPS

    # Validate training hyperparams
    assert cfg["training"]["num_ar_iters"] == tandem_runner.NUM_AR_ITERS
    assert cfg["training"]["num_slds_iters"] == tandem_runner.NUM_SLDS_ITERS
    assert cfg["training"]["total_iters"] == tandem_runner.TOTAL_ITERS
    assert cfg["training"]["save_every_n_iters"] == tandem_runner.SAVE_EVERY_N_ITERS
    assert float(cfg["training"]["init_kappa"]) == tandem_runner.INIT_KAPPA
    assert float(cfg["training"]["decrease_kappa_factor"]) == tandem_runner.DECREASE_KAPPA_FACTOR
    assert float(cfg["training"]["slds_kappa"]) == tandem_runner.SLDS_KAPPA


def test_cohort_static_21kp_config_equivalence():
    config_file = REPO_ROOT / "configs" / "cohort_static_21kp.yml"
    assert config_file.exists(), f"Config file missing: {config_file}"

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    runner_path = (
        REPO_ROOT
        / "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_static_21kp_production.py"
    )
    static_runner = load_module_from_path("static_runner", runner_path)

    # Validate constants
    assert cfg["model"]["target_fps"] == static_runner.TARGET_FPS
    assert cfg["model"]["min_segment_frames"] == static_runner.MIN_SEGMENT_FRAMES
    assert cfg["keypoints"]["dorsal"] == static_runner.DORSAL_BPS
    assert cfg["keypoints"]["paws"] == static_runner.PAW_BPS
    assert (cfg["keypoints"]["dorsal"] + cfg["keypoints"]["paws"]) == static_runner.BODYPARTS_21KP
