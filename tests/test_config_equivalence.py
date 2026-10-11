"""
tests.test_config_equivalence
=============================
Proves mathematical and parameter equivalence between centralized YAML configs
and historical production training runners and forwarders.
"""

from pathlib import Path
import importlib.util
import yaml

from fpcode.common.paths import REPO_ROOT


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


def test_cohort_static_21kp_v2_config_equivalence():
    config_file = REPO_ROOT / "configs/training/cohort_static_21kp_v2_seed101.yaml"
    assert config_file.exists()

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    runner_path = (
        REPO_ROOT
        / "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_static_21kp_v2_production.py"
    )
    runner = load_module_from_path("v2_runner", runner_path)

    assert cfg["model_name"] == runner.MODEL_NAME
    assert cfg["hyperparameters"]["num_ar_iters"] == runner.NUM_AR_ITERS
    assert cfg["hyperparameters"]["num_slds_iters"] == runner.NUM_SLDS_ITERS
    assert cfg["hyperparameters"]["total_iters"] == runner.TOTAL_ITERS
    assert cfg["hyperparameters"]["save_every_n_iters"] == runner.SAVE_EVERY_N_ITERS


def test_cohort_calibrated_retrain_config_equivalence():
    config_file = REPO_ROOT / "configs/training/cohort_calibrated_retrain_seed101.yaml"
    assert config_file.exists()

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    runner_path = (
        REPO_ROOT
        / "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py"
    )
    runner = load_module_from_path("calibrated_runner", runner_path)

    assert cfg["model_name"] == runner.MODEL_NAME
    assert cfg["hyperparameters"]["num_ar_iters"] == runner.NUM_AR_ITERS
    assert cfg["hyperparameters"]["num_slds_iters"] == runner.NUM_SLDS_ITERS
    assert cfg["hyperparameters"]["total_iters"] == runner.TOTAL_ITERS
    assert float(cfg["hyperparameters"]["init_kappa"]) == runner.INIT_KAPPA
    assert float(cfg["hyperparameters"]["slds_kappa"]) == runner.SLDS_KAPPA


def test_pilot_comparison_configs_equivalence():
    # 17kp vs 21kp
    cfg1_path = REPO_ROOT / "configs/training/pilot_17kp_vs_21kp_comparison.yaml"
    assert cfg1_path.exists()
    with open(cfg1_path, "r") as f:
        cfg1 = yaml.safe_load(f)

    runner1_path = (
        REPO_ROOT
        / "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_pilot_17kp_vs_21kp_comparison.py"
    )
    r1 = load_module_from_path("pilot1", runner1_path)
    assert cfg1["model_name"] == r1.MODEL_NAME
    assert cfg1["hyperparameters"]["num_ar_iters"] == r1.NUM_AR_ITERS
    assert cfg1["hyperparameters"]["num_slds_iters"] == r1.NUM_SLDS_ITERS
    assert float(cfg1["hyperparameters"]["init_kappa"]) == r1.INIT_KAPPA

    # Static comparison
    cfg2_path = REPO_ROOT / "configs/training/pilot_static_comparison.yaml"
    assert cfg2_path.exists()
    with open(cfg2_path, "r") as f:
        cfg2 = yaml.safe_load(f)

    runner2_path = (
        REPO_ROOT
        / "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_pilot_static_comparison.py"
    )
    r2 = load_module_from_path("pilot2", runner2_path)
    assert cfg2["model_name"] == r2.MODEL_NAME
    assert cfg2["hyperparameters"]["num_ar_iters"] == r2.NUM_AR_ITERS
    assert cfg2["hyperparameters"]["num_slds_iters"] == r2.NUM_SLDS_ITERS
    assert float(cfg2["hyperparameters"]["init_kappa"]) == r2.INIT_KAPPA


def test_fusion_configs_equivalence():
    # cohort static v2
    cfg_fused = REPO_ROOT / "configs/fusion/cohort_static_21kp_v2.yaml"
    assert cfg_fused.exists()
    with open(cfg_fused, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["arena_width"] == 972.0
    assert cfg["use_epochs"] is True

    # tandem
    cfg_tandem = REPO_ROOT / "configs/fusion/all_tandem_cohort.yaml"
    assert cfg_tandem.exists()
    with open(cfg_tandem, "r") as f:
        tc = yaml.safe_load(f)
    assert tc["keypoints"]["num_keypoints"] == 23
    assert tc["rules"]["reflect_botcam"] is True
