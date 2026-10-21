"""Study entry points preserve settings/argv/failure status through canonical CLI modules."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from fpcode.common import paths

ROOT = Path(__file__).resolve().parents[2]

CASES = (
    ("01_data_and_fusion/fuse_all_tandem_cohort.py", "fuse", "fusion/all_tandem_cohort.yaml"),
    ("01_data_and_fusion/fuse_cohort_static_v2.py", "fuse", "fusion/cohort_static_21kp_v2.yaml"),
    ("01_data_and_fusion/fuse_pilot_static.py", "fuse", "fusion/pilot_static.yaml"),
    ("02_training_runners/run_cohort_calibrated_retrain_seed101.py", "train", "training/cohort_calibrated_retrain_seed101.yaml"),
    ("02_training_runners/run_cohort_static_21kp_v2_production.py", "train", "training/cohort_static_21kp_v2_seed101.yaml"),
    ("02_training_runners/run_pilot_17kp_vs_21kp_comparison.py", "train", "training/pilot_17kp_vs_21kp_comparison.yaml"),
    ("02_training_runners/run_pilot_static_comparison.py", "train", "training/pilot_static_comparison.yaml"),
    ("02_training_runners/run_tandem_cohort_production_50ar_200slds.py", "train", "training/tandem_cohort_production_seed101.yaml"),
)


@pytest.mark.parametrize("relative,verb,config", CASES)
def test_study_launcher_forwards_settings_arguments_and_failure(relative, verb, config, tmp_path, monkeypatch):
    script = ROOT / "studies/multiview_experiments" / relative
    seen = []
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--dry-run", "--no-notify"])

    def capture(command):
        seen.append(command)
        return 19

    monkeypatch.setattr(subprocess, "call", capture)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 19
    assert seen == [[sys.executable, "-m", "fpcode.cli." + verb, "--config",
                     str(tmp_path / "configs" / config), "--dry-run", "--no-notify"]]
    assert not list(tmp_path.iterdir())
