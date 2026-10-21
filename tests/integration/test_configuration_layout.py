"""Preserved legacy profiles and actual execution of the photometry policy example."""

from pathlib import Path

import numpy as np
import pytest

from fpcode.common.config import load_yaml_config
from fpcode.contracts import EventInterval, SignalMetadata, SignalSeries, TimeAxis
from fpcode.workflows.photometry import event_photometry

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["cohort_static_21kp.yml", "tandem_cohort_50ar_200slds.yml"])
def test_legacy_training_config_path_resolves_to_unchanged_canonical_profile(filename):
    old = ROOT / "configs" / filename
    current = ROOT / "configs/training" / filename
    assert old.is_symlink() and old.resolve() == current
    assert old.read_bytes() == current.read_bytes()
    settings = load_yaml_config(current)
    assert settings["training"]["num_ar_iters"] == 50
    assert settings["training"]["num_slds_iters"] == 200


def test_photometry_yaml_runs_real_workflow_without_bridging_large_gaps():
    config = load_yaml_config(ROOT / "configs/photometry/event_windows.yaml")
    unit, clock = config["time"]["unit"], config["time"]["clock"]
    signal = SignalSeries(TimeAxis([0, 0.5, 1, 1.5, 2], unit, clock),
                          np.array([0., 1., 2., 3., 4.]), (SignalMetadata("example"),))
    event = EventInterval("onset", 1, 1, unit, clock)
    windows = event_photometry(signal, [event], config["offsets"], **config["alignment"])
    np.testing.assert_array_equal(windows.values[0, :, 0], [0, 1, 2, 3, 4])
    assert windows.valid.all()
    between = EventInterval("between", 1.25, 1.25, unit, clock)
    gaps = event_photometry(signal, [between], config["offsets"], **config["alignment"])
    assert not gaps.valid.any() and np.isnan(gaps.values).all()
