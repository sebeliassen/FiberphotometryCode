"""Independent import and artifact checks for canonical/compatibility ownership."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def isolated(script, *arguments):
    result = subprocess.run([sys.executable, "-I", "-B", "-c",
                             "import sys; sys.path.insert(0, sys.argv[1]);\n" + script,
                             str(ROOT / "src"), *map(str, arguments)],
                            text=True, capture_output=True, check=False, timeout=15)
    assert result.returncode == 0, result.stderr


def test_session_record_import_and_pickle_need_no_runtime_pandas():
    isolated("""
import importlib.abc
import pickle
class NoPandas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pandas':
            raise AssertionError('record import initialized pandas')
sys.meta_path.insert(0, NoPandas())
from fpcode.contracts.session_record import SessionRecord, VideoRef
record = SessionRecord('sample', video_refs=[VideoRef('example.avi', fps=30.)])
loaded = pickle.loads(pickle.dumps(record))
assert type(loaded) is SessionRecord and loaded.primary_fps() == 30.
assert 'pandas' not in sys.modules
""")


def test_generic_domains_do_not_load_compatibility_profiles():
    isolated("""
import importlib.abc
class NoCompatibility(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('fpcode.compat'):
            raise AssertionError('generic operation selected compatibility: ' + fullname)
sys.meta_path.insert(0, NoCompatibility())
from fpcode.photometry.event_windows import extract_event_windows
from fpcode.tracking.geometry import rot2d
from fpcode.viz.pose_geometry import egocentric_window
from fpcode.behavior.kinematics import smooth_heading_circular
assert rot2d(0.).shape == (2, 2)
assert smooth_heading_circular([0., 1.]).tolist() == [0., 1.]
""")


def test_pathconfig_uses_explicit_roots_and_keeps_pickle_identity(tmp_path):
    isolated("""
import os
import pickle
from pathlib import Path
os.environ['FIBERPHOTOMETRY_ROOT'] = sys.argv[2]
os.environ['FP_DATA_ROOT'] = sys.argv[3]
os.environ['FP_BOT_TRACKING_ROOT'] = sys.argv[4]
from fpcode.common.schemas import PathConfig
from fpcode.compat.schemas import PathConfig as CanonicalLegacyPaths
paths = PathConfig()
assert PathConfig is CanonicalLegacyPaths
assert PathConfig.__module__ == 'fpcode.common.schemas'
assert paths.repo_root == Path(sys.argv[2])
assert paths.top_h5_dir == Path(sys.argv[3]) / 'data/raw/karen_pff_vs_monomer/h5_files'
assert paths.prod_kpms_dir == Path(sys.argv[3]) / 'data/models/kpms_tandem_cohort_production'
assert paths.bot_prod_dir == Path(sys.argv[4])
assert type(pickle.loads(pickle.dumps(paths))) is PathConfig
""", tmp_path / "checkout", tmp_path / "external", tmp_path / "bottom")
    assert list(tmp_path.iterdir()) == []


def checkpoint(path):
    with h5py.File(path, "w") as output:
        output.attrs["dataset"] = "synthetic"
        output.create_group("metadata").attrs["fps"] = 60
        output.create_group("data").create_dataset("mask", data=[True, False])
        snapshots = output.create_group("model_snapshots")
        snapshots.attrs["units"] = "pixels"
        for name, value in (("1", 11), ("2", 22), ("0003", 33)):
            state = snapshots.create_group(name).create_dataset("state", data=[value, value + 1])
            state.attrs["identity"] = name


def test_export_cli_writes_selected_snapshots_without_changing_source(tmp_path):
    from fpcode.cli.export import main

    source = tmp_path / "input.h5"
    checkpoint(source)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "output" / "selected.h5"
    assert main(["-c", str(source), "--keep-final", "2", "-o", str(destination), "--no-notify"]) == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    with h5py.File(destination, "r") as result:
        assert set(result["model_snapshots"]) == {"2", "0003"}
        np.testing.assert_array_equal(result["model_snapshots/0003/state"][()], [33, 34])
        assert result["model_snapshots/0003/state"].attrs["identity"] == "0003"
        assert result["model_snapshots"].attrs["units"] == "pixels"
        assert result["metadata"].attrs["fps"] == 60
        np.testing.assert_array_equal(result["data/mask"][()], [True, False])
        assert result.attrs["dataset"] == "synthetic"
    record = json.loads((destination.parent / "run.json").read_text())
    assert record["status"] == "SUCCESS"
    assert record["output_path"] == str(destination)


def test_export_dry_run_and_invalid_retention_write_nothing(tmp_path):
    from fpcode.cli.export import main
    from fpcode.kpms.checkpoint_io import compact_kpms_checkpoint

    source = tmp_path / "input.h5"
    checkpoint(source)
    original = source.read_bytes()
    destination = tmp_path / "missing" / "selected.h5"
    assert main(["-c", str(source), "-o", str(destination), "--dry-run"]) == 0
    with pytest.raises(ValueError, match="not found"):
        compact_kpms_checkpoint(source, keep_iterations=[9], dest_path=destination)
    with pytest.raises(SystemExit) as error:
        main(["-c", str(source), "--keep-final", "0"])
    assert error.value.code == 2
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


def test_legacy_io_aliases_and_v2_fingerprints_follow_actual_implementations():
    from fpcode.common import file_io, io
    from fpcode.compat.viz import legacy_defaults
    from fpcode.kpms import checkpoint_io
    from fpcode.viz import hashing, v2

    assert io.compact_kpms_checkpoint is checkpoint_io.compact_kpms_checkpoint
    assert io.copy_group_bounded is file_io.copy_group_bounded
    assert v2.__name__ == "fpcode.viz.v2"
    hashes = hashing.hash_files([Path(v2.__file__)])
    assert "fpcode/compat/viz/legacy_api.py" in hashes
    assert "fpcode/viz/pose_geometry.py" in hashes
    assert all(path.is_file() for path in legacy_defaults.CALIBRATION_FILES)
    assert not list((ROOT / "src/fpcode/tracking").glob("*.json"))
