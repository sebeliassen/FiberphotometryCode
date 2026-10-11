"""Fast selection and lifecycle checks; no real backend imports or fitting."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ops.diagnostics import validate_two_sessions as diagnostic


def sample_data():
    values = np.arange(3 * 12 * 3 * 2, dtype=float).reshape(3, 12, 3, 2)
    data = {"Y": values, "mask": np.ones((3, 12)), "conf": np.ones((3, 12, 3))}
    metadata = (np.array([b"alpha_seg0_tandem", b"alpha_seg1_tandem", b"beta_seg2_tandem"]),
                np.array([[0, 12], [12, 24], [20, 32]]))
    return data, metadata


def test_two_session_selection_keeps_bounds_identity_and_owns_copies():
    data, metadata = sample_data()
    selected, (names, bounds), identities = diagnostic.select_two_sessions(data, metadata, 8)
    np.testing.assert_array_equal(selected["Y"], data["Y"][[0, 2], :8])
    np.testing.assert_array_equal(names, ["alpha_seg0_tandem", "beta_seg2_tandem"])
    np.testing.assert_array_equal(bounds, [[0, 8], [20, 28]])
    assert len(set(identities)) == 2
    selected["Y"][:] = -1
    bounds[:] = -1
    assert data["Y"].min() == 0 and metadata[1].min() == 0


@pytest.mark.parametrize("problem", ["gap", "nonfinite", "same_session", "shape", "bounds"])
def test_selection_cannot_fake_a_second_valid_session(problem):
    data, metadata = sample_data()
    if problem == "gap":
        data["mask"][2, 3] = 0
    elif problem == "nonfinite":
        data["conf"][2, 3, 0] = np.nan
    elif problem == "same_session":
        metadata[0][2] = b"alpha_seg2_tandem"
    elif problem == "shape":
        data["mask"] = np.ones((2, 12))
    else:
        metadata[1][2] = [30, 20]
    with pytest.raises(ValueError):
        diagnostic.select_two_sessions(data, metadata, 8)


def test_import_and_help_do_not_initialize_backend_or_write_files(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = f"""
import sys
sys.path.insert(0, {str(root)!r})
from pathlib import Path
before = set(sys.modules)
def forbidden(*args, **kwargs):
    raise AssertionError('import/help wrote a file')
Path.mkdir = Path.write_text = Path.write_bytes = forbidden
from ops.diagnostics import validate_two_sessions as module
assert not {{'jax', 'keypoint_moseq', 'jax_moseq', 'numpy'}} & (set(sys.modules) - before)
try:
    module.main(['--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('help started a worker')
assert not {{'jax', 'keypoint_moseq', 'jax_moseq', 'numpy'}} & (set(sys.modules) - before)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_input_size_limit_precedes_unpickling_or_output_creation(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    for name in ("config.yml", "pca.p"):
        (project / name).write_bytes(b"not a real input")
    with (project / "preprocessed_data.pkl").open("wb") as handle:
        handle.truncate(2 * 1024**2)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="size limit"):
        diagnostic.main(["--project", str(project), "--max-source-mib", "1", "--output-root", str(output)])
    assert not output.exists()


def test_timed_out_worker_group_is_killed_and_evidence_is_preserved(tmp_path, monkeypatch):
    project = tmp_path / "source"
    project.mkdir()
    for name in ("config.yml", "pca.p", "preprocessed_data.pkl"):
        (project / name).write_bytes(b"not unpickled by this test")
    waits, killed = [], []

    def wait(timeout=None):
        waits.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired("fake worker", timeout)
        return -9

    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123456, wait=wait))
    monkeypatch.setattr(diagnostic.os, "killpg", lambda *args: killed.append(args))
    output = tmp_path / "output"
    assert diagnostic.main(["--project", str(project), "--output-root", str(output), "--timeout", "0.01"]) == 1
    assert waits == [0.01, None] and killed == [(123456, diagnostic.signal.SIGKILL)]
    records = list(output.glob("*/validation.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["status"] == "TIMEOUT"
    assert records[0].with_name("validation_script.py").is_file()
