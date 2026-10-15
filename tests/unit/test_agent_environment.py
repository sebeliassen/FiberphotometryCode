"""Environment diagnostics/provenance observe inputs without initializing backends."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from fpcode.cli import doctor
from fpcode.common import environment
from fpcode.common.run_record import RunRecord

ROOT = Path(__file__).resolve().parents[2]


def test_doctor_missing_input_is_structured_and_does_not_create_roots(tmp_path):
    script = """
import importlib.abc, json, os, sys
startup_modules = set(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'jax', 'jaxlib', 'keypoint_moseq', 'jax_moseq', 'torch', 'numpy', 'pandas'}:
            raise AssertionError('metadata diagnostics imported ' + name)
sys.meta_path.insert(0, Block())
os.environ['FIBERPHOTOMETRY_ROOT'] = sys.argv[2] + '/code'
os.environ['FP_DATA_ROOT'] = sys.argv[2] + '/mirror'
from fpcode.cli.doctor import main
assert main(['--json', '--data-input', 'data/missing.pkl']) == 1
assert not {'jax', 'keypoint_moseq', 'numpy', 'pandas'} & (set(sys.modules) - startup_modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c",
                             "import sys; sys.path.insert(0, sys.argv[1]);\n" + script,
                             str(ROOT / "src"), str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["read_only"] and not report["ok"]
    assert report["roots"]["data_path_example"] == str(tmp_path / "mirror/data/models")
    assert report["inputs"][0]["path"] == str(tmp_path / "mirror/data/missing.pkl")
    assert list(tmp_path.iterdir()) == []


def test_doctor_reports_missing_requested_distribution(monkeypatch):
    monkeypatch.setattr(doctor, "package_versions", lambda names: {
        name: None if name == "keypoint-moseq" else "1.0" for name in names
    })
    report = doctor.diagnose(["kpms"])
    assert not report["ok"]
    assert report["issues"] == ["Missing kpms distribution: keypoint-moseq"]


def test_provenance_records_existing_jax_devices_and_distribution_versions(tmp_path, monkeypatch):
    bridge = ModuleType("jax._src.xla_bridge")
    bridge._backends = {"gpu": SimpleNamespace(devices=lambda: [
        SimpleNamespace(platform="gpu", id=2, device_kind="fixture accelerator")
    ])}
    monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", bridge)
    monkeypatch.setattr(environment.metadata, "version", lambda name: "fixture-version")
    estimates = {"input_array_bytes": 48, "method": "fixture shape times dtype size", "scope": ["tiny fixture"]}
    pending = RunRecord(output_dir=tmp_path, seed=19, config={"rate": 30}, resource_estimates=estimates)
    estimates["scope"].append("later mutation")
    path = pending.finish()
    record = json.loads(path.read_text())
    assert record["environment"]["packages"]["jax"] == "fixture-version"
    assert record["environment"]["gpu"]["inspection"] == "initialized_jax"
    assert record["environment"]["gpu"]["devices"] == [
        {"platform": "gpu", "id": 2, "kind": "fixture accelerator"}
    ]
    assert record["seed"] == 19 and record["config"] == {"rate": 30}
    assert record["resource_estimates"]["scope"] == ["tiny fixture"]
    assert record["resource_estimates"]["input_array_bytes"] == 48


@pytest.mark.parametrize("registry, inspection", [({}, "not_initialized"), (None, "unknown_jax_registry")])
def test_uninitialized_jax_registry_never_calls_initialization(monkeypatch, registry, inspection):
    bridge = ModuleType("jax._src.xla_bridge")
    bridge._backends = registry
    bridge.get_backend = lambda: (_ for _ in ()).throw(AssertionError("initialized backend"))
    monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", bridge)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert environment.initialized_devices()["inspection"] == inspection
