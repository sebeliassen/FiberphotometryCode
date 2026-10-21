"""Historical checkpoint I/O works without models; explicit patches restore identity."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest

from fpcode.kpms import checkpoints


@pytest.mark.parametrize("mode", ["full", "slim"])
def test_historical_compaction_keeps_its_independent_packing_policy(tmp_path, mode):
    source, dest = tmp_path / "source.h5", tmp_path / "packed.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data/Y", data=[7, 8])
        handle.create_dataset("metadata/frames", data=[0, 2])
        for iteration in (1, 2):
            handle.create_dataset(f"model_snapshots/{iteration}/states/x", data=[iteration, 4])
            handle.create_dataset(f"model_snapshots/{iteration}/states/s", data=[9])
            handle.create_dataset(f"model_snapshots/{iteration}/noise_prior", data=[5])
    before = source.read_bytes()
    report = checkpoints.compact_checkpoint(source, dest, mode=mode, keep_iterations=[2], verbose=False)
    assert report["mode"] == mode and report["retained_iterations"] == [2]
    assert report["dest_path"] == str(dest) and source.read_bytes() == before
    with h5py.File(dest) as handle:
        assert list(handle["model_snapshots"]) == ["2"]
        np.testing.assert_array_equal(handle["model_snapshots/2/states/x"][()], [2, 4])
        assert ("s" in handle["model_snapshots/2/states"]) == (mode == "full")
        assert ("noise_prior" in handle["model_snapshots/2"]) == (mode == "full")
    assert checkpoints.is_checkpoint_packed(dest)[1]["packing_mode"] == mode


def test_checkpoint_import_needs_no_backend_and_restore_without_apply_is_noop():
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'keypoint_moseq', 'jax', 'jaxlib', 'jax_moseq'}:
            raise AssertionError('I/O initialized ' + name)
sys.meta_path.insert(0, Block())
from fpcode.kpms.checkpoints import remove_kpms_checkpoint_patch
remove_kpms_checkpoint_patch()
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c",
                             "import sys; sys.path.insert(0, sys.argv[1]);\n" + script,
                             str(root / "src")], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("historical_first", [True, False])
def test_runtime_shim_preserves_identity_without_loading_modeling_backend(historical_first):
    root = Path(__file__).resolve().parents[2]
    legacy = "syllable_generation.syllabizers.keypoint_moseq_syll.utils.resample_batching"
    names = [legacy, "fpcode.kpms.runtime"] if historical_first else ["fpcode.kpms.runtime", legacy]
    script = """
import importlib, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'keypoint_moseq', 'jax', 'jaxlib', 'jax_moseq'}:
            raise AssertionError('compatibility import initialized ' + name)
sys.meta_path.insert(0, Block())
first, second = [importlib.import_module(name) for name in NAMES]
assert first is second
assert first.BatchConfig is second.BatchConfig
assert first.CONFIG is second.CONFIG
assert 'apply_optimizations' in dir(first)
"""
    bootstrap = "import sys; sys.path[:0] = sys.argv[1:4];\nNAMES = " + repr(names) + "\n"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap + script,
                             str(root / "src"), str(root / "compat"), str(root)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_explicit_patch_restores_distinct_prior_namespace_values(monkeypatch):
    backend = ModuleType("keypoint_moseq")
    io_loader, public_loader = object(), object()
    backend.io = SimpleNamespace(load_checkpoint=io_loader)
    backend.load_checkpoint = public_loader
    monkeypatch.setitem(sys.modules, "keypoint_moseq", backend)
    try:
        checkpoints.apply_kpms_checkpoint_patch()
        checkpoints.apply_kpms_checkpoint_patch()
        assert backend.io.load_checkpoint is backend.load_checkpoint is checkpoints.patched_load_checkpoint
    finally:
        checkpoints.remove_kpms_checkpoint_patch()
    assert backend.io.load_checkpoint is io_loader
    assert backend.load_checkpoint is public_loader
