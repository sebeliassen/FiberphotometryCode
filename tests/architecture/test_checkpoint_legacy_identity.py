"""Historical checkpoint callers share canonical objects without loading models."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("legacy_first", [True, False])
def test_checkpoint_alias_preserves_identity_without_modeling_or_artifacts(legacy_first, tmp_path):
    script = f"""
import importlib
import importlib.abc
import sys
import warnings

sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'compat')!r}, {str(ROOT)!r}]
blocked = {{'keypoint_moseq', 'jax', 'jaxlib', 'jax_moseq', 'torch',
           'tensorflow', 'tensorflow_probability'}}

class BlockModelingBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('compatibility import initialized ' + fullname)

sys.meta_path.insert(0, BlockModelingBackend())
warnings.simplefilter('ignore', DeprecationWarning)
legacy = 'syllable_generation.syllabizers.keypoint_moseq_syll.utils.patch_checkpoint'
canonical = 'fpcode.kpms.checkpoints'
first, second = (legacy, canonical) if {legacy_first!r} else (canonical, legacy)
one = importlib.import_module(first)
two = importlib.import_module(second)
assert one is two
assert one.apply_kpms_checkpoint_patch is two.apply_kpms_checkpoint_patch
assert one.remove_kpms_checkpoint_patch is two.remove_kpms_checkpoint_patch
assert callable(one.apply_kpms_checkpoint_patch)
assert callable(one.remove_kpms_checkpoint_patch)
assert not blocked & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
