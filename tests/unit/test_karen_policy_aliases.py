"""Karen policy adapters preserve canonical identity and cold physical loading."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEAVES = (
    ("enrich", ("enrich", "load_animal_map", "load_sessions_map", "load_exclusions",
                "apply_auto_masks", "apply_manual_exclusions_to_session")),
    ("tracking", ("discover_sessions_hybrid", "parse_ids_from_filename")),
)
ALIASES = (
    "studies.karen_pff_vs_monomer.session_policy",
    "studies.monoamine_interactions.projects.karen_project",
)
PHYSICAL_PARENTS = (
    "studies/karen_pff_vs_monomer/session_policy",
    "studies/monoamine_interactions/projects/karen_project",
)


def child(script, tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="2",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    setup = f"import sys\nsys.path[:0] = {[str(ROOT / 'src'), str(ROOT), str(ROOT / 'compat')]!r}\n"
    guards = '''
import builtins
import importlib
import importlib.abc
import importlib.util
import logging
import os
import pickle
import runpy
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

blocked = {'jax', 'jaxlib', 'jax_moseq', 'keypoint_moseq', 'torch', 'cuml',
           'tensorflow', 'tensorflow_probability', 'cv2', 'joblib', 'sklearn', 'h5py',
           'config', 'enrich', 'tracking', 'steps'}
class BlockBackendAndOldStudy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or fullname.startswith('studies.behavior'):
            raise AssertionError('cold policy import attempted ' + fullname)
sys.meta_path.insert(0, BlockBackendAndOldStudy())

def forbidden(*args, **kwargs):
    raise AssertionError('cold policy adapter attempted empirical I/O')
pd.read_pickle = pd.read_csv = np.load = pickle.load = pickle.loads = forbidden

def audit(event, arguments):
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.putenv', 'os.unsetenv'}:
        raise AssertionError('cold policy adapter attempted mutation: ' + event)
    if event == 'open':
        path, mode, flags = arguments
        writing = (mode is not None and any(mark in mode for mark in 'wax+'))
        writing |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            raise AssertionError('cold policy adapter opened output: ' + str(path))
        if isinstance(path, (str, bytes, os.PathLike)):
            suffix = Path(os.fsdecode(path)).suffix.lower()
            if suffix in {'.pkl', '.pickle', '.npz', '.npy', '.h5', '.hdf5', '.csv', '.avi', '.mp4'}:
                raise AssertionError('cold policy adapter opened empirical input: ' + str(path))
sys.addaudithook(audit)
warnings.simplefilter('ignore', DeprecationWarning)
paths, handlers = list(sys.path), list(logging.root.handlers)
level, env, filters = logging.root.level, dict(os.environ), list(warnings.filters)

def assert_inert():
    assert sys.path == paths
    assert logging.root.handlers == handlers and logging.root.level == level
    assert dict(os.environ) == env and warnings.filters == filters
    assert not blocked & set(sys.modules)
    assert not any(name.startswith('studies.behavior') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", setup + guards + script],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("leaf,functions", LEAVES)
@pytest.mark.parametrize("alias_parent", ALIASES)
@pytest.mark.parametrize("alias_first", [False, True])
def test_policy_aliases_share_canonical_module_and_functions(leaf, functions, alias_parent, alias_first, tmp_path):
    canonical = "studies.karen_pff_vs_monomer.analysis." + leaf
    alias = alias_parent + "." + leaf
    child(f'''
first, second = ({alias!r}, {canonical!r}) if {alias_first!r} else ({canonical!r}, {alias!r})
one, two = importlib.import_module(first), importlib.import_module(second)
assert one is two
canonical = importlib.import_module({canonical!r})
assert one is canonical and canonical.__name__ == {canonical!r}
assert Path(canonical.__file__).resolve() == Path({str(ROOT)!r}) / 'studies/karen_pff_vs_monomer/analysis' / {leaf + '.py'!r}
for name in {functions!r}:
    assert getattr(one, name) is getattr(two, name) is getattr(canonical, name)
    assert callable(getattr(one, name))
    assert getattr(one, name).__module__ == {canonical!r}
parent = importlib.import_module({alias_parent!r})
assert getattr(parent, {leaf!r}) is canonical
if {leaf!r} == 'tracking':
    policy = importlib.import_module('studies.karen_pff_vs_monomer.session_policy.config')
    assert canonical.JCFG is policy.JCFG
assert_inert()
''', tmp_path)


@pytest.mark.parametrize("leaf,functions", LEAVES)
@pytest.mark.parametrize("parent", PHYSICAL_PARENTS)
@pytest.mark.parametrize("loader", ["spec", "runpy", "main"])
def test_physical_policy_adapters_forward_without_running_empirical_functions(leaf, functions, parent, loader, tmp_path):
    canonical = "studies.karen_pff_vs_monomer.analysis." + leaf
    path = ROOT / parent / (leaf + ".py")
    assert path.is_file()
    child(f'''
canonical = importlib.import_module({canonical!r})
calls = []
prior_main = getattr(canonical, 'main', None)
had_main = hasattr(canonical, 'main')
def sentinel_main():
    calls.append('test-owned main')
canonical.main = sentinel_main
entry_main = sys.modules['__main__']
try:
    if {loader!r} == 'spec':
        name = '_karen_policy_adapter_under_test'
        spec = importlib.util.spec_from_file_location(name, {str(path)!r})
        wrapper = importlib.util.module_from_spec(spec)
        sys.modules[name] = wrapper
        spec.loader.exec_module(wrapper)
        assert sys.modules.pop(name) is canonical
        namespace = vars(wrapper)
    else:
        name = '__main__' if {loader!r} == 'main' else '_karen_policy_adapter_under_test'
        namespace = runpy.run_path({str(path)!r}, run_name=name)
    assert namespace['_real'] is canonical
    for name in {functions!r}:
        assert namespace['__getattr__'](name) is getattr(canonical, name)
        assert name in namespace['__dir__']()
    assert calls == (['test-owned main'] if {loader!r} == 'main' else [])
    assert sys.modules['__main__'] is entry_main
finally:
    if had_main:
        canonical.main = prior_main
    else:
        del canonical.main
assert_inert()
''', tmp_path)
