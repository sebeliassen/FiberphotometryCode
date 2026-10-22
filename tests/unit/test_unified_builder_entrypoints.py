"""Cold canonical/historical unified entry points without cohort execution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = "studies.session_building.build_sessions_unified"
LEGACY = "syllable_generation.loaders.build_sessions_unified"


def child(script, tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    setup = f"import sys\nsys.path[:0] = {[str(ROOT/'src'), str(ROOT), str(ROOT/'compat')]!r}\n"
    guard = '''
import importlib
import importlib.abc
import logging
import pickle
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','cv2','joblib','numpy','pandas','config','enrich','tracking','steps'} or fullname.startswith('studies.behavior'):
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0, Block())
def forbidden(*args, **kwargs):
    raise AssertionError('entry point attempted data loading')
pickle.load = pickle.loads = forbidden
paths, handlers = list(sys.path), list(logging.root.handlers)
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", setup+guard+script],
                            cwd=tmp_path, env=environment, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
    return result.stdout


@pytest.mark.parametrize("legacy_first", [False, True])
def test_unified_leaf_identity_is_cold_in_both_import_orders(legacy_first, tmp_path):
    assert child(f'''
first, second = ({LEGACY!r}, {CANONICAL!r}) if {legacy_first!r} else ({CANONICAL!r}, {LEGACY!r})
one, two = importlib.import_module(first), importlib.import_module(second)
assert one is two
assert one.build_sessions is two.build_sessions
assert one.build_sessions.__module__ == {CANONICAL!r}
assert sys.path == paths and logging.root.handlers == handlers
assert not any(name.startswith('studies.session_building.jonathan') for name in sys.modules)
''', tmp_path) == ""


@pytest.mark.parametrize("entrypoint", [CANONICAL, LEGACY])
def test_imported_help_does_not_resolve_projects_or_load_data(entrypoint, tmp_path):
    output = child(f'''
module = importlib.import_module({entrypoint!r})
module.build_sessions = forbidden
sys.argv = ['unified', '--help']
try:
    module.main()
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert sys.path == paths and logging.root.handlers == handlers
''', tmp_path)
    assert "Unified Sessions Builder" in output and "--discovery-mode" in output


@pytest.mark.parametrize("relative", [
    "studies/session_building/build_sessions_unified.py",
    "compat/syllable_generation/loaders/build_sessions_unified.py",
])
def test_physical_help_does_not_build_or_load_data(relative, tmp_path):
    output = child(f'''
import runpy
sys.argv = ['unified', '--help']
try:
    runpy.run_path({str(ROOT)!r}+'/'+{relative!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert sys.path == paths and logging.root.handlers == handlers
''', tmp_path)
    assert "Unified Sessions Builder" in output


def test_shim_registry_matches_the_physical_unified_owner():
    rows = (ROOT/"registry/SHIMS.tsv").read_text().splitlines()
    row, = [r.split("\t") for r in rows
            if r.startswith("syllable_generation/loaders/build_sessions_unified.py\t")]
    assert row[1] == "studies/session_building/build_sessions_unified.py"
