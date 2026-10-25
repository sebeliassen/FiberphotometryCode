"""Cold PPE routing and parser contract; no empirical analysis is executed."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = "studies.deep_research_experiments.action_prediction_error.run_ppe_analysis"
LEGACY = "syllable_generation.analysis.deep_research_experiments.action_prediction_error.run_ppe_analysis"
PARENT = LEGACY.rsplit(".", 1)[0]
PHYSICAL = (
    "studies/deep_research_experiments/action_prediction_error/run_ppe_analysis.py",
    "compat/syllable_generation/analysis/deep_research_experiments/action_prediction_error/run_ppe_analysis.py",
)
PUBLIC_FUNCTIONS = ("get_dls_signal_col", "get_best_signal_column", "run_analysis", "main")
THREAD_ENV = {name: "2" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)}

# NumPy, pandas and scipy.stats are preloaded before installing guards so their
# ordinary startup (including SciPy warning filters) is outside the measured own
# entry boundary. Modeling packages are never preloaded. True direct-file help
# below separately starts without any dependency or checkout preload.
COLD_GUARDS = '''
import builtins
import importlib
import importlib.abc
import importlib.util
import io
import json
import logging
import os
import pickle
import runpy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy.stats

blocked = {'jax', 'jaxlib', 'jax_moseq', 'keypoint_moseq', 'torch', 'cuml',
           'tensorflow', 'tensorflow_probability', 'cv2', 'joblib', 'sklearn', 'h5py'}
class NoBackendOrOldStudy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or fullname.startswith('studies.behavior'):
            raise AssertionError('entry boundary imported ' + fullname)
sys.meta_path.insert(0, NoBackendOrOldStudy())

def forbidden(*args, **kwargs):
    raise AssertionError('entry boundary attempted data decoding or serialization')
pd.read_pickle = pd.read_csv = np.load = pickle.load = pickle.loads = pickle.dump = forbidden

def audit(event, arguments):
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.putenv', 'os.unsetenv'}:
        raise AssertionError('entry boundary attempted mutation: ' + event)
    if event == 'open':
        path, mode, flags = arguments
        writing = mode is not None and any(mark in mode for mark in 'wax+')
        writing |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            raise AssertionError('entry boundary opened an output: ' + str(path))
        if isinstance(path, (str, bytes, os.PathLike)):
            suffix = Path(os.fsdecode(path)).suffix.lower()
            if suffix in {'.pkl', '.pickle', '.npz', '.npy', '.h5', '.hdf5', '.csv', '.avi', '.mp4'}:
                raise AssertionError('entry boundary opened a data file: ' + str(path))
sys.addaudithook(audit)
warnings.simplefilter('ignore', DeprecationWarning)
paths, handlers = list(sys.path), list(logging.root.handlers)
level, environment, filters = logging.root.level, dict(os.environ), list(warnings.filters)

def assert_inert():
    assert sys.path == paths, (sys.path, paths)
    assert logging.root.handlers == handlers and logging.root.level == level, 'logging state changed'
    assert dict(os.environ) == environment, 'environment changed'
    assert warnings.filters == filters, (warnings.filters, filters)
    assert not blocked & set(sys.modules)
    assert not any(name.startswith('studies.behavior') for name in sys.modules)
'''


def child(script, tmp_path, *, activate=True):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **THREAD_ENV)
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    setup = ""
    if activate:
        setup = f"import sys\nsys.path[:0] = {[str(ROOT / 'src'), str(ROOT), str(ROOT / 'compat')]!r}\n"
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", setup + COLD_GUARDS + script],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.iterdir())
    return result


@pytest.mark.parametrize("legacy_first", [False, True])
def test_cold_leaf_identity_and_structural_parent(legacy_first, tmp_path):
    result = child(f"""
first, second = ({LEGACY!r}, {CANONICAL!r}) if {legacy_first!r} else ({CANONICAL!r}, {LEGACY!r})
one, two = importlib.import_module(first), importlib.import_module(second)
assert one is two and one.__name__ == {CANONICAL!r}
for name in {PUBLIC_FUNCTIONS!r}:
    assert getattr(one, name) is getattr(two, name)
    assert getattr(one, name).__module__ == {CANONICAL!r}
namespace = {{}}
exec('from '+{PARENT!r}+' import run_ppe_analysis', namespace)
assert namespace['run_ppe_analysis'] is one
parent = importlib.import_module({PARENT!r})
assert [Path(p).resolve() for p in parent.__path__] == [Path({str(ROOT / PHYSICAL[1])!r}).parent]
assert parent._real is importlib.import_module({CANONICAL.rsplit('.', 1)[0]!r})
# The structural parent must not discover/execute an unsafe sibling or plotting.
assert not any(name.endswith(('.analyze_variance_distribution', '.lib.plotting')) for name in sys.modules)
assert_inert()
""", tmp_path)
    assert result.stdout == ""


DEFAULTS = dict(test_mode=False, use_history=False, use_kinematics=False,
                use_timewarp=False, use_force=False, suffix="", metric="variance")
CASES = [([], {})] + [([flag], {key: True}) for flag, key in (
    ("--test-mode", "test_mode"), ("--history", "use_history"),
    ("--kinematics", "use_kinematics"), ("--time-warp", "use_timewarp"),
    ("--force-vectors", "use_force"),
)] + [(["--test-mode", "--history", "--kinematics", "--time-warp", "--force-vectors",
        "--suffix", " selected;$(literal) result ", "--metric", "mean"],
       dict(test_mode=True, use_history=True, use_kinematics=True, use_timewarp=True,
            use_force=True, suffix=" selected;$(literal) result ", metric="mean"))]


@pytest.mark.parametrize("argv, overrides", CASES)
def test_main_forwards_exact_parser_values(argv, overrides, monkeypatch):
    owner = importlib.import_module(CANONICAL)
    calls = []
    monkeypatch.setattr(owner, "run_analysis", lambda **kwargs: calls.append(kwargs) or object())
    assert owner.main(argv) is None
    assert calls == [dict(DEFAULTS, **overrides)]


def test_main_none_uses_process_arguments(monkeypatch):
    owner = importlib.import_module(CANONICAL)
    calls = []
    monkeypatch.setattr(owner, "run_analysis", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["ppe", "--suffix", "process-argv", "--metric", "variance"])
    assert owner.main() is None
    assert calls == [dict(DEFAULTS, suffix="process-argv")]


@pytest.mark.parametrize("argv", [["--metric", "unknown"], ["--unknown"], ["--suffix"]])
def test_invalid_options_exit_before_analysis(argv, monkeypatch):
    owner = importlib.import_module(CANONICAL)
    calls = []
    monkeypatch.setattr(owner, "run_analysis", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as error:
        owner.main(argv)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("physical", PHYSICAL)
@pytest.mark.parametrize("loader", ["spec", "runpy"])
def test_nonmain_file_loading_is_inert(physical, loader, tmp_path):
    result = child(f"""
canonical = importlib.import_module({CANONICAL!r})
name = '_ppe_entry_under_test'
with warnings.catch_warnings(record=True) as emitted:
    warnings.simplefilter('always', DeprecationWarning)
    if {loader!r} == 'spec':
        spec = importlib.util.spec_from_file_location(name, {str(ROOT / physical)!r})
        wrapper = importlib.util.module_from_spec(spec)
        sys.modules[name] = wrapper
        spec.loader.exec_module(wrapper)
        namespace = vars(wrapper)
        loaded = sys.modules.pop(name)
    else:
        namespace = runpy.run_path({str(ROOT / physical)!r}, run_name=name)
        loaded = None
if {physical!r}.startswith('compat/'):
    assert len(emitted) == 1 and issubclass(emitted[0].category, DeprecationWarning)
    assert {CANONICAL!r} in str(emitted[0].message)
    assert namespace['_real'] is canonical
    assert loaded is None or loaded is canonical
    for function in {PUBLIC_FUNCTIONS!r}:
        assert namespace['__getattr__'](function) is getattr(canonical, function)
        assert function in namespace['__dir__']()
else:
    assert namespace['run_analysis'].__module__ == name
    assert callable(namespace['main'])
assert_inert()
""", tmp_path)
    assert result.stdout == ""


@pytest.mark.parametrize("physical", PHYSICAL)
def test_guarded_help_outside_checkout_restores_caller_state(physical, tmp_path):
    result = child(f"""
sys.argv = [{str(ROOT / physical)!r}, '--help']
try:
    runpy.run_path({str(ROOT / physical)!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert_inert()
""", tmp_path, activate=False)
    for option in ("--test-mode", "--history", "--kinematics", "--time-warp",
                   "--force-vectors", "--suffix", "--metric"):
        assert option in result.stdout


@pytest.mark.parametrize("physical", PHYSICAL)
def test_true_direct_file_help_without_preloads(physical, tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **THREAD_ENV)
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / physical), "--help"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--metric" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("physical", PHYSICAL)
@pytest.mark.parametrize("preexisting_paths", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_main_lifetime_and_cleanup_with_test_owned_analysis(physical, preexisting_paths, fail, tmp_path):
    result = child(f"""
root = Path({str(ROOT)!r})
if {preexisting_paths!r}:
    for selected in (str(root), str(root / 'src')):
        if selected not in sys.path:
            sys.path.insert(0, selected)
paths = list(sys.path)
expected_additions = {{str(root), str(root / 'src')}} - set(paths)
calls = []
class TestOwnedAnalysisFailure(RuntimeError):
    pass
def fake_analysis(**kwargs):
    assert str(root) in sys.path and str(root / 'src') in sys.path
    assert set(sys.path) - set(paths) == expected_additions
    calls.append(kwargs)
    if {fail!r}:
        raise TestOwnedAnalysisFailure('test-owned analysis failure')
    return object()
def install_stub(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == 'main' and frame.f_code.co_filename == {str(ROOT / PHYSICAL[0])!r}:
        frame.f_globals['run_analysis'] = fake_analysis
    return None
sys.argv = [{str(ROOT / physical)!r}, '--history', '--metric', 'mean', '--suffix', 'selected']
original_main = sys.modules['__main__']
sys.settrace(install_stub)
try:
    try:
        runpy.run_path({str(ROOT / physical)!r}, run_name='__main__')
    except TestOwnedAnalysisFailure:
        assert {fail!r}
    else:
        assert not {fail!r}
finally:
    sys.settrace(None)
assert calls == [{{'test_mode': False, 'use_history': True, 'use_kinematics': False,
                 'use_timewarp': False, 'use_force': False, 'suffix': 'selected', 'metric': 'mean'}}]
assert sys.modules['__main__'] is original_main
assert_inert()
""", tmp_path, activate=False)
    assert result.stdout == ""


@pytest.mark.parametrize("physical", PHYSICAL)
@pytest.mark.parametrize("preexisting_paths", [False, True])
def test_import_failure_restores_only_added_paths(physical, preexisting_paths, tmp_path):
    result = child(f"""
root = Path({str(ROOT)!r})
if {preexisting_paths!r}:
    for selected in (str(root), str(root / 'src')):
        if selected not in sys.path:
            sys.path.insert(0, selected)
paths = list(sys.path)
expected_additions = {{str(root), str(root / 'src')}} - set(paths)
class TestOwnedImportFailure(RuntimeError):
    pass
original_import = builtins.__import__
def import_failure(name, *args, **kwargs):
    if name.startswith('studies.deep_research_experiments.utils'):
        assert str(root) in sys.path and str(root / 'src') in sys.path
        assert set(sys.path) - set(paths) == expected_additions
        raise TestOwnedImportFailure('test-owned loader import failure')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_failure
sys.argv = [{str(ROOT / physical)!r}, '--help']
try:
    try:
        runpy.run_path({str(ROOT / physical)!r}, run_name='__main__')
    except TestOwnedImportFailure:
        pass
    else:
        raise AssertionError('synthetic import failure did not occur')
finally:
    builtins.__import__ = original_import
assert_inert()
""", tmp_path, activate=False)
    assert result.stdout == ""
