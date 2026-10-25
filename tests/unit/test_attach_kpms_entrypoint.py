"""Base attachment entry routes and independently specified synthetic timelines.

All decoded objects and backend responses below are test owned. No empirical
pickle, checkpoint, recording or model is loaded, and no fitting is performed.
"""

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = "studies.session_model_exports.attach_kpms"
LEGACY = "syllable_generation.analysis.attach_kpms"
PHYSICAL = (
    "studies/session_model_exports/attach_kpms.py",
    "compat/syllable_generation/analysis/attach_kpms.py",
)
PUBLIC_FUNCTIONS = ("get_sessions", "_attach_kpms_full_to_sessions", "run_attach", "main")
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


@pytest.fixture
def attach():
    return importlib.import_module(CANONICAL)


@pytest.mark.parametrize("legacy_first", [False, True])
def test_cold_both_order_identity_and_canonical_utilities(legacy_first, tmp_path):
    result = child(f'''
first, second = ({LEGACY!r}, {CANONICAL!r}) if {legacy_first!r} else ({CANONICAL!r}, {LEGACY!r})
one, two = importlib.import_module(first), importlib.import_module(second)
assert one is two and one.__name__ == {CANONICAL!r}
assert Path(one.__file__).resolve() == Path({str(ROOT)!r}) / {PHYSICAL[0]!r}
for name in {PUBLIC_FUNCTIONS!r}:
    assert getattr(one, name) is getattr(two, name)
    assert getattr(one, name).__module__ == {CANONICAL!r}
assert importlib.import_module('syllable_generation.analysis').attach_kpms is one
assert importlib.import_module('studies.session_model_exports').attach_kpms is one
sampling = importlib.import_module('studies.session_model_exports.utils.sampling')
results = importlib.import_module('studies.session_model_exports.utils.syll_results')
assert one.flatten_results_to_timeline is sampling.flatten_results_to_timeline
for name in ('build_result_to_session_map', 'compute_actual_lengths_from_sessions', 'load_kpms_results_and_vfi'):
    assert getattr(one, name) is getattr(results, name)
assert_inert()
''', tmp_path)
    assert result.stdout == ""


@pytest.mark.parametrize("physical", PHYSICAL)
@pytest.mark.parametrize("loader", ["spec", "runpy"])
def test_nonmain_physical_load_is_inert_and_legacy_delegates(physical, loader, tmp_path):
    result = child(f'''
canonical = importlib.import_module({CANONICAL!r})
calls = []
canonical.main = lambda *args, **kwargs: calls.append((args, kwargs))
name = '_attach_entry_under_test'
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
    if loaded is not None:
        assert loaded is canonical
    assert namespace['_real'] is canonical
    for function in {PUBLIC_FUNCTIONS!r}:
        assert namespace['__getattr__'](function) is getattr(canonical, function)
        assert function in namespace['__dir__']()
else:
    assert namespace['run_attach'].__module__ == name
    assert callable(namespace['main'])
    assert namespace['flatten_results_to_timeline'] is canonical.flatten_results_to_timeline
assert calls == []
assert_inert()
''', tmp_path)
    assert result.stdout == ""


def test_legacy_runpy_main_delegates_once_to_test_owned_main(tmp_path):
    result = child(f'''
root = Path({str(ROOT)!r})
sys.path[:0] = [str(root), str(root / 'src')]
canonical = importlib.import_module({CANONICAL!r})
sys.path[:] = paths
calls = []
sys.argv = [{str(ROOT / PHYSICAL[1])!r}, '--sessions', 'test-owned.pkl', '--results', 'fixture-model']
def sentinel_main(argv=None):
    assert str(root) in sys.path and str(root / 'src') in sys.path
    calls.append((argv, list(sys.argv)))
canonical.main = sentinel_main
original_main = sys.modules['__main__']
namespace = runpy.run_path({str(ROOT / PHYSICAL[1])!r}, run_name='__main__')
assert namespace['_real'] is canonical
assert calls == [(None, sys.argv)]
assert sys.modules['__main__'] is original_main
assert_inert()
''', tmp_path, activate=False)
    assert result.stdout == ""


def test_canonical_runpy_main_uses_parser_with_only_test_owned_decoding(tmp_path):
    result = child(f'''
root = Path({str(ROOT)!r})
sys.path[:0] = [str(root), str(root / 'src')]
results_owner = importlib.import_module('studies.session_model_exports.utils.syll_results')
sys.path[:] = paths
session = SimpleNamespace(session_id='fixture', metadata={{}}, video_refs_df=pd.DataFrame({{'n_frames': [8]}}))
sessions, calls = [session], []
def read(path):
    assert str(root) in sys.path and str(root / 'src') in sys.path
    assert Path(path) == Path('test-owned.pkl')
    calls.append('read')
    return sessions
def load(path, *, use_cpu):
    assert Path(path) == Path('fixture-model') and use_cpu is False
    calls.append('legacy')
    return {{'record_fixture': {{'syllable': np.array([2, 5])}}}}, {{'record_fixture': np.array([3, 6])}}
def save(obj, stream):
    assert obj is sessions
    assert session.syllables.index.tolist() == list(range(8))
    assert session.syllables.loc[3, 'syllable'] == 2
    assert session.syllables.loc[6, 'syllable'] == 5
    calls.append('save')
original_open = builtins.open
def open_output(path, mode='r', *args, **kwargs):
    if Path(path) == Path('selected-output.pkl'):
        assert mode == 'wb'
        return io.BytesIO()
    return original_open(path, mode, *args, **kwargs)
pd.read_pickle, results_owner.load_kpms_results_and_vfi, pickle.dump = read, load, save
builtins.open = open_output
sys.argv = [{str(ROOT / PHYSICAL[0])!r}, '--sessions', 'test-owned.pkl', '--results', 'fixture-model',
            '--out', 'selected-output.pkl']
original_main = sys.modules['__main__']
runpy.run_path({str(ROOT / PHYSICAL[0])!r}, run_name='__main__')
assert calls == ['read', 'legacy', 'save']
assert sys.modules['__main__'] is original_main
assert_inert()
''', tmp_path, activate=False)
    json_start = result.stdout.index('{\n')
    assert json.loads(result.stdout[json_start:]) == {"attached": 1, "save_path": "selected-output.pkl"}


@pytest.mark.parametrize("physical", PHYSICAL)
def test_guarded_physical_help_bootstraps_only_for_main_outside_checkout(physical, tmp_path):
    result = child(f'''
root = Path({str(ROOT)!r})
assert str(root) not in sys.path
sys.argv = [{str(ROOT / physical)!r}, '--help']
try:
    runpy.run_path({str(ROOT / physical)!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert str(root) not in sys.path
assert_inert()
''', tmp_path, activate=False)
    for option in ("--sessions", "--results", "--model-name", "--overwrite", "--cpu", "--key-prefix",
                   "--use-lib", "--downsample", "--out"):
        assert option in result.stdout


@pytest.mark.parametrize("physical", PHYSICAL)
def test_main_bootstrap_restores_paths_after_test_owned_import_failure(physical, tmp_path):
    result = child(f'''
root = Path({str(ROOT)!r})
assert str(root) not in sys.path and str(root / 'src') not in sys.path
class TestOwnedImportFailure(RuntimeError):
    pass
original_import = builtins.__import__
def import_failure(name, *args, **kwargs):
    if name.startswith('studies.session_model_exports.utils'):
        assert str(root) in sys.path and str(root / 'src') in sys.path
        raise TestOwnedImportFailure('test-owned utility import failure')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_failure
sys.argv = [{str(ROOT / physical)!r}, '--help']
original_main = sys.modules['__main__']
try:
    runpy.run_path({str(ROOT / physical)!r}, run_name='__main__')
except TestOwnedImportFailure:
    pass
else:
    raise AssertionError('synthetic dependency failure did not occur')
assert sys.modules['__main__'] is original_main
assert_inert()
''', tmp_path, activate=False)
    assert result.stdout == ""


@pytest.mark.parametrize("physical", PHYSICAL)
def test_main_bootstrap_lasts_through_selected_execution_and_restores_on_failure(physical, tmp_path):
    result = child(f'''
root = Path({str(ROOT)!r})
sys.path[:0] = [str(root), str(root / 'src')]
canonical = importlib.import_module({CANONICAL!r})
sys.path[:] = paths
class TestOwnedMainFailure(RuntimeError):
    pass
calls = []
def selected_execution(*args, **kwargs):
    assert str(root) in sys.path and str(root / 'src') in sys.path
    calls.append('selected execution')
    raise TestOwnedMainFailure('test-owned main or decode failure')
if {physical!r}.startswith('compat/'):
    canonical.main = selected_execution
else:
    pd.read_pickle = selected_execution
sys.argv = [{str(ROOT / physical)!r}, '--sessions', 'test-owned.pkl', '--results', 'fixture-model']
original_main = sys.modules['__main__']
try:
    runpy.run_path({str(ROOT / physical)!r}, run_name='__main__')
except TestOwnedMainFailure:
    pass
else:
    raise AssertionError('synthetic selected-execution failure did not occur')
assert calls == ['selected execution']
assert sys.modules['__main__'] is original_main
assert_inert()
''', tmp_path, activate=False)
    assert result.stdout == ""


@pytest.mark.parametrize("physical", PHYSICAL)
def test_true_direct_file_help_from_outside_checkout(physical, tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **THREAD_ENV)
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / physical), "--help"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--sessions" in result.stdout and "--results" in result.stdout and "--out" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("from_sys_argv", [False, True])
@pytest.mark.parametrize("all_flags", [False, True])
def test_main_forwards_exact_cli_values_and_prints_json(attach, monkeypatch, capsys, from_sys_argv, all_flags):
    calls = []

    def sentinel(*args, **kwargs):
        calls.append((args, kwargs))
        return {"attached": 2, "save_path": "test-owned selected output.pkl"}

    monkeypatch.setattr(attach, "run_attach", sentinel)
    arguments = ["--sessions", "test-owned input.pkl", "--results", "fixture project"]
    if all_flags:
        arguments += ["--model-name", "fixture model", "--overwrite", "--cpu", "--key-prefix", "tag",
                      "--use-lib", "--downsample", "2.75", "--out", "test-owned selected output.pkl"]
    if from_sys_argv:
        monkeypatch.setattr(sys, "argv", ["test-entry"] + arguments)
    assert attach.main(None if from_sys_argv else arguments) is None
    assert calls == [(("test-owned input.pkl", "fixture project"), {
        "model_name": "fixture model" if all_flags else None,
        "overwrite": all_flags, "use_cpu": all_flags,
        "key_prefix": "tag" if all_flags else "kpms", "use_kpms_lib": all_flags,
        "downsample_rate": 2.75 if all_flags else 1.0,
        "out_path": "test-owned selected output.pkl" if all_flags else None,
    })]
    assert capsys.readouterr().out == json.dumps(
        {"attached": 2, "save_path": "test-owned selected output.pkl"}, indent=2,
    ) + "\n"


def test_api_defaults_remain_true_while_library_mode_remains_opt_in(attach):
    parameters = inspect.signature(attach.run_attach).parameters
    assert parameters["overwrite"].default is True
    assert parameters["use_cpu"].default is True
    assert parameters["use_kpms_lib"].default is False
    assert parameters["downsample_rate"].default == 1.0
    assert parameters["out_path"].default is None


@pytest.mark.parametrize("arguments", [[], ["--sessions", "fixture.pkl"], ["--results", "fixture-model"],
                                       ["--sessions", "fixture.pkl", "--results", "model", "--downsample", "bad"]])
def test_main_rejects_missing_required_or_invalid_arguments_before_execution(attach, monkeypatch, capsys, arguments):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid arguments reached attachment execution")

    monkeypatch.setattr(attach, "run_attach", forbidden)
    with pytest.raises(SystemExit) as error:
        attach.main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("empty", [False, True])
def test_get_sessions_returns_stub_objects_or_rejects_empty_input(attach, monkeypatch, empty):
    sessions = [] if empty else [SimpleNamespace(session_id="fixture")]
    paths = []

    def read(path):
        paths.append(path)
        return sessions

    monkeypatch.setattr(attach.pd, "read_pickle", read)
    if empty:
        with pytest.raises(ValueError, match="No sessions in test-owned.pkl"):
            attach.get_sessions("test-owned.pkl")
    else:
        assert attach.get_sessions("test-owned.pkl") is sessions
    assert paths == ["test-owned.pkl"]


def make_session(identifier="fixture", length=14):
    # The first row is authoritative; the larger second row must not be selected.
    return SimpleNamespace(session_id=identifier, metadata={},
                           video_refs_df=pd.DataFrame({"n_frames": [length, length + 100]}))


def test_normal_flipped_pair_uses_real_matching_and_keeps_dataframe_identity(attach):
    session = make_session(length=100)
    normal, flipped = pd.DataFrame({"syllable": [2, 3]}), pd.DataFrame({"syllable": [8, 5, 1]})
    frames = {"record_fixture": normal, "record_fixture_flipped": flipped}
    assert attach._attach_kpms_full_to_sessions([session], frames) == 2
    assert session.syllables is normal and session.syllables_flipped is flipped
    assert session.results_dict_key == "record_fixture_flipped"
    # A mismatch between expected length=100 and either dataframe is retained.
    assert len(session.syllables) == 2 and len(session.syllables_flipped) == 3


@pytest.mark.parametrize("flipped", [False, True])
def test_single_stream_destination_and_unmatched_recordings(attach, flipped):
    session, untouched = make_session(), make_session("other")
    frame = pd.DataFrame({"syllable": [7]})
    key = "record_fixture" + ("_flipped" if flipped else "")
    frames = {key: frame, "record_unknown": pd.DataFrame({"syllable": [999]})}
    assert attach._attach_kpms_full_to_sessions([session, untouched], frames) == 1
    assert getattr(session, "syllables_flipped" if flipped else "syllables") is frame
    assert not hasattr(session, "syllables" if flipped else "syllables_flipped")
    assert not hasattr(untouched, "syllables") and untouched.results_dict_key == ""


@pytest.mark.parametrize("existing_dict", [False, True])
def test_dictionary_attachment_preserves_key_prefix_and_existing_entries(attach, existing_dict):
    session = make_session()
    old = pd.DataFrame({"syllable": [99]})
    session.syllables = {"preexisting": old} if existing_dict else old
    frames = {"record_fixture": pd.DataFrame({"syllable": [2]}),
              "record_fixture_flipped": pd.DataFrame({"syllable": [8]})}
    assert attach._attach_kpms_full_to_sessions([session], frames, key_prefix="selected", single_df=False) == 2
    assert set(session.syllables) == ({"preexisting"} if existing_dict else set()) | {
        "selected:record_fixture", "selected:record_fixture_flipped",
    }
    for key, frame in frames.items():
        assert session.syllables["selected:" + key] is frame
    if existing_dict:
        assert session.syllables["preexisting"] is old
    assert not hasattr(session, "syllables_flipped")


def test_explicit_two_stream_map_allows_late_nonflipped_assignment(attach):
    session = make_session()
    first, last = pd.DataFrame({"syllable": [1]}), pd.DataFrame({"syllable": [2]})
    assert attach._attach_kpms_full_to_sessions(
        [session], {"one": first, "two": last}, result_to_session={"one": 0, "two": 0},
        expected_lengths={"one": 999, "two": 999},
    ) == 2
    assert session.syllables is last


def test_more_than_two_streams_are_rejected_before_attachment(attach):
    session = make_session()
    frames = {key: pd.DataFrame({"syllable": [i]}) for i, key in enumerate(("one", "two", "three"))}
    with pytest.raises(ValueError, match=r"single_df=True but >2 recs map to sessions:.*fixture"):
        attach._attach_kpms_full_to_sessions(
            [session], frames, result_to_session={key: 0 for key in frames}, expected_lengths={},
        )
    assert not hasattr(session, "syllables") and not hasattr(session, "syllables_flipped")


class TestOwnedNPZ(dict):
    __test__ = False

    @property
    def files(self):
        return list(self)


def controlled_attachment(attach, monkeypatch, tmp_path, *, library):
    """Stub only external decoding/backend/serialization; run owned mapping/math."""
    source = tmp_path / "test-owned-input.pkl"
    source.write_bytes(b"TEST OWNED: deliberately not a pickle")
    sessions = [make_session()]
    model = tmp_path / "fixture-project"
    keys = ("record_fixture", "record_fixture_flipped")
    arrays = {keys[0]: np.array([4.0, 7.0, 11.0]), keys[1]: np.array([4.0, 7.0, 11.0])}
    results = {
        keys[0]: {"syllable": np.array([3, 6, 9]), "centroid": np.array([[10., 20.], [30., 40.], [50., 60.]])},
        keys[1]: {"syllable": np.array([8, 5, 2]), "centroid": np.array([[1., 2.], [3., 4.], [5., 6.]])},
    }
    calls = []

    def read(path):
        assert Path(path) == source
        calls.append(("read", Path(path)))
        return sessions

    def legacy(path, *, use_cpu):
        assert not library
        calls.append(("legacy", Path(path), use_cpu))
        return results, arrays

    def compute(project, model_name, *, smooth_heading):
        assert library
        calls.append(("library", project, model_name, smooth_heading))
        # Deliberately unsorted model rows, with an out-of-range row to be clipped.
        return pd.DataFrame({
            "name": [keys[0]] * 4 + [keys[1]] * 4 + ["no_vfi"] * 1,
            "frame_index": [2, 0, 1, 9, 1, 2, 0, 9, 0],
            "syllable": [9, 3, 6, 777, 5, 2, 8, 777, 555],
            "centroid_x": [50., 10., 30., 777., 3., 5., 1., 777., 555.],
        })

    def load(path):
        assert library
        calls.append(("vfi", Path(path)))
        assert Path(path) == model / "test-model" / "video_frame_indexes.npz"
        return TestOwnedNPZ(arrays)

    def dump(objects, stream):
        assert objects is sessions
        calls.append(("dump", Path(stream.name)))
        stream.write(b"TEST OWNED SERIALIZATION SENTINEL")

    monkeypatch.setattr(attach.pd, "read_pickle", read)
    monkeypatch.setattr(attach, "load_kpms_results_and_vfi", legacy)
    monkeypatch.setattr(attach.np, "load", load)
    monkeypatch.setattr(attach.pickle, "dump", dump)
    # A test-owned module satisfies the explicit optional import even under the
    # CPU suite's backend finder; no actual modeling distribution is imported.
    monkeypatch.setitem(sys.modules, "keypoint_moseq", SimpleNamespace(compute_moseq_df=compute))
    return SimpleNamespace(source=source, sessions=sessions, model=model, keys=keys, arrays=arrays, calls=calls)


@pytest.mark.parametrize("library", [False, True])
@pytest.mark.parametrize("downsample", [1.0, 2.75])
def test_run_attach_preserves_distinct_native_timelines_and_selected_output(attach, monkeypatch, tmp_path, library, downsample):
    fixture = controlled_attachment(attach, monkeypatch, tmp_path, library=library)
    output = tmp_path / "selected-output.pkl"
    environment = dict(os.environ)
    use_cpu = downsample > 1
    result = attach.run_attach(
        fixture.source, fixture.model, model_name="test-model", use_kpms_lib=library,
        use_cpu=use_cpu, downsample_rate=downsample, out_path=output, key_prefix="chosen",
    )
    assert result == {"attached": 2, "save_path": str(output)}
    assert fixture.source.read_bytes() == b"TEST OWNED: deliberately not a pickle"
    assert output.read_bytes() == b"TEST OWNED SERIALIZATION SENTINEL"
    assert dict(os.environ) == environment
    assert fixture.calls == [("read", fixture.source)] + (
        [("library", str(fixture.model), "test-model", True),
         ("vfi", fixture.model / "test-model" / "video_frame_indexes.npz")]
        if library else [("legacy", fixture.model, use_cpu)]
    ) + [("dump", output)]
    normal, flipped = fixture.sessions[0].syllables, fixture.sessions[0].syllables_flipped
    for frame, values, centroids in ((normal, [3, 6, 9], [10., 30., 50.]),
                                     (flipped, [8, 5, 2], [1., 3., 5.])):
        if library:
            assert frame.index.tolist() == list(range(4, 12))
            assert frame.syllable.tolist() == [values[0]] * 3 + [values[1]] * 4 + [values[2]]
            np.testing.assert_array_equal(frame.centroid_x, [centroids[0]] * 3 + [centroids[1]] * 4 + [centroids[2]])
            # Model-row positions remain model-row fields; native frames are the
            # table index. This test does not reinterpret one clock as the other.
            assert frame.frame_index.tolist() == [0.] * 3 + [1.] * 4 + [2.]
        else:
            assert frame.index.tolist() == list(range(14))
            assert frame.loc[[4, 7, 11], "syllable"].tolist() == values
            assert frame.loc[[0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13], "syllable"].isna().all()
            assert frame.syllable.dtype == pd.Int64Dtype()
            np.testing.assert_array_equal(frame.loc[[4, 7, 11], "centroid_0"], centroids)
    for array in fixture.arrays.values():
        np.testing.assert_array_equal(array, [4., 7., 11.])


@pytest.mark.parametrize("explicit_output", [False, True])
def test_overwrite_failure_is_late_but_never_serializes_or_changes_existing_output(attach, monkeypatch, tmp_path, explicit_output):
    fixture = controlled_attachment(attach, monkeypatch, tmp_path, library=False)
    output = tmp_path / "existing-output.pkl" if explicit_output else fixture.source
    if explicit_output:
        output.write_bytes(b"EXISTING TEST OWNED OUTPUT")
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="exists and overwrite=False"):
        attach.run_attach(fixture.source, fixture.model, overwrite=False,
                          out_path=output if explicit_output else None)
    assert fixture.calls == [("read", fixture.source), ("legacy", fixture.model, True)]
    assert hasattr(fixture.sessions[0], "syllables") and hasattr(fixture.sessions[0], "syllables_flipped")
    assert output.read_bytes() == before
    assert set(tmp_path.iterdir()) == ({fixture.source, output} if explicit_output else {fixture.source})


def test_api_default_output_overwrites_only_selected_test_owned_input(attach, monkeypatch, tmp_path):
    fixture = controlled_attachment(attach, monkeypatch, tmp_path, library=False)
    result = attach.run_attach(fixture.source, fixture.model)
    assert result == {"attached": 2, "save_path": str(fixture.source)}
    assert fixture.calls[-1] == ("dump", fixture.source)
    assert fixture.source.read_bytes() == b"TEST OWNED SERIALIZATION SENTINEL"
    assert list(tmp_path.iterdir()) == [fixture.source]


def test_missing_optional_library_is_reported_only_on_explicit_library_call(attach, monkeypatch, tmp_path):
    source = tmp_path / "never-decoded.pkl"
    calls = []
    original_import = __import__

    def controlled_import(name, *args, **kwargs):
        if name == "keypoint_moseq":
            calls.append("optional-import")
            raise ImportError("test-owned missing optional dependency")
        return original_import(name, *args, **kwargs)

    def read(path):
        assert Path(path) == source
        calls.append("read")
        return [make_session()]

    monkeypatch.setattr(attach.pd, "read_pickle", read)
    monkeypatch.setattr("builtins.__import__", controlled_import)
    with pytest.raises(ImportError, match="^keypoint_moseq library not installed$"):
        attach.run_attach(source, tmp_path / "test-owned-model", use_kpms_lib=True)
    assert calls == ["read", "optional-import"]
    assert not list(tmp_path.iterdir())
