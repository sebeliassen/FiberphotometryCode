"""Synthetic stage policies, direct-call fallbacks and bounded process transport."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.contracts.session_record import SessionRecord
from studies.session_building.jonathan import config as lib_config
from studies.session_building.jonathan.config import JonathanConfig
from studies.session_building.jonathan.steps import fiberphotometry, tracking, video

ROOT = Path(__file__).resolve().parents[2]
STAMP = "2026-03-05T11_01_01"


@pytest.mark.parametrize("explicit", [False, True])
def test_tracking_uses_selected_chamber_policy_and_retains_direct_fallback(explicit, tmp_path, monkeypatch):
    path = tmp_path / f"Arena1and2_LeftTopCamVid_{STAMP}_upperDLC_fixture.h5"
    path.touch()
    selected = JonathanConfig(
        camera_token_map={"lefttopcamvid": {"upper": "owned_chamber"}},
        chamber_to_index={"owned_chamber": 7},
    )
    ctx = SimpleNamespace(h5_dir=tmp_path, project="synthetic")
    if explicit:
        ctx.cfg = selected
    else:
        monkeypatch.setattr(lib_config, "JCFG", selected)
    prior, values = tracking.JCFG, asdict(tracking.JCFG)
    sessions = tracking.discover_sessions(ctx, [])
    chamber, index = ("owned_chamber", 7) if explicit else ("top_left", 0)
    assert [s.session_id for s in sessions] == [f"{STAMP}_{chamber}"]
    assert sessions[0].metadata["chamber"] == chamber
    assert sessions[0].metadata["chamber_index"] == index
    row = sessions[0].tracking_refs_df.to_dict("records")[0]
    assert row["path"] == str(path) and row["chamber_index"] == index
    assert row["camera_key"] == f"left_cam:{chamber}"
    assert tracking.JCFG is prior and asdict(prior) == values
    assert tracking.H5_PATTERN.pattern == prior.h5_filename_regex
    assert path.stat().st_size == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_video_refs_use_selected_tokens_with_controlled_metadata_probe(explicit, tmp_path, monkeypatch):
    path = tmp_path / f"Arena1and2_LeftTopCamVid_{STAMP}_upper.avi"
    path.touch()
    calls = []

    class Capture:
        def __init__(self, selected_path):
            calls.append(selected_path)

        def isOpened(self):
            return True

        def get(self, key):
            return {1: 40.0, 2: 80.0}[key]

        def release(self):
            pass

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=Capture, CAP_PROP_FPS=1, CAP_PROP_FRAME_COUNT=2))
    selected = JonathanConfig(
        camera_token_map={"lefttopcamvid": {"upper": "owned_chamber"}},
        camset_to_slot={"lefttopcamvid": "owned_camera"},
    )
    chamber, slot = ("owned_chamber", "owned_camera") if explicit else ("top_left", "left_cam")
    ctx = SimpleNamespace(root=tmp_path, video_dir=tmp_path)
    if explicit:
        ctx.cfg = selected
    else:
        monkeypatch.setattr(lib_config, "JCFG", selected)
    record = SessionRecord(session_id=f"{STAMP}_{chamber}", project="synthetic")
    sessions = [record]
    assert video.attach_video_refs(ctx, sessions) is sessions
    assert calls == [str(path)]
    assert record.video_refs_df.to_dict("records") == [dict(
        path=str(path), path_type="video", kind="avi", role="aux", ts=STAMP,
        chamber=chamber, camset="LeftTopCamVid", cam_slot=slot, arena="Arena1and2",
        camera_key=f"{slot}:{chamber}", fps=40.0, n_frames=80, duration_s=2.0,
        meta_origin="opencv",
    )]
    assert path.stat().st_size == 0


def test_photometry_discovery_uses_selected_filename_and_tolerance(tmp_path):
    selected = JonathanConfig(
        phot_combined_csv_glob="owned_*.csv",
        phot_combined_csv_regex=r"^owned_(\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})\.csv$",
        tolerance_seconds=0.5,
    )
    trial = tmp_path / "T1_6.2.4.e"
    trial.mkdir()
    path = trial / "owned_2026-03-05T11.01.02.csv"
    path.touch()
    ctx = SimpleNamespace(csv_root=tmp_path, cfg=selected)
    record = SessionRecord(session_id="fixture", project="synthetic",
                           metadata={"timestamp": STAMP, "chamber_index": 1})
    sessions = [record]
    assert fiberphotometry.attach_photometry_refs(ctx, sessions) is sessions
    assert "photometry_csv_path" not in record.metadata
    assert fiberphotometry.attach_photometry_refs(ctx, sessions, tolerance_seconds=1.0) is sessions
    assert record.metadata == dict(
        timestamp=STAMP, chamber_index=1, trial_id="T1_6.2.4.e", animal_id="2",
        n_chambers=4, photometry_ts_csv="2026-03-05T11:01:02",
        photometry_csv_path=str(path), photometry_source="csv_combined",
    )
    assert path.stat().st_size == 0


def test_realization_and_split_use_selected_column_clock_and_led_policy(tmp_path, monkeypatch):
    selected = JonathanConfig(
        fiber_col_regex=r"^([A-Z]+)_(\d+)$", phot_system_ts_col="OwnedTS",
        allts_system_ts_col="VideoTS", frame_index_col_name="NativeFrame",
        led_col_name="OwnedLED", led_state_map={10: ("iso", "405"), 20: ("G", "555")},
    )
    path = tmp_path / "photometry.csv"
    pd.DataFrame({
        "OwnedTS": [0.1, 1.8, 2.9, np.nan], "OwnedLED": [10, 20, 10, 20],
        "G_0": [11.0, 12.0, 13.0, 14.0], "G_1": [21.0, 22.0, 23.0, 24.0],
        "G_99": [31.0, 32.0, 33.0, 34.0],
    }).to_csv(path, index=False)
    allts = tmp_path / f"CenterTopCam_AllTimestampsCombined{STAMP}.csv"
    pd.DataFrame({"VideoTS": [0.0, 1.0, 2.0, 3.0]}).to_csv(allts, index=False)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    # Parser-engine availability is unrelated to policy selection. Read only our
    # two tiny CSV fixtures with the standard installed parser in either env.
    original_read = pd.read_csv

    def fixture_read(*args, **kwargs):
        kwargs.pop("engine", None)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fixture_read)
    record = SessionRecord(session_id="fixture", project="synthetic", metadata={
        "timestamp": STAMP, "chamber_index": 0, "photometry_csv_path": str(path),
        "signal_0": "L-DLS", "signal_1": "R-DMS",
    })
    ctx = SimpleNamespace(csv_root=tmp_path, cfg=selected)
    sessions = [record]
    assert fiberphotometry.realize_photometry_columns(ctx, sessions) is sessions
    actual = record.photometry["combined_filtered"]
    assert list(actual.columns) == ["OwnedTS", "OwnedLED", "signal_0", "signal_1", "NativeFrame"]
    assert actual["NativeFrame"].tolist() == [0, 2, 3, -1]
    assert record.metadata["effective_indices"] == [0, 1, 2, 3]
    assert record.metadata["signal_meta"]["signal_0"]["orig_fiber"] == "G_0"
    assert fiberphotometry.split_photometry(ctx, sessions) is sessions
    assert record.metadata["photometry_channels"] == ["405", "555"]
    assert record.photometry["405"]["signal_0"].tolist() == [11.0, 13.0]
    assert record.photometry["555"]["signal_1"].tolist() == [22.0, 24.0]
    assert record.photometry["405"]["NativeFrame"].tolist() == [0, 3]
    assert record.photometry["555"]["NativeFrame"].tolist() == [2, -1]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def selected_compute_policy():
    return JonathanConfig(
        phot_system_ts_col="OwnedTS", include_ransac=False, include_wallace=False,
        include_manual=True, include_zscore=False, include_dff=True, include_prominence=False,
    )


def compute_fixture():
    times = np.arange(32, dtype=float)
    frame = pd.DataFrame({"OwnedTS": times * 0.5, "SystemTimestamp": times * 0.1,
                          "signal_0": 20.0 + times * 0.25})
    reference = frame.copy(deep=True)
    reference["signal_0"] = 10.0
    return SessionRecord(session_id="fixture", project="synthetic", photometry={
        "415": reference, "470": frame,
    })


def assert_selected_compute(record):
    target = record.photometry["470"]
    assert list(target.columns) == ["OwnedTS", "SystemTimestamp", "signal_0",
                                    "signal_0_zscore_manual", "signal_0_dff"]
    np.testing.assert_allclose(np.asarray(target["signal_0_dff"], dtype=float),
                               1.0 + np.arange(32) * 0.025, rtol=1e-13, atol=1e-13)
    raw = pd.Series(20.0 + np.arange(32) * 0.25)
    baseline = raw.rolling(window=10, center=True, min_periods=1).quantile(0.1)
    cleaned = (raw - baseline) / baseline
    rolling = cleaned.rolling(window=40, center=True, min_periods=1)
    expected = (cleaned - rolling.mean()) / rolling.std()
    np.testing.assert_allclose(np.asarray(target["signal_0_zscore_manual"], dtype=float),
                               expected.to_numpy(), rtol=1e-13, atol=1e-13, equal_nan=True)
    assert record.metadata == dict(dff_iso_channel="415", dff_computed_channels=["470"],
                                   datta_replication_enabled=True, reference_interpolation=True)


@pytest.mark.parametrize("selection", ["context", "explicit", "live_fallback"])
def test_actual_compute_uses_selected_time_flags_and_live_direct_fallback(selection, monkeypatch):
    selected = selected_compute_policy()
    ctx = SimpleNamespace()
    kwargs = {}
    retained = fiberphotometry.JCFG
    if selection == "context":
        ctx.cfg = selected
    elif selection == "explicit":
        ctx.cfg = JonathanConfig()
        kwargs["cfg"] = selected
    else:
        # Unlike other stages, this function historically read config.JCFG at
        # invocation. Keep that behavior for direct callers without ctx.cfg.
        monkeypatch.setattr(lib_config, "JCFG", selected)
    record = compute_fixture()
    fiberphotometry.compute_dff_zscore_session(ctx, record, freqs=("470",), **kwargs)
    assert_selected_compute(record)
    assert fiberphotometry.JCFG is retained


@pytest.mark.parametrize("has_context_policy", [False, True])
@pytest.mark.parametrize("joblib_available", [False, True])
def test_process_wrapper_explicitly_forwards_context_or_live_fallback(has_context_policy, joblib_available, monkeypatch):
    selected, other = selected_compute_policy(), JonathanConfig()
    monkeypatch.setattr(lib_config, "JCFG", other if has_context_policy else selected)
    ctx = SimpleNamespace(cfg=selected) if has_context_policy else SimpleNamespace()
    calls = []

    def compute(ctx_value, session, *, freqs, iso, cfg):
        assert ctx_value is ctx and cfg is selected
        calls.append((session, freqs, iso))

    class InlineParallel:
        def __init__(self, *, n_jobs):
            assert n_jobs == -1

        def __call__(self, tasks):
            return [function(*args, **kwargs) for function, args, kwargs in tasks]

    controlled_joblib = ModuleType("joblib")
    controlled_joblib.Parallel = InlineParallel
    controlled_joblib.delayed = lambda function: lambda *args, **kwargs: (function, args, kwargs)
    monkeypatch.setitem(sys.modules, "joblib", controlled_joblib if joblib_available else None)
    monkeypatch.setattr(fiberphotometry, "compute_dff_zscore_session", compute)
    first, second = object(), object()
    assert fiberphotometry.compute_dff_zscore(ctx, [first, second], freqs=("555",), iso="405") == [first, second]
    assert calls == [(first, ("555",), "405"), (second, ("555",), "405")]


def test_actual_two_process_joblib_path_transports_policy_after_fresh_imports(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="2",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    script = f'''
import sys
sys.path[:0] = {[str(ROOT / 'src'), str(ROOT)]!r}
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','sklearn'}}:
            raise AssertionError('unexpected backend: '+fullname)
sys.meta_path.insert(0, Block())
import json
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
from fpcode.contracts.session_record import SessionRecord
from studies.session_building.jonathan.config import JonathanConfig
from studies.session_building.jonathan.steps import fiberphotometry as stage
from studies.oft_baseline.config import JCFG as baseline_policy
real_parallel = joblib.Parallel
configured = []
def guard_worker():
    import importlib.abc
    import sys
    banned={{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','sklearn'}}
    assert not set(sys.modules)&banned
    class WorkerBlock(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in banned:
                raise AssertionError('unexpected worker backend: '+fullname)
    sys.meta_path.insert(0,WorkerBlock())
def bounded_parallel(*, n_jobs):
    assert n_jobs == -1
    configured.append(2)
    return real_parallel(n_jobs=2, backend='loky', inner_max_num_threads=1,
                         initializer=guard_worker)
joblib.Parallel = bounded_parallel
cfg = JonathanConfig(phot_system_ts_col='OwnedTS', include_ransac=False,
    include_wallace=False, include_manual=True, include_zscore=False,
    include_dff=True, include_prominence=False)
records=[]
for index in range(2):
    times=np.arange(32,dtype=float)
    target=pd.DataFrame({{'OwnedTS':times*0.5,'SystemTimestamp':times*0.1,
                         'signal_0':20.0+times*0.25}})
    reference=target.copy(deep=True)
    reference['signal_0']=10.0
    records.append(SessionRecord(session_id=str(index),project='synthetic',
                   photometry={{'415':reference,'470':target}}))
prior=stage.JCFG
result=stage.compute_dff_zscore(SimpleNamespace(cfg=cfg),records,freqs=('470',))
assert configured==[2] and stage.JCFG is prior
assert len(result)==2 and all(one is not two for one,two in zip(records,result))
for original,computed in zip(records,result):
    assert list(original.photometry['470'].columns)==['OwnedTS','SystemTimestamp','signal_0']
    target=computed.photometry['470']
    assert list(target.columns)==['OwnedTS','SystemTimestamp','signal_0','signal_0_zscore_manual','signal_0_dff']
    np.testing.assert_allclose(np.asarray(target['signal_0_dff'],dtype=float),1+np.arange(32)*0.025,rtol=1e-13,atol=1e-13)
    raw=pd.Series(20+np.arange(32)*0.25)
    floor=raw.rolling(window=10,center=True,min_periods=1).quantile(0.1)
    clean=(raw-floor)/floor
    rolling=clean.rolling(window=40,center=True,min_periods=1)
    expected=(clean-rolling.mean())/rolling.std()
    np.testing.assert_allclose(np.asarray(target['signal_0_zscore_manual'],dtype=float),expected.to_numpy(),rtol=1e-13,atol=1e-13,equal_nan=True)
    assert computed.metadata['reference_interpolation'] is True
assert not set(sys.modules)&{{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','sklearn'}}
def inspect_named_policy(selected):
    import os
    import sys
    from studies.session_building.jonathan.config import JonathanConfig as CanonicalConfig
    assert any(type(finder).__name__=='WorkerBlock' for finder in sys.meta_path)
    assert not set(sys.modules)&{{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','sklearn'}}
    assert type(selected) is CanonicalConfig
    return dict(chamber=selected.token_to_chamber('Arena2_LeftTopCamVid',None),
                regex=selected.video_filename_regex,indices=selected.chamber_to_index,
                type=type(selected).__module__,pid=os.getpid(),parent_pid=os.getppid(),
                threads={{name:os.environ.get(name) for name in
                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}})
named=real_parallel(n_jobs=2,backend='loky',inner_max_num_threads=1,initializer=guard_worker)(
    joblib.delayed(inspect_named_policy)(baseline_policy) for _ in range(2))
for observed in named:
    assert observed['chamber']=='bottom_left' and observed['regex']==baseline_policy.video_filename_regex
    assert observed['indices']=={{'top_left':0,'bottom_left':1,'bottom_right':2,'top_right':3}}
    assert observed['type']=='studies.session_building.jonathan.config'
    assert observed['pid']!=os.getpid() and observed['parent_pid']==os.getpid()
print(json.dumps({{'records':len(result),'workers':2,'policy':'OwnedTS/manual+dff',
                  'script_pid':os.getpid(),'worker_evidence':named}}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert (observed["records"], observed["workers"], observed["policy"]) == (2, 2, "OwnedTS/manual+dff")
    assert 1 <= len({worker["pid"] for worker in observed["worker_evidence"]}) <= 2
    assert observed["script_pid"] != os.getpid()
    for worker in observed["worker_evidence"]:
        assert worker["pid"] != observed["script_pid"]
        assert worker["parent_pid"] == observed["script_pid"]
        assert set(worker["threads"].values()) <= {"1", "2"}
    print(json.dumps({"bounded_worker_evidence": observed["worker_evidence"]}))
    assert not list(tmp_path.iterdir())
