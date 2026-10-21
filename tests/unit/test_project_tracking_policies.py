"""Independent cohort policy and filename-only discovery checks."""

import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from studies.karen_pff_vs_monomer.analysis import tracking as karen_tracking
from studies.karen_pff_vs_monomer.session_policy import config as karen_config
from studies.monoamine_interactions.steps import tracking as baseline_tracking
from studies.oft_baseline import config as baseline_config
from studies.session_building.jonathan import config as generic_config
from studies.session_building.jonathan.utils.files import parse_h5_path, parse_video_path

ROOT = Path(__file__).resolve().parents[2]
STAMP = "2026-03-05T11_01_01"


def test_project_policies_retain_distinct_ranges_and_independent_instances():
    assert karen_config.VALID_FRAME_RANGE == (0, 999999)
    assert baseline_config.VALID_FRAME_RANGE == (50000, 250000)
    assert karen_tracking.JCFG is karen_config.JCFG
    assert baseline_tracking.JCFG is baseline_config.JCFG
    assert karen_config.JCFG is not baseline_config.JCFG
    assert karen_config.JCFG is not generic_config.JCFG
    assert baseline_config.JCFG is not generic_config.JCFG
    assert asdict(karen_config.JCFG) == asdict(generic_config.JonathanConfig())
    generic_values = asdict(generic_config.JonathanConfig())
    baseline_values = asdict(baseline_config.JCFG)
    assert baseline_values.pop("video_filename_regex") != generic_values.pop("video_filename_regex")
    assert baseline_values == generic_values
    assert karen_config.JCFG.chamber_to_index == {
        "top_left": 0, "bottom_left": 1, "bottom_right": 2, "top_right": 3,
    }


@pytest.mark.parametrize("arena,chamber", [
    (1, "top_left"), (2, "bottom_left"), (3, "bottom_right"), (4, "top_right"),
])
def test_baseline_raw_video_arena_fallback_retains_explicit_chamber_order(arena, chamber):
    path = Path("/filename-only") / f"T1_6.2.4.e_Arena{arena}_LeftTopCamVid_{STAMP}.avi"
    result = parse_video_path(path, baseline_config.JCFG)
    assert result == {
        "arena": f"Arena{arena}", "camset": f"Arena{arena}_LeftTopCamVid",
        "ts": STAMP, "token": None, "ext": "avi", "chamber": chamber,
        "cam_slot": f"Arena{arena}_LeftTopCamVid", "session_id": f"{STAMP}_{chamber}",
        "path": path,
    }
    assert parse_video_path(path, karen_config.JCFG) is None


@pytest.mark.parametrize("camset,token,chamber,slot", [
    ("LeftTopCamVid", "upper", "top_left", "left_cam"),
    ("LeftTopCamVid", "lower", "bottom_left", "left_cam"),
    ("RightTopCamVid", "upper", "top_right", "right_cam"),
    ("RightTopCamVid", "lower", "bottom_right", "right_cam"),
])
def test_native_split_names_keep_explicit_tokens_ahead_of_arena_fallback(camset, token, chamber, slot):
    prefix = f"T1_6.2.4.e_Arena1and2_{camset}_{STAMP}_{token}"
    video = Path("/filename-only") / (prefix + ".avi")
    track = Path("/filename-only") / (prefix + "DLC_fixture.h5")
    for policy in (karen_config.JCFG, baseline_config.JCFG):
        parsed = parse_h5_path(track, policy)
        assert parsed["chamber"] == chamber and parsed["cam_slot"] == slot
        assert parsed["ts"] == STAMP and parsed["token"] == token
        parsed = parse_video_path(video, policy)
        assert parsed["chamber"] == chamber and parsed["session_id"] == f"{STAMP}_{chamber}"
        # Historical baseline video regex includes Arena in camset; preserve it.
        expected_camset = f"Arena1and2_{camset}" if policy is baseline_config.JCFG else camset
        assert parsed["camset"] == expected_camset


@pytest.mark.parametrize("baseline_first", [False, True])
@pytest.mark.parametrize("cached_config", [False, True])
def test_imports_use_owned_policy_despite_foreign_config_and_stay_cold(baseline_first, cached_config, tmp_path):
    poison = tmp_path / "config.py"
    poison.write_text("raise AssertionError('foreign bare config was imported')\n")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="2",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for name in ("PYTHONPATH", "FIBERPHOTOMETRY_ROOT", "FP_DATA_ROOT"):
        environment.pop(name, None)
    script = f'''
import sys
sys.path[:0] = {[str(ROOT / 'src'), str(ROOT), str(tmp_path)]!r}
import importlib
import importlib.abc
import logging
import types
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'jax','jaxlib','keypoint_moseq','jax_moseq','torch','cuml','cv2','config'}}:
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0, Block())
poison = types.ModuleType('config')
def unexpected(name):
    raise AssertionError('foreign config accessed: '+name)
poison.__getattr__ = unexpected
if {cached_config!r}:
    sys.modules['config'] = poison
from studies.session_building.jonathan import config as generic
from studies.juvenile_intruder import config as juvenile
from dataclasses import asdict
generic_object, juvenile_object = generic.JCFG, juvenile.JCFG
generic_before, juvenile_before = asdict(generic.JCFG), asdict(juvenile.JCFG)
paths, handlers = list(sys.path), list(logging.root.handlers)
modules = ['studies.monoamine_interactions.steps.tracking',
           'studies.karen_pff_vs_monomer.analysis.tracking']
if not {baseline_first!r}:
    modules.reverse()
for module in modules:
    importlib.import_module(module)
from studies.oft_baseline import config as baseline
from studies.karen_pff_vs_monomer.session_policy import config as karen
from studies.monoamine_interactions.steps import tracking as baseline_tracking
from studies.karen_pff_vs_monomer.analysis import tracking as karen_tracking
assert baseline_tracking.JCFG is baseline.JCFG
assert karen_tracking.JCFG is karen.JCFG
assert baseline.VALID_FRAME_RANGE == (50000,250000)
assert karen.VALID_FRAME_RANGE == (0,999999)
assert generic.JCFG is generic_object and asdict(generic.JCFG)==generic_before
assert juvenile.JCFG is juvenile_object and asdict(juvenile.JCFG)==juvenile_before
assert (sys.modules.get('config') is poison) if {cached_config!r} else ('config' not in sys.modules)
assert sys.path==paths and logging.root.handlers==handlers
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == [poison]


def fixture_directories(tmp_path, names, *, with_h5):
    video_dir, h5_dir = tmp_path / "videos", tmp_path / "tracks"
    video_dir.mkdir()
    h5_dir.mkdir()
    for name in names:
        (video_dir / name).touch()
    track_names = [f"T1_6.2.4.e_Arena1and2_LeftTopCamVid_{STAMP}_upperDLC_{tail}.h5"
                   for tail in ("z", "a")]
    track_names += [f"T1_6.2.4.e_Arena3and4_RightTopCamVid_{STAMP}_lowerDLC_fixture.h5",
                    "invalid.h5"]
    if with_h5:
        for name in track_names:
            (h5_dir / name).touch()
    return video_dir, h5_dir, track_names


@pytest.mark.parametrize("with_h5", [False, True])
def test_karen_discovery_groups_camera_views_once_and_excludes_empty_fourth_id(tmp_path, with_h5):
    names = [f"T1_6.2.4.e_Arena1and2_LeftTopCamVid_{STAMP}_{token}.avi"
             for token in ("upper", "lower")]
    names += [f"T1_6.2.4.e_Arena3and4_RightTopCamVid_{STAMP}_{token}.avi"
              for token in ("upper", "lower")]
    names.append("unparsable.mp4")
    video_dir, h5_dir, tracks = fixture_directories(tmp_path, names, with_h5=with_h5)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file())
    sessions = karen_tracking.discover_sessions_hybrid(
        SimpleNamespace(video_dir=video_dir, h5_dir=h5_dir, project="synthetic-karen"), [])
    assert [s.session_id for s in sessions] == [
        f"{STAMP}_bottom_left", f"{STAMP}_bottom_right", f"{STAMP}_top_left",
    ]
    for session, chamber, index, video_name, expected_tracks in [
        (sessions[0], "bottom_left", 1, names[1], []),
        (sessions[1], "bottom_right", 2, names[3], [tracks[2]]),
        (sessions[2], "top_left", 0, names[0], sorted(tracks[:2])),
    ]:
        assert session.project == "synthetic-karen" and session.session_type == "openfield"
        assert session.metadata == dict(
            timestamp=STAMP, chamber=chamber, chamber_index=index,
            camsets_present=["LeftTopCamVid", "RightTopCamVid"],
            cam_slots_present=["left_cam", "right_cam"], arenas_present=["Arena1and2", "Arena3and4"],
            original_video_path=str(video_dir / video_name),
        )
        rows = session.tracking_refs_df.to_dict("records")
        assert [Path(row["path"]).name for row in rows] == (expected_tracks if with_h5 else [])
        for row in rows:
            assert row["chamber"] == chamber and row["chamber_index"] == index
            assert row["ts"] == STAMP and row["kind"] == "dlc_h5"
            assert row["path_type"] == "tracking"
            assert row["camera_key"] == f"{row['cam_slot']}:{chamber}"
            assert row["role"] == ("upper" if index == 0 else "lower")
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()) == before
    assert all(p.stat().st_size == 0 for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("with_h5", [False, True])
def test_baseline_discovery_uses_optional_token_and_skips_named_empty_chambers(tmp_path, with_h5):
    names = [f"T1_6.2.e.e_Arena{arena}_LeftTopCamVid_{STAMP}.avi" for arena in (1, 2, 3, 4)]
    video_dir, h5_dir, tracks = fixture_directories(tmp_path, names, with_h5=with_h5)
    sessions = baseline_tracking.discover_sessions_hybrid(
        SimpleNamespace(video_dir=video_dir, h5_dir=h5_dir, project="synthetic-baseline"), [])
    assert [s.session_id for s in sessions] == [f"{STAMP}_bottom_left", f"{STAMP}_top_left"]
    for session, chamber, index, arena in [
        (sessions[0], "bottom_left", 1, 2), (sessions[1], "top_left", 0, 1),
    ]:
        assert session.project == "synthetic-baseline" and session.session_type == "openfield"
        assert session.metadata == dict(
            timestamp=STAMP, chamber=chamber, chamber_index=index,
            camsets_present=[f"Arena{arena}_LeftTopCamVid"],
            cam_slots_present=[f"Arena{arena}_LeftTopCamVid"], arenas_present=[f"Arena{arena}"],
            original_video_path=str(video_dir / names[arena - 1]),
        )
        rows = session.tracking_refs_df.to_dict("records")
        expected = sorted(tracks[:2]) if with_h5 and index == 0 else []
        assert [Path(row["path"]).name for row in rows] == expected
        for row in rows:
            assert row["chamber"] == "top_left" and row["chamber_index"] == 0
            assert row["ts"] == STAMP and row["role"] == "upper"
    assert sorted(p.name for p in video_dir.iterdir()) == sorted(names)
    assert sorted(p.name for p in h5_dir.iterdir()) == (sorted(tracks) if with_h5 else [])
    assert all(p.stat().st_size == 0 for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("tracking", [karen_tracking, baseline_tracking])
def test_no_parsable_videos_produce_no_sessions(tracking, tmp_path):
    (tmp_path / "invalid.mp4").touch()
    assert tracking.discover_sessions_hybrid(
        SimpleNamespace(video_dir=tmp_path, h5_dir=None, project="synthetic"), [object()]) == []
