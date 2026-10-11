"""Small clock fixtures prove read-only lookup and isolated source selection."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpcode.tracking import clock_sync
from fpcode.tracking.camera_correspondence import (
    CameraMappingContext, get_session_mapping, get_session_calibration_details,
    top_to_bot_frame_continuous, mapping_fingerprint,
)


def timestamps(tmp_path, name, shift=0.):
    path = tmp_path / name
    pd.DataFrame({"Cam_Bonsai_Timestamp": np.arange(30) * .02 + shift}).to_csv(path, index=False)
    return path


def forbid_discovery(*args, **kwargs):
    raise AssertionError("A normal clock read attempted cohort discovery")


def test_missing_explicit_cache_does_not_build_or_write(tmp_path, monkeypatch):
    monkeypatch.setattr(clock_sync, "build_and_save_ttl_cache", forbid_discovery)
    monkeypatch.setattr(clock_sync, "legacy_timestamp_root", forbid_discovery)
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        clock_sync.load_ttl_cache(path)
    assert list(tmp_path.iterdir()) == []


def test_explicit_cache_ignores_legacy_global_and_keeps_offset_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(clock_sync, "_OFFSET_CACHE", {"same": {"bot0_in_top": 100}})
    path = tmp_path / "offsets.json"
    path.write_text(json.dumps({"same": {"bot0_in_top": 3, "top0_in_bot": 1}}))
    assert clock_sync.get_session_ttl_offset("same", cache_path=path)["bot0_in_top"] == 3
    assert clock_sync.top_to_bot_frame("same", 1, cache_path=path) == 0  # old clamp
    assert clock_sync.top_to_bot_frame("same", 9, cache_path=path) == 7
    assert clock_sync.bot_to_top_frame("same", 9, cache_path=path) == 11
    assert clock_sync.get_session_ttl_offset("absent")["bot0_in_top"] == 0
    with pytest.raises(KeyError, match="absent.*supplied TTL cache"):
        clock_sync.get_session_ttl_offset("absent", cache={})


def test_contexts_with_same_session_id_do_not_share_mapping_or_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(clock_sync, "legacy_timestamp_root", forbid_discovery)
    top = timestamps(tmp_path, "top.csv")
    bottom = timestamps(tmp_path, "bottom.csv")
    shifted = timestamps(tmp_path, "shifted.csv", .02)
    first = CameraMappingContext(ttl_cache={"same": {"top_csv": top, "bot_csv": bottom}})
    second = CameraMappingContext(ttl_cache={"same": {"top_csv": top, "bot_csv": shifted}})
    np.testing.assert_array_equal(get_session_mapping("same", context=first), np.arange(30))
    assert top_to_bot_frame_continuous("same_seg2_tandem_23kp", 1, context=second) == 0
    assert top_to_bot_frame_continuous("same", 0, context=second) == -1
    assert top_to_bot_frame_continuous("same", 30, context=first) == -1
    details = get_session_calibration_details("same", context=first)
    assert details["status"] == "FALLBACK_HOST_TIMESTAMPS"
    details["is_valid_crystal"] = True
    with pytest.raises(RuntimeError, match="not valid crystal"):
        get_session_mapping("same", strict=True, context=first)
    with pytest.raises(ValueError):
        get_session_mapping("same", context=first)[0] = 7
    timestamps(tmp_path, "bottom.csv", .04)
    assert top_to_bot_frame_continuous("same", 2, context=first) == 0  # input change invalidates


def test_explicit_builder_reads_only_supplied_pairs_and_publishes_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(clock_sync, "legacy_timestamp_root", forbid_discovery)
    top, bottom = tmp_path / "top.csv", tmp_path / "bottom.csv"
    pd.DataFrame({"FP3002_System_Timestamp": np.arange(30) * .02}).to_csv(top, index=False)
    pd.DataFrame({"FP3002_System_Timestamp": np.arange(25) * .02 + .04}).to_csv(bottom, index=False)
    target = tmp_path / "output" / "offsets.json"
    data = clock_sync.build_and_save_ttl_cache(target, timestamp_pairs={"same": {"top_csv": top, "bot_csv": bottom}})
    assert data["same"]["bot0_in_top"] == 2
    assert data["same"]["top0_in_bot"] == 0
    assert data["same"]["dt_start_sec"] == pytest.approx(.04)
    assert data["same"]["mean_res_ms"] == pytest.approx(0., abs=1e-10)
    assert clock_sync.load_ttl_cache(target) == data
    assert list(target.parent.iterdir()) == [target]


def test_importing_tracking_engines_does_not_read_cohort_resources(repo_root):
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
original_open = Path.open
def guarded_open(self, *args, **kwargs):
    assert self.name not in {'ttl_cohort_offsets.json', 'static_arena_transforms.json', 'epoch_arena_transforms.json'}
    return original_open(self, *args, **kwargs)
Path.open = guarded_open
from fpcode.tracking import static_fusion, clock_sync
assert static_fusion.STATIC_ARENA_TRANSFORMS._data is None
assert static_fusion.EPOCH_ARENA_TRANSFORMS._data is None
assert clock_sync._OFFSET_CACHE is None
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(repo_root / "src")],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("options", [
    {}, {"ttl_cache": {}, "ttl_cache_path": "anything"},
])
def test_mapping_context_requires_one_explicit_source(options):
    with pytest.raises(ValueError, match="exactly one"):
        CameraMappingContext(**options)


@pytest.mark.parametrize("options", [
    {"cache": {}, "cache_path": "anything"},
    {"offset_info": {"bot0_in_top": 0}, "cache": {}},
    {"lookup": lambda sid: {"bot0_in_top": 0}, "cache_path": "anything"},
])
def test_offset_lookup_rejects_ambiguous_sources(options):
    with pytest.raises(ValueError, match="Provide one"):
        clock_sync.get_session_ttl_offset("same", **options)


@pytest.mark.parametrize("record", [None, {}, [], {"unrelated": 1}])
def test_context_lookup_rejects_missing_or_malformed_records(record):
    context = CameraMappingContext(offset_lookup=lambda sid: record)
    with pytest.raises(ValueError, match="Malformed TTL.*same"):
        context.resolve("same")


def test_mapping_fingerprint_distinguishes_equal_bytes_with_different_layouts():
    two_frames = np.array([0, 1], dtype="<i4")
    one_frame = np.array([4294967296], dtype="<i8")
    assert two_frames.tobytes() == one_frame.tobytes()
    assert mapping_fingerprint(two_frames) != mapping_fingerprint(one_frame)
    assert mapping_fingerprint(two_frames) != mapping_fingerprint(two_frames.reshape(1, 2))
    assert mapping_fingerprint(two_frames) == mapping_fingerprint(two_frames.copy())
    np.testing.assert_array_equal(two_frames, [0, 1])
    np.testing.assert_array_equal(one_frame, [4294967296])
