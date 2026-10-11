"""Actual tiny HDF fusion using supplied clocks and affine calibration."""

import json

import numpy as np
import pandas as pd
import pytest

from fpcode.common.cache_manifest import read_manifest
from fpcode.common.schemas import TOP_USE_BODYPARTS, BOT_RAW_BODYPARTS
from fpcode.tracking import fusion, static_fusion
from fpcode.workflows.fusion import execute_fusion


def tracking_file(path, keypoints, *, bottom=False):
    values = {}
    for name in keypoints:
        x, y = ((500., 300.) if bottom else (100., 100.))
        if name == "tail_base":
            y += 135. if bottom else 50.
        for coordinate, data in (("x", x + np.arange(3)), ("y", np.full(3, y)),
                                 ("likelihood", np.full(3, .95))):
            values[("scorer", name, coordinate)] = data
    frame = pd.DataFrame(values)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["scorer", "bodyparts", "coords"])
    frame.to_hdf(path, key="df_with_missing", mode="w")


def input_pair(tmp_path):
    top, bottom = tmp_path / "top.h5", tmp_path / "bottom.h5"
    tracking_file(top, TOP_USE_BODYPARTS)
    tracking_file(bottom, BOT_RAW_BODYPARTS, bottom=True)
    return {"sid": "same", "arena": "custom_arena", "top_h5": top, "bot_h5": bottom}


def forbid_defaults(*args, **kwargs):
    raise AssertionError("Explicit fusion attempted historical resource lookup")


@pytest.mark.parametrize("static", [False, True])
def test_supplied_mapping_bypasses_cohort_lookup_and_preserves_missing_frames(tmp_path, monkeypatch, static):
    pair = input_pair(tmp_path)
    sources = pair["top_h5"].read_bytes(), pair["bot_h5"].read_bytes()
    implementation = static_fusion if static else fusion
    monkeypatch.setattr(implementation, "get_session_mapping", forbid_defaults)
    monkeypatch.setattr(implementation, "get_session_ttl_offset", forbid_defaults)
    if static:
        monkeypatch.setattr(static_fusion.STATIC_ARENA_TRANSFORMS, "_load", forbid_defaults)
        monkeypatch.setattr(static_fusion.EPOCH_ARENA_TRANSFORMS, "_load", forbid_defaults)
    pair.update(camera_mapping=np.array([-1, 2, 0]),
                calibration_details={"status": "VALID_CRYSTAL", "is_valid_crystal": True})
    if static:
        pair["transform_info"] = {"matrix_2x2": [[1., 0.], [0., 1.]], "translation_2d": [0., 0.]}
    result = execute_fusion([pair], {"mode": "static_21kp" if static else "tandem_23kp"}, tmp_path / "out")
    assert result["status"] == "SUCCESS", result
    output = tmp_path / "out" / ("same_tandem_21kp.h5" if static else "same_tandem_23kp.h5")
    frame = pd.read_hdf(output)
    np.testing.assert_array_equal(frame[("fused_multiview", "nose", "x")], [100., 101., 102.])
    assert np.isnan(frame[("fused_multiview", "left_front_paw", "x")].iloc[0])
    assert frame[("fused_multiview", "left_front_paw", "likelihood")].iloc[0] == 0.
    if static:
        np.testing.assert_array_equal(frame[("fused_multiview", "left_front_paw", "x")].iloc[1:], [470., 472.])
    assert read_manifest(output)["config"]["camera_mapping_sha256"]
    assert (pair["top_h5"].read_bytes(), pair["bot_h5"].read_bytes()) == sources


def test_custom_epoch_transform_and_timestamp_files_are_manifest_dependencies(tmp_path, monkeypatch):
    pair = input_pair(tmp_path)
    transform = tmp_path / "affine.json"
    transform.write_text(json.dumps({"custom_phase": {"custom_arena": {
        "matrix_2x2": [[1., 0.], [0., 1.]], "translation_2d": [1., 2.],
    }}}))
    top, bottom = tmp_path / "top.csv", tmp_path / "bottom.csv"
    basler = np.arange(60) * 16_666_667.
    host = basler / 1e9
    pd.DataFrame({"Cam_Basler_Timestamp": basler, "Cam_Bonsai_Timestamp": host}).to_csv(top, index=False)
    pd.DataFrame({"Cam_Basler_Timestamp": basler + 42e6, "Cam_Bonsai_Timestamp": host}).to_csv(bottom, index=False)
    cache = tmp_path / "ttl.json"
    cache.write_text(json.dumps({"same": {"top_csv": str(top), "bot_csv": str(bottom)}}))
    monkeypatch.setattr(static_fusion.STATIC_ARENA_TRANSFORMS, "_load", forbid_defaults)
    monkeypatch.setattr(static_fusion.EPOCH_ARENA_TRANSFORMS, "_load", forbid_defaults)
    config = {"mode": "static_21kp", "transform_path": str(transform), "epoch": "custom_phase",
              "use_epochs": True, "ttl_cache_path": str(cache)}
    result = execute_fusion([pair], config, tmp_path / "out")
    assert result["status"] == "SUCCESS", result
    output = tmp_path / "out" / "same_tandem_21kp.h5"
    manifest = read_manifest(output)
    paths = {item["path"] for item in manifest["inputs"]}
    assert paths == {str(path.resolve()) for path in (pair["top_h5"], pair["bot_h5"], transform, top, bottom)}
    assert manifest["config"]["epoch"] == "custom_phase"
    frame = pd.read_hdf(output)
    np.testing.assert_array_equal(frame[("fused_multiview", "left_front_paw", "x")], [473., 472., 471.])


def test_supplied_degraded_mapping_cannot_claim_strict_calibration(tmp_path):
    pair = input_pair(tmp_path)
    result = static_fusion.fuse_single_pair_static(pair, tmp_path / "out",
        camera_mapping=np.arange(3), calibration_details={"status": "FALLBACK_HOST_TIMESTAMPS"},
        transform_info={"matrix_2x2": np.eye(2), "translation_2d": [0., 0.]})
    assert result["status"] == "ERROR"
    assert "not valid crystal" in result["error"]
    assert list((tmp_path / "out").iterdir()) == []


def test_generic_workflow_rejects_missing_clock_context_before_output(tmp_path):
    pair = input_pair(tmp_path)
    result = execute_fusion([pair], {"mode": "tandem_23kp"}, tmp_path / "out")
    assert result["status"] == "FAILED"
    assert "explicit clock inputs" in result["errors"][0]["error"]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("timestamp, arena, expected_epoch, expected_scale", [
    ("2026-03-31T12_00_00", "Arena1", "epoch1_winter_spring_2026", 2.),
    ("2026-04-01T12_00_00", "Arena1", "epoch2_spring_post_shift_2026", 3.),
    ("2026-04-01T12_00_00", "Arena9", "fallback_static", 1.),
])
def test_explicit_legacy_profile_keeps_epoch_boundary_and_arena_fallback(
    tmp_path, monkeypatch, timestamp, arena, expected_epoch, expected_scale,
):
    from fpcode.adapters import multiview_compatibility as compatibility
    def transform(scale):
        return {"matrix_2x2": [[scale, 0.], [0., scale]], "translation_2d": [0., 0.]}
    monkeypatch.setattr(compatibility, "_STATIC_TRANSFORMS", {"Arena1": transform(1.)})
    monkeypatch.setattr(compatibility, "_EPOCH_TRANSFORMS", {
        "epoch1_winter_spring_2026": {"Arena1": transform(2.)},
        "epoch2_spring_post_shift_2026": {"Arena1": transform(3.)},
    })
    received = []
    def backend(pair, **options):
        received.append(options)
        return {"status": "FUSED"}
    result = execute_fusion([{"sid": timestamp + "_top_left", "ts": timestamp, "arena": arena}],
        {"mode": "static_21kp", "use_epochs": True, "compatibility_profile": "legacy_multiview"},
        tmp_path, static_backend=backend)
    assert result["status"] == "SUCCESS"
    assert received[0]["epoch"] == expected_epoch
    assert received[0]["transform_info"]["matrix_2x2"][0][0] == expected_scale
    assert received[0]["ttl_cache_path"] == str(compatibility.TTL_CACHE_PATH)
    expected_path = compatibility.STATIC_TRANSFORMS_PATH if expected_epoch == "fallback_static" else compatibility.EPOCH_TRANSFORMS_PATH
    assert received[0]["transform_source_path"] == expected_path


def test_cli_requires_explicit_input_directories_before_discovery(tmp_path, monkeypatch):
    from fpcode.cli import fuse
    config = tmp_path / "fusion.yaml"
    config.write_text("mode: tandem_23kp\n")
    monkeypatch.setattr(fuse, "find_matched_pairs", forbid_defaults)
    assert fuse.main(["--config", str(config), "--no-notify"]) == 1
    assert list(tmp_path.iterdir()) == [config]


def test_static_workflow_missing_explicit_ttl_session_publishes_nothing(tmp_path):
    pair = input_pair(tmp_path)
    pair.update(ttl_cache={}, transform_info={"matrix_2x2": np.eye(2), "translation_2d": [0., 0.]})
    result = execute_fusion([pair], {"mode": "static_21kp"}, tmp_path / "out")
    assert result["status"] == "FAILED"
    assert "absent from the supplied TTL cache" in result["errors"][0]["error"]
    assert list((tmp_path / "out").iterdir()) == []
