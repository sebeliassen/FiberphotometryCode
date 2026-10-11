"""Tiny real HDF5 static-fusion fixtures; no recordings or model backends."""

import numpy as np
import pandas as pd

from fpcode.common.cache_manifest import UNVERIFIED_NO_MANIFEST, manifest_path_for
from fpcode.common.schemas import TOP_USE_BODYPARTS, BOT_RAW_BODYPARTS
from fpcode.tracking import static_fusion


def write_pose(path, keypoints):
    columns = pd.MultiIndex.from_product([["scorer"], keypoints, ["x", "y", "likelihood"]],
                                         names=["scorer", "bodyparts", "coords"])
    values = np.tile([100., 50., .95], (3, len(keypoints)))
    pd.DataFrame(values, columns=columns).to_hdf(path, key="df_with_missing", mode="w")


def test_static_unmanifested_cache_trust_or_rebuild(tmp_path, monkeypatch):
    top, bottom = tmp_path / "top.h5", tmp_path / "bottom.h5"
    write_pose(top, TOP_USE_BODYPARTS)
    write_pose(bottom, BOT_RAW_BODYPARTS)
    pair = {"sid": "tiny", "arena": "Arena1", "top_h5": str(top), "bot_h5": str(bottom)}
    output = tmp_path / "tiny_tandem_21kp.h5"
    # Existing cache-size threshold is crossed with a sparse fixture, not a recording.
    with output.open("wb") as handle:
        handle.write(b"legacy")
        handle.seek(500_001)
        handle.write(b"x")
    previous = output.read_bytes()
    trusted = static_fusion.fuse_single_pair_static(pair, tmp_path, unmanifested_policy="trust")
    assert trusted["status"] == "CACHED"
    assert trusted["cache_verification"] == UNVERIFIED_NO_MANIFEST
    assert output.read_bytes() == previous
    monkeypatch.setattr(static_fusion, "get_session_mapping", lambda *args, **kwargs: np.arange(3))
    monkeypatch.setattr(static_fusion, "get_session_calibration_details", lambda *args: {"status": "VALID_CRYSTAL"})
    rebuilt = static_fusion.fuse_single_pair_static(pair, tmp_path, unmanifested_policy="rebuild")
    assert rebuilt["status"] == "FUSED", rebuilt
    assert len(pd.read_hdf(output)) == 3
    assert manifest_path_for(output).is_file()


def test_static_fusion_uses_shared_projection_without_changing_float32_or_likelihood(tmp_path, monkeypatch):
    top, bottom = tmp_path / "top.h5", tmp_path / "bottom.h5"
    write_pose(top, TOP_USE_BODYPARTS)
    write_pose(bottom, BOT_RAW_BODYPARTS)
    detections = pd.read_hdf(bottom)
    detections[("scorer", "right_front_paw", "x")] = 743.6666
    detections[("scorer", "right_front_paw", "y")] = 91.11223
    detections[("scorer", "right_front_paw", "likelihood")] = .2
    detections.to_hdf(bottom, key="df_with_missing", mode="w")
    source_bytes = (top.read_bytes(), bottom.read_bytes())
    monkeypatch.setitem(static_fusion.STATIC_ARENA_TRANSFORMS, "Arena1", {
        "matrix_2x2": [[1.234567, -.098765], [.23456, .999991]],
        "translation_2d": [21.555, -4.333],
    })
    monkeypatch.setattr(static_fusion, "get_session_mapping", lambda *args, **kwargs: np.arange(3))
    monkeypatch.setattr(static_fusion, "get_session_calibration_details", lambda *args: {"status": "VALID_CRYSTAL"})
    pair = {"sid": "tiny", "arena": "Arena1", "top_h5": str(top), "bot_h5": str(bottom)}
    result = static_fusion.fuse_single_pair_static(pair, tmp_path, keypoint_mode="23kp")
    assert result["status"] == "FUSED", result
    fused = pd.read_hdf(result["path"])
    x = fused[("fused_multiview", "left_front_paw", "x")].to_numpy()
    y = fused[("fused_multiview", "left_front_paw", "y")].to_numpy()
    confidence = fused[("fused_multiview", "left_front_paw", "likelihood")].to_numpy()
    assert x.dtype == y.dtype == confidence.dtype == np.float32
    np.testing.assert_array_equal(x, np.full(3, 294.44915771484375, dtype=np.float32))
    np.testing.assert_array_equal(y, np.full(3, 140.33627319335938, dtype=np.float32))
    np.testing.assert_array_equal(confidence, np.full(3, .2, dtype=np.float32))
    np.testing.assert_array_equal(fused[("fused_multiview", "nose", "x")], np.full(3, 100., dtype=np.float32))
    assert (top.read_bytes(), bottom.read_bytes()) == source_bytes
