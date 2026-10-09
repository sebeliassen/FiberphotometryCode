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
