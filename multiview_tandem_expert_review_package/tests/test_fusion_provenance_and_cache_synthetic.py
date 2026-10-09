#!/usr/bin/env python3
"""
tests/test_fusion_provenance_and_cache_synthetic.py
=============================================================================
Synthetic, CPU-only tests for core.fusion_engine.fuse_single_pair:

  * keypoint schema/order of the fused 23-kp output
  * strict calibration rejects non-crystal mappings and writes nothing
  * permissive mode is explicit and accurately labeled (never 'continuous_calibrated')
  * fused H5 is published atomically with a manifest; failed writes leave the
    previous file intact and no temp files behind
  * manifest-verified cache: OK -> CACHED; changed config -> rebuilt;
    legacy unmanifested caches are UNVERIFIED (kept; 'trust' or 'rebuild' policy)
=============================================================================
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parent.parent
for p in [PKG_ROOT, PKG_ROOT / "syllable_generation/multiview_top_bot_experiments", Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import camera_correspondence as cc
from core import cache_manifest as cm
from core.fusion_engine import fuse_single_pair
from core.schemas import TOP_USE_BODYPARTS, BOT_RAW_BODYPARTS, BOT_RENAME_MAP, TANDEM_BODYPARTS

SID = "synthetic_session_0001"
N = 300

VALID_CALIB = {
    "status": "VALID_CRYSTAL", "mode": "rate_aware_basler_crystal", "is_valid_crystal": True,
    "strict_validated": True, "slope": 1.000012, "intercept": 1.0, "ppm_drift": 12.0,
    "n_anchors_retained": 20, "n_anchors_rejected": 0, "anchor_max_res_ms": 0.1, "anchor_mean_res_ms": 0.05,
}
DEGRADED_CALIB = dict(VALID_CALIB, status="DEGRADED_INSUFFICIENT_ANCHORS",
                      mode="insufficient_anchors_fallback", is_valid_crystal=False,
                      slope=1.0, ppm_drift=0.0, n_anchors_retained=2)


def _write_dlc(path: Path, bodyparts, n=N, seed=0, nose=(100.0, 100.0), tail=(100.0, 150.0)):
    rng = np.random.default_rng(seed)
    cols = {}
    for bp in bodyparts:
        base = {"nose": nose, "nose_tip": nose, "tail_base": tail}.get(bp, (nose[0] + 5, nose[1] + 25))
        cols[("sc", bp, "x")] = base[0] + rng.normal(0, 0.2, n)
        cols[("sc", bp, "y")] = base[1] + rng.normal(0, 0.2, n)
        cols[("sc", bp, "likelihood")] = np.full(n, 0.95)
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["scorer", "bodyparts", "coords"])
    df.to_hdf(str(path), key="df_with_missing", mode="w", format="fixed")


class _FusionBase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)
        self.top = self.d / "top.h5"
        self.bot = self.d / "bot.h5"
        _write_dlc(self.top, TOP_USE_BODYPARTS, seed=1)
        _write_dlc(self.bot, BOT_RAW_BODYPARTS, seed=2, nose=(500.0, 300.0), tail=(500.0, 435.0))
        self.out = self.d / "fused"
        self.pair = {"sid": SID, "top_h5": str(self.top), "bot_h5": str(self.bot)}
        self._saved = (dict(cc._MAPPING_CACHE), dict(cc._CALIBRATION_DETAILS_CACHE))
        self.set_calibration(VALID_CALIB)

    def tearDown(self):
        cc._MAPPING_CACHE.clear(); cc._MAPPING_CACHE.update(self._saved[0])
        cc._CALIBRATION_DETAILS_CACHE.clear(); cc._CALIBRATION_DETAILS_CACHE.update(self._saved[1])
        self._td.cleanup()

    def set_calibration(self, calib):
        cc._MAPPING_CACHE[SID] = np.arange(N)
        cc._CALIBRATION_DETAILS_CACHE[SID] = dict(calib)

    def fuse(self, **kw):
        return fuse_single_pair(self.pair, out_dir=self.out, **kw)

    @property
    def out_h5(self):
        return self.out / f"{SID}_tandem_23kp.h5"


class TestFusionSchemaAndProvenance(_FusionBase):
    def test_valid_session_fuses_with_calibrated_label_and_schema(self):
        res = self.fuse()
        self.assertEqual(res["status"], "FUSED", res)
        self.assertEqual(res["sync_mode"], "continuous_calibrated")
        self.assertEqual(res["calibration_status"], "VALID_CRYSTAL")
        df = pd.read_hdf(self.out_h5)
        self.assertEqual(len(df), N)
        order = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
        self.assertEqual(order, list(TOP_USE_BODYPARTS) + list(BOT_RENAME_MAP.values()))
        self.assertEqual(order, TANDEM_BODYPARTS)
        self.assertEqual(len(order), 23)
        coords = list(dict.fromkeys(df.columns.get_level_values("coords")))
        self.assertEqual(coords, ["x", "y", "likelihood"])

    def test_topcam_keypoints_preserved_verbatim(self):
        self.fuse()
        fused = pd.read_hdf(self.out_h5)
        top = pd.read_hdf(self.top)
        for bp in ("nose", "tail_base", "mouse_center"):
            np.testing.assert_array_equal(
                fused["fused_multiview", bp, "x"].to_numpy(),
                top["sc", bp, "x"].to_numpy(dtype=np.float32),
            )

    def test_strict_default_rejects_degraded_calibration_and_writes_nothing(self):
        self.set_calibration(DEGRADED_CALIB)
        res = self.fuse()  # require_continuous_sync=True => strict by default
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("not valid crystal calibration", res["error"])
        self.assertFalse(self.out_h5.exists())
        self.assertEqual(list(self.out.glob("*")), [])

    def test_permissive_mode_is_explicit_and_accurately_labeled(self):
        self.set_calibration(DEGRADED_CALIB)
        res = self.fuse(strict_calibration=False)
        self.assertEqual(res["status"], "FUSED", res)
        self.assertNotEqual(res["sync_mode"], "continuous_calibrated")
        self.assertEqual(res["sync_mode"], "continuous_uncalibrated_degraded_insufficient_anchors")
        m = cm.read_manifest(self.out_h5)
        self.assertEqual(m["extra"]["sync_mode"], res["sync_mode"])
        self.assertFalse(m["extra"]["calibration"]["is_valid_crystal"])

    def test_strict_flag_is_part_of_cache_identity(self):
        self.set_calibration(VALID_CALIB)
        self.assertEqual(self.fuse(strict_calibration=True)["status"], "FUSED")
        res = self.fuse(strict_calibration=False)
        self.assertEqual(res["status"], "FUSED", "different strictness must not reuse the cache")


class TestFusionCacheAndAtomicity(_FusionBase):
    def test_manifest_written_and_second_call_is_verified_cache_hit(self):
        res1 = self.fuse()
        self.assertEqual(res1["status"], "FUSED")
        self.assertTrue(cm.manifest_path_for(self.out_h5).exists())
        self.assertEqual(list(self.out.glob(".*.tmp")), [])
        m = cm.read_manifest(self.out_h5)
        self.assertEqual(m["kind"], "fused_h5")
        self.assertEqual(m["config"]["top_keypoint_order"], list(TOP_USE_BODYPARTS))
        self.assertTrue(m["code_hash"])
        status, _ = cm.verify_manifest(self.out_h5, kind="fused_h5")
        self.assertEqual(status, cm.OK)

    def _grow_and_republish(self):
        """Pad the published H5 past 1 MB and re-publish a matching manifest (simulates a real session file)."""
        with open(self.out_h5, "ab") as f:
            f.write(b"\0" * 1_100_000)
        cfg = {
            "arena_width": 972.0, "require_continuous_sync": True, "strict_calibration": True,
            "confidence_gating": True, "truncate_aborted_tail": True,
            "top_keypoint_order": list(TOP_USE_BODYPARTS), "bot_rename_map": dict(BOT_RENAME_MAP),
        }
        from core import fusion_engine as fe
        cm.write_manifest(self.out_h5, kind="fused_h5", inputs=[self.top, self.bot], config=cfg,
                          code_files=fe._FUSION_CODE_FILES)

    def test_large_verified_file_is_cached_and_not_rewritten(self):
        self.fuse()
        self._grow_and_republish()
        before = self.out_h5.stat().st_mtime_ns
        res = self.fuse()
        self.assertEqual(res["status"], "CACHED", res)
        self.assertEqual(res["cache_verification"], cm.OK)
        self.assertEqual(self.out_h5.stat().st_mtime_ns, before)

    def test_changed_config_rebuilds_instead_of_reusing(self):
        self.fuse()
        self._grow_and_republish()
        res = self.fuse(confidence_gating=False)
        self.assertEqual(res["status"], "FUSED", res)

    def test_modified_input_rebuilds(self):
        self.fuse()
        self._grow_and_republish()
        _write_dlc(self.top, TOP_USE_BODYPARTS, seed=99)  # TopCam tracking changed
        res = self.fuse()
        self.assertEqual(res["status"], "FUSED", res)

    def test_legacy_unmanifested_cache_trust_policy_flags_unverified_and_keeps_file(self):
        self.out.mkdir(parents=True)
        self.out_h5.write_bytes(b"LEGACY" * 300_000)  # 1.8 MB, no manifest
        before = self.out_h5.read_bytes()
        res = self.fuse()  # default policy: trust (backward compatible)
        self.assertEqual(res["status"], "CACHED")
        self.assertEqual(res["cache_verification"], cm.UNVERIFIED_NO_MANIFEST)
        self.assertEqual(self.out_h5.read_bytes(), before)
        self.assertFalse(cm.manifest_path_for(self.out_h5).exists())

    def test_legacy_unmanifested_cache_rebuild_policy_replaces_atomically(self):
        self.out.mkdir(parents=True)
        self.out_h5.write_bytes(b"LEGACY" * 300_000)
        res = self.fuse(unmanifested_policy="rebuild")
        self.assertEqual(res["status"], "FUSED", res)
        self.assertTrue(cm.manifest_path_for(self.out_h5).exists())
        self.assertEqual(list(self.out.glob(".*.tmp")), [])
        self.assertEqual(len(pd.read_hdf(self.out_h5)), N)

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            self.fuse(unmanifested_policy="delete")

    def test_failed_write_keeps_previous_file_and_leaves_no_temp(self):
        self.out.mkdir(parents=True)
        self.out_h5.write_bytes(b"PREVIOUS" * 200_000)
        previous = self.out_h5.read_bytes()
        with mock.patch.object(pd.DataFrame, "to_hdf", side_effect=OSError("disk full")):
            res = self.fuse(overwrite=True)
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(self.out_h5.read_bytes(), previous, "failed write must not corrupt the published file")
        self.assertEqual(list(self.out.glob(".*.tmp")), [])

    def test_partial_temp_never_matches_artifact_glob(self):
        seen = []
        real = pd.DataFrame.to_hdf

        def spy(self_, path, *a, **k):
            seen.append(Path(path).name)
            real(self_, path, *a, **k)
            seen.append(sorted(p.name for p in Path(path).parent.glob("*.h5")))

        with mock.patch.object(pd.DataFrame, "to_hdf", spy):
            self.fuse()
        self.assertTrue(seen[0].startswith(".") and seen[0].endswith(".tmp"))
        self.assertEqual(seen[1], [], "temp file must not be visible to *.h5 globbing mid-write")


if __name__ == "__main__":
    unittest.main(verbosity=2)
