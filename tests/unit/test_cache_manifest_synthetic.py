#!/usr/bin/env python3
"""
tests/test_cache_manifest_synthetic.py
=============================================================================
Synthetic, CPU-only tests for core.cache_manifest:
  * atomic publication (failed / killed writers never appear valid)
  * manifest verification: OK / STALE / UNVERIFIED / MISSING / CORRUPT
  * invalidation on input, config, code, set_id and artifact modification
  * unmanifested legacy artifacts are reported UNVERIFIED and never deleted
  * artifact-set consistency
  * huge inputs are fingerprinted by stat only (never content-hashed)
=============================================================================
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fpcode.common import cache_manifest as cm


class TestAtomicWrites(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_success_publishes_and_leaves_no_temp(self):
        target = self.dir / "out.h5"
        with cm.atomic_path(target) as tmp:
            self.assertNotEqual(Path(tmp), target)
            Path(tmp).write_bytes(b"payload")
            self.assertFalse(target.exists(), "must not be visible before publication")
        self.assertEqual(target.read_bytes(), b"payload")
        self.assertEqual(list(self.dir.glob(".*.tmp")), [])

    def test_failure_leaves_target_untouched_and_cleans_temp(self):
        target = self.dir / "out.h5"
        target.write_bytes(b"GOOD")
        with self.assertRaises(RuntimeError):
            with cm.atomic_path(target) as tmp:
                Path(tmp).write_bytes(b"PARTIAL")
                raise RuntimeError("writer crashed")
        self.assertEqual(target.read_bytes(), b"GOOD")
        self.assertEqual(list(self.dir.glob(".*.tmp")), [])

    def test_failed_first_write_never_creates_target(self):
        target = self.dir / "new.h5"
        with self.assertRaises(ValueError):
            with cm.atomic_path(target) as tmp:
                Path(tmp).write_bytes(b"x")
                raise ValueError("boom")
        self.assertFalse(target.exists())

    def test_writer_that_creates_nothing_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            with cm.atomic_path(self.dir / "nothing.h5"):
                pass

    def test_unique_temp_names(self):
        target = self.dir / "a.h5"
        names = set()
        for _ in range(5):
            with cm.atomic_path(target) as tmp:
                names.add(Path(tmp).name)
                Path(tmp).write_bytes(b"x")
        self.assertEqual(len(names), 5)

    def test_temp_does_not_match_artifact_globs(self):
        target = self.dir / "s.h5"
        with cm.atomic_path(target) as tmp:
            Path(tmp).write_bytes(b"x")
            self.assertEqual(list(self.dir.glob("*.h5")), [])

    def test_cleanup_stale_temps(self):
        old = self.dir / ".x.h5.1.deadbeef.tmp"
        old.write_bytes(b"x")
        past = time.time() - 7200
        os.utime(old, (past, past))
        fresh = self.dir / ".y.h5.2.cafebabe.tmp"
        fresh.write_bytes(b"x")
        removed = cm.cleanup_stale_temps(self.dir, older_than_s=3600)
        self.assertEqual([p.name for p in removed], [old.name])
        self.assertTrue(fresh.exists())


class TestManifestVerification(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        self.inp = self.dir / "input.csv"
        self.inp.write_text("a,b\n1,2\n")
        self.code = self.dir / "impl.py"
        self.code.write_text("X = 1\n")
        self.art = self.dir / "artifact.h5"
        self.art.write_bytes(b"A" * 100)
        self.cfg = {"gating": True, "keypoints": ["nose", "tail"], "seed": 101}
        cm.write_manifest(
            self.art, kind="fused_h5", inputs=[self.inp], config=self.cfg,
            code_files=[self.code], set_id="setA",
        )

    def tearDown(self):
        self._td.cleanup()

    def _verify(self, **overrides):
        kw = dict(kind="fused_h5", inputs=[self.inp], config=self.cfg,
                  code_files=[self.code], set_id="setA")
        kw.update(overrides)
        return cm.verify_manifest(self.art, **kw)

    def test_ok_when_nothing_changed(self):
        self.assertEqual(self._verify(), (cm.OK, []))

    def test_config_change_invalidates(self):
        status, reasons = self._verify(config={**self.cfg, "gating": False})
        self.assertEqual(status, cm.STALE)
        self.assertTrue(any("configuration" in r for r in reasons))

    def test_keypoint_order_change_invalidates(self):
        status, _ = self._verify(config={**self.cfg, "keypoints": ["tail", "nose"]})
        self.assertEqual(status, cm.STALE)

    def test_input_content_change_invalidates(self):
        self.inp.write_text("a,b\n9,9\n")
        status, reasons = self._verify()
        self.assertEqual(status, cm.STALE)
        self.assertTrue(any("input" in r for r in reasons))

    def test_input_removed_or_added_invalidates(self):
        extra = self.dir / "extra.csv"
        extra.write_text("x")
        status, reasons = self._verify(inputs=[self.inp, extra])
        self.assertEqual(status, cm.STALE)
        self.assertTrue(any("input set changed" in r for r in reasons))

    def test_code_change_invalidates(self):
        self.code.write_text("X = 2\n")
        status, reasons = self._verify()
        self.assertEqual(status, cm.STALE)
        self.assertTrue(any("implementation" in r for r in reasons))

    def test_artifact_modified_or_truncated_after_publication(self):
        self.art.write_bytes(b"A" * 40)  # truncated
        status, reasons = self._verify()
        self.assertEqual(status, cm.STALE)
        self.assertTrue(any("artifact size/mtime" in r for r in reasons))

    def test_kind_and_set_id_mismatch(self):
        self.assertEqual(self._verify(kind="results_h5")[0], cm.STALE)
        self.assertEqual(self._verify(set_id="setB")[0], cm.STALE)

    def test_missing_artifact(self):
        self.art.unlink()
        self.assertEqual(self._verify()[0], cm.MISSING_ARTIFACT)

    def test_unmanifested_legacy_artifact_is_unverified_and_kept(self):
        legacy = self.dir / "legacy.h5"
        legacy.write_bytes(b"OLD" * 50)
        status, reasons = cm.verify_manifest(legacy, kind="fused_h5", config=self.cfg)
        self.assertEqual(status, cm.UNVERIFIED_NO_MANIFEST)
        self.assertTrue(legacy.exists(), "legacy cache must never be deleted")
        self.assertEqual(legacy.read_bytes(), b"OLD" * 50)
        self.assertFalse(cm.manifest_path_for(legacy).exists())

    def test_corrupt_manifest(self):
        cm.manifest_path_for(self.art).write_text("{not json")
        self.assertEqual(self._verify()[0], cm.CORRUPT_MANIFEST)

    def test_unsupported_schema_version(self):
        mp = cm.manifest_path_for(self.art)
        m = json.loads(mp.read_text())
        m["schema_version"] = 999
        mp.write_text(json.dumps(m))
        self.assertEqual(self._verify()[0], cm.CORRUPT_MANIFEST)

    def test_manifest_requires_published_artifact(self):
        with self.assertRaises(FileNotFoundError):
            cm.write_manifest(self.dir / "ghost.h5", kind="x")

    def test_manifest_write_is_atomic_no_temp_left(self):
        self.assertEqual(list(self.dir.glob(".*.tmp")), [])


class TestArtifactSets(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        self.cfg = {"pca_dims": 10, "seed": 101}
        self.paths = {}
        for n in ("fused", "pca", "init"):
            p = self.dir / f"{n}.dat"
            p.write_bytes(n.encode() * 10)
            self.paths[n] = p

    def tearDown(self):
        self._td.cleanup()

    def _build(self, cfg_by_name=None, set_id="S1"):
        for n, p in self.paths.items():
            cfg = (cfg_by_name or {}).get(n, self.cfg)
            cm.write_manifest(p, kind=n, config=cfg, set_id=set_id)

    def test_consistent_set_ok(self):
        self._build()
        overall, per = cm.verify_artifact_set(self.paths, set_id="S1")
        self.assertEqual(overall, cm.OK)
        self.assertTrue(all(s == cm.OK for s, _ in per.values()))

    def test_mixed_config_hashes_rejected(self):
        self._build({"pca": {"pca_dims": 99, "seed": 101}})
        overall, per = cm.verify_artifact_set(self.paths, set_id="S1")
        self.assertEqual(overall, cm.STALE)
        self.assertTrue(all("mixes different config" in " ".join(r) for _, r in per.values()))

    def test_one_missing_manifest_makes_set_unverified(self):
        self._build()
        cm.manifest_path_for(self.paths["pca"]).unlink()
        overall, per = cm.verify_artifact_set(self.paths, set_id="S1")
        self.assertEqual(overall, cm.UNVERIFIED_NO_MANIFEST)
        self.assertEqual(per["pca"][0], cm.UNVERIFIED_NO_MANIFEST)

    def test_wrong_set_id_rejected(self):
        self._build(set_id="S1")
        overall, _ = cm.verify_artifact_set(self.paths, set_id="OTHER")
        self.assertEqual(overall, cm.STALE)


class TestFingerprintCost(unittest.TestCase):
    def test_huge_file_is_not_content_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            big = Path(td) / "big.bin"
            big.write_bytes(b"0")
            real_stat = Path.stat
            huge = cm.SMALL_FILE_HASH_LIMIT + 1

            class FakeStat:
                def __init__(self, st):
                    self._st = st
                    self.st_size = huge

                def __getattr__(self, k):
                    return getattr(self._st, k)

            with mock.patch.object(Path, "stat", lambda self, *a, **k: FakeStat(real_stat(self, *a, **k)) if self == big else real_stat(self, *a, **k)):
                with mock.patch.object(cm, "_sha256_file", side_effect=AssertionError("must not hash huge file")):
                    fp = cm.fingerprint_file(big)
            self.assertNotIn("sha256", fp)
            self.assertEqual(fp["size"], huge)

    def test_small_file_is_content_hashed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text("hello")
            self.assertIn("sha256", cm.fingerprint_file(p))

    def test_huge_input_change_detected_by_stat(self):
        rec = [{"path": "/x", "exists": True, "size": 10**9, "mtime_ns": 1}]
        same = [{"path": "/x", "exists": True, "size": 10**9, "mtime_ns": 1}]
        diff = [{"path": "/x", "exists": True, "size": 10**9, "mtime_ns": 2}]
        self.assertIsNone(cm._inputs_match(rec, same))
        self.assertIsNotNone(cm._inputs_match(rec, diff))

    def test_module_import_does_not_load_numerical_or_modeling_backends(self):
        # Explicit NPZ operations may load NumPy; metadata-only import must not.
        # A source substring incorrectly rejected lazy operation-local imports
        # and could miss indirect imports entirely. Check real fresh-process state.
        source_root = str(Path(cm.__file__).resolve().parents[2])
        script = (f"import sys; sys.path.insert(0, {source_root!r}); from fpcode.common import cache_manifest; "
                  "assert not any(name in sys.modules for name in ('jax', 'h5py', 'pandas', 'numpy'))")
        # -S excludes workstation site hooks that can preload NumPy themselves.
        completed = subprocess.run([sys.executable, "-B", "-S", "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
