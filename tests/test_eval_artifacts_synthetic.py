#!/usr/bin/env python3
"""
tests/test_eval_artifacts_synthetic.py
=============================================================================
Synthetic, CPU-only tests for evaluation-artifact protection (core.eval_artifacts
and the EML script's derived-results / dry-run behaviour):

  * stable label permutation is idempotent (the old argsort[::-1] was not)
  * old->new mapping is a true permutation and is recorded/invertible
  * source checkpoints opened read-only are never modified (stat guard)
  * incompatible models (dataset/mask/segment order/shape/iteration) are rejected
  * derived results cache is invalidated by a changed checkpoint, iteration or
    label mapping (stale label mappings are never reused)
  * --dry-run / --help never import JAX and write nothing
=============================================================================
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
EML_SCRIPT = EXP_DIR / "03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import h5py

from core import cache_manifest as cm
from core import eval_artifacts as ea


# --------------------------------------------------------------------------- #
# Synthetic checkpoint builder
# --------------------------------------------------------------------------- #
def make_checkpoint(path, *, n_seq=6, seg_len=40, latent=4, nlags=3, num_states=8,
                    iterations=(50, 100), seed=0, keys=None, bounds=None, z=None):
    rng = np.random.default_rng(seed)
    if keys is None:
        keys = np.array([f"rec{i // 2}_seg{i % 2}".encode() for i in range(n_seq)], dtype=object)
    if bounds is None:
        bounds = np.array([[i * 30, i * 30 + seg_len - (i % 3)] for i in range(n_seq)], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("metadata/arr0", data=keys, dtype=h5py.string_dtype())
        f.create_dataset("metadata/arr1", data=bounds)
        for it in iterations:
            g = f.create_group(f"model_snapshots/{it}")
            g.create_dataset("states/x", data=rng.normal(size=(n_seq, seg_len, latent)))
            zz = z if z is not None else rng.integers(0, num_states, size=(n_seq, seg_len - nlags)).astype(np.int32)
            g.create_dataset("states/z", data=zz)
            g.create_dataset("params/pi", data=np.full((num_states, num_states), 1.0 / num_states))
    return path


class TestStablePermutation(unittest.TestCase):
    def test_sorts_descending(self):
        index = ea.stable_frequency_order([5, 20, 1, 10])
        self.assertEqual(list(index), [1, 3, 0, 2])

    def test_ties_keep_original_order(self):
        index = ea.stable_frequency_order([3, 7, 3, 3, 7])
        self.assertEqual(list(index), [1, 4, 0, 2, 3])

    def test_idempotent_on_already_sorted_with_ties(self):
        freqs = np.array([9, 5, 5, 5, 2, 0, 0])
        index = ea.stable_frequency_order(freqs)
        self.assertEqual(list(index), list(range(len(freqs))))
        # applying it again to the relabeled frequencies is still identity
        new_freqs = freqs[index]
        self.assertEqual(list(ea.stable_frequency_order(new_freqs)), list(range(len(freqs))))

    def test_previous_argsort_reversal_was_not_idempotent(self):
        """Documents the bug being fixed: reversing argsort flips tied states every run."""
        freqs = np.array([9, 5, 5, 2])
        old_index = np.argsort(freqs)[::-1]
        self.assertFalse(np.array_equal(old_index, np.arange(4)), "tied sorted input was relabeled")
        self.assertTrue(np.array_equal(ea.stable_frequency_order(freqs), np.arange(4)))

    def test_random_idempotence(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            freqs = rng.integers(0, 6, size=30)
            idx = ea.stable_frequency_order(freqs)
            self.assertTrue(ea.is_permutation(idx))
            again = ea.stable_frequency_order(freqs[idx])
            self.assertTrue(np.array_equal(again, np.arange(30)))

    def test_inverse_and_relabel_roundtrip(self):
        freqs = np.array([1, 8, 3, 8, 0])
        idx = ea.stable_frequency_order(freqs)
        inv = ea.invert_permutation(idx)
        z = np.array([[0, 1, 2, 3, 4, 1]])
        z_new = ea.relabel_states(z, inv)
        # new label i must correspond to old state idx[i]
        self.assertTrue(np.array_equal(idx[z_new], z))
        self.assertEqual(ea.old_to_new_map(idx), {int(o): int(inv[o]) for o in range(5)})

    def test_mapping_hash_distinguishes_mappings(self):
        a = ea.mapping_hash([0, 1, 2, 3])
        b = ea.mapping_hash([0, 1, 3, 2])
        self.assertNotEqual(a, b)
        self.assertEqual(a, ea.mapping_hash(np.array([0, 1, 2, 3])))


class TestCheckpointIdentityAndCompatibility(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _ident(self, name, **kw):
        return ea.read_checkpoint_identity(make_checkpoint(self.d / name, **kw))

    def test_identity_fields(self):
        i = self._ident("a.h5")
        self.assertEqual(i["iteration"], 100)
        self.assertEqual(i["x_shape"], (6, 40, 4))
        self.assertEqual(i["nlags"], 3)
        self.assertEqual(i["num_states"], 8)
        self.assertEqual(i["n_recordings"], 6)  # one unique key per segment in this fixture

    def test_compatible_models_pass_even_with_different_parameters(self):
        a, b = self._ident("a.h5", seed=1), self._ident("b.h5", seed=2)
        ea.validate_model_compatibility({"a": a, "b": b})  # no raise

    def test_different_segment_order_rejected(self):
        a = self._ident("a.h5")
        keys = np.array([f"rec{i // 2}_seg{i % 2}".encode() for i in range(6)], dtype=object)[::-1].copy()
        b = self._ident("b.h5", keys=keys)
        with self.assertRaises(ea.IncompatibleModelsError) as cm_:
            ea.validate_model_compatibility({"a": a, "b": b})
        self.assertIn("segment order", str(cm_.exception))

    def test_different_bounds_mask_rejected(self):
        a = self._ident("a.h5")
        bounds = np.array([[i * 30, i * 30 + 39] for i in range(6)], dtype=np.int64)
        b = self._ident("b.h5", bounds=bounds)
        with self.assertRaises(ea.IncompatibleModelsError):
            ea.validate_model_compatibility({"a": a, "b": b})

    def test_different_shapes_and_latent_dim_rejected(self):
        a = self._ident("a.h5")
        b = self._ident("b.h5", latent=7)
        with self.assertRaises(ea.IncompatibleModelsError) as cm_:
            ea.validate_model_compatibility({"a": a, "b": b})
        self.assertIn("latent", str(cm_.exception))

    def test_different_iteration_rejected_unless_allowed(self):
        a = self._ident("a.h5", iterations=(50, 250))
        b = self._ident("b.h5", iterations=(50, 100))
        with self.assertRaises(ea.IncompatibleModelsError):
            ea.validate_model_compatibility({"a": a, "b": b})
        ea.validate_model_compatibility({"a": a, "b": b}, require_same_iteration=False)

    def test_all_mismatches_reported_together(self):
        a = self._ident("a.h5")
        b = self._ident("b.h5", latent=7, iterations=(50, 250))
        with self.assertRaises(ea.IncompatibleModelsError) as cm_:
            ea.validate_model_compatibility({"a": a, "b": b})
        msg = str(cm_.exception)
        self.assertIn("latent", msg)
        self.assertIn("iteration", msg)

    def test_needs_two_models(self):
        with self.assertRaises(ea.IncompatibleModelsError):
            ea.validate_model_compatibility({"a": self._ident("a.h5")})

    def test_mask_reconstruction_matches_bounds(self):
        bounds = np.array([[0, 5], [10, 18]])
        m = ea.reconstruct_mask(bounds, 10)
        self.assertEqual(m.sum(axis=1).tolist(), [5, 8])
        self.assertTrue(m[0, :5].all() and not m[0, 5:].any())


class TestImmutabilityGuard(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_readonly_reads_do_not_change_checkpoint(self):
        ck = make_checkpoint(self.d / "c.h5")
        before = ea.stat_signature([ck])
        time.sleep(0.02)
        ea.read_checkpoint_identity(ck)
        with h5py.File(ck, "r") as f:
            _ = f["model_snapshots/100/states/z"][()]
        ea.assert_unchanged(before, ea.stat_signature([ck]))  # no raise

    def test_modification_is_detected(self):
        ck = make_checkpoint(self.d / "c.h5")
        before = ea.stat_signature([ck])
        time.sleep(0.02)
        with h5py.File(ck, "r+") as f:  # what the old fast_reindex_checkpoint did
            f["model_snapshots/100/states/z"][...] = 0
        with self.assertRaises(RuntimeError) as cm_:
            ea.assert_unchanged(before, ea.stat_signature([ck]))
        self.assertIn("read-only", str(cm_.exception))

    def test_reindex_pipeline_leaves_source_bytes_identical(self):
        ck = make_checkpoint(self.d / "c.h5")
        sha_before = cm._sha256_file(ck)
        with h5py.File(ck, "r") as f:
            z = f["model_snapshots/100/states/z"][()]
        freqs = np.bincount(z.ravel(), minlength=8)
        idx = ea.stable_frequency_order(freqs)
        _ = ea.relabel_states(z, ea.invert_permutation(idx))
        self.assertEqual(sha_before, cm._sha256_file(ck))


def _load_eml_module():
    spec = importlib.util.spec_from_file_location("eml_script_under_test", str(EML_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDerivedResultsStaleness(unittest.TestCase):
    """The derived results cache must not survive a different checkpoint/iteration/mapping."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)
        self.eml = _load_eml_module()
        self.ck = make_checkpoint(self.d / "checkpoint.h5")
        self.res = self.d / self.eml.DERIVED_RESULTS_NAME
        self.res.write_bytes(b"R" * 64)
        self.idx = np.arange(8)
        self._publish(self.idx)

    def tearDown(self):
        self._td.cleanup()

    def _cfg(self, idx, ck=None):
        return self.eml._results_config(ea.read_checkpoint_identity(ck or self.ck), idx)

    def _publish(self, idx):
        cm.write_manifest(self.res, kind="results_h5", inputs=[self.ck], config=self._cfg(idx),
                          code_files=[Path(ea.__file__)])

    def _status(self, idx, ck=None):
        return cm.verify_manifest(self.res, kind="results_h5", inputs=[self.ck],
                                  config=self._cfg(idx, ck), code_files=[Path(ea.__file__)])[0]

    def test_same_checkpoint_and_mapping_is_cache_hit(self):
        self.assertEqual(self._status(self.idx), cm.OK)

    def test_different_label_mapping_invalidates(self):
        swapped = self.idx.copy()
        swapped[[2, 3]] = swapped[[3, 2]]
        self.assertEqual(self._status(swapped), cm.STALE)

    def test_checkpoint_modified_after_results_invalidates(self):
        time.sleep(0.02)
        with h5py.File(self.ck, "r+") as f:
            ds = f["model_snapshots/100/states/z"]
            ds[0, 0] = (int(ds[0, 0]) + 1) % 8  # guaranteed content change
        self.assertEqual(self._status(self.idx), cm.STALE)

    def test_large_checkpoint_rewrite_detected_by_stat_only(self):
        """Real 33 GB checkpoints are never content-hashed; any rewrite must still invalidate."""
        from unittest import mock
        with mock.patch.object(cm, "SMALL_FILE_HASH_LIMIT", 0):  # force stat-only fingerprint
            cm.write_manifest(self.res, kind="results_h5", inputs=[self.ck], config=self._cfg(self.idx),
                              code_files=[Path(ea.__file__)])
            m = cm.read_manifest(self.res)
            self.assertNotIn("sha256", m["inputs"][0])
            self.assertEqual(self._status(self.idx), cm.OK)
            time.sleep(0.02)
            with h5py.File(self.ck, "r+") as f:  # the old in-place reindex pattern: same bytes rewritten
                ds = f["model_snapshots/100/states/z"]
                ds[...] = ds[()]
            os.utime(self.ck, None)
            self.assertEqual(self._status(self.idx), cm.STALE)

    def test_different_iteration_invalidates(self):
        other = make_checkpoint(self.d / "other.h5", iterations=(50, 250))
        cfg_other = self.eml._results_config(ea.read_checkpoint_identity(other), self.idx)
        self.assertNotEqual(cm.config_hash(cfg_other), cm.config_hash(self._cfg(self.idx)))

    def test_legacy_unmanifested_results_are_unverified_not_trusted(self):
        legacy = self.d / "results.h5"
        legacy.write_bytes(b"L" * 4096)
        status = cm.verify_manifest(legacy, kind="results_h5", inputs=[self.ck], config=self._cfg(self.idx))[0]
        self.assertEqual(status, cm.UNVERIFIED_NO_MANIFEST)
        self.assertTrue(legacy.exists())

    def test_derived_name_never_collides_with_legacy_results(self):
        self.assertNotEqual(self.eml.DERIVED_RESULTS_NAME, "results.h5")


class TestDryRunAndHelp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)
        self.proj = self.d / "proj"
        for s in (101, 202):
            sub = self.proj / f"tandem_model_seed{s}"
            sub.mkdir(parents=True)
            make_checkpoint(sub / "checkpoint.h5", iterations=(50, 250), seed=s)
        self.out = self.d / "out"

    def tearDown(self):
        self._td.cleanup()

    def _run(self, *args):
        code = (
            "import sys, runpy\n"
            f"sys.argv = [{str(EML_SCRIPT)!r}] + {list(args)!r}\n"
            "try:\n"
            f"    runpy.run_path({str(EML_SCRIPT)!r}, run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    print('EXIT', e.code)\n"
            "print('JAX_IMPORTED', 'jax' in sys.modules)\n"
            "print('KPMS_IMPORTED', 'keypoint_moseq' in sys.modules)\n"
        )
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    def test_help_does_not_import_jax(self):
        r = self._run("--help")
        self.assertIn("--dry-run", r.stdout)
        self.assertIn("JAX_IMPORTED False", r.stdout)

    def test_dry_run_no_jax_no_writes_and_readonly_sources(self):
        cks = [self.proj / f"tandem_model_seed{s}/checkpoint.h5" for s in (101, 202)]
        before = ea.stat_signature(cks)
        r = self._run("--dry-run", "--project-dir", str(self.proj), "--output-dir", str(self.out), "--min-iters", "200")
        self.assertIn("[COMPATIBLE]", r.stdout, r.stdout + r.stderr)
        self.assertIn("[DRY RUN]", r.stdout)
        self.assertIn("JAX_IMPORTED False", r.stdout)
        self.assertIn("KPMS_IMPORTED False", r.stdout)
        self.assertFalse(self.out.exists(), "dry-run must not create output dirs")
        self.assertEqual(list(self.proj.rglob("results*.h5")), [])
        ea.assert_unchanged(before, ea.stat_signature(cks))

    def test_dry_run_rejects_incompatible_models(self):
        sub = self.proj / "tandem_model_seed303"
        sub.mkdir()
        make_checkpoint(sub / "checkpoint.h5", iterations=(50, 250), latent=9, seed=3)
        r = self._run("--dry-run", "--project-dir", str(self.proj), "--output-dir", str(self.out), "--min-iters", "200")
        self.assertIn("not comparable", r.stdout)
        self.assertIn("EXIT 2", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
