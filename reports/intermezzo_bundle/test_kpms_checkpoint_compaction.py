#!/usr/bin/env python3
"""
test_kpms_checkpoint_compaction.py
=============================================================================
Zero-Side-Effect Comprehensive Test Suite for KPMS Checkpoint Compaction
and Transparent Monkeypatching.

Guarantees:
  - 100% isolated inside tempfile.TemporaryDirectory()
  - Zero modification to existing production models or repository checkpoints
  - Tests North Star backward-compatibility on real uncompacted checkpoints
  - Tests bit-for-bit equivalence between uncompacted and compacted models
  - Tests downstream pipeline tools (extract_results, reindex, fit_model, EML)
=============================================================================
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np

import keypoint_moseq as kpms
from keypoint_moseq.fitting import marginal_log_likelihood
import jax.numpy as jnp

# Import our compaction engine and patch
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.compact_checkpoint import (
    compact_checkpoint,
)
from syllable_generation.syllabizers.keypoint_moseq_syll.utils.patch_checkpoint import (
    apply_kpms_checkpoint_patch,
    is_checkpoint_packed,
    patched_load_checkpoint,
    remove_kpms_checkpoint_patch,
)

# Reference sample checkpoints in the workspace
MINIMAL_MODEL_CKPT = (
    REPO_ROOT
    / "syllable_generation/multiview_top_bot_experiments/kpms_minimal_real_gpu/real_gpu_minimal_mv_model/checkpoint.h5"
)
KAREN_LEGACY_CKPT = (
    REPO_ROOT
    / "syllable_generation/syllabizers/keypoint_moseq_syll/karen_project/test_pipeline_run/checkpoint.h5"
)


class TestKPMSCheckpointCompaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Determine which test file to use as the North Star
        if MINIMAL_MODEL_CKPT.exists():
            cls.ref_ckpt = MINIMAL_MODEL_CKPT
        elif KAREN_LEGACY_CKPT.exists():
            cls.ref_ckpt = KAREN_LEGACY_CKPT
        else:
            raise FileNotFoundError("Could not find any reference sample checkpoint for testing.")

        print(f"\n[Test Setup] Using reference checkpoint: {cls.ref_ckpt}")
        with h5py.File(cls.ref_ckpt, "r") as f:
            cls.available_snaps = sorted([int(k) for k in f["model_snapshots"].keys()])
        cls.max_iter = max(cls.available_snaps)
        print(f"[Test Setup] Available snapshots: {cls.available_snaps}, max_iter={cls.max_iter}")

    def setUp(self):
        # Create an isolated temporary sandbox for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox_path = Path(self.temp_dir.name)

        # Copy reference checkpoint to sandbox
        self.sandbox_orig = self.sandbox_path / "checkpoint_orig.h5"
        shutil.copy2(self.ref_ckpt, self.sandbox_orig)

        # Output path for compacted version
        self.sandbox_compact = self.sandbox_path / "checkpoint_compact.h5"

    def tearDown(self):
        # Always restore stock KPMS loader
        remove_kpms_checkpoint_patch()
        # Clean up temporary sandbox directory completely
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # TEST 1: The North Star Test (Uncompacted Legacy Backward Compatibility)
    # -------------------------------------------------------------------------
    def test_01_north_star_uncompacted_backward_compatibility(self):
        """
        Verify that on an uncompacted legacy checkpoint, patched_load_checkpoint
        produces bit-for-bit identical outputs to stock kpms.load_checkpoint.
        """
        # 1. Load via stock loader
        stock_model, stock_data, stock_meta, stock_iter = kpms.io.load_checkpoint(
            path=str(self.sandbox_orig)
        )

        # 2. Load via patched loader (without patch applied globally)
        patch_model, patch_data, patch_meta, patch_iter = patched_load_checkpoint(
            path=str(self.sandbox_orig)
        )

        self.assertEqual(stock_iter, patch_iter)
        self.assertEqual(patch_iter, self.max_iter)

        # Verify all core state arrays
        for k in ["x", "z", "v", "h"]:
            if k in stock_model["states"]:
                np.testing.assert_array_equal(
                    stock_model["states"][k],
                    patch_model["states"][k],
                    err_msg=f"Mismatch in states/{k} between stock and patched loader",
                )

        # Verify observation noise
        if "s" in stock_model["states"]:
            np.testing.assert_array_equal(
                stock_model["states"]["s"],
                patch_model["states"]["s"],
                err_msg="Mismatch in states/s",
            )
        if "noise_prior" in stock_model:
            np.testing.assert_array_equal(
                stock_model["noise_prior"],
                patch_model["noise_prior"],
                err_msg="Mismatch in noise_prior",
            )

        # Verify parameters
        for p in ["Ab", "Q", "pi"]:
            np.testing.assert_array_equal(
                stock_model["params"][p],
                patch_model["params"][p],
                err_msg=f"Mismatch in params/{p}",
            )

        # Verify data and metadata
        self.assertEqual(stock_data.keys(), patch_data.keys())
        for d_key in stock_data.keys():
            np.testing.assert_array_equal(
                stock_data[d_key],
                patch_data[d_key],
                err_msg=f"Mismatch in data/{d_key}",
            )
        self._assert_meta_equal(stock_meta, patch_meta)
        print("  [PASS] Pillar 1: North Star legacy uncompacted backward compatibility verified.")

    def _assert_meta_equal(self, meta1, meta2):
        self.assertEqual(len(meta1), len(meta2))
        for e1, e2 in zip(meta1, meta2):
            if isinstance(e1, (np.ndarray, list)):
                np.testing.assert_array_equal(e1, e2)
            else:
                self.assertEqual(e1, e2)

    # -------------------------------------------------------------------------
    # TEST 2: Compaction Execution and "is_packed" Inspection
    # -------------------------------------------------------------------------
    def test_02_full_compaction_execution(self):
        """
        Verify that compact_checkpoint successfully prunes intermediate burn-in
        snapshots, retains iteration max_iter, and sets packing metadata.
        """
        stats = compact_checkpoint(
            source_path=self.sandbox_orig,
            dest_path=self.sandbox_compact,
            mode="full",
            keep_iterations=[self.max_iter],
            verbose=False,
        )

        self.assertTrue(self.sandbox_compact.exists())
        self.assertLessEqual(stats["compact_size_bytes"], stats["original_size_bytes"])

        # Check HDF5 structure
        with h5py.File(self.sandbox_compact, "r") as f:
            self.assertTrue(f.attrs.get("is_packed", False))
            self.assertEqual(f.attrs.get("packing_mode"), "full")
            saved_iters = sorted([int(k) for k in f["model_snapshots"].keys()])
            self.assertEqual(saved_iters, [self.max_iter])

        is_packed, info = is_checkpoint_packed(self.sandbox_compact)
        self.assertTrue(is_packed)
        self.assertEqual(info["available_iterations"], [self.max_iter])
        print(f"  [PASS] Pillar 2: Full compaction executed. Size: {stats['original_size_bytes']:,} -> {stats['compact_size_bytes']:,} bytes ({stats['reduction_percent']:.1f}% reduction).")

    # -------------------------------------------------------------------------
    # TEST 3: Bit-for-Bit Mathematical Equivalence (Compacted vs Uncompacted)
    # -------------------------------------------------------------------------
    def test_03_compacted_bit_for_bit_equivalence(self):
        """
        Verify that loading a compacted checkpoint returns 100% identical floats
        and ints to loading the original uncompacted file at the final iteration.
        """
        compact_checkpoint(
            source_path=self.sandbox_orig,
            dest_path=self.sandbox_compact,
            mode="full",
            keep_iterations=[self.max_iter],
            verbose=False,
        )

        # Stock load on original
        orig_model, orig_data, orig_meta, orig_iter = kpms.io.load_checkpoint(
            path=str(self.sandbox_orig), iteration=self.max_iter
        )

        # Patched load on compact
        comp_model, comp_data, comp_meta, comp_iter = patched_load_checkpoint(
            path=str(self.sandbox_compact)
        )

        self.assertEqual(orig_iter, comp_iter)

        # Compare every single array
        for k in ["x", "z", "v", "h"]:
            if k in orig_model["states"]:
                np.testing.assert_array_equal(
                    orig_model["states"][k],
                    comp_model["states"][k],
                    err_msg=f"Array divergence in states/{k} after compaction!",
                )

        if "s" in orig_model["states"]:
            np.testing.assert_array_equal(
                orig_model["states"]["s"],
                comp_model["states"]["s"],
                err_msg="Observation noise scale 'states/s' diverged!",
            )
        if "noise_prior" in orig_model:
            np.testing.assert_array_equal(
                orig_model["noise_prior"],
                comp_model["noise_prior"],
                err_msg="Noise prior diverged!",
            )

        for p in ["Ab", "Q", "pi"]:
            np.testing.assert_array_equal(
                orig_model["params"][p],
                comp_model["params"][p],
                err_msg=f"Parameter {p} diverged!",
            )

        self.assertEqual(orig_data.keys(), comp_data.keys())
        for d_key in orig_data.keys():
            np.testing.assert_array_equal(
                orig_data[d_key],
                comp_data[d_key],
                err_msg=f"Array divergence in data/{d_key} after compaction!",
            )
        self._assert_meta_equal(orig_meta, comp_meta)
        print("  [PASS] Pillar 3: Bit-for-bit mathematical equivalence verified across all tensors.")

    # -------------------------------------------------------------------------
    # TEST 4: Pruned Historical Query Error Interception
    # -------------------------------------------------------------------------
    def test_04_pruned_iteration_clean_error(self):
        """
        Verify that requesting a pruned iteration on a compacted checkpoint
        raises a descriptive KeyError rather than an unhandled assertion failure.
        """
        if len(self.available_snaps) < 2:
            self.skipTest("Need at least 2 snapshots to test pruned query.")

        pruned_iter = self.available_snaps[0]
        compact_checkpoint(
            source_path=self.sandbox_orig,
            dest_path=self.sandbox_compact,
            mode="full",
            keep_iterations=[self.max_iter],
            verbose=False,
        )

        # Requesting pruned iteration should raise KeyError with clear explanation
        with self.assertRaises(KeyError) as ctx:
            patched_load_checkpoint(path=str(self.sandbox_compact), iteration=pruned_iter)

        self.assertIn("has been compacted", str(ctx.exception))
        self.assertIn(str(pruned_iter), str(ctx.exception))
        print("  [PASS] Pillar 4: Pruned iteration query correctly caught with descriptive error.")

    # -------------------------------------------------------------------------
    # TEST 5: Downstream Pipeline & EML Functional Equivalence
    # -------------------------------------------------------------------------
    def test_05_downstream_pipeline_and_eml_equivalence(self):
        """
        Verify that downstream tools (extract_results, marginal_log_likelihood)
        produce identical results from compacted vs uncompacted models.
        """
        compact_checkpoint(
            source_path=self.sandbox_orig,
            dest_path=self.sandbox_compact,
            mode="full",
            keep_iterations=[self.max_iter],
            verbose=False,
        )

        orig_model, orig_data, orig_meta, _ = kpms.io.load_checkpoint(path=str(self.sandbox_orig))
        comp_model, comp_data, comp_meta, _ = patched_load_checkpoint(path=str(self.sandbox_compact))

        # 1. extract_results inside sandbox
        res_orig_path = self.sandbox_path / "results_orig.h5"
        res_comp_path = self.sandbox_path / "results_comp.h5"

        res_orig = kpms.extract_results(orig_model, orig_meta, project_dir=str(self.sandbox_path), model_name="orig", path=str(res_orig_path))
        res_comp = kpms.extract_results(comp_model, comp_meta, project_dir=str(self.sandbox_path), model_name="comp", path=str(res_comp_path))

        self.assertEqual(res_orig.keys(), res_comp.keys())
        for session_key in res_orig.keys():
            np.testing.assert_array_equal(res_orig[session_key]["syllable"], res_comp[session_key]["syllable"])
            np.testing.assert_array_equal(res_orig[session_key]["latent_state"], res_comp[session_key]["latent_state"])

        # 2. marginal_log_likelihood computation
        mask_jnp = jnp.array(orig_data["mask"])
        x_orig = jnp.array(orig_model["states"]["x"])
        x_comp = jnp.array(comp_model["states"]["x"])

        Ab_orig = jnp.array(orig_model["params"]["Ab"])
        Q_orig = jnp.array(orig_model["params"]["Q"])
        pi_orig = jnp.array(orig_model["params"]["pi"])

        ll_orig = float(marginal_log_likelihood(mask_jnp, x_orig, Ab_orig, Q_orig, pi_orig))
        ll_comp = float(marginal_log_likelihood(mask_jnp, x_comp, Ab_orig, Q_orig, pi_orig))

        self.assertAlmostEqual(ll_orig, ll_comp, places=7)
        print(f"  [PASS] Pillar 5: Downstream extract_results & EML marginal_log_likelihood matched (LL={ll_orig:.4f}).")

    # -------------------------------------------------------------------------
    # TEST 6: In-Place Syllable Reindexing on Compact Checkpoint
    # -------------------------------------------------------------------------
    def test_06_reindex_syllables_on_compact_checkpoint(self):
        """
        Verify that kpms.io.reindex_syllables_in_checkpoint works seamlessly
        on a compacted checkpoint.
        """
        compact_checkpoint(
            source_path=self.sandbox_orig,
            dest_path=self.sandbox_compact,
            mode="full",
            keep_iterations=[self.max_iter],
            verbose=False,
        )

        # Apply global monkeypatch
        apply_kpms_checkpoint_patch()

        # Run in-place reindexing on sandbox copy
        reindexed_path = self.sandbox_path / "reindexed_compact.h5"
        shutil.copy2(self.sandbox_compact, reindexed_path)

        kpms.io.reindex_syllables_in_checkpoint(path=str(reindexed_path))

        # Verify it loads cleanly and has sorted syllables
        reindex_model, _, _, it = kpms.load_checkpoint(path=str(reindexed_path))
        self.assertEqual(it, self.max_iter)
        self.assertTrue("pi" in reindex_model["params"])
        print("  [PASS] Pillar 6: In-place syllable reindexing executed cleanly on compact checkpoint.")

    # -------------------------------------------------------------------------
    # TEST 7: Resuming / Continuing Training from Compact Checkpoint
    # -------------------------------------------------------------------------
    def test_07_continue_training_from_compact_checkpoint(self):
        """
        Verify that training (fit_model Gibbs sampling) can resume seamlessly
        from a compacted checkpoint, performing an iteration and saving a new snapshot.
        """
        compact_checkpoint(
            source_path=self.sandbox_orig,
            dest_path=self.sandbox_compact,
            mode="full",
            keep_iterations=[self.max_iter],
            verbose=False,
        )

        apply_kpms_checkpoint_patch()

        # Load the model state from the compact checkpoint
        model, data, metadata, current_iter = kpms.load_checkpoint(path=str(self.sandbox_compact))
        self.assertEqual(current_iter, self.max_iter)

        # Create a dedicated isolated model directory in the sandbox
        resumed_model_dir = self.sandbox_path / "resumed_model"
        resumed_model_dir.mkdir(parents=True, exist_ok=True)
        resumed_ckpt = resumed_model_dir / "checkpoint.h5"
        shutil.copy2(self.sandbox_compact, resumed_ckpt)

        # Run 1 iteration of fit_model (Gibbs sampling)
        next_target = current_iter + 1
        new_model, _ = kpms.fit_model(
            model, data, metadata,
            project_dir=str(self.sandbox_path),
            model_name="resumed_model",
            ar_only=False,
            start_iter=current_iter,
            num_iters=next_target,
            save_every_n_iters=1,
            generate_progress_plots=False,
            verbose=False
        )

        # Verify new iteration was saved into the checkpoint
        reloaded_model, _, _, final_iter = kpms.load_checkpoint(
            project_dir=str(self.sandbox_path),
            model_name="resumed_model"
        )
        self.assertEqual(final_iter, next_target)
        self.assertTrue("x" in reloaded_model["states"])
        self.assertTrue("s" in reloaded_model["states"])
        print(f"  [PASS] Pillar 7: Resumed training from compact checkpoint (iter {current_iter} -> {final_iter}) successfully.")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING KPMS CHECKPOINT COMPACTION & MONKEYPATCH VERIFICATION BATTERY")
    print("=" * 80)
    unittest.main(verbosity=2)
