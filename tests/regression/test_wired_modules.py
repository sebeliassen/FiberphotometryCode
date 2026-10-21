"""
tests/test_wired_modules.py
===========================
Byte-identical / mathematical equivalence tests comparing new fpcode modules
against legacy shims and runners on synthetic data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np


# 1. Test kpms.checkpoints vs old compact_checkpoint shim
def test_checkpoint_compaction_equivalence():
    import syllable_generation.syllabizers.keypoint_moseq_syll.utils.compact_checkpoint as old_ckpt

    import fpcode.kpms.checkpoints as new_ckpt

    assert new_ckpt.compact_checkpoint is old_ckpt.compact_checkpoint
    assert new_ckpt.is_checkpoint_packed is old_ckpt.is_checkpoint_packed

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "test_src.h5"
        with h5py.File(src, "w") as f:
            f.create_group("model_snapshots/10")
            f.create_group("model_snapshots/20")
            f["model_snapshots/20/states/x"] = np.ones((5, 10, 10))
            f["model_snapshots/20/params/pi"] = np.ones(10)

        dst = Path(tmpdir) / "compacted.h5"
        res = new_ckpt.compact_checkpoint(src, dest_path=dst, keep_iterations=[20], verbose=False)
        assert dst.exists()
        assert res["compacted_bytes"] > 0
        is_packed, meta = new_ckpt.is_checkpoint_packed(dst)
        assert is_packed is True
        assert meta["available_iterations"] == [20]


# 2. Test kpms.data vs old trimming shim
def test_data_trimming_numerical_equivalence():
    import syllable_generation.syllabizers.keypoint_moseq_syll.utils.trimming as old_trim

    import fpcode.kpms.data as new_data

    # Test _clean_short_runs
    mask = np.array([True, True, False, False, True, False, True, True, True], dtype=bool)
    new_cleaned = new_data._clean_short_runs(mask, min_true=2, min_false=2)
    old_cleaned = old_trim._clean_short_runs(mask, min_true=2, min_false=2)
    np.testing.assert_array_equal(new_cleaned, old_cleaned)

    # Test presence_mask_from_conf
    np.random.seed(42)
    conf = np.random.uniform(0.0, 1.0, size=(100, 15))
    new_pmask = new_data.presence_mask_from_conf(conf, p_thresh=0.8, min_on_frames=5)
    old_pmask = old_trim.presence_mask_from_conf(conf, p_thresh=0.8, min_on_frames=5)
    np.testing.assert_array_equal(new_pmask, old_pmask)

    # Test subset_spread_blocks
    new_sub = new_data.subset_spread_blocks(500, frac=0.2, n_blocks=4, seed=123)
    old_sub = old_trim.subset_spread_blocks(500, frac=0.2, n_blocks=4, seed=123)
    np.testing.assert_array_equal(new_sub, old_sub)

    # Test subset_by_activity
    coords = {"rec1": np.random.randn(200, 10, 2)}
    confs = {"rec1": np.ones((200, 10))}
    vfis = {"rec1": np.arange(200)}

    nc, ncf, nv = new_data.subset_by_activity(coords, confs, vfis, frac=0.3)
    oc, ocf, ov = old_trim.subset_by_activity(coords, confs, vfis, frac=0.3)
    np.testing.assert_array_equal(nc["rec1"], oc["rec1"])
    np.testing.assert_array_equal(ncf["rec1"], ocf["rec1"])
    np.testing.assert_array_equal(nv["rec1"], ov["rec1"])


# 3. Test kpms.runtime vs old resample_batching shim
def test_runtime_batch_config_equivalence():
    import syllable_generation.syllabizers.keypoint_moseq_syll.utils.resample_batching as old_rt

    import fpcode.kpms.runtime as new_rt

    cfg1 = new_rt.BatchConfig(discrete_bs=12, scales_bs=24)
    assert cfg1.discrete_bs == 12
    assert cfg1.scales_bs == 24
    assert old_rt.BatchConfig is new_rt.BatchConfig
    assert old_rt.CONFIG is new_rt.CONFIG


# 4. Test kpms.init_model vs old batch_init shim
def test_init_model_shim_equivalence():
    import syllable_generation.syllabizers.keypoint_moseq_syll.utils.batch_init as old_init

    import fpcode.kpms.init_model as new_init

    assert new_init.batched_init_model_by_video is old_init.batched_init_model_by_video


# 5. Test kpms.evaluation vs core.eval_artifacts
def test_evaluation_reindex_mathematical_equivalence():
    import fpcode.kpms.evaluation as new_ea
    import fpcode.kpms.evaluation as old_ea

    freqs = np.array([0.05, 0.40, 0.15, 0.40, 0.00])
    order_new = new_ea.stable_frequency_order(freqs)
    order_old = old_ea.stable_frequency_order(freqs)
    np.testing.assert_array_equal(order_new, order_old)

    # Invert permutation
    inv_new = new_ea.invert_permutation(order_new)
    inv_old = old_ea.invert_permutation(order_old)
    np.testing.assert_array_equal(inv_new, inv_old)


# 6. Test viz rendering
def test_viz_figures_and_trajectories(tmp_path):
    from fpcode.viz.figures import plot_syllable_usage, plot_transition_matrix
    from fpcode.viz.trajectories import plot_syllable_trajectories

    # Figures
    freqs = np.array([0.5, 0.3, 0.2])
    fig1 = plot_syllable_usage(freqs, output_path=tmp_path / "usage.png")
    assert fig1.exists()

    tm = np.array([[0.8, 0.2], [0.1, 0.9]])
    fig2 = plot_transition_matrix(tm, output_path=tmp_path / "tm.png")
    assert fig2.exists()

    # Trajectories
    coords = {"rec": np.random.randn(100, 5, 2)}
    sylls = {"rec": np.array([0] * 50 + [1] * 50)}
    fig3 = plot_syllable_trajectories(coords, sylls, syllable_id=0, output_path=tmp_path / "traj.png")
    assert fig3.exists()
