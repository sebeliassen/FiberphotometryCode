import numpy as np
import pytest

from fpcode.behavior.kinematics import (
    clean_short_runs, presence_mask_from_conf, presence_mask_from_conf_kpms,
    smooth_heading_circular, smooth_heading_circular_legacy_fusion,
)
from fpcode.behavior.syllables import (
    extract_bouts, extract_syllable_runs, flatten_results_to_timeline,
    LEGACY_SPARSE_NO_FILL, tile_results,
)
from fpcode.common import schemas
from fpcode.contracts import pose


def test_presence_algorithms_remain_distinct_explicit_policies():
    confidence = np.tile([1., 1., 0., 0., 0.], (12, 1))
    kwargs = dict(smooth_win=1, min_on_frames=1, min_off_frames=1)
    assert not presence_mask_from_conf(confidence, **kwargs).any()
    assert presence_mask_from_conf_kpms(confidence, **kwargs).all()
    np.testing.assert_array_equal(confidence[0], [1, 1, 0, 0, 0])


def test_cleanup_bouts_and_compatibility_layout_identity():
    mask = np.array([0, 1, 0, 1, 1, 0], dtype=bool)
    np.testing.assert_array_equal(clean_short_runs(mask, min_true=2), [0, 0, 0, 1, 1, 0])
    np.testing.assert_array_equal(mask, [0, 1, 0, 1, 1, 0])
    bouts = extract_bouts([2, 2, 3, 3, 3])
    assert [(b.syllable, b.onset, b.offset, b.duration) for b in bouts] == [(2, 0, 2, 2), (3, 2, 5, 3)]
    assert extract_syllable_runs([2, 2, 3], min_length=2) == [dict(syllable=2, onset=0, offset=2, duration=2)]
    assert schemas.TOP_USE_BODYPARTS is pose.TOP_USE_BODYPARTS
    assert schemas.TANDEM_23KP is pose.TANDEM_23KP


def test_smoothing_preserves_angular_wrap_and_historical_short_array_policy():
    theta = np.array([3.1, -3.1, 3.1])
    result = smooth_heading_circular(theta, window=3)
    assert np.all(np.abs(result) > 3)
    short = np.array([0., 1.])
    np.testing.assert_array_equal(smooth_heading_circular(short, window=2), short)
    np.testing.assert_allclose(smooth_heading_circular_legacy_fusion(short, window=2), [0, 0.6995216443485196])


def test_sparse_and_dense_timeline_policies_and_padding_remain_separate():
    results = {"s": {"syllable": np.array([2, 3]), "centroid": np.array([[1, 2], [3, 4]])}}
    vfi = {"s": np.array([1, 3])}
    dense = flatten_results_to_timeline(results, vfi, actual_lengths={"s": 5})["s"]
    np.testing.assert_array_equal(dense.syllable, [2, 2, 2, 3, 3])
    sparse = flatten_results_to_timeline(results, vfi, actual_lengths={"s": 5}, policy=LEGACY_SPARSE_NO_FILL)["s"]
    assert str(sparse.syllable.dtype) == "Int64"
    assert sparse.syllable.isna().tolist() == [True, False, True, False, True]
    assert sparse.loc[1, "centroid_0"] == 1 and sparse.loc[3, "centroid_1"] == 4
    np.testing.assert_array_equal(results["s"]["syllable"], [2, 3])
    padded = flatten_results_to_timeline({"s": {"syllable": np.array([4])}}, vfi, policy=LEGACY_SPARSE_NO_FILL)["s"]
    assert padded.syllable.tolist() == [4, 4]
    with pytest.raises(ValueError, match="unknown"):
        flatten_results_to_timeline(results, vfi, policy="guess")
    tile_results(results, 2)
    np.testing.assert_array_equal(results["s"]["syllable"], [2, 2, 3, 3])
