"""Physical feature projection and training-only categorical/feature encoders."""

import numpy as np
import pytest

from fpcode.behavior.analysis.bout_features import summarize_source_bouts
from fpcode.behavior.primitives.segmentation import WholeBoutCodebook


def test_physical_summaries_preserve_gaps_bouts_and_circular_heading():
    frames = np.array([0, 1, 2, 10, 11, 12])
    result = summarize_source_bouts(frames, np.array([4, 4, 4, 4, 4, 4]),
              np.array([[0., 0.], [1., 0.], [2., 0.], [100., 0.], [101., 0.], [102., 0.]]),
              fps=10, heading=np.array([3.1, -3.1, -3., 0., 0., 0.]))
    np.testing.assert_array_equal(result['start_frames'], [0, 10])
    np.testing.assert_array_equal(result['stop_frames'], [3, 13])
    np.testing.assert_allclose(result['values'][:, 0], [.3, .3])
    np.testing.assert_allclose(result['values'][:, 1], [10., 10.])
    np.testing.assert_allclose(result['values'][:, 3], [2., 2.])
    assert 0 < result['values'][0, 6] < .3


def test_bout_codebooks_change_types_without_fitting_on_held_values():
    values = np.r_[np.zeros((20, 1)), np.full((20, 1), 10.)]
    parents = ['groom']*40
    model = WholeBoutCodebook.fit(values, parents, feature_names=('speed',), clusters=2, seed=2)
    before = model.location.copy()
    assigned = model.assign([[0.], [10.], [np.nan], [1000.]], ['groom']*4)
    assert assigned[0] != assigned[1] and assigned[2] == ''
    np.testing.assert_array_equal(model.location, before)
    assert assigned[3] in model.catalog_ids


def test_joint_ids_are_categories_and_subtypes_preserve_parent_identity():
    values = np.zeros((40, 1))
    first = WholeBoutCodebook.fit(values, ['1']*20+['99']*20, feature_names=('x',), route='joint', clusters=2, seed=1)
    second = WholeBoutCodebook.fit(values, ['elephant']*20+['mouse']*20, feature_names=('x',), route='joint', clusters=2, seed=1)
    a = first.assign([[0.], [0.]], ['1', '99'])
    b = second.assign([[0.], [0.]], ['elephant', 'mouse'])
    assert (a[0] == a[1]) == (b[0] == b[1]) is False
    subtype = WholeBoutCodebook.fit(values, ['1']*20+['99']*20, feature_names=('x',), route='within_parent', clusters=2)
    labels = subtype.assign([[0.], [0.], [0.]], ['1', '99', 'unseen'])
    assert labels[0].startswith('1:v') and labels[1].startswith('99:v') and labels[2] == ''


def test_reconstructed_bout_codebook_is_frozen_and_assignments_are_batched():
    location, scale, centers = np.array([5.]), np.array([2.]), np.array([[-1.], [1.]])
    model = WholeBoutCodebook('feature', ('speed',), location, scale, ('groom',), {'all': centers})
    location[0], scale[0], centers[0, 0] = 99., 99., 99.
    np.testing.assert_array_equal(model.location, [5.])
    np.testing.assert_array_equal(model.centers['all'], [[-1.], [1.]])
    with pytest.raises(TypeError):
        model.centers['other'] = np.zeros((2, 1))
    with pytest.raises(ValueError):
        model.centers['all'][0, 0] = 9.
    values = np.arange(101, dtype=float)[:, None]
    parents = ['groom']*len(values)
    expected = np.where(values[:, 0] < 5., 'type_0', 'type_1')
    # The equidistant observation at 5 keeps the lower-index center.
    expected[5] = 'type_0'
    np.testing.assert_array_equal(model.assign(values, parents, batch_size=3), expected)
    np.testing.assert_array_equal(model.assign(values, parents, batch_size=23), expected)
    with pytest.raises(ValueError, match='budget'):
        model.assign(values, parents, memory_budget_bytes=100)
    with pytest.raises(ValueError, match='fitted arrays'):
        WholeBoutCodebook('feature', ('speed',), [0.], [0.], ('groom',), {'all': [[1.]]})


def test_bout_projection_budget_scales_with_latent_schema_and_invalid_policies():
    frames, labels, centroid = np.arange(100), np.zeros(100, dtype=int), np.zeros((100, 2))
    # Basic summaries fit; 64-dimensional result/scratch arrays exceed this cap.
    summarize_source_bouts(frames, labels, centroid, fps=10, memory_budget_bytes=60000)
    with pytest.raises(MemoryError, match='budget'):
        summarize_source_bouts(frames, labels, centroid, fps=10, latent=np.zeros((100, 64)),
                               memory_budget_bytes=60000)
    for field in ({'fps': True}, {'fps': 10, 'minimum_support': True}, {'fps': 10, 'memory_budget_bytes': True}):
        with pytest.raises(ValueError):
            summarize_source_bouts(frames, labels, centroid, **field)
