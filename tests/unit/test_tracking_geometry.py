"""Independent tiny geometry/support expectations; no recordings or rendering."""

import ast
from pathlib import Path

import numpy as np
import pytest

from fpcode.contracts import PoseSeries, TimeAxis
from fpcode.tracking.geometry import apply_affine, project_reflected_points, rot2d
from fpcode.tracking.quality import confidence_mask, supported_track_mask
from fpcode.viz.selection.eligibility import valid_track_mask

ROOT = Path(__file__).resolve().parents[2]


def test_rotation_places_matrix_axes_after_angle_axes():
    angles = np.array([[0., np.pi / 2], [-np.pi / 2, np.pi]])
    actual = rot2d(angles, dtype=np.float32)
    expected = np.array([[[[1., 0.], [0., 1.]], [[0., -1.], [1., 0.]]],
                         [[[0., 1.], [-1., 0.]], [[-1., 0.], [0., -1.]]]], dtype=np.float32)
    assert actual.shape == (2, 2, 2, 2)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(rot2d(0), np.eye(2))
    assert rot2d(np.array([]), dtype=np.float32).shape == (0, 2, 2)


@pytest.mark.parametrize("relative_path", [
    "src/fpcode/tracking/fusion.py",
    "studies/multiview_experiments/01_data_and_fusion/fuse_multiview_cohort.py",
    "studies/multiview_experiments/01_data_and_fusion/fuse_reflection_corrected_sessions.py",
])
def test_production_rotation_wrappers_preserve_float32_values_without_running_studies(relative_path):
    # Execute only the actual import and wrapper, avoiding historical scripts'
    # import-time output-directory creation and empirical main functions.
    source = ROOT / relative_path
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name == "rot2d")
             or (isinstance(node, ast.ImportFrom) and node.module == "fpcode.tracking.geometry")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    actual = namespace["rot2d"](np.array([0., np.pi / 2, np.pi]))
    assert actual.shape == (3, 2, 2)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, [[[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]]], atol=1e-7)


def test_reflection_affine_preserves_sequential_float32_rounding_and_inputs():
    points = np.array([[743.6666, 91.11223]], dtype=np.float32)
    points.setflags(write=False)
    matrix = np.array([[1.234567, -.098765], [.23456, .999991]], dtype=np.float32)
    translation = np.array([21.555, -4.333], dtype=np.float32)
    actual = project_reflected_points(points, matrix, translation, width=972., dtype=np.float32)
    # Saved explicit values from the original two-stage float32 arithmetic.
    # Float64 promotion, even followed by a final cast, changes the second value.
    expected = np.array([[294.44915771484375, 140.33627319335938]], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(points, np.array([[743.6666, 91.11223]], dtype=np.float32))
    assert not points.flags.writeable


def test_affine_default_precision_remains_float64_and_nan_support_is_preserved():
    np.testing.assert_array_equal(apply_affine([[1, 2]], [[2, 0], [0, 3]], [4, 5]), [[6, 11]])
    assert apply_affine(np.ones((1, 2), np.float32), np.eye(2), [0, 0]).dtype == np.float64
    source = np.array([[[1., 2.], [np.nan, 3.]], [[4., 5.], [6., np.nan]]], dtype=np.float32)
    actual = project_reflected_points(source, np.eye(2), [0, 0], width=10, dtype=np.float32)
    np.testing.assert_array_equal(actual, [[[9., 2.], [np.nan, np.nan]], [[6., 5.], [np.nan, np.nan]]])
    assert actual.shape == source.shape


@pytest.mark.parametrize("width", [0., -1., np.nan, np.inf])
def test_reflection_rejects_invalid_calibration_width(width):
    with pytest.raises(ValueError, match="width"):
        project_reflected_points([[1, 2]], np.eye(2), [0, 0], width=width, dtype=np.float32)


def test_shared_array_support_keeps_threshold_boundary_finite_checks_and_inputs():
    coordinates = np.array([[[1, 1], [np.nan, 1], [1, np.inf]], [[1, 1], [1, 1], [1, 1]]])
    confidence = np.array([[.5, .9, .9], [.499, np.nan, .7]])
    coordinates.setflags(write=False)
    confidence.setflags(write=False)
    expected = np.array([[True, False, False], [False, False, True]])
    np.testing.assert_array_equal(supported_track_mask(coordinates, confidence, threshold=.5), expected)
    np.testing.assert_array_equal(valid_track_mask(coordinates, confidence, .5), expected)
    assert not coordinates.flags.writeable and not confidence.flags.writeable


def test_pose_quality_keeps_explicit_validity_and_three_dimensional_missingness():
    pose = PoseSeries(TimeAxis([0], "frame", "camera"), np.array([[[1, 1, 1], [1, 1, np.nan], [1, 1, 1]]]),
                      np.ones((1, 3)), ("masked", "missing_z", "supported"), "pixel", "camera",
                      valid=np.array([[False, False, True]]))
    np.testing.assert_array_equal(confidence_mask(pose, threshold=.5), [[False, False, True]])


def test_legacy_array_threshold_policy_remains_distinct_from_strict_pose_policy():
    np.testing.assert_array_equal(valid_track_mask([[[1, 1]]], [[.5]], -1), [[True]])
    np.testing.assert_array_equal(valid_track_mask([[[1, 1]]], [[.5]], np.nan), [[False]])
    pose = PoseSeries(TimeAxis([0], "frame", "camera"), np.ones((1, 1, 2)), np.ones((1, 1)),
                      ("point",), "pixel", "camera")
    with pytest.raises(ValueError, match="threshold"):
        confidence_mask(pose, threshold=-1)
