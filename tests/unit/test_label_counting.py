"""Tiny independent expectations, including the real study aggregation loops."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fpcode.behavior.sequences.counting import label_counts


def test_frame_and_bout_counts_preserve_missing_labels_and_original_frame_gaps():
    labels = np.array([0, 0, -5, 0, 0, 8, 1, 1, 1], dtype=np.int32)
    frames = np.array([100, 101, 102, 103, 110, 111, 112, 113, 114])
    np.testing.assert_array_equal(label_counts(labels, 3), [4, 3, 0])
    np.testing.assert_array_equal(label_counts(labels, 3, mode="bouts"), [2, 1, 0])
    np.testing.assert_array_equal(label_counts(labels, 3, mode="bouts", frame_indices=frames), [3, 1, 0])
    np.testing.assert_array_equal(labels, [0, 0, -5, 0, 0, 8, 1, 1, 1])
    np.testing.assert_array_equal(label_counts(np.array([], dtype=int), 3, mode="bouts"), [0, 0, 0])


@pytest.mark.parametrize("labels,states,options", [
    ([1., 2.], 3, {}), ([[1, 2]], 3, {}), ([1, 2], True, {}),
    ([1, 2], 0, {}), ([1, 2], 3, {"mode": "unknown"}),
    ([1, 2], 3, {"frame_indices": [0, 0]}),
])
def test_counting_rejects_ambiguous_labels_vocabulary_or_frame_map(labels, states, options):
    with pytest.raises(ValueError):
        label_counts(labels, states, **options)


@pytest.mark.parametrize("filename", ["run_animal_permutation_tests.py", "transparent_animal_permutation_analysis.py"])
def test_actual_study_loop_preserves_pooled_frames_and_full_recording_duration(filename):
    # These historical scripts execute full-cohort permutations at module level.
    # Execute only their actual aggregation loop against tiny fake sessions.
    path = Path(__file__).resolve().parents[2] / "studies/behavior/karen_studies" / filename
    tree = ast.parse(path.read_text())
    loops = [node for node in tree.body if isinstance(node, ast.For)
             and ast.unparse(node.target) == "(i, aid)"
             and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                     and call.func.id == "label_counts" for call in ast.walk(node))]
    assert len(loops) == 1
    namespace = dict(np=np, pd=pd, label_counts=label_counts, K=3, animals=["A", "B"],
                     animal_occupancy=np.zeros((2, 3)), animal_bout_rate=np.zeros((2, 3)),
                     animal_sessions={
                         "A": [SimpleNamespace(syllables=pd.DataFrame({"syllable": [0, 0, -5, 1, 1, 0]})),
                               SimpleNamespace(syllables=np.array(["1", "bad", "1", "8"]))],
                         "B": [SimpleNamespace(syllables=np.array([2, 2, 0]))],
                     })
    module = ast.Module(body=loops, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    np.testing.assert_allclose(namespace["animal_occupancy"], [[3/7, 4/7, 0], [1/3, 0, 2/3]])
    if filename == "run_animal_permutation_tests.py":
        # Ten recorded frames for A and three for B, even where labels are invalid.
        np.testing.assert_allclose(namespace["animal_bout_rate"], [[960, 1440, 0], [1600, 0, 1600]])
