"""Independent alignment expectations plus the exact pre-refactor wrapper policy."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest

from fpcode.photometry.alignment import build_aligned_moseq_phot, combined_photometry_and_syllables

ROOT = Path(__file__).resolve().parents[2]


def session(name="one", present=True):
    return SimpleNamespace(session_id=name,
                           syllables=pd.DataFrame({"syllable": [1, 2, 3, 4]}, index=[10, 11, 12, 13]),
                           photometry={"470": pd.DataFrame({"frame_idx": [10, 12],
                                                            "SystemTimestamp": [0., 2.],
                                                            "signal_0": [1., 5.]})} if present else {})


def fixed():
    return pd.DataFrame({"frame_index": [12, 10, 14], "syllable": [3, 1, 9]}, index=[7, 8, 9])


def test_alignment_preserves_native_frame_order_missing_support_and_inputs():
    first, missing = session(), session("missing", False)
    frame = fixed()
    before = frame.copy(deep=True), first.photometry["470"].copy(deep=True)
    results = build_aligned_moseq_phot([first, missing], [frame, frame])
    assert list(results[0].index) == [0, 1, 2]
    np.testing.assert_array_equal(results[0].frame_index, [12, 10, 14])
    np.testing.assert_array_equal(results[0].signal_0, [5., 1., np.nan])
    np.testing.assert_array_equal(results[0].SystemTimestamp, [2., 0., np.nan])
    assert results[1][["SystemTimestamp", "signal_0"]].isna().all().all()
    pd.testing.assert_frame_equal(frame, before[0])
    pd.testing.assert_frame_equal(first.photometry["470"], before[1])


def test_missing_photometry_retains_historical_fallback_columns():
    result, = build_aligned_moseq_phot([session(present=False)], [fixed()])
    assert list(result.columns) == ["frame_index", "syllable", "SystemTimestamp",
                                    "signal_0", "signal_1", "signal_2", "signal_3"]
    assert result.iloc[:, 2:].isna().all().all()


def test_alignment_requires_explicit_frame_coordinates():
    with pytest.raises(KeyError, match="frame_index"):
        build_aligned_moseq_phot([session()], [fixed().drop(columns="frame_index")])


def test_restored_wrapper_matches_original_source_and_retains_legacy_alias():
    fixture = json.loads((ROOT / "tests/golden/aligned_photometry_reference.json").read_text())
    function = ast.parse(fixture["function"]).body[0]
    assert isinstance(function, ast.FunctionDef) and function.name == "build_aligned_moseq_phot"
    namespace = {"np": np, "pd": pd, "List": List,
                 "combined_photometry_and_syllables": combined_photometry_and_syllables}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pre_refactor_alignment", "exec"), namespace)
    sessions = [session(), session("missing", False)]
    frames = [fixed(), fixed()]
    expected = namespace["build_aligned_moseq_phot"](sessions, frames)
    actual = build_aligned_moseq_phot(sessions, frames)
    for new, old in zip(actual, expected):
        pd.testing.assert_frame_equal(new, old)
    code = f"""
import sys
import warnings
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'compat')!r}]
warnings.simplefilter('ignore', DeprecationWarning)
from fpcode.photometry.alignment import build_aligned_moseq_phot as canonical
from syllable_generation.analysis.replications.datta_utils import build_aligned_moseq_phot as legacy
assert legacy is canonical
assert not {{'jax', 'keypoint_moseq', 'torch'}} & set(sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
