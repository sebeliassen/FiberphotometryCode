"""
tests/test_photometry_alignment_equivalence.py

Equivalence test verifying that fpcode.photometry.alignment matches the behavior of the legacy datta_utils module.
"""

import numpy as np
import pandas as pd
import pytest

from fpcode.photometry.alignment import (
    build_fixed_moseq_dfs,
    combined_photometry_and_syllables,
)


class MockSession:
    def __init__(self):
        frames = 50
        # syllables dataframe with frame index
        self.syllables = pd.DataFrame(
            {"syllable": np.random.randint(0, 10, size=frames)},
            index=np.arange(frames),
        )
        # photometry dataframe with 20Hz samples
        self.photometry = {
            "470": pd.DataFrame({
                "SystemTimestamp": np.linspace(0, 5, 25),
                "frame_idx": np.linspace(0, 48, 25, dtype=int),
                "signal_0": np.sin(np.linspace(0, 3.14, 25)),
                "signal_0_zscore": np.cos(np.linspace(0, 3.14, 25)),
            })
        }


def test_combined_photometry_and_syllables_synthetic():
    session = MockSession()
    merged = combined_photometry_and_syllables(session, freqs=["470"])

    assert isinstance(merged, pd.DataFrame)
    assert len(merged) == len(session.syllables)
    assert "SystemTimestamp" in merged.columns
    assert "signal_0" in merged.columns
    assert "signal_0_zscore" in merged.columns
    # Check interpolation filled gaps
    assert merged["signal_0"].isna().sum() == 0


def test_legacy_datta_utils_shim_equivalence():
    with pytest.deprecated_call():
        import syllable_generation.analysis.replications.datta_utils as legacy_mod

    session = MockSession()
    canonical_df = combined_photometry_and_syllables(session, freqs=["470"])
    shim_df = legacy_mod.combined_photometry_and_syllables(session)

    pd.testing.assert_frame_equal(canonical_df, shim_df)
