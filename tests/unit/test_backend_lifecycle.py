from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from fpcode.kpms.backend.configuration import CONFIG, set_batch_sizes
from fpcode.kpms.backend.patch_registry import PatchSet, temporary_patches
from fpcode.kpms.backend.sizing import precompute_optimal_batch_config


def test_nested_patches_restore_each_namespace_and_exception():
    first, second = SimpleNamespace(fn="first"), SimpleNamespace(fn="second")
    with temporary_patches([(first, "fn", "outer"), (second, "fn", "outer")]):
        with pytest.raises(RuntimeError, match="failure"):
            with temporary_patches([(first, "fn", "inner")]):
                assert first.fn == "inner"
                raise RuntimeError("failure")
        assert first.fn == second.fn == "outer"
    assert first.fn == "first" and second.fn == "second"


def test_partial_patch_install_rolls_back():
    target = SimpleNamespace(fn="original")
    with pytest.raises(AttributeError):
        PatchSet([(target, "fn", "patched"), (target, "absent", "bad")]).apply()
    assert target.fn == "original"


def test_batch_overrides_are_atomic_and_boolean_settings_work():
    before = asdict(CONFIG)
    try:
        set_batch_sizes(use_numba_hmm=False, discrete_bs=3)
        assert CONFIG.use_numba_hmm is False and CONFIG.discrete_bs == 3
        expected = asdict(CONFIG)
        with pytest.raises(ValueError):
            set_batch_sizes(discrete_bs=8, heading_bs=-1)
        assert asdict(CONFIG) == expected
    finally:
        for key, value in before.items():
            setattr(CONFIG, key, value)


def test_explicit_memory_budgets_use_independent_expected_batch_sizes():
    before = asdict(CONFIG)
    try:
        data = {"Y": np.zeros((8, 100, 5, 2))}
        model = {"params": {"pi": np.zeros((3, 3)), "Ab": np.zeros((3, 2, 5))}}
        result = precompute_optimal_batch_config(data, model, gpu_budget_gb=0.00006,
                                                 cpu_budget_gb=0.00008, verbose=False)
        assert (result.discrete_bs, result.heading_bs, result.gpu_loc_bs,
                result.scales_bs, result.continuous_chunk_size) == (3, 1, 2, 1, 3)
    finally:
        for key, value in before.items():
            setattr(CONFIG, key, value)
