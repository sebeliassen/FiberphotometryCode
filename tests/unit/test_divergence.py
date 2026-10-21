"""Analytical values and exact parity with the preserved presentation implementation."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from fpcode.statistics.divergence import js_divergence

ROOT = Path(__file__).resolve().parents[2]


def test_divergence_has_analytical_bit_units_and_preserves_inputs():
    p = np.array([.25, .75])
    q = np.array([.75, .25])
    before = p.copy(), q.copy()
    assert js_divergence(p, p) == 0.
    assert js_divergence(p, q) == pytest.approx(0.18872187554086717, abs=1e-15)
    assert js_divergence([1., 0.], [0., 1.]) == pytest.approx(1., abs=1e-9)
    assert js_divergence(p * 7, q * 13) == pytest.approx(js_divergence(p, q), abs=1e-15)
    np.testing.assert_array_equal(p, before[0])
    np.testing.assert_array_equal(q, before[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("eps", [1e-12, 1e-4])
def test_divergence_matches_frozen_formula_exactly(dtype, eps):
    fixture = json.loads((ROOT / "tests/golden/js_divergence_reference.json").read_text())
    function = ast.parse(fixture["function"]).body[0]
    assert isinstance(function, ast.FunctionDef) and function.name == "js_divergence"
    namespace = {"np": np, "__builtins__": {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "preserved_js_divergence", "exec"), namespace)
    rng = np.random.default_rng(42)
    p, q = rng.uniform(-.1, 1., (2, 13, 7)).astype(dtype)
    p[0, 0] = 0
    q[0, 1] = 0
    expected = namespace["js_divergence"](p, q, eps)
    np.testing.assert_array_equal(js_divergence(p, q, eps), expected)
    np.testing.assert_array_equal(js_divergence(p, q, eps), js_divergence(q, p, eps))
