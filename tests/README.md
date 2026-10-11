# Test discovery and validation

`make test` discovers `tests/` and runs the fast functional suite. Synthetic
checks live in `unit/`, `integration/`, `regression/` and `architecture/`;
`golden/` holds bounded, reviewed fixtures. Moving these files does not change
their assertions or remove the established functional coverage.

`empirical/` contains tests requiring workstation datasets or the modeling
backend. Pytest ignores that directory before importing modules unless
`--empirical` is supplied. Marker filtering alone cannot prevent imports during
collection. Empirical modules defer dataset loading and optional backend imports
until execution, so this command checks discovery without fitting models:

```bash
python -B -m pytest tests/empirical --empirical --collect-only -q \
  -m 'needs_data or gpu or slow' -p no:cacheprovider
```

To run a particular empirical check, supply its path, `--empirical` and an
explicit marker expression such as `-m needs_data`. Inspect its inputs and
runtime before running it. `FPCODE_CPT_DATA` supplies the CPT dataset directory
for `test_simple_session.py`.

Research notebooks, manual visualization checks and examples now live with
their study under `studies/photometry_exploration/` or
`studies/multiview_experiments/`. Their original filenames and source are
preserved; `registry/TEST_MOVES.tsv` records their destinations.

CPU CI uses `configs/environments/ci-cpu-requirements.txt`, which pins the active
dependency closure observed in the validated Python 3.10 environment. It runs
the functional suite, architecture audit, empirical collection, maintained
source lint and an installed-wheel check. GPU and empirical scientific
validation remain explicit additional gates.

`make lint` enforces full rules on the installed library and core entry points,
and checks all maintained study, test and operations source against the explicit
debt inventory in `registry/LINT_BASELINE.json`. New diagnostics fail. Existing
diagnostics remain visible and can be removed from the inventory after repair;
CI cannot refresh it. `make lint-baseline` is an explicit review operation.

The wheel includes `py.typed`. `make typecheck` is the initial optional contract
check, using the `typing` extra. This gate is separate from CPU CI until the
annotations and checker are verified locally; the marker alone does not claim
complete interface typing.
