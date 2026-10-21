# Select validation for the change

Run checks at useful milestones, not after every small edit. Start with focused
expectations for the changed behaviour; batch required broader gates against the
completed candidate. Do not repeat successful checks without relevant changes,
failures or unresolved concerns. Documentation-only changes do not require the
full numerical suite or an installed-wheel build. Apply the root
[resource-use guidance](../../AGENTS.md) to repeated or substantial workloads,
including configurable memory caps, vectorization, bounded parallelism and
representative evidence before adopting GPU acceleration.

Use a selected interpreter, not an assumed system Python. Run `python -B -m fpcode.cli.doctor --json --require cpu` for metadata-only dependency/root diagnostics. `--input` uses the code root; `--data-input data/...` uses the data mirror root. Missing requested inputs/distributions produce exit code 1. The doctor neither decodes inputs nor initializes backends; installed metadata does not prove version compatibility.

| Change | Checks |
| --- | --- |
| Numerical/library behavior | Relevant unit/regression tests with independent expectations; full functional suite; architecture and lint |
| Historical imports/relocation | Relevant fresh-process import-order/identity and direct-script checks; filesystem references and shim registry |
| API/CLI/package resources | Relevant integration tests; help/dry-run; `make package-check` outside checkout |
| Maps/instructions/skills | Generator `--check`; local reference checks; skill frontmatter validator; realistic harmless workflow exercises |
| GPU/runtime kernels | Synthetic/state-restoration tests first, then explicitly scoped empirical validation matching the change |

```bash
export PYTHONDONTWRITEBYTECODE=1
make test-cpu PYTHON=/path/to/python
make architecture PYTHON=/path/to/python
make lint PYTHON=/path/to/python
make folder-maps-check PYTHON=/path/to/python
make package-check PYTHON=/path/to/python
```

The default suite skips `tests/empirical` before imports. `make test-cpu` additionally forbids modeling-backend imports in the test process and disables automatic third-party pytest plugin loading. `make test` runs the same functional selection without the import guard. Explicit empirical discovery with the same guard:

```bash
python -B ops/repo/cpu_check.py --collect-empirical
```

This collects only; executing test bodies requires a selected test and runtime markers. The full-cohort GPU workload is not a routine repository check. See [runtime validation](../architecture/VALIDATION.md) for bounded evidence.

The lint guard accepts an explicit historical inventory. Repair introduced diagnostics; refreshing the inventory is a separate reviewed operation grounded in the previous source. Do not infer zero debt from a pass.

Use `make folder-maps` when maintained source/config files change, then inspect the generated diff and run its check. With Git metadata, the generator includes tracked and unignored new source/config files; ignored local scripts are omitted. Data/cache directories and symlinks are pruned before traversal. Without Git metadata, it indexes the nonexcluded source tree. The check reports obsolete maps; regeneration removes those generated files while preserving other notes/source. The check verifies synchronization, not correctness of a scientific description. Report command, interpreter, working-tree/commit, outcome and material limitations in [current task state](../tasks/CURRENT.md).
