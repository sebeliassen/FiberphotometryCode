# Capability and dependency contract

The canonical source is `src/fpcode`. Libraries are grouped by scientific or
engineering capability, and split recursively when responsibilities differ.
Each leaf needs a clear API and meaningful tests; it does not need its own CLI,
configuration file or boilerplate directory template.

## Dependencies

Contracts define shapes, units, clocks, validity, events, identity and artifacts.
Adapters decode external formats. Alignment operates on explicit coordinates.
Domain algorithms consume these foundations. Workflows compose algorithms and
adapters. CLI and studies are outer consumers; the shared library never imports
studies, legacy runners, tests or archive material.

`common` contains generic infrastructure. The existing `common.schemas` facade
may re-export contracts for compatibility; it must not import domain algorithms.
Exact allowed boundaries are encoded in `ops/repo/dependencies.py` and tested in
`tests/architecture`. Relative imports, cycles, syntax errors and argument parsing
at import time are checked. Computed dynamic imports still need human review.

## Execution and configuration

Importing the base package, CLI or runtime configuration must not initialize
JAX/CUDA, train a model, parse arguments, send notifications or write artifacts.
Explicit backend calls may load optional dependencies. Runtime patches must be
scoped and restore each namespace's entry value on success and failure.

New workflows accept explicit input/output locations and scientific policies.
Historical cohort defaults are isolated behind compatibility APIs. The supplied
environment root is honored even when a destination has not been created yet.
Installed packages do not assume this workstation's home directory.

## Tests and artifacts

`make test` selects small functional tests through explicit pytest testpaths.
CPU-only does not imply fast. Empirical files, training and GPU execution require
separate explicit validation. Compatibility identity tests supplement independent
numerical expectations; they cannot prove numerical equivalence by themselves.

Small validation references live in `registry/checkpoints`; disposable outputs
live under `outputs/<run-id>`. Models and data retain their existing paths during
migration. Module length is a review signal, not a reason to split cohesive
algorithms mechanically. See `docs/README.md` for current guidance and the dated
historical records, and `docs/architecture/VALIDATION.md` for explicit runtime checks.
