# Capability and dependency contract

The canonical shared library is `src/fpcode`. Libraries are grouped by scientific
or engineering capability and split when responsibilities differ. A module needs
a clear API and meaningful validation, rather than its own boilerplate CLI or
configuration directory.

## Repository ownership

| Location | Responsibility |
| --- | --- |
| `src/fpcode` | Reusable contracts, algorithms, adapters, workflows and installed CLI |
| `studies/<project>` | Project analyses, methods, hypotheses and scientific cohort policy |
| `configs` | Explicit reusable run profiles and calibration configuration |
| `data/{raw,sessions,models,fused,analysis_cache}` | Ignored empirical inputs, model states and caches |
| `outputs/<run-id>` | Disposable results and runtime validation evidence |
| `ops/{monitoring,dlc,compaction,repo,backup,diagnostics}` | Operational implementations, organized by purpose |
| `compat` | Live historical import paths and compatibility entry points |
| `archive` | Preserved historical source, reports and frozen bundles |
| `registry` | Source inventories, migration records and validation references |

Studies are grouped by project, without a top-level behavior/photometry split.
Domain distinctions remain useful inside a project when they clarify ownership.
Shared visualization and photometry mechanisms belong in the library; a project's
particular figures, assumptions and scientific settings remain in that project.
`ops/scripts` contains aliases for older operational callers.

## Dependencies

Contracts define shapes, units, clocks, validity, events, identity and artifacts.
Adapters decode external formats. Alignment operates on explicit coordinates.
Domain algorithms consume these foundations. Workflows compose algorithms and
adapters. CLI and studies are outer consumers; the shared library never imports
studies, historical runners, tests or archive material.

`common` contains generic infrastructure. Historical library interfaces live in
`fpcode.compat`; canonical capabilities must not acquire dependencies on those
interfaces. Capability bridges that deliberately serve an old API retain an
explicit audit exception and migration rationale until their consumers move.
The precise rules live in `ops/repo/dependencies.py` and are checked by the
architecture tests. An audit passing means the checked graph complies with those
rules; computed imports and runtime behavior still require review.

## Execution and configuration

Importing the base package, CLI or runtime configuration must not initialize
JAX/CUDA, train a model, parse arguments, send notifications or write artifacts.
Explicit backend calls may load optional dependencies. Runtime patches must
restore each namespace's entry value on success and failure.

New workflows accept explicit input/output locations and scientific policies.
Cohort defaults and calibration belong in project policy or configuration, rather
than reusable algorithms. Installed packages do not assume a workstation's home
directory. A file relocation must account for literal paths as well as imports:
import aliases alone do not repair a script's filesystem assumptions.

## Tests, compatibility and scientific evidence

The default test suite uses small functional fixtures; expensive empirical and
GPU checks are separate. Compatibility identity tests supplement independent
numerical expectations but cannot prove numerical equivalence by themselves.
Historical pickle reconstruction is demonstrated only for artifacts actually
checked. Retaining a class import path does not establish compatibility with
all historical serialized objects.

Small validation references live in `registry/checkpoints`. Each explicit runtime
check stores evidence under `outputs/<run-id>`. Four GPU updates on two excerpts
verify those execution paths; they do not establish convergence, full-cohort
memory behavior or universal scientific invariance. See
[validation](VALIDATION.md), [recovery](../recovery.md) and the
[documentation index](../README.md) for the available evidence and its limits.
