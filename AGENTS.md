# Working in FiberphotometryCode

Start with [current task state](docs/tasks/CURRENT.md) and the [capability contract](docs/architecture/UNIT_CONTRACT.md). Historical reports describe their dated checkpoint, not the current completion status. Use [the navigation index](ARCHITECTURE_INDEX.md) or `rg` to find the owner; read only relevant folder maps and source.

## Ownership and specialized guidance

- Reusable implementation: `src/fpcode`; read [library guidance](src/fpcode/AGENTS.md) before changing it.
- Scientific policy, cohort choices and analyses: `studies/<project>`; read [study guidance](studies/AGENTS.md).
- Operations: `ops/<purpose>`; read [operations guidance](ops/AGENTS.md).
- Historical namespaces: root `compat`; read [compatibility guidance](compat/AGENTS.md). `fpcode.compat` instead owns historical library implementations/profiles. `core` and `pipelines` are physical adapters.
- Empirical inputs/models belong under ignored `data` or configured external storage; generated products belong at explicitly selected output locations. Do not stage raw recordings, pickles or model payloads with source.

## Setup and checks

Select an interpreter with the required dependencies. Set `PYTHONDONTWRITEBYTECODE=1` for repository checks. The local lab example is `/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python`; other contributors should use their own environment. Do not upgrade the shared lab environment incidentally.

```bash
python -B -m fpcode.cli.doctor --json --require cpu
make test-cpu PYTHON=/path/to/python
make architecture PYTHON=/path/to/python
make lint PYTHON=/path/to/python
make folder-maps-check PYTHON=/path/to/python
make package-check PYTHON=/path/to/python
```

Choose relevant tests first using [validation guidance](docs/agent-workflows/validation.md); run the complete functional suite when shared behavior changes. Empirical collection needs `--empirical`; empirical execution additionally needs explicit runtime markers and requested inputs/scope. A lint pass includes accepted historical debt; do not regenerate its inventory to hide new diagnostics. Regenerate derived maps with `make folder-maps` after relevant source changes.

## Validation cadence and resource use

- Batch validation around meaningful implementation milestones. Use focused checks during development and run required broader checks once against the final candidate. Repeat a passed check when relevant code changes, a failure or unresolved concern justifies it; do not rerun every Make target after each edit. Select package and navigation checks for changes that affect their owners.
- Frequently repeated or substantial operations should use available resources when this materially improves end-to-end runtime. Prefer vectorized algorithms, batched calculations and reuse of invariant inputs/results over repeated Python loops, loading, fitting or recomputation. Use exact analytical calculations where they replace simulations without changing the scientific estimand.
- Set configurable RAM caps and batch/chunk sizes for large workloads; budget intermediate arrays and worker copies, leave headroom and stream results when appropriate. Select CPU threads/processes from actual availability and avoid oversubscription from nested worker and BLAS/OpenMP pools. Configure thread limits before numerical imports and record actual settings.
- Consider GPU acceleration for repeated suitable numerical workloads when available VRAM, transfer costs and precision requirements allow a meaningful speedup. Check on a representative bounded workload before committing a large run. Keep a suitable CPU path; do not initialize accelerators during import/help/metadata checks or upgrade the shared environment incidentally.
- Performance changes must preserve units, clocks, masks, exclusions, endpoint definitions and numerical accuracy. Record dtype/tolerances, seeds, resource caps and representative timing evidence. Reproducible simulations need not be bitwise identical across different backends, but changes in results require an explained, validated numerical or Monte Carlo tolerance.

`FIBERPHOTOMETRY_ROOT` selects the code/config root. `FP_DATA_ROOT` is a mirror root **containing `data/`**, not the `data/` directory itself. For example, `resolve_data_path("data/models/example")` joins that mirror root. Test environment overrides in fresh processes.

## Scientific and execution invariants

Preserve units, time/frame conventions, masks, missing-frame semantics, cohort exclusions and historical numerical policies. Use objective physical/kinematic/statistical descriptions for unsupervised states; see [nomenclature](studies/language_models/mds/01_theory_and_strategy/04_objective_ethological_nomenclature_guidelines.md).

Import/help/metadata checks must not initialize JAX/CUDA, fit models, send notifications or write empirical artifacts. Inspect dry-run support per command; do not infer that every maintenance script has it. Existing user authorization governs execution; a request to inspect code does not imply a cohort training or maintenance run.

Keep numerical golden fixtures and identity assertions substantive. Record exact checks and their limits: a synthetic pass, static audit or a few GPU updates is not universal scientific equivalence.

For work spanning sessions, update `docs/tasks/CURRENT.md` with the baseline, changes, completed checks, unresolved work and next step. Preserve unrelated uncommitted work. Never interpret unstaged move deletions as permission to clean/reset the tree.
