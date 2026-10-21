# Architecture & Navigation Index

Generated file/symbol maps support selective navigation. Start with [AGENTS.md](AGENTS.md),
[current task state](docs/tasks/CURRENT.md) and [the capability contract](docs/architecture/UNIT_CONTRACT.md).
Read the relevant map or use `rg`; there is no requirement to read every map.

## 1. Subsystem Directory Topology

| Directory | Role | Primary Responsibility |
| :--- | :--- | :--- |
| [`src/fpcode/`](src/fpcode/.folder-map.md) | **Canonical Library** | 202 Python modules; implementations and installed CLI |
| [`pipelines/`](pipelines/.folder-map.md) | **Historical Entry Points** | Adapters to `fpcode.cli`; preserve physical script execution |
| [`ops/`](ops/.folder-map.md) | **Operations & Tooling** | Verification gates, diagnostics, repository hygiene, backups, and metrics |
| [`studies/`](studies/.folder-map.md) | **Research Studies** | Domain analysis notebooks, active replication packages, and experimental units |
| [`tests/`](tests/.folder-map.md) | **Test Suite** | Synthetic functional checks; explicit empirical opt-in |
| [`configs/`](configs/.folder-map.md) | **Configuration** | Model hyperparameters, training profiles, environment locks, and backup policy |
| [`compat/`](compat/AGENTS.md) | **Historical Namespaces** | Top-level import/file adapters; distinct from historical library implementations in `fpcode.compat` |

## 2. Core Architectural Principles
- The dependency audit enforces the configured matrix, not a single total ordering of capabilities.
- Empirical inputs/models belong in ignored `data`, generated run products in `outputs`, or explicitly configured external storage. Transitional aliases remain.
- Root `compat` preserves old namespaces; `fpcode.compat` owns historical implementations/profiles. Identity and filesystem obligations both matter.
- Summaries derived from docstrings/imports are hints, not scientific policy or evidence of universal pickle compatibility.

## 3. Autonomous Agent Routing Protocol
When the owner is unfamiliar, use this route; direct symbol search is also appropriate:
1. **Zoom 1 (Root Topology)**: Read this `ARCHITECTURE_INDEX.md` to identify candidate subsystem directories.
2. **Zoom 2 (Subsystem Map)**: Open the candidate subsystem's `.folder-map.md` (e.g. `src/fpcode/.folder-map.md`) to locate the relevant subpackage.
3. **Zoom 3 (Leaf Map)**: Follow a symbol/file link into source. Listed implementation symbols are not necessarily public APIs; import summaries are static hints.

## 4. Key Developer Entrypoints & Gates
```bash
# Select an interpreter with the required dependencies; timings are environment-dependent.
make test-cpu

# 2. Architectural dependency verification (0 cycles, strict layering)
make architecture

# 3. Maintain source code lint checks
make lint

# 4. Verify generated maps (pre-commit and CPU CI)
make folder-maps-check

# 5. Regenerate all hierarchical folder maps
make folder-maps
```
