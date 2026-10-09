# Recursive Modularity Unit Contract

**Document**: `docs/architecture/UNIT_CONTRACT.md`  
**Applies To**: All packages in `src/fpcode/`, all experiment studies in `studies/`, and all operational tools in `ops/`.  
**Status**: Active Standard  

---

## 1. Principle of Recursive Modularity

In this repository, "recursive modularity" means that every directory containing Python code must adhere to the **same uniform structural and behavioral contract**, regardless of whether it is a core library package (`src/fpcode/`), an operational utility (`ops/`), or a domain research experiment (`studies/`).

Each directory is a self-contained, auditable **Unit**. A unit must be fully comprehensible, testable, and runnable without requiring tribal knowledge, arbitrary path adjustments (`sys.path.append`), or implicit file system side-effects.

```text
<unit-root>/
├── README.md               # Unit documentation, purpose, status, and usage
├── __init__.py             # Clean public API; zero side-effects at import time
├── run.py (or __main__.py) # Thin CLI entry point accepting declarative configuration
├── config.yaml             # Canonical configuration schema / defaults
└── tests/                  # Mirrored unit tests under tests/<relative-path>/
```

---

## 2. The Seven Pillars of the Unit Contract

### 2.1. Pillar 1: Purpose & Status Documentation (`README.md`)
Every unit folder **must** contain a `README.md` defining:
1. **Title and Purpose**: 1–2 paragraphs summarizing the biological or engineering goal.
2. **Lifecycle Status**: Exactly one of:
   - `ACTIVE`: Actively developed and maintained.
   - `FROZEN`: Completed experiment or publication artifact. Maintained for bit-for-bit reproducibility; no new features.
   - `LEGACY`: Superseded or slated for deprecation. Point to canonical successor.
3. **Inputs**: Explicit data schemas, input files, or upstream dependencies required.
4. **Outputs**: Generated artifacts, plots, or processed tensors.
5. **How to Run**: A concrete CLI command or python snippet showing how to execute the unit.

---

### 2.2. Pillar 2: Clean Public API (`__init__.py`)
1. **Explicit Exports**: Define `__all__` listing all intended public functions and classes.
2. **Zero Side-Effects at Import**: Importing the package (`import <unit>`) must **never**:
   - Parse command-line arguments (`argparse.parse_args()` or `sys.argv`).
   - Execute heavy computation, matrix factorization, or model fitting.
   - Perform network I/O, database connections, or send push notifications (`send_ntfy`).
   - Write files to disk or mutate global variables.
3. **No `sys.path` Manipulation**: Never invoke `sys.path.append()`, `sys.path.insert()`, or path hacks. All imports must resolve through standard editable package installation (`pip install -e .` for `fpcode`).

---

### 2.3. Pillar 3: Declarative Configuration & Thin Entry Points
1. **Configuration Over Hardcoding**:
   - Zero hardcoded paths (especially `/home/fsp585/`).
   - All hyperparameters, file paths, cohort IDs, and seeds must live in declarative YAML configurations (stored under `configs/` or local `config.yaml`).
2. **Thin Entry Point**:
   - Entry points (`run.py`, `__main__.py`, or `pipelines/*.py`) should be thin orchestrators (<100 lines) that:
     1. Load and validate the YAML configuration against schemas (`fpcode.common.schemas`).
     2. Invoke pure library functions from `fpcode`.
     3. Record execution provenance via `fpcode.common.run_record`.

---

### 2.4. Pillar 4: Strict Downward-Only Layering
Dependencies and imports must strictly flow **downward**:

$$\text{studies/} \longrightarrow \text{pipelines/} \longrightarrow \text{src/fpcode/} \longrightarrow \text{stdlib / vendor}$$

- **Rule 1**: `studies/` may import from `fpcode.*`, `pipelines.*`, and `ops.*`.
- **Rule 2**: `src/fpcode/` may **NEVER** import from `studies/`, `notebooks/`, or legacy test scripts.
- **Rule 3**: `src/fpcode/common/` must have zero internal dependencies on domain logic (`fpcode.kpms`, `fpcode.photometry`, `fpcode.tracking`).
- **Enforcement**: Automated in `tests/test_unit_contracts.py`.

---

### 2.5. Pillar 5: Standard Output Routing
1. **Output Root**: All runtime artifacts, figure plots, and extracted summaries must route through the environment-controlled directory `$FP_DATA_ROOT` or `outputs/`.
2. **No Repository Pollution**:
   - Unit runs must never dump loose logs, `.h5` files, `.pkl` files, or `.png` images directly into the source tree or workspace root.
   - Any temporary caches must use `scratch/` or `tempfile`.

---

### 2.6. Pillar 6: Mirrored Test Coverage
Every code unit must have a corresponding test suite located under `tests/` mirroring the unit's relative path:
- Code at `src/fpcode/behavior/kinematics.py` $\longleftrightarrow$ Test at `tests/behavior/test_kinematics.py`
- Code at `studies/behavior/language_models/` $\longleftrightarrow$ Test at `tests/studies/test_language_models.py`
- Tests must execute under standard fast CPU pytest runs without requiring multi-gigabyte raw video data unless marked with `@pytest.mark.needs_data`.

---

### 2.7. Pillar 7: Module Size & Single Responsibility Cap
1. **500-Line Threshold**: Any single module exceeding **500 lines** of code is automatically flagged during metrics audits.
2. **Decomposition Protocol**: Modules exceeding 500 lines should be factored into cohesive submodules:
   - Data / schema parsing (`schemas.py` or `io.py`)
   - Algorithmic mathematical kernels (`kernels.py` or `engine.py`)
   - Evaluation / metrics computation (`metrics.py`)
3. **No `utils.py` Dumping Grounds**: Generic catch-all utility modules are forbidden. Helpers must be explicitly named according to domain purpose (e.g., `timing.py`, `filtering.py`, `geometry.py`).

---

## 3. Automated Contract Enforcement

Compliance with the Unit Contract is enforced through:
- **`make test`**: Runs `tests/test_unit_contracts.py` to assert import direction, clean importability, and no top-level side effects.
- **`make metrics`**: Scans the codebase for files exceeding 500 lines, untracked shims, and hardcoded absolute paths.
- **`make shims-report`**: Displays the active registry of backward-compatibility deprecation shims.
