# Deep Research Exploratory Experiments

**Location**: `studies/deep_research_experiments`
**Status**: `FROZEN`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Action prediction error (APE) architectures, latent space embeddings, and experimental sequence representations.

## 2. Inputs
Syllable sequence tokens and latent behavioral factors

## 3. Outputs
Predictability curves and exploratory embeddings

## 4. Loading and diagnostics

The study's scientific methods remain frozen. Infrastructure imports now use the
explicit `studies.replications_datta.panel_1z.lib` owners. Importing its loader,
diagnostics or analysis help does not select or open a cohort.

Select a trusted session pickle explicitly when calling the loader:

```python
from studies.deep_research_experiments.utils.loaders import load_sessions

sessions = load_sessions("amph", apply_filter=True, pkl_path="/selected/trusted.pkl")
```

Without `pkl_path`, discovery retains the historical newest project-matching
pickle policy and generic `*sessions*all_data*.pkl` fallback. It looks under
`resolve_data_path("data")`; `FP_DATA_ROOT` is the mirror root **containing data/**.
Set root overrides before starting a fresh Python process. Cohort exclusions,
metadata standardization, enrichment, the optional 1 Hz filter and AMPH's
five-minute post-injection delay retain their original policies.

From a checkout activated as described in the repository README, explicitly run:

```bash
python -m studies.deep_research_experiments.utils.check_project_attr
python -m studies.deep_research_experiments.utils.summarize_regions
python -m studies.deep_research_experiments.action_prediction_error.run_ppe_analysis --help
```

The first two commands intentionally load IFDN/AMPH when executed; importing them
is inert. Historical compatibility files still forward to their guarded `main`
functions. Tests use controlled synthetic sessions and do not validate historical
pickles or reproduce cohort analyses. PPE accepts selected paths as described
below; omitted paths retain the historical discovery and output defaults.

## PPE entry points

The canonical PPE driver exposes `main(argv=None)` and retains its original CLI
flags and analysis defaults. The historical leaf forwards to that same canonical
module; its compatibility parent retains its physical search path so ordinary
imports reach the leaf proxy. Canonical and historical direct files support
`--help` from another working directory, activating this checkout only through
explicit execution and restoring only paths they added on success/help/failure.

Select a trusted session pickle and output directory for an explicit execution:

```bash
python -m studies.deep_research_experiments.action_prediction_error.run_ppe_analysis \
  --sessions-pkl /selected/trusted.pkl --output-dir /selected/ppe-output
```

The API exposes the same optional paths as keyword-only `pkl_path` and
`output_dir` arguments to `run_analysis`. Omitting either retains its historical
default. An explicitly selected input never falls back to discovery when missing;
an empty input string selects the current directory and fails as an input file.
Empty results create no output directory or files. Nonempty results use the
selected directory for the CSV and both plots, retaining their original names
and suffix behavior.

Running without `--help` loads trusted pickle data; `--test-mode` still loads
sessions and is not a dry run. AMPH selection, filtering, frame/time, regression,
slope and inference policies remain frozen. Path selection does not validate
historical artifacts or scientific results. Sibling diagnostics and saved-output
analyses retain their own execution obligations; importing PPE does not run them.
