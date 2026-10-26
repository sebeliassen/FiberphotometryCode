# FiberphotometryCode

Computational neuroscience platform supporting multi-camera posture tracking, unsupervised behavioral syllable segmentation (Keypoint-MoSeq), and synchronized multi-channel fiber photometry analysis. Developed in the Gether Laboratory, Department of Neuroscience, University of Copenhagen (Panum Institute).


## Current code and documentation

The reusable implementation is in `src/fpcode`, organized into contracts,
adapters, alignment, tracking, photometry, behavior, KPMS, statistics,
visualization, workflows and installed CLI modules. Study-specific analyses
live in `studies/<project>`, with methods and cohort policy kept alongside their
project. Historical import/file adapters, including `core`, `pipelines`,
`syllable_generation` and `fiberphotometry`, live under `compat`. Current commands
use the installed `fpcode` CLI. Ignored historical storage may remain locally
under `archive`; its [retrieval index](docs/reports/ARCHIVE_INDEX.md) is documentation.

Start with the [documentation index](docs/README.md) for current guidance and
preserved historical reports. The agreed architecture is documented in
[the capability contract](docs/architecture/UNIT_CONTRACT.md); configuration
ownership is described in [configs/README.md](configs/README.md), and backup and
rollback procedures are in [recovery.md](docs/recovery.md). The existing
[October 9 handoff](docs/reports/handoff.md) remains a dated record.

For the current seqNMF project, start with the
[Karen analysis guide](studies/karen_pff_vs_monomer/analysis/README.md),
[delivered theme report and recipe](studies/karen_pff_vs_monomer/analysis/EXPLORATORY_THEME_DELIVERY_20261021.md)
and [atlas inputs/methods](studies/karen_pff_vs_monomer/analysis/BEHAVIOR_ATLAS.md).
The source from the original modular and Karen delivery branches is integrated;
new work should start from `main`. The reference and shared seqNMF solvers retain
their different numerical policies. Existing exploratory findings are not new
validated disease endpoints.

Empirical inputs/models remain under `data` or configured external storage;
generated reports/media remain under their original `outputs` locations. The
preserved delivery checkout has its own outputs. `FP_DATA_ROOT`, when used, is a
mirror root containing `data/`. Source consolidation does not move these products
or refit models. See [recovery](docs/recovery.md) for source and empirical limits.

Coding/research agents should start with [AGENTS.md](AGENTS.md), the
[navigation index](ARCHITECTURE_INDEX.md) and [current task state](docs/tasks/CURRENT.md).
Task recipes for [validation](docs/agent-workflows/validation.md),
[KPMS execution/optimizations](docs/agent-workflows/kpms.md) and
[scientific media](docs/agent-workflows/visualization.md) point to supported APIs.
Run `python -B -m fpcode.cli.doctor --json --require cpu` for read-only setup
diagnostics; a refreshed installation also provides `fpcode-doctor`.

```bash
# Installation: optional model/rendering dependencies remain explicit.
python -m pip install -e '.[test,kpms]'
make test PYTHON=/path/to/the/project/environment/bin/python
make architecture PYTHON=/path/to/the/project/environment/bin/python
fpcode-train --help
fpcode-evaluate --help
fpcode-fuse --help
fpcode-render --help
```

The installed wheel provides `fpcode` and its CLI commands. Historical namespaces
under `compat` require the source checkout because some aliases target study or
vendor modules. For old notebooks/scripts or named historical loaders, activate
these paths from the repository directory before starting Python:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/compat:$PWD/vendor/keypoint_moseq_syll${PYTHONPATH:+:$PYTHONPATH}"
```

The paths expand to this checkout and remain valid after changing directories.
Use a fresh Python process when switching checkouts. This exposes retained
aliases; their presence does not prove that every historical artifact or script
works with current dependencies. Prefer `fpcode` imports and CLI commands for
new work. Tests explicitly exercise this checkout setup in a separate process;
pytest's own path configuration is not an installation mechanism. Physical
historical CLI files are under `compat/pipelines/<verb>.py`; use the installed
commands above or `python -m fpcode.cli.<verb>` for maintained workflows.

The default test suite uses tiny functional fixtures. Empirical scientific
reproduction and GPU model fitting are separate checks. Existing models, videos,
recordings and session files live under `data`, including `data/raw` for
project recordings. Compatibility paths preserve legacy consumers where needed;
Git snapshots do not cover these ignored empirical files.

For an explicit real-data GPU check, see [runtime validation](docs/architecture/VALIDATION.md):

```bash
make validate-two-sessions PYTHON=/path/to/the/project/environment/bin/python \
  VALIDATION_PROJECT=/path/to/pilot/project
```

This samples two sessions and runs exactly two AR plus two SLDS updates in a
fresh process, preserving its evidence under `outputs/<run-id>`.

Unsupervised state names follow `.agents/rules/nomenclature.md`: use objective
kinematic, spatial and statistical descriptors.
