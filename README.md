# FiberphotometryCode

Computational neuroscience platform supporting multi-camera posture tracking, unsupervised behavioral syllable segmentation (Keypoint-MoSeq), and synchronized multi-channel fiber photometry analysis. Developed in the Gether Laboratory, Department of Neuroscience, University of Copenhagen (Panum Institute).


## Current code and documentation

The reusable implementation is in `src/fpcode`, organized into contracts,
adapters, alignment, tracking, photometry, behavior, KPMS, statistics,
visualization, workflows and installed CLI modules. Study-specific analyses
live in `studies/<project>`, with methods and cohort policy kept alongside their
project. Historical `core`, `pipelines`, `syllable_generation` and photometry
imports are retained in `compat`; `archive` contains preserved historical material.

Start with the [documentation index](docs/README.md) for current guidance and
preserved historical reports. The agreed architecture is documented in
[the capability contract](docs/architecture/UNIT_CONTRACT.md); configuration
ownership is described in [configs/README.md](configs/README.md), and backup and
rollback procedures are in [recovery.md](docs/recovery.md). The existing
[October 9 handoff](docs/reports/handoff.md) remains a dated record.

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
