# FiberphotometryCode

Computational neuroscience platform supporting multi-camera posture tracking, unsupervised behavioral syllable segmentation (Keypoint-MoSeq), and synchronized multi-channel fiber photometry analysis. Developed in the Gether Laboratory, Department of Neuroscience, University of Copenhagen (Panum Institute).


## Current code and documentation

The reusable implementation is in `src/fpcode`, organized into contracts,
adapters, alignment, tracking, photometry, behavior, KPMS, statistics,
visualization, workflows and installed CLI modules. Study-specific analyses
live in `studies`. Historical `core`, `pipelines`, `syllable_generation` and
photometry entry points remain compatibility paths during migration.

Start with the [documentation index](docs/README.md) for current guidance and
preserved historical reports. The agreed architecture is documented in
[the capability contract](docs/architecture/UNIT_CONTRACT.md); configuration
ownership is described in [configs/README.md](configs/README.md), and backup and
rollback procedures are in [recovery.md](docs/recovery.md). The existing
[October 9 handoff](handoff.md) remains a dated record.

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
recordings and session files retain their locations.

For an explicit real-data GPU check, see [runtime validation](docs/architecture/VALIDATION.md):

```bash
make validate-two-sessions PYTHON=/path/to/the/project/environment/bin/python \
  VALIDATION_PROJECT=/path/to/pilot/project
```

This samples two sessions and runs exactly two AR plus two SLDS updates in a
fresh process, preserving its evidence under `outputs/<run-id>`.

Unsupervised state names follow `.agents/rules/nomenclature.md`: use objective
kinematic, spatial and statistical descriptors.
