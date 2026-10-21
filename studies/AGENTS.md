# Study ownership

Each `studies/<project>` owns scientific choices: cohort membership/exclusions, chamber and channel mapping, rate policy, calibration selection, event definitions and hypotheses. Reusable parsing/rendering/processing mechanisms should receive explicit configuration and belong in `fpcode`; the library must not import studies to discover globals.

Read the project's README and relevant scientific notes, not every historical report. Preserve explicit clock/coordinate conventions: model-row positions, native video frames and timestamps are different quantities. Never infer a 40/80 Hz conversion from recording length or the largest frame index.

Use objective descriptors for unsupervised components; see [nomenclature](language_models/mds/01_theory_and_strategy/04_objective_ethological_nomenclature_guidelines.md). Keep validated scientific policy changes separate from mechanical relocation.

Resolve inputs through explicit roots/configuration. A repaired import does not repair literal paths, `__file__.parents` depth or pickle reconstruction. Avoid whole-tree global-config mutation, bare `config` imports and silent fallback to another cohort.

Record selected inputs, settings, outputs and evidence. Start with relevant synthetic tests; use empirical data/GPU only within the user's requested scope. Do not unpickle arbitrary data merely to inventory it.
