# Open-field baseline policy

`config.py` preserves the original monoamine-interactions baseline policy:
`VALID_FRAME_RANGE = (50000, 250000)`, video filenames with optional chamber
tokens, and the original Arena 1–4 chamber fallback. Its `JCFG` is a separate
instance from the juvenile, Karen and generic configurations, even though the
historical baseline and juvenile policy source was identical.

`studies.monoamine_interactions.steps.tracking.discover_sessions_hybrid` imports
this policy explicitly. It discovers sessions from video filenames and optionally
attaches matching H5 references; it receives the selected directories and project
through its context. The policy and discovery imports do not read recording
contents, load cohorts or write outputs. Synthetic tests use empty files and
verify chamber mapping, empty-animal filtering and matching references.

The retained `studies/monoamine_interactions/projects/jonathan_oft_baseline`
symlink leads here. The [policy regressions](../../tests/unit/test_project_tracking_policies.py)
preserve cohort choices, and [unified-builder tests](../../tests/unit/test_unified_builder_projects.py)
cover supported project selection. These synthetic checks do not establish
complete empirical execution or compatibility with arbitrary saved sessions.
