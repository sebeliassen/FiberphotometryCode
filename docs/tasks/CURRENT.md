# Current repository state — consolidation

Updated 26 October 2026. See the single
[consolidation checklist](CONSOLIDATION_AND_MAIN_PLAN_20261021.md).
Scope is the original five branches' functionality; no new research features.

## Delivered by this change

- Original modular (`31326cc`) and Karen delivery (`2c43b75`) histories are
  integrated on main. Original primary notes are preserved on origin under
  `codex/pre-consolidation-20261026` (`16c5337`).
- Removed 18 disposable worktrees, 20 merged local branches and 19 remote
  branches. Post-merge cleanup retires the temporary runtime/development names;
  the primary checkout uses main and the delivery viewer keeps its frozen tag.
- Removed 21 redundant inventory/checkpoint files and shortened stale notes.
  Original scientific reports, recovered-source manifest and numerical fixtures
  remain. Detailed historical records are available through Git and old outputs.
- Root core/pipelines adapters now live together under compat; duplicate metrics
  and record implementations forward to canonical owners. Maintained consumers
  use canonical imports. Old imports require the README's compat path setup.
- Archive retrieval index is docs/reports/ARCHIVE_INDEX.md. Ignored archive,
  data/models, output products, external targets and both viewers are preserved.

## Validation

Lab Python: /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python; bytecode disabled,
numerical threads capped at two. Full backend-free CPU suite: 1,298 passed with
six existing deprecation warnings (55.23 s). Architecture: 202 modules, no
violations/cycles. Lint: 3,060 accepted diagnostics, no new debt; inventory unchanged.
Navigation: 149 directories synchronized. Fresh wheel/resources and all six CLI
help/doctor checks passed outside checkout. No empirical fitting/cohort/media run.
GitHub Actions records required PR and main checks; its ffmpeg-dependent skip is
covered by the local suite. Source backup/restore uses the existing original
bundle plus final_consolidation_20261026.bundle under outputs/consolidation_20261021.
These source bundles exclude ignored/external empirical payloads and browser feedback.

## Handoff

Start from main and the README's setup and seqNMF links. Use one task branch for
new work and remove it after merging. Keep compat as a thin boundary for existing
historical names, not an implementation owner. Scientific follow-up and empirical
backup gaps are separate from this completed source consolidation.
