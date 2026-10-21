# Consolidation execution state — 21 October 2026

## Scope and recovery

Executing [the staged plan](CONSOLIDATION_AND_MAIN_PLAN_20261021.md). Primary branch is codex/modular-refactor; delivery worktree is .worktrees/karen-seqnmf-delivery. Do not reset, clean or switch either original worktree. Preserve running viewer processes 3693918 and 3768245.

Source-only snapshots, per-file SHA256 manifests, source restore evidence, initial artifact inventory and detailed check logs are in outputs/consolidation_20261021 in the primary checkout. Snapshots include 58 ignored potential source files from the primary checkout for recovery only; these files have not been added to active source. They need individual classification. No empirical artifact has moved, and backup coverage for large artifacts/browser feedback remains unverified.

## Completed checks

- Refreshed origin refs: main67468c1, modular3ec30fb, legacyb83298f; local history relationships unchanged.
- Primary CPU846passed/6warnings; delivery CPU933passed/6warnings.
- Architecture, maintained lint and installed-wheel checks passed in both worktrees.
- Delivery maps passed; primary had two stale ops maps, regenerated and rechecked successfully. No numerical source changed for that repair.
- Interpreter: /home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python; numerical pools limited to two threads. See baseline_checks.json and logs for exact commands.
- Source restore exercise passed for named regular files in both snapshots; directory symlinks were not followed.

## Remote publication limitation

Git fetch works. A dry-run push failed because GitHub HTTPS credentials are unavailable: could not read Username. No remote refs changed. Continue preparing tested local integration branches; publication/PR creation and remote main promotion need an authenticated route. Do not work around branch protection or rewrite remote history.

## Next operation

Checkpoint explicitly reviewed source files in the original branches, create preservation tags and a verified all-refs Git bundle, then prepare codex/main-foundation in a separate worktree from origin/main. Keep main unchanged until the integration candidate has passed its gates and can be reviewed/published. Later local candidates can be stacked while remote publication remains unavailable.
