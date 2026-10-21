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

## Foundation candidate validated

Isolated worktree .worktrees/main-foundation / codex/main-foundation merges the preserved modular checkpoint into origin/main without changing main. A clean checkout revealed20 ignored Python source files in photometry data loading and language-model data/models. registry/SOURCE_RECOVERY_20261021.json records original hashes and numerical-definition baselines. All original import sets and numerical definitions are preserved; changes are import ordering, whitespace and explicit retained module attributes. Git exceptions expose source only; test payloads remain ignored. Navigation recognizes the two named source data packages while pruning empirical directories/symlinks.

Final foundation checks:850CPU tests passed/6existingwarnings; architecture, maintained lint(no newdebt), maps149directories and installed-wheel checks passed. Eight focused recovery/navigation tests passed. No lint inventory/golden fixture was regenerated. See primary outputs/consolidation_20261021/foundation_recovered_*.log. Source checkpoints31326cc/2c43b75 and all three tags restore from the verified703830120-byte Gitbundle.

NEXT: checkpoint this source recovery, then create a stacked codex/main-seqnmf candidate and merge preserved delivery2c43b75. Resolve task-state notes without losing dated provenance. Keep main unchanged while GitHub publication remains unavailable.
