# Consolidation execution state — 21 October 2026

## Current source and Git layout

The original four-PR integration stack and the checkpoint-import/handoff followup
PR #6 are merged into main at `50a722c6a129f21769d3b454f632729e8f2c0cca`. New work starts
from current main, with one main-based branch/worktree per coherent change. On
Woody, .worktrees/main-integration verifies main without switching the preserved
primary modular or Karen delivery checkouts and their running review viewers.
Original research branches and source checkpoints remain recoverable.

| PR | Actual main merge | Previous main | Reviewed/tested head |
| --- | --- | --- | --- |
| [#2](https://github.com/sebeliassen/FiberphotometryCode/pull/2) | `41ec464` | `67468c1` | `aac9e77` |
| [#3](https://github.com/sebeliassen/FiberphotometryCode/pull/3) | `ecb9f67` | `41ec464` | `78e2dfd` |
| [#4](https://github.com/sebeliassen/FiberphotometryCode/pull/4) | `c053117` | `ecb9f67` | `17d1381` |
| [#5](https://github.com/sebeliassen/FiberphotometryCode/pull/5) | `1ceb47b` | `c053117` | `7399ad6` |
| [#6](https://github.com/sebeliassen/FiberphotometryCode/pull/6) | `50a722c` | `1ceb47b` | `0c73ed3` |

Every merge has two preserved parents and a tree identical to its passing head.
The next candidate included current main before its checks. GitHub automatically
marked the older overlapping PR #1 merged once its commits reached main. No
squash, rebase, force push, branch deletion or protection bypass was used.

## Completed implementation

- Preserved original source as checkpoints31326cc/2c43b75, source-only snapshots
  and restore-tested Git bundles. Recovered20 accidentally ignored Python files,
  with narrow ignore exceptions and numerical-definition checks.
- Integrated the reusable framework, capability ownership and validation tools,
  then the preserved Karen seqNMF/theme/gallery source and complete task record.
- Migrated8 CLI launchers,6 JS-statistic callers,6 photometry callers and now3
  checkpoint-patch imports to canonical owners, preserving their policies.
- Foundation review caught the missing historical alignment helper and omitted
  legacy setup instructions. The exact pinned wrapper and substantive independent
  tests were brought forward before the first main merge. Setup now distinguishes
  the canonical wheel from explicitly activated checkout aliases.
- Preserved all additive package and compatibility assertions during conflicts;
  regenerated derived maps. A hosted scheduler fixture needed more dispatch/poll
  headroom; its deliberate timeout and every result/STOP/ledger assertion remain.
- Three final runner migrations change only import modules and narrow import-order
  lint comments. All17 function/class ASTs and all other normalized module ASTs
  match their originals. Existing golden fixtures and lint baseline are intact.

## Validation and scientific limits

Exact commands, interpreters, source hashes and results are in
registry/checkpoints/main_promotion_20261021.json. Lab checks use
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python with bytecode disabled and
numerical pools capped at two. An isolated pinned CPU environment provides an
independent installation; the shared lab environment was not upgraded. The
outside-checkout wheel check verifies resources, CLI entry points and origins.
Hosted CI has a known ffmpeg-dependent synthetic-video skip, separately exercised
in the local validation environments; report that limit with its exact counts.

The6 frozen runtime symbols across3 modules retain their source hashes. Installed
library and candidate study origins are checked separately. This is source and
fixture evidence; no empirical fit, replay, export, conversion, media generation
or new statistical inference was performed during consolidation. The scientific
interpretation and held-out validation limits of the Karen atlas remain unchanged.

## Recovery and remaining gates

Primary outputs/consolidation_20261021 contains source manifests/snapshots,
preserved_all_refs.bundle, incremental milestone bundles, restore evidence,
promotion_20261021.json and detailed logs. Git does not back up ignored recordings,
models, media, browser-local feedback or external symlink targets. Their recovery
coverage still needs verification before any physical move or deletion.

Root core/pipelines/compat/archive remain. The static ledger records aliases,
active imports, historical notebooks/review bundles and scan limits; it cannot
prove the absence of external callers or universal pickle compatibility.

Next capabilities to review:

1. Deep-research loaders: obsolete hardcoded directory paths and bare loader/processing
   imports; diagnostic modules can load cohorts during import.
2. Karen/monoamine tracking: bare config and obsolete paths; Jonathan parser/config
   aliases require module/class identity checks across the directory symlinks.
3. Session export utilities: reconcile shim targets and SHIMS.tsv, missing
   legacy_support, historical densification/extrapolation and result-write policy.
4. Named trusted artifacts, physical script/runpy obligations, external archive
   links, backup coverage and exported human feedback before retiring adapters.

The code integration milestone is complete; the whole legacy retirement project
is still in progress. Do not interpret a green CPU suite as authority to delete
legacy roots, load arbitrary pickles or launch empirical maintenance workloads.

Final full suites: 958 pass in lab and isolated pinned CPU environments (six existing warnings each), all repository/package/origin gates pass. See primary `outputs/consolidation_20261021/promotion_20261021.json` for the final main workflow and `main_bundle_restore_check.json` for the latest recovery proof. Main is available in `.worktrees/main-integration`; preserved primary/delivery sources and viewers remain intact.

Final merged-main workflow also passes: [CPU checks](https://github.com/sebeliassen/FiberphotometryCode/actions/runs/39638140445). Hosted result:957 passed,1 ffmpeg-dependent skip,6 existing warnings; local suites958 pass each. The integrated source remains main `50a722c6a129f21769d3b454f632729e8f2c0cca`.
