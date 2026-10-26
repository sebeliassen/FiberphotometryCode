# Consolidation checklist

Updated 26 October 2026. This is the single remaining-work plan. Complete the
milestones in order; there is no daily schedule and no new functionality goal.
The user has about four days before handoff.

## Scope and baseline

Keep the functionality captured in the original five branches: modular snapshot
`31326cc`, Karen delivery `2c43b75`, and their ancestors. Current main `ce281cc`
already contains both source histories. Preserve useful fixes to existing
behavior, scientific policies, numerical fixtures and recovered source; remove
redundant process, duplicated implementations and obsolete development copies.
Do not roll back main merely to reduce the visible branch count.

Original audit: 24 local branches / 21 worktrees. All tips except the primary's
20 documentation/checkpoint-only commits were already ancestors of main.
Preservation tag `codex/pre-consolidation-20261026` now protects those unique
notes on origin; existing bundles retain original branch names and source.

## 1. Git consolidation

- [x] Reuse the existing integration checkout as `codex/consolidation`.
- [x] Remove 18 clean merged worktrees, after checking ignored files and running
  processes. Removed products were only disposable caches/build metadata.
- [x] Delete 20 merged local branches and 19 remote branches. Current layout is
  four local branches / three checkouts; primary and delivery viewers remain.
After merging the final change, put the primary checkout on main and retire
  the old runtime branch names. Keep the delivery viewer/products in a frozen
  tagged/detached checkout until safely movable. Normal layout: main plus at
  most one task branch, with no accumulated merged PR branches.

## 2. Remove bloat against the original source snapshots

- [x] Remove dated inventory dumps and per-change checkpoint JSON. Keep the
  original recovered-source manifest and independent numerical expectations.
- [x] Replace sprawling CURRENT and stale continuation prose with concise
  setup/status pointers. Preserve historical detail in Git and existing outputs.
- [x] Keep original seqNMF delivery code: its apparent size increase on main is
  merged pre-existing functionality, not a reason to delete analyses.
- [x] Remove duplicate historical metrics and datamodel implementations by
  forwarding to their existing canonical owners. Preserve class/pickle identity.

## 3. Consolidate the existing source owners

- [x] Route remaining maintained study imports directly to canonical owners.
- [x] Consolidate the 20 tiny root core/pipelines adapters under the existing
  compat boundary. Keep old import names usable through the documented checkout
  setup; current commands use installed fpcode CLI. Do not add another framework.
- [x] Keep root compat as a forwarding boundary where it preserves original
  functionality. Most of its 584 Python files were already proxies in the old
  source snapshot; deleting them blindly would remove old imports/file routes.
  Removing every historical spelling is not required for a coherent handoff.
- [x] Remove the tracked archive compatibility link and move its historical index
  into documentation. Leave ignored historical payloads/external links intact.
- [x] Update affected shim/move ledgers, setup instructions and generated maps.
  Do not invent new APIs or scientific support to close old speculative gates.

## 4. Validate, merge and hand off

- [x] Focused import/pickle identity, numeric, CLI and consumer regressions for
  actual changes; preserve existing independent goldens and meaningful tests.
- [x] One complete CPU, architecture, lint, map and outside-checkout package pass
  on the completed candidate. Use the selected lab interpreter without upgrades;
  repeat passed checks only after relevant changes, failures or unresolved issues.
Release procedure: merge PR A (source/process consolidation). Verify required PR and actual-main
  CI. A second PR is reserved for necessary handoff fixes; omit it if unnecessary.
- [x] Leave a short handoff covering installation, current seqNMF workflow, input/
  output locations, known support limits and source/data recovery references.
Then preserve one final source recovery reference using existing tools. Report
  empirical backup gaps separately; Git does not protect ignored data or feedback.

## Rules and completion

One active migration branch, existing checkouts, at most two remaining PRs.
No per-helper branches, completion JSON, new tracking system, general converters,
extra research features or universal compatibility project. A surprising blocker
gets a bounded investigation, then a required fix or a named retained adapter;
it does not generate several new mandatory milestones. No implicit cohort runs,
model fitting, large media generation, bulk pickle/data changes or environment
upgrades. No force pushes, blanket clean/reset/staging or forced worktree removal.

Done means maintained original functionality is usable from main, implementations
have canonical owners, development clutter is removed, and a successor can start
from the short handoff. Compatibility aliases and ignored historical storage may
remain where they serve existing users; describe them plainly. Update this file
in place at real milestones. CURRENT stays a short pointer, not another history.

## Validation for this consolidation

1,298 local CPU tests pass (six existing warnings); architecture, strict/debt
lint, 149-directory navigation and fresh outside-checkout wheel/CLI checks pass.
No numerical implementation or scientific policy was added. Historical source
records remain in Git; source backups do not protect all empirical products.
