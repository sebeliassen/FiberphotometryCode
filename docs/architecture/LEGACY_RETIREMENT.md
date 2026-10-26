# Canonical owners and historical interfaces

Reusable implementation belongs in `src/fpcode`, project analysis and cohort
policy in `studies/<project>`, and operational tools in `ops/<purpose>`.
Maintained callers should import these owners directly. Root `compat` contains
historical interfaces, including `compat/core` and `compat/pipelines`; `src/fpcode/compat` separately owns
historical library implementations and numerical policies.

Follow the [current consolidation checklist](../tasks/CONSOLIDATION_AND_MAIN_PLAN_20261021.md).
Preserve the workflows from the original source and seqNMF delivery; unsupported
historical experiments do not require new functionality as part of consolidation.

Before retiring an adapter, check its maintained imports, literal script paths
and explicitly supported serialized objects. Preserve units, clocks, masks,
cohort policy and numerical expectations when moving callers. Keep independent
behavioral tests and required import identities; use the
[validation guidance](../agent-workflows/validation.md) to select checks.

The removed dated inventory and per-slice reports remain available in Git history
at `ce281cc`. They described their recorded checkpoint and are not current
completion gates. A static import scan or passing CPU suite does not establish
universal compatibility with historical pickles or empirical workflows.

Keep recovery bundles and ignored empirical products separate from source
cleanup. Removing an obsolete tracked adapter is not permission to delete
recordings, models, output directories, browser feedback or external link targets.
