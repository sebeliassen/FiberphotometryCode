# Legacy retirement status

21 October 2026. The foundation, seqNMF delivery and first consumer migrations are integrated into main through reviewed normal merges. Legacy retirement remains in progress. See [execution state](../tasks/CONSOLIDATION_PROGRESS_20261021.md) and [the plan](../tasks/CONSOLIDATION_AND_MAIN_PLAN_20261021.md).

The canonical library is src/fpcode. Current study launchers should invoke fpcode.cli modules or installed commands; new study code should import the owning capability directly. Root core/pipelines/compat aliases remain supported while their consumers and artifact obligations are reviewed. A root directory is not retired merely because its imports are deprecated.

## Completed migrations

- Eight multiview training/fusion launchers now invoke `python -m fpcode.cli.train` or `python -m fpcode.cli.fuse`. Configurations, forwarded arguments and child failure status are preserved by direct-entry tests.
- CLI orchestration tests now exercise canonical modules. Fresh-process tests separately preserve all five historical pipeline alias identities in both import orders.
- Six active language-model analyses now import Jensen-Shannon divergence from fpcode.statistics.divergence, avoiding an empirical presentation renderer for that calculation. Its base-2 logarithms, clipping and last-axis normalization match the frozen original exactly in float32/float64 tests. The renderer reexports the shared helper. A frozen review-package copy remains historical source; its old imports are not evidence of a supported current entry point.
- Twenty accidentally ignored source files in project data/models packages have been recovered. Their numerical definitions and original import sets are preserved; source-only Git exceptions continue excluding empirical payloads.

- Six study alignment consumers now import fpcode.photometry.alignment directly. Four had referenced a lost public helper, build_aligned_moseq_phot. Its exact original function AST was restored from the pinned pre-scaffolding source into fpcode.compat.photometry.frame_merge. Independent native-frame order, missing-support, missing-photometry and fallback-column expectations pass, with original wrapper parity and legacy alias identity. The preserved default470/historical interpolation profile is explicit; complete export jobs and alternate historical configurations were not executed.

## Inventory and limits

The versioned inventory is [LEGACY_PATHS_20261021.tsv](../../registry/LEGACY_PATHS_20261021.tsv), [LEGACY_CONSUMERS_20261021.tsv](../../registry/LEGACY_CONSUMERS_20261021.tsv) and [scan metadata](../../registry/LEGACY_INVENTORY_20261021.json). It records current aliases, targets, imports, literal path candidates, code-cell references and per-consumer source hashes. Static references are review candidates, not proof of live execution. Notebook line numbers refer to concatenated code cells. Forty-four parse/budget limitations are recorded; unknown external users are not detectable from this scan.

Root compat is distinct from src/fpcode/compat, which still owns historical policies and calibration resources. Neither migration of study imports nor a successful CPU suite establishes that every historical pickle can be loaded without its old namespace.

## Next retirement gates

1. Classify the remaining active syllable_generation/fiberphotometry imports by owner and migrate one capability at a time, keeping scientific defaults explicit.
2. Review historical notebook, exported-bundle and operational consumers separately. Some empirical presentation scripts still contain old workstation/output paths and import-time filesystem effects; do not import or execute them as metadata checks.
3. Inventory named trusted session/model artifacts and prove a canonical loader, separate converter or retained small legacy component for each supported family.
4. Update shim registries and historical shim-generation maps before deleting adapters so tooling cannot recreate retired paths. Preserve physical direct-script/runpy obligations where supported.
5. Verify external archive/legacy links, large-artifact backups and human feedback exports before any physical relocation. Git/source bundles do not contain the ignored empirical products.
6. Continue reviewed changes from integrated main. The foundation, seqNMF, CLI/statistics and photometry integration PRs #2–#5 are merged with preserved ancestry; retirement still requires the capability and artifact gates above.

Legacy roots remain intact at this checkpoint. No migration of recordings, models, generated media or human feedback was performed.

## Checkpoint callers and remaining implementation gaps

Three training runners now import checkpoint patching from fpcode.kpms.checkpoints.
Both import orders retain legacy/canonical module and apply/remove-function
identity without initializing modeling backends. Runner algorithms and import
execution order are preserved; their empirical workflows were not executed.
See registry/checkpoints/checkpoint_consumer_imports_20261021.json for the AST audit.

The README documents explicit checkout activation for historical namespaces;
those paths are not installed by the canonical wheel. Known next repairs include
deep-research hardcoded/bare loaders, tracking's bare config/symlink identity and
session-export utility alias/registry disagreements. Some target implementations
are missing. Alias files and a passing CPU suite do not establish executable
historical cohort pipelines; repair and validate those named routes separately.
