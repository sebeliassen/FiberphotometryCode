# Session/model export boundary

The utility owner is `studies.session_model_exports.utils`. Historical sampling,
syll_results and frame_map_placeholder files proxy to those exact modules through
a structural physical parent. Importing these utilities does not load models,
initialize modeling backends or write products; their algorithms are unchanged.

Synthetic checks preserve internal forward filling, leading missing values and
masking after the last supported VFI even when video/photometry extends the target.
They also preserve trial-aware/flipped result matching, unmatched-key handling,
strict ambiguity errors and first-row video length selection. Those historical
policies need explicit scientific review when adopting a new workflow.

`syll_results.load_kpms_results_and_vfi` is an explicit empirical operation. With
no saved results.h5 it requests extraction with save_results=True. It can write
model results; it is not a read-only inspection helper. Tests stub the backend and
verify requests without running it. Placeholder identity/spread maps retain their
separate filename and must not be mistaken for true video_frame_indexes.

Full exporter drivers still contain old workstation paths and import-time
backends/global setup. Their help/import is not a safe
metadata probe. These aliases do not establish executable cohort export or support
for arbitrary historical artifacts. Inputs, outputs, clock policies and recovery
coverage must be resolved before explicit empirical execution or legacy deletion.

## Historical VFI support

`utils.legacy_support.migrate_legacy_vfi` restores the last pre-refactor helper.
Despite its historical name, it returns independent array copies with unchanged
values/dtypes. Its earlier length-based rescaling heuristic had already been
removed after synchronization errors on trimmed recordings. Expected lengths and
scale_factor remain accepted for old callers and do not trigger inference. The
historical verbose message does not prove that coordinates were detected/validated.
An actual relative-index artifact needs an explicitly justified conversion.

Two drivers now import this exact study owner; the physical old leaf proxy shares
module/function identity. Original quarantine history remains in MOVES.tsv with
an additive source-restoration record. Checkpoint absolute_vfi_support_20261021.json
pins the original source and normalized AST; only documentation/unused imports and
the two import paths change. Full exporters remain unexecuted and are not cold
metadata probes. Legacy roots, symlinks and empirical data remain in place.
