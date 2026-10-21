# Preserved evidence for the October 11 Antigravity handoff

Read [the handoff](../ANTIGRAVITY_HANDOFF_20261011.md) first. These files preserve source-only continuation evidence from the interrupted session. They are not canonical implementation modules, a production backup, or instructions to rerun historical migration scripts.

- `claude_review.txt`: original user-supplied external review.
- `checkpoint_state.json`: Git-status counts immediately before the handoff artifacts, plus exact dangling symlinks.
- `architecture.json`: final read-only 140-module dependency audit and graph.
- `reference_counts.json`: textual indicators; matching files are not necessarily broken consumers.
- `symlinks_before_cleanup.json`: earlier link inventory, including external pointers.
- `link_repairs_reviewed.json`: reviewed subset of repaired link decisions.
- `fpcode_migrate_paths.py`: partially applied scoped helper. Its hardcoded workspace and `/tmp` backup destination require review before further use. A dry run prints counts, not full diffs.
- `path_migration_originals.json`: 77 original UTF-8 source files captured immediately before the scoped path rewrites, at their already-moved canonical paths. Each record carries source text and its byte SHA-256. These are not the Git HEAD baseline.
- `prior_lint.json`: old HEAD diagnostic snapshot for tracked studies/tests/ops, used to distinguish inherited debt from newly introduced errors after path mapping. It excludes strict library/pipeline paths.
- `fpcode_prior_lint.py`: historical snapshot extraction helper; preserve as evidence rather than rerunning blindly.
- `SHA256_MANIFEST.json`: hashes/sizes of the evidence above. This README and the main handoff are intentionally outside that evidence manifest.

The originals can be inspected with `json.loads(...)["files"]`; do not automatically restore every record over current source. Compare only the specific file you are investigating. The unchanged baseline commit is `409407a06fb29e9658b6d75c958703dd6ab22978`.

All support files were copied or generated from explicitly selected local source/evidence. No empirical payload, arbitrary temporary-directory contents, credentials or data pickle was included. The evidence totals about 3 MB and avoids dependence on `/tmp` surviving.
