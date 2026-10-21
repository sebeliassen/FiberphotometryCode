# Operations

Implementations belong in purpose-specific directories; `ops/scripts` preserves old callers. Before invoking a script, inspect its inputs and effects. Monitoring can launch jobs or send notifications; compaction/relinking/backup/notebook utilities can mutate artifacts. A help/import check is different from execution.

Use explicit inputs/output destinations and the current interpreter. Preserve existing external symlink targets and empirical data. Do not create empty replacement inputs to make existence checks pass. Once-off migration/triage tools are historical unless deliberately adapted to current ledgers.

For repository tooling, validate syntax/lint and meaningful tiny-fixture behavior. The folder-map generator must prune empirical/cache directories and directory symlinks before traversal, and parse source without importing it. Derived maps are navigation hints; the dependency audit and scientific contracts are authoritative.

Record bounded operation evidence and exact scope. No bulk production maintenance is implied by adding agent instructions or running the functional suite.
