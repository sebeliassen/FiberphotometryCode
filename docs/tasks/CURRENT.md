# Current status

Updated 26 October 2026. **Source consolidation is complete** ([PR #20](https://github.com/sebeliassen/FiberphotometryCode/pull/20), main `4161eec`).

- Original modular and Karen seqNMF source are integrated; no additional functionality is required.
- Final layout: main only, plus the unchanged tagged delivery viewer checkout. Both viewers and empirical products are preserved.
- Verification: 1,298 local CPU tests; architecture, lint, maps, wheel/CLI and merged-main CI passed.
- Recovery: original/final source bundles under `outputs/consolidation_20261021`; original notes tagged `codex/pre-consolidation-20261026`.

Handoff starts with the [README](../../README.md) and its seqNMF links. The recipient still needs access to the environment/data; ignored empirical products and browser feedback need separate recovery coverage before relocation.

Keep compat for existing historical names. Future updates remove completed items; history belongs in Git, not appended here.
