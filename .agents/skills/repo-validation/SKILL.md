---
name: repo-validation
description: Diagnose FiberphotometryCode setup and select or run appropriate checks for a repository change; report evidence without launching empirical workloads implicitly.
---

# Repository validation

Read the root `AGENTS.md` and [validation selection](../../../docs/agent-workflows/validation.md). Choose checks for the changed owner and intended behavior; do not require every historical report or every skill.

Use the selected interpreter and read-only `python -B -m fpcode.cli.doctor --json`. Add `--require cpu`, `kpms` or `render` for the requested workflow, and name actual inputs with `--input`/`--data-input`. Missing distributions/paths are explicit; metadata does not prove CUDA readiness or semantic input validity. Preserve the lab environment.

Relevant small regressions come first. Shared behavior also needs the default suite and architecture/lint gates. Package/entry-point changes need installed-wheel verification outside the source checkout. Navigation changes need generator synchronization and accurate ownership references. Do not refresh lint debt to accept newly introduced errors or regenerate numerical goldens to accept an unexplained change.

Empirical collection is separate from execution; marker filtering alone cannot prevent imports. A normal code check does not imply a cohort fit, maintenance operation or notification.

Report exact command/interpreter, commit or dirty working tree, result and limitations. For ongoing work, update `docs/tasks/CURRENT.md`. A fixture pass or static audit does not establish universal scientific equivalence.
