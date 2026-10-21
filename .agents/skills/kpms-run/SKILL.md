---
name: kpms-run
description: Configure, validate and execute a requested KPMS fitting or optimization-benchmark workflow in FiberphotometryCode with explicit inputs, device, output and scope.
---

# KPMS execution

Read [the KPMS workflow](../../../docs/agent-workflows/kpms.md) for the supported training schema, trusted preprocessed inputs, dry-run and scoped optimization API.

Resolve input/PCA/profile, seed, device, model output and requested scale. Ask for missing consequential scientific or execution choices; infer routine implementation details from the request and existing project policy. Do not create placeholder empirical inputs, change cohorts or expand one requested run into a sweep.

Use the read-only doctor for named input/distribution checks, then the training CLI's dry-run. Existing models need explicit compatible resume/overwrite intent; this CLI has no general resume flag. Configuration/input existence is not proof of CUDA readiness, sufficient memory or convergence.

Execute only the requested scope. Preserve scientific settings and established patch restoration. The canonical fitter already scopes its memory optimizations; custom calls use `fpcode.kpms.runtime.kpms_optimizations`. Backend patches are process-global and should not be configured concurrently.

Inspect the requested checkpoint/evaluation artifacts and `run.json`. Report effective settings, dependency/device evidence and what was actually tested. Use bounded independent numerical checks and measured workloads before making optimization speed/parity claims. Persist long-running task state in `docs/tasks/CURRENT.md`.
