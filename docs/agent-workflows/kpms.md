# Configure and run KPMS explicitly

The fitting CLI consumes already-preprocessed data, normally `preprocessed_data.pkl`. Fresh initialization uses a matching PCA artifact if present; otherwise it fits/saves PCA at the configured path. Resuming an initialized checkpoint needs no PCA artifact. Only load trusted project artifacts in an explicit fitting operation; input diagnostics use existence/metadata checks. Raw tracking preprocessing belongs to a separate workflow.

Resolve the user's input, profile, seed, device, model name/output and requested scale. `project_dir`/`data_path` in training configuration are resolved against the code root by the fitting CLI; set explicit paths when data lives on an external mirror. Do not assume `FP_DATA_ROOT` rewrites every profile string.

```bash
python -B -m fpcode.cli.doctor --json --require kpms \
  --input /path/to/preprocessed_data.pkl --input /path/to/pca.p
python -B -m fpcode.cli.train --config /path/to/profile.yaml \
  --device cpu --dry-run --no-notify
```

Dry-run validates settings and preprocessed-data existence without unpickling/fitting or creating outputs. Use the doctor to check the named PCA path separately. Model/PCA content and checkpoint compatibility are validated during the explicit fitting operation. Check for an existing model directory and clarify resume intent if it conflicts with the requested action; the CLI has no general resume flag. The canonical fitter resumes an earlier checkpoint toward the requested endpoint and rejects checkpoints already at or beyond it. A CPU dry-run does not test CUDA readiness or estimate cohort memory accurately.

Execute the requested run with its selected device, output and iteration counts. Do not add a sweep or compaction operation to a training request. `--no-notify` disables completion notifications. Preserve cohort/preprocessing settings and record any deliberate changes separately.

`run.json` records effective configuration, source-config hash, seed, interpreter, dependency versions and already-initialized backend/device information. Inspect checkpoint existence and requested evaluation evidence; a completed file is not proof of convergence.

Optional top-level `resource_estimates` in a training profile is copied to its own `run.json` field. Supply numerical values with explicit units (for example `peak_device_memory_bytes`), the measurement/estimation method, matched workload scope and assumptions. Derive planning bounds from an authorized representative pilot or a documented analytical model. When unavailable, this field is null; the doctor and dry-run do not invent memory/runtime predictions from installed hardware or pickle size. Other workflows can pass the same mapping directly to `RunRecord(resource_estimates=...)`.

## Optimizations

Public entry points are in `fpcode.kpms.runtime`; implementations live in `fpcode.kpms.backend`. The canonical fitting wrapper selects the memory patch when `use_memory_patch` is true and restores its scope. For an explicit custom backend workflow:

```python
from fpcode.kpms.runtime import kpms_optimizations

with kpms_optimizations():
    # Invoke the already-configured backend fitting operation here.
    ...
```

Entering this context loads optional backends and installs process-global patches. Serialize configuration/patch activation; do not apply a second unrelated patch set concurrently. Preserve original functions on success and failure. Batch sizing and kernel selection remain explicit runtime choices, not universal hardware defaults.

The optional `kpms` dependency extra is broad; compatible versions depend on the tested environment. `fpcode-doctor` reports installed versions without claiming they are all compatible. See [KPMS benchmark history](../KPMS_SPEEDUP_RECONCILIATION_AND_BENCHMARK_PLAN.md) and [bounded validation](../architecture/VALIDATION.md). Benchmark a defined workload and independently validate changed numerical behavior before advertising speed or sampler parity.
