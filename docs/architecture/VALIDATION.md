# Functional and runtime validation

Run these commands from the repository in its Python environment. The default
functional tests use small arrays and temporary files; GPU jobs and empirical
artifacts are separate, explicitly requested checks.

```bash
make test PYTHON=/path/to/environment/bin/python
make architecture PYTHON=/path/to/environment/bin/python
make smoke-runtime PYTHON=/path/to/environment/bin/python
```

`smoke-runtime` runs real batched initialization, one AR and one SLDS update,
checkpoint endpoints, patch restoration and a tiny encoded movie in a fresh CPU
process. Its files are temporary and its default timeout is 60 seconds.

## Two real sessions on the GPU

Supply a pilot KPMS project containing `preprocessed_data.pkl`, `config.yml` and
`pca.p`. The source project is read-only input. The preprocessed file must contain
the observation/confidence/mask arrays and recording names with segment bounds.

```bash
make validate-two-sessions \
  PYTHON=/path/to/environment/bin/python \
  VALIDATION_PROJECT=/path/to/pilot/project

# Optional smaller excerpts, CPU validation, or a different time limit:
make validate-two-sessions \
  PYTHON=/path/to/environment/bin/python \
  VALIDATION_PROJECT=/path/to/pilot/project \
  VALIDATION_ARGS='--frames 128 --device cpu --timeout 120'
```

The default is 256 valid prefix frames from each of two distinct sessions,
exactly two AR updates and two SLDS updates, seed 101, and a 180-second timeout.
Repeated `_seg<number>` fragments from one session cannot satisfy the two-session
requirement. The default input size limit is 1024 MiB, checked before unpickling;
use an existing pilot, not a full-cohort cache. No frames are joined across gaps.

The worker inherits the source anatomy, PCA and model dimensions. It uses the
existing optimized GPU/CPU hybrid sampler with explicit AR kappa `1e6` and SLDS
kappa `1.5e15`. It checks float64 execution on the requested device, actual update
counts, finite model arrays, snapshots 0 through 4, patch restoration, checkpoint
reload, valid-frame evaluation and result extraction. CUDA requests fail if the
GPU backend is unavailable instead of silently running on the CPU.

Each invocation creates a unique `outputs/<run-id>/` directory containing the
executed diagnostic script, sampled inputs, effective configuration, checkpoint,
results, `worker.log` and `validation.json`. Failed runs retain their evidence;
timeout termination covers the worker process group. No production checkpoint
is resumed or modified, and source size/mtime/inode signatures are checked again.

This verifies the training and export paths on small empirical excerpts. Four
updates do not establish convergence, scientific equivalence of samplers, or
memory behavior for a full cohort. It is intentionally absent from `make test`.
