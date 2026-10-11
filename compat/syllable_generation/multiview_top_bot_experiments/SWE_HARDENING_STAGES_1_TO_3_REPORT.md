# SWE hardening — stages 1–3 (+ parts of 4–5): changes, compatibility, verification

Date: 2026-10-05. Scope: `syllable_generation/multiview_top_bot_experiments`.
No production training launched, no jobs stopped, no existing checkpoints/datasets
rewritten or deleted by this work (see "Incident" for the one earlier exception).

## Incident found during this work (needs your decision)

The **first EML run (10:31–10:46) modified all 5 source checkpoints in place**
(`fast_reindex_checkpoint` opened them `r+`; mtimes 10:32). It is a pure,
consistent relabeling (params `betas/pi/Ab/Q` + `states/z`), the same operation as KPMS's own
reindex, and EML is permutation-invariant, so **the EML ranking is unaffected**.
But two consequences:

1. **Seed 101 `results.h5` (made 10:29, before the run) is now stale vs. its checkpoint.**
   Recovered the old→new map from the data: a clean one-to-one permutation in which only
   **labels 67↔68 swapped** (tied frequency). Saved:
   `04_reports_and_figures/audits/seed101_old_to_new_label_map_recovered_20261005.json`.
   Anything built from the old seed-101 labels (existing pkl, `visualizations_seed101`) uses
   the *old* numbering; checkpoint-derived outputs now use the new one. Seeds 202–505 have
   no such problem (their `results.h5` was regenerated after the relabel; 0/60 sampled
   recordings differ).
2. The old reindex was **not idempotent**: `argsort(freqs)[::-1]` reverses tied states every
   call. Fixed (stable sort).

I cannot restore the pre-relabel checkpoints (no backup of those 33 GB files) and did not
modify them further.

## Changes

| Area | File | Behavior change |
|---|---|---|
| Calibration provenance | `core/camera_correspondence.py` | Returns `status` / `is_valid_crystal`; `<5` anchors is `DEGRADED_INSUFFICIENT_ANCHORS` (no longer labeled `rate_aware_basler_crystal`); missing Basler cols = `FALLBACK_HOST_TIMESTAMPS`; `strict=True` raises on any non-crystal calibration; validates finite/non-empty/monotonic timestamps. Anchor sampling margin now scales with recording length (identical for real sessions, step stays 50). Math for valid sessions unchanged. |
| Fusion | `core/fusion_engine.py` | `strict_calibration` (defaults to `require_continuous_sync`); `sync_mode` labeled from real provenance; atomic H5 write + manifest; unmanifested caches flagged `UNVERIFIED_NO_MANIFEST` (kept; `unmanifested_policy="trust"` default or `"rebuild"`); stale caches rebuilt atomically. Fused values unchanged. |
| Cache layer (new) | `core/cache_manifest.py` | `atomic_path`, sidecar manifests (inputs, config, keypoint order, code hash, dependency versions), `verify_manifest`, `verify_artifact_set`. Stat-only identity for files >16 MiB (no re-hashing of huge inputs). stdlib only. |
| Eval protection (new) | `core/eval_artifacts.py` | Read-only checkpoint identity, stable label permutation, model-compatibility validation (dataset/mask/segment order/shapes/iteration), immutability guard. |
| EML | `03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py` | Checkpoints never opened `r+`; reindex in memory; derived `results_reindexed.h5` (new name, atomic, manifest with old→new map keyed on checkpoint size/mtime + iteration + mapping hash); compatibility check before EML; stat-signature assertion after GPU phase; outputs tagged per project and never overwritten without `--overwrite`; lazy JAX/KPMS imports; `--dry-run`, `--skip-results`, `--allow-mixed-iterations`. |
| Audit (new) | `01_data_and_fusion/audit_calibration_validity.py` | Read-only calibration audit across all TTL sessions. |

## Findings

* Calibration audit: **258/258 sessions** in the TTL cache are `VALID_CRYSTAL`;
  ppm drift −0.5…3.4, max anchor residual ≤ 0.229 ms, min mapped fraction 0.9997.
  Strict mode therefore changes **no existing session**.
* EML (continuous-calibrated): seed 404 first (225,001,430 ± 234,745), seed 101 within error
  (224,940,120 ± 302,124). Treat as a soft ranking.
* `--help` 0.26 s / 142 MB; `--dry-run` 0.30 s / 144 MB peak (no JAX/GPU, validates the 5 real
  checkpoints as comparable: 3473 segments, 17,374,468 active frames, latent_dim 10).

## Compatibility commands

```bash
PY=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
E=syllable_generation/multiview_top_bot_experiments
$PY $E/03_evaluation_and_eml/compute_tandem_cohort_eml_rankings.py --dry-run       # safe plan
$PY $E/01_data_and_fusion/audit_calibration_validity.py                              # read-only
# Real EML (cgroup-isolated, unchanged entry point):
$E/03_evaluation_and_eml/launch_eml_evaluation.sh
# Legacy fusion remains permissive when asked explicitly:
#   fuse_single_pair(..., require_continuous_sync=False, strict_calibration=False)
```

## Tests (all CPU, `PYTHONPATH=. $PY tests/<name>.py`)

| Test | Result |
|---|---|
| `test_camera_correspondence_synthetic` (5) | PASS |
| `test_cache_manifest_synthetic` (29) | PASS |
| `test_eval_artifacts_synthetic` (29) | PASS |
| `test_fusion_provenance_and_cache_synthetic` (14) | PASS |
| `test_multiview_sync_and_renderer_regression` (existing) | PASS |
| `test_multiview_north_stars_4_1_to_4_3` (existing; run after restoring legacy rankings files) | PASS |
| `test_kpms_checkpoint_compaction` (7 pillars, 32 s) | PASS |

Not run / unavailable: a full-scale EML re-run with the new code (would repeat ~14 min of GPU
work; only `--dry-run` was exercised on real data); GPU path unchanged except lazy imports.

## Remaining issues

* Cache manifests are wired into fused H5 and EML derived results only. **Not yet** applied to
  preprocessing, PCA, initialization, and rendered outputs.
* Stage 4 (consolidating training/rendering into `core/`, parameterized wrappers) not started;
  `--dry-run` exists only for the EML script.
* Existing fused H5s and results files have no manifests → reported UNVERIFIED (kept).
* Gap-boundary, VFI-correspondence synthetic tests (stage 5) not yet written.
* `test_north_star_4_1` hard-codes legacy 118-session EML values and reads the legacy rankings path.
* Decision needed: how to treat seed-101 old-label artifacts (see Incident).
