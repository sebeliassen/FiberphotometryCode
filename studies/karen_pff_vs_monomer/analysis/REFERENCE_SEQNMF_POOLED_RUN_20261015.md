# Pooled Karen reference-seqNMF panel

Executed15October2026 on `codex/karen-seqnmf-delivery`. **New exploratory Karen
fits, not lab-reviewed findings.** This supersedes the earlier proposed
train/validation design for this delivery analysis: the user explicitly requested
one common chain generator fitted over the whole dataset, as intended for KPMS.

## Frozen scope and design

- All99recordings,21mice and five waves, including L-DOPA. Every eligible native
  bout is included. No animal split or bout-count subsampling is used for fitting.
- Original historical split JSON is retained as provenance only. Former reserved
  animals contribute to this pooled fit; no held-out validation claim is made.
- Each candidate is ONE shared generator across all groups/waves. Separate
  models are alternatives in the small parameter comparison, not cohort-specific
  vocabularies. No group-effect statistics select a candidate.
- Original vocabulary L1/100states and existing L3_unresolved_separate/69states.
  Known synonyms merge within valid support only; unlabelled/dump IDs remain
  individually distinct rather than being treated as one common behaviour.
- K10,L3bout steps,80iterations, lambda0/.1/.3/1/3, seeds42/43/44:30fits.
  Component count20at selected lambda settings is a proposed follow-up; it was
  **not** included in these30fits.

Input view:
`/home/fsp585/Desktop/FiberphotometryCode/outputs/karen_reviewed_analysis_view_20261012/reviewed_boundaries`.
Its existing reviewed handling masks were checked against the exclusion intervals
for all99recordings. Native60Hz clocks and half-open endpoints are preserved.
RLE splits at invalid labels, handling exclusions and missing frames. Support
blocks receive zero padding, and unsupported H onsets are exactly zero at every
update. Detection never crosses those barriers; each end uses the actual final
included bout. Grouped RLE cannot bridge an excluded interval.

`reference_seqnmf.py` preserves the recovered historical activation competition,
ratio-based multiplicative updates and per-lag W normalisation. It is explicitly
distinct from the newer shared-library data-dependent seqNMF penalty. This study
adaptation adds onset masks rather than importing the old side-effectful runner.

## Execution and evidence

Output root:
`/home/fsp585/Desktop/FiberphotometryCode/.worktrees/karen-seqnmf-delivery/outputs/karen_pooled_reference_seqnmf_20261015`.

```bash
tail -f /home/fsp585/Desktop/FiberphotometryCode/.worktrees/karen-seqnmf-delivery/outputs/karen_pooled_reference_seqnmf_20261015/run.log
```

The job was launched through `nohup setsid timeout`, independently of the app
terminal. LauncherPID/PGID3727659, driverPID3727660; it has now exited0.
`driver_status.json` and `status.json` are COMPLETE,30/30.

Prepared input files/refs and `recipe.json` freeze actual label/frame/mask array
hashes, source hashes, mapping, all included recordings, parameters and detection
rules. L1 has315,035valid onsets; grouped has307,651. Packed padding is explicitly
excluded from onset support. The actual bout counts differ after synonym merging.

CPU/GPU preflight used8iterations on the actual L1 matrix, float32. Tolerances
atol3e-5/rtol3e-4 passed (max Wdifference1.20e-5,Hdifference5.97e-6). Timed8updates:
CPU16.75s,GPU0.075s; initial GPU compilation/warmup0.622s. RTX4090/JAX0.6.2,
highest matmul precision was selected. This bounded check is not a claim of
universal backend equivalence or full convergence. Resource settings:2numerical
threads,8GiB RSS guard,25%GPU allocation fraction,80GiB free-disk floor,4GiB output
guard,3hour outer timeout. Final products occupied about45MiB.

Key outputs:

| Artifact | Meaning |
| --- | --- |
| `results.csv` | Every parameter/seed/result correspondence; fit error, physical union coverage, support and redundancy |
| `seed_stability.csv` | Hungarian-matched, phase-locked template cosine across seed pairs within each vocabulary/lambda |
| `capacity_metrics_K10.csv` | Global reconstruction error and uncentred score1−relative_error², alongside the other diagnostics |
| `factor_catalogue.csv` | All300candidate factors, actual observed path counts, mouse support and dominant-template exact counts |
| `trials/<id>/model.npz` | Saved W,pooled H,thresholds and exact parameters |
| `trials/<id>/pooled_calls.csv.gz` | Actual retrieved executions with native boundaries/durations |
| `panel_audit.json` | Independent persisted-call correspondence checks for all320,307calls across30fits |

The saved-call audit checked frozen source hashes, complete parameter coverage,
finite model arrays, hard-zero unsupported H, saved90th-percentile thresholds,
observed tokens, native starts/ends, same support block, internal adjacency and
60Hz durations. All checks passed. Dominant argmax templates are not presented as
observed behaviour: the catalogue separately records their actual occurrence
counts and the most frequent real path.

## Interpretation and next step

These diagnostics are **in-sample**, with a jointly learned vocabulary. A high
reconstruction score is not a percentage of behaviour explained, centred PCA
variance explained or biological validation. Reconstruction targets differ
between raw/grouped vocabularies, so use separate curves for each. Physical union
time coverage is a separate measure, conditional on the stated detection rules.
The threshold/error/support settings were held fixed across this panel.

Best detected-time coverage within each vocabulary was8.35%(L1,lambda.3,seed42)
and17.31%(grouped,lambda.3,seed43). These are descriptive maxima, not accepted
winners or evidence of biological superiority. Retain alternatives along the
coherence/coverage/redundancy/stability tradeoff. No group p-values were used.

K10alone cannot establish sufficient capacity. Next compare K20at selected lambda
settings against matching K10fits: reconstruction gains, physical coverage,
additional nonduplicate multi-mouse families and seed stability. Add another K
only if useful gains remain substantial. A PCA-style component-count curve needs
those additional Kvalues; the present table has one component count only.

Then project/review the selected shared catalogue using p100/P80/random grids.
No new qualitative acceptance or chain-movie generation occurred in this run.
After qualitative catalogue freeze, compare group occurrence and bounded intra/
inter-chain readouts at the animal level.

Checks:6new focused boundary/identity tests passed; complete CPU suite852passed
with6existing deprecation warnings. Lint3078reviewed/no new debt, architecture
201modules/no violations/cycles, folder maps146directories and CLI help passed.
No shared-library numerical behaviour changed. No lint baseline was refreshed.
