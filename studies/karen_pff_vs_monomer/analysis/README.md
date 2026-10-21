# Karen PFF vs Monomer Longitudinal Behavioral Study

The stopped overnight-search analysis is owned by `report_pipeline_search.py`.
It freezes one twice-successful winner per method/resolution before cohort
comparison, projects saved fold models without refitting, and reports inclusive
and named-member organisation at 0/1/5-second gaps, execution, first-order nulls,
mouse-level effect sizes, week contrasts and separate L-DOPA-wave changes.
Use explicit `--search`, `--atlas`, `--output`, resource limits and `--execute`;
without execution it only prints the plan. `--max-jobs` bounds a projection check.
Outputs belong in the selected run folder; generic projection/null/statistical
mechanisms remain in the existing `fpcode` owners. The source search is stopped
at the user's request and must not be resumed without later authorisation.

The exploratory syllable/chain/frame atlas is documented in
[BEHAVIOR_ATLAS.md](BEHAVIOR_ATLAS.md), including current owners, input provenance,
reproduction commands, effect-size follow-up tables and interpretation limits.

The executed overall self-transition, repetition, initiation and short-loop
extension is documented in [GENERAL_CHAIN_ORGANISATION.md](GENERAL_CHAIN_ORGANISATION.md),
with its original [execution plan](GENERAL_CHAIN_ORGANISATION_PLAN.md).

The bounded layer pilot is run by `run_layer_pilot.py`. It reuses the verified
frame cache, latest cached descriptions, frozen animal splits and L1–L3 templates;
only held-out mice enter the comparison. Two configured interpretably labelled
raw routines are projected across levels. Their reusable two-action parts can
form bounded assemblies. No KPMS, seqNMF-template or HSMM-parameter fitting occurs.
Default scope is 12 one-minute windows from four mice over 3w, 16w and L-DOPA,
starting 120 seconds after the first modelled frame. Native video spot checks are
required: early modelled frames can still include handling.

Screen bounds and pooled training-only nose envelopes are explicit spatial proxies,
stratified by chamber, calibration epoch and video size. Distances retain body
midpoint, nose and supported ventral-paw-centre references, all four image corners
and TopCam camera-near/far corner interpretations. The shared central TopCam and
local BotCams have different perspective geometry; tandem21 contains dorsal body
points plus ventral paws, not two independent nose measurements. No inferred
parallax correction or physical-wall calibration is claimed.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 /path/to/python -B \
  studies/karen_pff_vs_monomer/analysis/run_layer_pilot.py \
  --atlas outputs/karen_behavior_atlas_20261011 \
  --clustering outputs/karen_clustering_levels_20261012 \
  --fused-dir data/fused/multiview/cohort_fused_static_21kp_v2 \
  --video-dir data/raw/karen_pff_vs_monomer/video_files \
  --output outputs/your_new_pilot --workers 2 --ram-cap-mb 1536
```

This is a read-only plan until `--execute` is added. Use `--max-recordings 1
--max-clips 2` for a smoke scope. Scheduling budgets include pose-table copies;
the RAM cap is a conservative scheduling limit, not an operating-system RSS cap.
`report_layer_pilot.py --output outputs/your_new_pilot` validates saved intervals,
roles, exclusive streams and clips, then exports the method/proxy sensitivity report.
Add `--assembly-preview --video-dir <native-video-root>` for one additional assembly
clip. The executed [2026-10-12 pilot report](../../../outputs/karen_layer_pilot_20261012/pilot/report.md)
records the scope, handling QC, configured-model limits and next scientific steps.

**Location**: `studies/behavior/karen_studies`  
**Status**: `ACTIVE`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Longitudinal behavioral phenotyping comparing alpha-synuclein PFF vs Monomer models across multi-week timepoints.

## 2. Inputs
BotCam ventral pose estimations, TopCam dorsal tracks, and MoSeq syllable annotations

## 3. Outputs
Longitudinal progression curves, motor impairment metrics, transition graphs

## 4. How to Run
```bash
python run_karen_analysis.py
```

## Confidence exclusion audit

`audit_confidence_exclusions.py` scans the verified atlas cache and tandem pose
confidence tables, separating dorsal and ventral support and checking long
low-confidence streaks against existing native-frame labels. It exports candidates
and review stills; it does not apply exclusions or fit models. Pass `--atlas`,
`--fused-dir`, `--video-dir`, `--output`, `--workers 2` and `--ram-cap-mb 1536`;
execution requires `--execute`. Use four numerical threads as in the pilot command.
The [2026-10-12 audit](../../../outputs/karen_handling_audit_20261012/report.md)
includes user-confirmed still labels and a candidate-only mask proposal. Full
interval boundaries still require review before application to behavioural analyses.

`review_confidence_boundaries.py` produces a bounded group-blind review of six
large later-session candidates plus the known initial handling witness. It accepts
`--audit`, `--atlas`, `--video-dir`, `--output`, `--workers` (one or two), and
`--execute`; without execution it reports the planned scope. It verifies native
video seeks/rates and encoded clip counts. Reviewed edge-behaviour false positives
show that confidence candidate intervals must not automatically become handling
exclusions. Retain separate presence, pose-support and handling masks.

`apply_reviewed_exclusions.py` builds a separate frame-cache analysis view from a
CSV of explicitly confirmed half-open native intervals. It accepts `--atlas`,
`--reviewed-exclusions`, `--output`, and `--execute`; the default is a read-only
plan. Only affected recordings are copied; untouched caches are linked to their
source. Canonical and exported labels are invalidated together, and centroid,
heading, latent and exported velocity channels become unsupported in excluded rows.
Native clocks are preserved. Downstream consumers must use the new view's cache
and recalculate exposure; frozen chain calls must be revalidated against its valid
blocks. This does not clean model templates or imply cohort-wide QC completion.

## Routine-quality search

`run_overnight_random_search.py` now delegates scientific evaluation to
`fpcode.workflows.sequence_search` and process watchdogs to
`fpcode.common.trial_runner`. It requires explicit `--atlas`/`--output`; its default
is a read-only plan and `--execute` fits the requested routines. `--hours 0` runs
until stopped. Search is grouped by animal, reserves the saved test animals from
this search, ranks within method/resolution, and averages deterministic replicates.
Short-bout filters preserve native gaps. Disease/L-DOPA and first-order-null checks
are downstream finalist analyses. Use `python -B -m
studies.karen_pff_vs_monomer.analysis.run_overnight_random_search` from the code root.

`ops/monitoring/launch_behavior_search.py` provides a reviewable default plan and an
explicit `--execute` user-service/nohup launch with numerical thread, memory and log
controls. The [2026-10-13 repair report](../../../outputs/karen_search_repair_20261013/REPAIR_REPORT.md)
records the actual running service, frozen inputs/code, validation and stop commands.
