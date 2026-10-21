# Karen exploratory themes: delivered 21 October 2026

The deadline-focused plan produced five provisional, human-labelled ordered
themes from existing pooled seqNMF catalogues. Definitions were frozen before
group statistics. Their complete observed cores were then retrieved throughout
all valid recordings, separately from the original broad factor detections.
This is an explicitly refined theme library, not acceptance of every generator
component or independent qualitative validation.

| Theme | Complete overlapping windows | Mice |
| --- | ---: | ---: |
| Travel–turn–travel | 2,614 | 21 |
| Orientation–sniff–orientation | 1,539 | 21 |
| Sniff–observe–sniff | 616 | 21 |
| Rear–back-off | 1,736 | 21 |
| Groom–reorient–groom | 814 | 21 |

Source suggestions come from the already-fitted cleaned pooled K10/K18 L3/L5
catalogues (grouped69, lambda0.3, seed43,80updates). No new model or parameter
search was required. The two sensory themes may share a broader family;
back-off is the supplied label, not evidence for descent. Rest/sniff/rest and
uncertain scrunch themes remain recorded reserves rather than primary outcomes.

## Deliverables and findings

Numerical output: `outputs/karen_theme_delivery_20261021/` in the delivery worktree.
Portable report: `report/index.html`, `report/report.md`, eight PNG/PDF figure
pairs and full downloadable tables. `report/media/` has15six-panel movies and a
small feedback gallery: p100/P80/uniform random, five themes,90executions.
All P80 files together last26.35seconds once (~4m24s at10plays/file).

Measurements cover recruitment/union coverage, whole/stage kinematics and timing,
constituent substitutions, same-raw-syllable inside/outside comparisons matched
on valid block/300s/occupancy quadrant, branching after all eligible roots,
exclusive-event repetition/transitions/returns, preceding-theme context,
frequency and first-order controls, primitive frequencies and movement baselines.
Unknown frames remain in modelled exposure but interrupt inter-theme adjacency.
Only299primary one-second-gap successor opportunities exist; many inter-chain
conditional estimates are sparse. Original overlapping events and all timeline
decisions remain available.

8,300finite primary tests:462nominal p<.05,146within-analysis-family BH results,
zero global-BH results. Of146family results,144are within-group longitudinal
changes,2are cross-sectional group differences and0are differences in changes.
Related endpoints, mirrors and sensitivities are not independent discoveries.
The two group results are early grooming-stage duration and a zero-gap
repetition-excess sensitivity. No broad corrected disease finding is claimed.

Examples worth inspecting:

- Pooled untreated travel–turn–travel: monomer0.705 versusPFF0.488events/min;
  difference−0.217,95%CI[−0.389,−0.045],p=.014,familyq=.168,globalq=.511;
  9monomer/12PFFmice. Independent enumeration of293,930mouse partitions gives
  p=.014565, within the stated Monte Carlo tolerance.
- First grooming stage at3weeks: PFF−monomer+1.039seconds,
  95%CI[.611,1.467],p=.0007,familyq=.0308,globalq=.277;8monomer/11PFFmice.
  This remains an exploratory, narrower-correction association.

Full results retain unsupported/nonsignificant rows, effect sizes, CIs, mouse
counts, opportunities and correction families. Pooled untreated means weight
available waves equally per mouse; complete-four-wave sensitivity is separate.
At24weeks only3monomer/6PFFmice remain. L-dopa confounds age/drug/wave. No Karen
DA analysis is implied; Jonathan-WT photometry is a separate optional modality.

## Reproduction and provenance

Use the existing lab interpreter; no environment was upgraded. Study execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python -B \
  -m studies.karen_pff_vs_monomer.analysis.run_theme_delivery \
  --fits outputs/karen_coherence_repair_20261016/fits \
  --output outputs/karen_theme_delivery_20261021 --stage all --execute
```

The group-blind library/index must exist first; their frozen JSON and inputs
are included as provenance. The media entry point `render_theme_delivery.py`
accepts the frozen index, explicit source/fits/delivery/output and `--execute`.
See launch JSONs and logs for exact commands, CPU threads, RAM/disk caps/timeouts.
The initial measurement stage took97.4seconds; fitting was not performed.

The lab editable install resolves shared fpcode functions from the primary
checkout, while study drivers run in the delivery worktree. Actual runtime
module paths/hashes are recorded in `runtime_library_sources.json`; copies of
those owners and the study/test sources accompany the portable report. Retain
the primary checkout revision as well as the dirty delivery source snapshot;
the output is not claimed reproducible from a clean delivery commit alone.

A floating-point zero-order-hold bug in the existing movie sampler was repaired
with a small float64 boundary tolerance. The new90selectedclips have exact1:1
native60Hzindices. Core definitions, all7,319descriptors/ranks and selected
occurrence identities stayed unchanged. The initial media/helper are archived;
older catalogues were not rerendered. Irregular-clock hold/duplicate-timestamp
semantics have independent regressions.

## Verification and limits

- `native_audit.json`: independent direct float64 calculations for all7,319
  occurrences and every prefix opportunity; raw order/masks/clocks, stage time
  conservation and physical measurements passed (rtol1e-7/atol2e-6).
- `statistics_audit.json`: every family/global q independently recomputed;
  mouse counts, travel-rate aggregation and exact-versus-Monte-Carlo p checked.
- `baseline_metadata.json`:315,608rawbouts conserve audited exposure; primitive
  merges preserve blocks; within-bout movement-pair counts/units are explicit.
- `media/media_audit.json`, `sampling_repair_validation.json` and browser checks:
  native mappings, all source dots, encodings, unchanged ranking, gallery saving,
  import/export and actual playback passed. Report search/pagination/filters,
  all12,160rows and eight images were browser checked.
- Final CPU922tests passed with6existing warnings. Lint retains reviewed debt;
  architecture and navigation checks are recorded separately. No golden fixtures
  or lint inventory were refreshed.

Computational checks do not establish the supplied behavioural labels, causal
effects, unseen-animal generalisation or a globally good chain generator. This
delivery fulfils the narrower exploratory-report scope; qualitative interpretation
and independent confirmation remain open. Existing work and labels are preserved.
