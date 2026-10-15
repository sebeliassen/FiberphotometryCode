# Karen behaviour atlas

The 2026-10-11 output root is `outputs/karen_behavior_atlas_20261011`.
`report.html` is a portable report with embedded scientific figures; `report.md`
is its editable counterpart. CSVs retain individual sessions, animal/wave
summaries, uncertainty, support and exploratory follow-up rankings.

This study owns the PFF/monomer roster, wave definitions, latest annotations,
100-state vocabulary and candidate/report policy. Reusable implementation lives
in `src/fpcode`: `behavior/phenotyping.py`, `behavior/sequences/{chains,seqnmf,
prediction}.py`, `behavior/variants.py`, `tracking/pose_measurements.py`,
`statistics/{behavior_comparisons,followup_planning}.py`, `workflows/*atlas*.py`,
`workflows/temporal_motifs.py` and `viz/{behavior_atlas,event_clips}.py`.
The new CPU seqNMF solver implements data-dependent cross-factor competition;
it does not claim MATLAB-reference numerical identity or replace old studies.

## Reproduce the authorised run

Use an interpreter with NumPy, pandas, SciPy, h5py, matplotlib and OpenCV for
clips. The report's Markdown tables use `tabulate`. Do not upgrade the shared
environment as part of this run. Pickle loading executes Python reconstruction;
only load an explicitly trusted/authorised study file. Extraction verifies all
included labels against the named model results and native-frame mapping.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ATLAS_PYTHON=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
ATLAS_OUTPUT=outputs/karen_behavior_atlas_20261011
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/extract_atlas_inputs.py --help
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/run_behavior_atlas.py --output "$ATLAS_OUTPUT" --stage measurements
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/run_behavior_atlas.py --output "$ATLAS_OUTPUT" --stage chains
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/run_behavior_atlas.py --output "$ATLAS_OUTPUT" --stage extensions
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/run_behavior_atlas.py --output "$ATLAS_OUTPUT" --stage pose --fused-dir data/fused/multiview/cohort_fused_static_21kp_v2
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/run_behavior_atlas.py --output "$ATLAS_OUTPUT" --stage temporal --workers 8
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/run_behavior_atlas.py --output "$ATLAS_OUTPUT" --stage statistics
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/report_behavior_atlas.py --output "$ATLAS_OUTPUT"
"$ATLAS_PYTHON" -B studies/karen_pff_vs_monomer/analysis/render_atlas_examples.py --help
```

Exact extraction paths and their size/mtime are recorded in the output
`input_paths.json` and extraction provenance. The approved source was
`data/sessions/sessions_karen_project_tandem_cohort_static_21kp_seed101.pkl`,
with results and VFI under `data/models/cohort_static_21kp_v2`; annotations came
from `compat/syllable_generation/metadata/karen_static_21kp_v2_syllable_labels.csv`.
The original video references were relocated by exact basename to the explicitly
selected `data/raw/karen_pff_vs_monomer/video_files` root for clips. No source
pickle or video reference was rewritten.

`chain-refresh` recomputes movement on frozen calls and adds held-out order nulls,
without rediscovering templates. `temporal --reuse-native-templates` reuses saved
native templates and reruns inference; `--reuse-native` requires completed native
usage and refreshes only micro-motifs. Cache options are explicit and recorded.
The first temporal inference was interrupted and restarted with saved templates
and eight CPU workers; final products use corrected movement estimates.

## Reading the outputs

- `csv_manifest.csv`: file index; per-session bout CSVs are in
  `measurements/bouts_by_session` and are omitted from this compact inventory.
- `feature_dictionary.csv`: feature names, annotations and units.
- `statistics/session_features_long.csv`: full session feature atlas.
- `statistics/animal_wave_measurements.csv`: independent-mouse panel.
- `statistics/group_contrasts.csv`: per-wave, pooled and difference-of-change
  contrasts; PFF minus monomer, bootstrap intervals, permutations, Hedges g,
  family/scope and global BH q.
- `statistics/wave_adjusted_contrasts.csv`: additive group + wave regression
  with CR1 covariance clustered by mouse; separate from permutation inference.
- `statistics/paired_wave_changes.csv`: within-group paired changes, including
  untreated progression and the separate L-DOPA wave.
- `statistics/promising_effects.csv` and `followup_shortlist.csv`: include
  nonsignificant effects. Ranking considers magnitude, week direction and
  leave-one-animal-out stability. The heuristic is not independent replication.
- `statistics/prospective_*_scenarios.csv`: independent equal-size t-test design
  sensitivities, including fixed true d and explicit pilot-magnitude shrinkage
  assumptions. These are not achieved power or promised replication.
- `chains`: frozen templates, train/validation/test mouse split, all calls,
  exclusive-stream grouping/transition summaries, motif-prefix edits and nulls.
- `measurements`: usage, execution, phase and native-onset profiles, recording
  time bins, variants, history prediction, chain context and 2D pose/micro-motifs.
- `examples/gallery.html`: 40 native-frame clips, not behaviourally reviewed.
- `figures`: PNGs and standalone PDFs. `provenance`: executed stage records.

## Interpretation limits

Pre/post is earlier weeks versus the separate L-DOPA wave. Age, wave and drug
are confounded. The closest 24-week baseline has only nine mice. Repeated
sessions do not increase independent N. Pooling averages available waves per
mouse; the wave-adjusted panel is exported separately.

RLE already collapses repeated frames into bouts. Preserve that protection and
overlap/peak suppression. A 0–5-second grouping sensitivity addresses successive
chain calls; it does not establish biological restart intent. Prefix completion
is observed sequence continuation, with censored prefixes excluded.

Nine-frame position/circular-heading smoothing precedes derivatives; speed uses
an additional five-frame median. Native video rate is 60 Hz, but this does not
make 150-ms-filtered derivatives high-frequency physiological measures. Pose is
projected 2D, not calibrated millimetres, height or paw contact. Inspect tracking
before interpreting relative paw-motion candidates. Unassigned/Dump states and
seqNMF factors require video review. Exact templates and model calls are frozen
for the current comparisons, but the entire atlas remains an exploratory pilot.
