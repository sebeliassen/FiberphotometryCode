# General chain organisation: executed workflow

Executed 2026-10-12 against the verified Karen frame cache and frozen calls.
The [execution plan](GENERAL_CHAIN_ORGANISATION_PLAN.md) records the scientific
definitions. The current deliverables are in
`outputs/karen_general_chain_organisation_20261012/report.html` and `report.md`.
Earlier atlases and empirical inputs are preserved.

## Owners

- `fpcode.behavior.sequences.organisation`: deterministic native-frame overlap
  decisions, complete interval timelines, exclusion barriers, successor masks,
  censored repetition runs, return windows and sufficient counts.
- `fpcode.statistics.chain_nulls`: exact finite-population frequency expectations,
  batched count-preserving shuffles and explicit first-order simulations.
- `fpcode.workflows.chain_organisation`: session/library measurement, compressed
  audits, all-library leave-chain-out and mouse/wave count pooling.
- `fpcode.statistics.behavior_comparisons.compare_feature_atlas_parallel`:
  bounded independent feature batches, existing mouse contrasts and joint FDR
  after joining chunks. Serial/parallel results have an independent equality test.
- Study `run_general_chain_organisation.py`: frozen Karen libraries, scopes,
  comparisons, resource configuration, provenance and output selection.
- Study `report_general_chain_organisation.py`: interpretation, figures and CSV index.

## Reproduce

Select your interpreter; the lab interpreter below is an example. Thread settings
precede startup, and the runner also limits already loaded pools at runtime.
Eight workers with four numerical threads each match the observed 32 CPUs.
`--memory-mb` caps conservative surrogate allocations per worker, including
Markov state matrices; interpreters/input/output tables are additional. Statistical
input chunks have a separate 3x allocation allowance using the same budget.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python=/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python
"$python" -B -m studies.karen_pff_vs_monomer.analysis.run_general_chain_organisation \
  --atlas outputs/karen_behavior_atlas_20261011 \
  --clustering outputs/karen_clustering_levels_20261012 \
  --output outputs/karen_general_chain_organisation_20261012 \
  --stage all --workers 8 --threads 4 --memory-mb 256 --replicates 1999
"$python" -B -m studies.karen_pff_vs_monomer.analysis.report_general_chain_organisation \
  --output outputs/karen_general_chain_organisation_20261012
```

Use a new output root to retain a checkpoint. Individual `prepare`, `measure`
and `statistics` stages are supported. `--limit-sessions` is an explicit profiling
scope and should use a separate output; it is not a complete-cohort result.
There is no pickle loading, KPMS/seqNMF refitting or accelerator initialization.

## Scope and interpretation

Fourteen frozen libraries cover six anchor resolutions, four exact discovery
libraries and four seqNMF libraries. Each has inclusive and named-member scopes,
with 0/0.5/1/2/5-second gaps. The original RLE and call detection protections are
retained. New deterministic overlap selection uses the inclusive library before
membership exclusions; the primary resolver comparison is exported.

Primary: named L3 anchors, one-second gap, self-transition frequency excess,
singleton-inclusive mean run length and two-step return frequency excess.
Exact frequency expectations cover every gap. Frequency simulation intervals
cover g1 in every scope/library. First-order and recording-quarter diagnostics
cover L3 anchors g1. All-library leave-chain-out retains removed calls as barriers.
First-order fitting uncertainty is not integrated; duration shuffles are not used
to claim duration significance. Null replicates and overlapping return windows
are never independent mice.

The primary library covers only median 4.1% of modelled time, with 203 eligible
pairs and 21 two-step windows across 99 recordings. The untreated loop comparison
has only three monomer/two PFF mice with opportunities. Most primary runs are
censored and singleton. The report therefore describes detected routines and
shows endpoint-specific support rather than implying universal behavioural
persistence or interpreting all ends as exits.

Mouse/wave probabilities pool counts/opportunities, rates pool counts/exposure
and run means pool call sums/run counts. Nonadditive diagnostics use recording
means or longest-run maximum. Available waves receive equal weights per mouse
in pooled summaries. Equal-session and balanced early-attendance sensitivities,
paired untreated/later-wave changes and separate wave-adjusted regression are
exported. PFF/monomer are experimental groups. The later L-DOPA wave confounds
age, drug and wave; the 24-week paired baseline has at most nine mice.

Non-significant effects remain visible. Lower routine-start rates largely track
lower detection rates. A named L3 seqNMF self-excess candidate is positive at
weeks 8/16 and negative at week 3; named L4 repeat-run means tend slightly lower
in PFF. These are selected pilot patterns, with broad correction, uncertainty,
mouse/chain influence and previous limited seqNMF order-null evidence shown.

## Products and validation

Top-level CSVs include policy/library/support, counts and mouse panels, null
summaries, contrasts, candidate rankings, sensitivity and future-cohort scenarios.
`audit/<library>/<session>/` contains compressed call decisions, selected calls,
full interval coverage, successor decisions and repetition runs. Primary anchors
also retain return-window audits. Original overlapping calls are retained in the
decision files. `csv_index.csv` inventories top-level tables;
`feature_definitions.csv` describes units/aggregation. Figures have PNG/PDF pairs,
and the HTML embeds PNGs for portable reading.

`validation.json` records final repository checks, numerical/count/clock audits,
input/source fingerprints, actual thread pools, memory bounds and measured timing.
Synthetic checks establish the stated toy expectations, not universal scientific
equivalence. Review labels, coverage, detector/overlap sensitivity and an independent
cohort before treating exploratory patterns as established biological routines.
