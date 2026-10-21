# Karen clustering-resolution comparison

Output: `outputs/karen_clustering_levels_20261012/report.html` and CSVs beneath
the same root. The existing raw atlas is retained.

Policy lives in `karen_hierarchy.py`: existing L1/L2/L3/L4 memberships (100/93/
22/14 states), plus unresolved-preserving L3/L4 sensitivities (69/61 states).
These are hand-curated annotation groupings. Generic remapping, conditional
member composition, fixed-mixture raw execution and anchored-template detection
live in `fpcode.behavior.resolutions`, `fpcode.workflows.resolution_atlas` and
`fpcode.workflows.anchored_sequences`. `extract_measurements` accepts an explicit
`label_map` and recomputes RLE after mapping while preserving missing samples.

Run `python -B -m studies.karen_pff_vs_monomer.analysis.run_clustering_atlas`
with explicit `--atlas`, `--output`, `--stage` and `--level`. Stages are prepare,
measurements, chains, anchors and statistics. L1 measurements/chains reuse the
raw atlas. Fresh discovery is scoped to L2/L3/L4 and uses the same animal split;
unresolved-preserving sensitivities use measurements and frozen raw anchors.
`report_clustering_atlas.py --atlas ... --output ...` combines completed results.
The statistics stage should follow chains/anchors so all planned features exist.

The original 30 training-derived exact motifs are mapped, RLE-collapsed and
deduplicated at every level; collapsed motifs shorter than three bouts are
explicitly logged. Factor numbers are local identities. Interpretability flags
identify unresolved members. Full outputs retain them, but the main shortlist
excludes them. Fixed-mixture execution uses untreated discovery-mouse weights
and ≥80% reference-weight coverage; it is a descriptive raw-member sensitivity,
not merged-bout execution or causal mediation.

The final comparisons ran independent levels in parallel with
`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`; both NumPy/SciPy OpenBLAS pools were
checked at four threads. Earlier discovery/measurement processes were already
running with one BLAS thread per process when the user requested multithreading;
their finished numerical results were retained rather than fitted again.
No raw pickle reload, KPMS fit, original-atlas overwrite or environment upgrade
was needed. Preserve age/wave/L-DOPA confounding and the nine-mouse 24-week
baseline limitation. Per-level q and across-primary-level family/scope q are
different corrections; correlated aliases and resolutions are not independent
discoveries. Lower absolute reconstruction error in fewer states is not proof
of better routine discovery.
