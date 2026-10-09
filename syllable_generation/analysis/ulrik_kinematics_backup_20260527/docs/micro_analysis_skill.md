# SKILL DOC: Micro-Kinematics Extraction

This document outlines the architecture and math behind the Micro-Kinematics module (`lib/micro_kinematics.py`) in the `ulrik_kinematics` pipeline. It acts as an AI reference guide when extending or modifying individual continuous behavioral physics bounds.

## 1. Goal & Execution
The goal of this module is to move beyond the macro-structural *choices* made by the animal (e.g. Motif loops) and to investigate the *micro-physics* of execution. If both Genotypes execute identical behavioral frames, does the Knock-In phenotype execute the sequence string mechanically faster or with rougher torque?

### Endpoints
The pipeline hooks directly into the `run_analysis.py` orchestrator:
- `python run_analysis.py --verify-micro` (Executes deterministic physics aggregation and duration crush testing).
- `python run_analysis.py --run-micro-discovery --dataset all` (Generates split violin distributions in both `alphabet` and `standard` label modes).

## 2. Core Mining Logic (`extract_bouts`)
The engine parses the continuous raw telemetry into distinct independent units of behavior:
1. **Contiguous Chunking**: It identifies independent 'bouts' by tracking run-length encoded boundary splits in the `syllable` timestamp array.
2. **Raw Array Tracing**: For each contiguous chunk, it pulls velocity and angular velocity, computing:
   - `Mean_Velocity` — typical effort for that bout.
   - `Median_Velocity` — robust (outlier-resistant) effort.
   - `Mean_Angular` / `Median_Angular` — typical and robust turning rates.

## 3. Angular Velocity Wrap-Around Fix (CRITICAL)

> ⚠️ The pre-baked `angular_velocity` column in `all_data_df` contains wrap-around spikes reaching ±188 rad/s. These are caused by the heading value jumping from ≈ -π to ≈ +π, which produces a raw delta of ≈ 2π × fs ≈ 188 rad/s.

**The fix**, implemented in `extract_bouts`, is to always recompute from `heading` using the `arctan2` method:
```python
if "heading" in df.columns:
    dh = np.diff(h)
    dh_fixed = np.arctan2(np.sin(dh), np.cos(dh))  # Clamps delta to [-π, +π]
    ang_arr = np.concatenate(([0], dh_fixed * fs))
```
Do **not** trust or use the `angular_velocity` column directly for bout-level analysis. The remaining long tails in violin plots are real biological variance (fast head-turning bouts), not artifacts.

## 4. Dual-Style Output
`run_discovery(sessions, project_name, style)` is called twice per dataset:
- `style="alphabet"` → unbiased frequency-rank identifiers (A, B, C...)
- `style="standard"` → human-curated names from `manual_labels.csv`

Output filenames are tagged with the style: `{project}_{drug}_{style}_micro_violin.png`.

## 5. Group-Mean-Centred Residual Plots (Normalised)

Beyond the **absolute** violin plots, each run also generates **normalised residual** figures:
```python
group_means = df_norm.groupby("Genotype")[col].transform("mean")
df_norm[col + "_norm"] = df_norm[col] - group_means
```
- **Purpose**: Factors out "this genotype just moves faster globally" and reveals whether a kinematic difference is **syllable-specific** or purely a group-level confound.
- **Zero line**: A dashed black line marks the group mean (at 0), making residual direction immediately interpretable.
- Normalised files are tagged: `*_micro_violin_normalised.png`.

## 6. Verification Check Framework (`verify_micro_kinematics`)

1. **Global Mathematical Conservation (Invariance Check)**
   - The script multiplies each micro-bout's mean velocity by its frame count and sums to reconstruct the global mean. Must match to within `< 0.01`.

2. **Desync Shredder (Structural Ejection)**
   - Inserts random frame deletions masking 20% of the sequence stream. Requires a `> 40%` crash in mean bout duration (empirically confirms `93.0%` collapse).
