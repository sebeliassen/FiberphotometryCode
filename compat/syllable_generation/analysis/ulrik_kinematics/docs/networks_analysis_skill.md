# SKILL DOC: Behavioral Transition Networks

This document explains the architecture and math behind the Networks module (`lib/networks.py`) in the `ulrik_kinematics` pipeline. It acts as an AI reference guide when modifying or extending the Markov transition graph analysis.

## 1. Goal & Execution
The goal of this module is to map the **global topology** of the behavioral repertoire as a directed graph. While Motifs examine specific 3-gram sequences, Networks reveal the bird's-eye view: which behaviors act as "hubs" and which act as "exit ramps", and whether the Knock-In phenotype structurally rewires these pathways.

### Endpoints
The pipeline hooks into the `run_analysis.py` orchestrator:
- `python run_analysis.py --verify-networks --dataset amph` (Runs the row-stochastic invariance check).
- `python run_analysis.py --run-network-discovery --dataset all` (Generates all absolute and differential graphs for both styles/datasets).

## 2. Tunable Constants

Defined at the **top of `networks.py`** for easy adjustment:
```python
EDGE_CUTOFF = 0.03        # Min transition probability to draw an edge (absolute graphs)
EDGE_DELTA_CUTOFF = 0.02  # Min absolute shift to draw an edge (delta graphs)
```

## 3. Core Logic (`build_transition_matrix`)

1. Collapses the syllable stream into a run-length-encoded sequence (self-transitions removed).
2. Drops gap frames (syllable = -1) and unmapped/UNK syllables.
3. Computes bigrams for all consecutive pairs, filtering to the Top N most common behaviors.
4. Normalises by row to produce a **row-stochastic matrix** (each row sums to 1.0).
5. Returns `(trans_df, freq_series)` — the matrix and per-node usage frequency.

## 4. Graph Layout

All graphs use `nx.circular_layout` (polygon geometry):
- **Nodes** sit at equidistant polygon corners, sized by usage frequency.
- **Labels** are pushed radially outward using unit-norm vector offset: `label_pos = (x + dx/‖d‖×0.5, y + dy/‖d‖×0.5)`.
- `ax.margins(0.2)` is essential to prevent label clipping at polygon extremes.
- Edge arc curvature set to `connectionstyle='arc3,rad=0.15'` to keep overlapping bidirectional arrows distinguishable.

## 5. Differential (Delta) Graphs (`plot_delta_network`)

For comparing two groups, matrices are subtracted element-wise:
```python
delta = mat_test.subtract(mat_reference, fill_value=0)
```
`fill_value=0` is critical — it ensures nodes present in only one matrix are treated as zero probability in the other, preventing NaN bleed.

- **Red edges** = positive shift (test group amplified this transition).
- **Blue edges** = negative shift (test group suppressed this transition).
- Edge width scales with `abs(delta) * 15.0`.
- Threshold uses `EDGE_DELTA_CUTOFF` (not `EDGE_CUTOFF`).

## 6. Three Analytical Routes (AMPH)

All computed and saved within `run_discovery` after the per-drug loop:

| Output File | Formula | Scientific Question |
|---|---|---|
| `*_{drug}_{style}_network.png` | Raw P(WT) and P(KI) | What does each group's backbone look like? |
| `*_delta_baseline_{style}_network.png` | P(KI\_Saline) - P(WT\_Saline) | What is the innate genotypic rewiring? |
| `*_dod_amphetamine_{style}_network.png` | (KI\_Amph - KI\_Saline) - (WT\_Amph - WT\_Saline) | What pathways does Amphetamine uniquely rewire in KI? |

## 7. Verification (`verify_networks_module`)

Asserts that all rows in the computed transition matrix sum to exactly 1.0 (tolerance `< 0.0001`). This confirms the Markov probability conservation law is not violated by any filtering or mapping step.
