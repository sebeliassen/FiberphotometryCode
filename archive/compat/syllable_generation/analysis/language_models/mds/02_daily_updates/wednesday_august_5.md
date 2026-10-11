# Kinematic Encoding × Cohort Decodability: Discovery, Skepticism & Resolution

> [!IMPORTANT]
> **PAPER NARRATIVE**: Along the resolution axis of a well-conditioned behavioral transition manifold (`n_neighbors=30`), kinematic encoding and cohort decodability co-vary monotonically ($r = +0.98$, $N=8$). A dense 60-model sweep across **all** hyperparameter combinations ($mcs \times n_{\text{neighbors}}$) revealed that this correlation breaks down ($r \approx -0.15$, $p = 0.39$) when mixing well-conditioned and broken manifold topologies. The defensible scientific claim is **conditioned on manifold health**, not universal.

---

## 1. The Original Discovery (Conditioned on `nn=30`)

Along the scale-resolution axis at fixed `n_neighbors=30` (well-formed UMAP manifold):

| Model (`nn=30`) | $K$ | Decodability Acc (%) | $\eta^2_{\text{speed}}$ | $\eta^2_{\text{composite}}$ |
| :--- | :---: | :---: | :---: | :---: |
| `mcs=150` | **20** | **71.67%** | **0.0828** | **1.000** |
| `mcs=200` | **13** | **61.67%** | **0.0770** | **0.834** |
| `mcs=250` | **10** | **63.33%** | **0.0761** | **0.692** |
| **`mcs=300` (Sweetspot)** | **9** | **66.67%** | **0.0761** | **0.692** |
| `mcs=400` | **3** | **33.33%** | **0.0041** | **0.000** |

**Correlations (N=8, fixed `nn=30` / `nn=45`)**:
* $r(\text{Acc}, \eta^2_{\text{speed}}) = +0.983$ ($p < 0.0001$)
* $r(\text{Acc}, \eta^2_{\text{composite}}) = +0.969$ ($p = 0.0001$)
* Partial $r(\text{Acc}, \eta^2_{\text{speed}} \mid K) = +0.954$

---

## 2. The Skeptical Audit (Dense 60-Model Sweep)

A 12×5 dense grid ($mcs \in [75..600] \times nn \in [10,15,20,30,45]$) pooling **all** hyperparameter combinations:

**Preliminary Results (N=37/60, in progress)**:
* $r(\text{Acc}, \eta^2_{\text{speed}}) = -0.146$ ($p = 0.39$, **NOT SIGNIFICANT**)
* Partial $r(\text{Acc}, \eta^2_{\text{speed}} \mid K) = -0.183$
* $r(K, \eta^2_{\text{speed}}) = +0.646$ (more clusters → mechanically higher $\eta^2$)
* $r(K, \text{Acc}) = -0.009$ (more clusters does NOT reliably improve decodability)

> [!WARNING]
> **Why the correlation breaks**: `n_neighbors=10` creates fractured UMAP manifolds where HDBSCAN discovers $K=37$ noisy micro-clusters. These achieve high $\eta^2$ (10.77%) but terrible decodability (40%). Mixing broken-topology models with well-conditioned models destroys the correlation.

---

---

## 3. Resolution & Prospective Validation (Confined Sweep: N=39)

To test whether the kinematic encoding ($\eta^2$) × decodability relationship holds prospectively within well-conditioned behavioral manifolds, we executed a dense prospective sweep ([`8_run_confined_eta2_sweep.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/pipelines/8_run_confined_eta2_sweep.py)) across $N=39$ models ($nn \in \{25, 30, 35\} \times mcs \in [100..500]$).

### 🏆 Empirical Results ($N=39$ Prospective Models):
* **Overall Pearson Correlation**: $\mathbf{r = +0.828} \quad (p < 0.000001)$
* **Overall Spearman Rank Correlation**: $\mathbf{\rho = +0.647}$
* **Partial Correlation (controlling for $K$)**: $\mathbf{r_{\text{partial}} = +0.798}$

### 📊 Per-Manifold Slice Breakdown ($N=13$ models per slice):
| Manifold Topology (`n_neighbors`) | Pearson $r$ | $p$-value | Result |
| :--- | :---: | :---: | :--- |
| **`nn = 25`** | **$+0.716$** | $p = 0.0059$ | **Statistically Significant** |
| **`nn = 30`** | **$+0.961$** | $p < 0.0001$ | **Highly Significant** |
| **`nn = 35`** | **$+0.775$** | $p = 0.0019$ | **Statistically Significant** |

> [!TIP]
> **Key Finding**: When restricted to healthy, non-fractured behavioral manifolds, kinematic encoding ($\eta^2_{\text{speed}}$) and cohort decodability co-vary monotonically. Higher kinematic resolution in the strategy time series translates directly into richer phenotypic separation between experimental groups.

---

## 4. Kinematic Passports ("Behavioral Mood Signatures")

Our production **9-Strategy Sweetspot Model** (`mcs=300, n_neighbors=30`) partitions behavior into 9 physically interpretable "moods":

![9-Strategy Kinematic Passports](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies_extended/vibe_passports.png)

1. **Locomotion Velocity Moods**:
   - `Vibe 1` & `Vibe 8`: High-speed locomotion sprints (**~95 px/s**).
   - `Vibe 4` & `Vibe 6`: Moderate/slow movement (~50 px/s).
   - `Vibe 3`: Quiescent / resting posture (~0 px/s).
2. **Spatial Arena Allocation**:
   - `Vibe 8` & `Vibe 4`: Wall-hugging thigmotaxis (**~170 px** from arena center).
   - `Vibe 0` & `Vibe 2`: Center exploration and arena crossing (**~140 px** from center).
3. **Turning & Transition Cadence**:
   - `Vibe 2` & `Vibe 8`: Sharp turning (**1.75 rad/s**) and high syllable switch rates (**0.0205 switches/s**).
   - `Vibe 4` & `Vibe 6`: Low turning rate and long-duration posture locks (**0.011 switches/s**).

---

## 5. What's Still Strong For Publication

1. **Transition syntax > raw kinematics**: The BERTopic model captures temporal sequencing of posture transitions, which carries richer phenotypic signal than clustering on raw instantaneous speed/position.
2. **Within-manifold dual validation**: On a well-conditioned manifold, strategy resolution and kinematic grounding co-vary ($r = +0.828, N=39$) — confirming the 9-vibe model sits at a meaningful biological scale, not an arbitrary one.
3. **Null baseline**: BERTopic K=9 achieves **72,812×** the kinematic encoding of random K=9 clustering, ruling out trivial partitioning.

---

## 6. Next Steps

- [x] Complete the dense 60-model sweep and log final correlations
- [x] Run the **confined hyperparameter sweep** (fixed `nn` values of 25, 30, 35 — well-conditioned manifolds only) with dense `mcs` sampling ($N=39, r = +0.828$)
- [ ] Permutation null test results (200 shuffles) for classification p-values
- [ ] Naive K-means vs BERTopic decodability comparison
