# Controlled Benchmark Report: 17kp vs 21kp Keypoint Selection on 10-Session Pilot

**Date:** 2026-10-06  
**Cohort:** 10 Pilot Sessions (18 continuous segments, 1,735,420 valid frames)  
**Hardware:** NVIDIA GeForce RTX 4090 GPU  
**Regimen:** 50 AR + 150 SLDS = 200 Gibbs Sampling Iterations, $\kappa_{\text{init}} = 1.0\times 10^6 \to \kappa_{\text{SLDS}} = 1.5\times 10^{12}$, Snapshots every 25 iterations.

---

## 1. Executive Summary & Verdict

We executed a fully controlled, 200-iteration head-to-head benchmark across three architectures:
1. **Arm 1 (`17kp` Baseline):** Dorsal TopCam only (17 keypoints).
2. **Arm 2 (`21kp_static` Occam's Razor):** 17 dorsal TopCam + 4 ventral paws projected via a rigid static affine transformation ($r \ge 0.9993$).
3. **Arm 3 (`21kp_dynamic` Control):** 17 dorsal TopCam + 4 ventral paws from dynamic re-anchoring with bottom nose/tail removed.

### Primary Conclusions:
1. **Static vs. Dynamic 21kp is Decided:**  
   `21kp_static` outperforms `21kp_dynamic` across every postural metric:
   - **Dorsal $R^2$:** $+34.2\%$ improvement ($0.1655$ vs $0.1233$).
   - **Same-$K$ Ceiling Ratio:** $+28.8\%$ improvement ($0.2322$ vs $0.1803$).
   - **Absolute Dorsal Variance in Top 10 PCs:** $+54.5\%$ gain ($329.33\text{ px}^2$ vs $213.14\text{ px}^2$).  
   *Dynamic per-frame re-anchoring introduces artificial noise even when bottom anchors are removed. Static affine registration is definitively superior.*

2. **17kp vs. 21kp_static Decision:**
   - **Temporal Scale:** At $\kappa = 1.5\times 10^{12}$, `17kp` segments **macro-postures** (median duration **1,633 ms = 98 frames**), whereas `21kp_static` segments **micro-gait stride cycles** (median duration **266.7 ms = 16 frames** = ~4 strides/sec).
   - **Dorsal Posture Retention:** `21kp_static` preserves **$80.7\%$** ($329.33\text{ px}^2$ of $407.95\text{ px}^2$) of absolute dorsal posture variance in the top 10 PCs, while unlocking **$762.27\text{ px}^2$** of paw gait dynamics.
   - **Syllable Homogeneity Trade-off:** Dorsal snippet $R^2$ drops by only **$0.031$** ($0.1963 \to 0.1655$) when adding paws.

---

## 2. Complete Head-to-Head Scorecard

| Metric | Arm 1: `17kp` Baseline | Arm 2: `21kp_static` | Arm 3: `21kp_dynamic` | Target / Direction |
| :--- | :---: | :---: | :---: | :--- |
| **Total Keypoints** | 17 | **21** | 21 | — |
| **Active Syllables** ($\ge 0.1\%$) | 38 | 62 | 50 | 40–80 |
| **Median Bout Duration** | 1,633.3 ms (98 frames) | **266.7 ms (16 frames)** | 250.0 ms (15 frames) | 350–450 ms (21–27 fr) |
| **Syllable Entropy** | 4.44 bits | **5.34 bits** | 4.76 bits | Higher = richer repertoire |
| **Late Convergence ARI** ($\ge 100$ iters) | **0.9511** | 0.8838 | 0.9333 | $\ge 0.85$ (Stable plateau) |
| **Late Frame Flip Rate** | **2.27%** | 6.53% | 3.34% | $\le 10\%$ between snapshots |
| **Dorsal Snippet $R^2$** | **0.1963** | **0.1655** | 0.1233 | Higher = purer posture |
| **Same-$K$ KMeans Ceiling** | 0.6789 | 0.7126 | 0.6839 | Theoretical upper bound |
| **Same-$K$ Ceiling Ratio** | **0.2891** | **0.2322** | 0.1803 | Higher = closer to ceiling |
| **Abs. Dorsal Var. (Top 10 PCs)** | **$407.95\text{ px}^2$** | **$329.33\text{ px}^2$** | $213.14\text{ px}^2$ | Maximize retained signal |
| **Abs. Paw Var. (Top 10 PCs)** | $0.00\text{ px}^2$ | **$762.27\text{ px}^2$** | $433.37\text{ px}^2$ | Gait stride amplitude |
| **Dorsal PCA Share (%)** | 100.0% | 30.2% | 33.0% | Compositional metric |
| **Paw PCA Share (%)** | 0.0% | 69.8% | 67.0% | Compositional metric |
| **Training Runtime (200 iters)** | 15.85 min | 13.74 min | 11.50 min | — |

---

## 3. Syllable Landscape Convergence & Smoothing Analysis

As Gibbs sampling progresses, the syllable landscape rapidly stabilizes following the SLDS transition:

### Consecutive Snapshot ARI & Frame Flip Rate (%)

| Snapshot Interval | Arm 1 (`17kp`) ARI / Flip | Arm 2 (`21kp_static`) ARI / Flip | Arm 3 (`21kp_dynamic`) ARI / Flip |
| :--- | :---: | :---: | :---: |
| **Iter 0 $\to$ 25** (AR) | $0.003$ / $97.3\%$ | $0.004$ / $96.9\%$ | $0.011$ / $97.3\%$ |
| **Iter 25 $\to$ 50** (AR) | $0.456$ / $33.0\%$ | $0.461$ / $34.2\%$ | $0.373$ / $45.1\%$ |
| **Iter 50 $\to$ 75** (SLDS onset) | $0.256$ / $47.6\%$ | $0.262$ / $49.8\%$ | $0.258$ / $58.5\%$ |
| **Iter 75 $\to$ 100** | $0.913$ / $4.2\%$ | $0.852$ / $8.9\%$ | $0.868$ / $7.3\%$ |
| **Iter 100 $\to$ 125** | $0.937$ / $3.0\%$ | $0.871$ / $7.4\%$ | $0.908$ / $4.7\%$ |
| **Iter 125 $\to$ 150** | $0.949$ / $2.4\%$ | $0.884$ / $6.6\%$ | $0.928$ / $3.5\%$ |
| **Iter 150 $\to$ 175** | $0.956$ / $2.0\%$ | $0.890$ / $6.1\%$ | $0.948$ / $2.7\%$ |
| **Iter 175 $\to$ 200** | **$0.962$ / $1.7\%$** | **$0.894$ / $5.8\%$** | **$0.949$ / $2.4\%$** |

All three models reach a plateau where fewer than **$6\%$** of frames change label over 25 Gibbs iterations.

---

## 4. Architectural Recommendation for Full Cohort Training

If the experimental objective is:
1. **Pure Macro-Posture Analysis (Turning, Grooming, Rearing, Freezing):**  
   $\to$ **Choose 17kp**. It yields higher within-syllable dorsal posture purity ($R^2 = 0.196$), eliminates all cross-camera tracking noise, and avoids centroid gait wobble.
2. **Multiview Locomotion & Gait Analysis (Stride dynamics, Stance/Swing phase, Paw alternation):**  
   $\to$ **Choose 21kp_static**. It preserves **$80.7\%$** of dorsal posture variance while providing rich 16-frame gait syllables. Never use dynamic re-anchoring.
