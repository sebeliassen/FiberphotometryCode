# Data Science & Mathematical Guide: The 7 Core Phenotyping Metrics

This document provides a clean, empirical data-science reference for the **7 core behavioral metrics** extracted from tracking coordinates and Keypoint-MoSeq discrete syllable sequences in the Karen dataset. 

It explains **how each metric is computed mathematically**, gives an **intuitive pedagogical explanation**, and documents the **exact observed differences between PFF and Monomer cohorts** without biological speculation.

---

## Summary Matrix of the 7 Metrics

| # | Metric Name | Timescale & Level | Mathematical Symbol | Formula | Observed Data Difference (PFF vs. Monomer) | $p$-value | Permutation ($p_{\text{perm}}$) |
| :-: | :--- | :-: | :-: | :--- | :--- | :-: | :-: |
| **1** | **Mean Locomotor Speed** | Global (40 min) | $\langle v \rangle$ | $\frac{1}{T} \sum_{t=1}^T v(t)$ | **$-18.2\%$ slower in PFF** ($61.2$ vs $74.8\text{ px/s}$) | $p = 0.0310$ | $p = 0.0110$ |
| **2** | **Peak Burst Velocity** | Burst ($v_{95}$) | $v_{95}$ | $\inf \{v : F_V(v) \ge 0.95\}$ | **$-24.7\%$ lower sprint peak in PFF** ($175.2$ vs $232.5\text{ px/s}$) | $p = 0.0320$ | $p = 0.0150$ |
| **3** | **Normalized Movement Jerk** | Kinematic (80 Hz) | $\mathcal{J}_{\text{norm}}$ | $\frac{\langle \|\dddot{\mathbf{x}}(t)\| \rangle}{\langle v(t) \rangle + \epsilon}$ | **$+11.0\%$ more erratic acceleration in PFF** ($2627.8$ vs $2366.9$) | $p = 0.0341$ | $p = 0.0264$ |
| **4** | **Macro Locomotor Bout Length** | Macro (~2.5–4.0 s) | $\bar{\tau}_{\text{macro\_bout}}$ | $\frac{1}{B} \sum_{b=1}^B (t_{\text{stop}}^{(b)} - t_{\text{start}}^{(b)})$ | **$-12.0\%$ shorter continuous runs in PFF** ($2.61$ vs $2.97\text{ s}$) | $p = 0.0287$ | $p = 0.0287$ |
| **5** | **Micro Syllable Dwell Time** | Micro (~250–500 ms) | $\bar{\tau}_{\text{syll\_dwell}}$ | $\frac{1}{M} \sum_{m=1}^M \frac{L_m}{f_s}$ | **$+14.9\%$ longer state dwell in PFF** ($0.569$ vs $0.495\text{ s}$) | $p = 0.1066$ | $p = 0.1628$ |
| **6** | **Stationary Immobility** | Global (%) | $\Phi_{\text{freeze}}$ | $\frac{1}{T}\sum_{t=1}^T \mathbb{I}(v(t) < 5\text{ px/s})$ | **$+11.1\%$ more stationary time in PFF** ($1.05\%$ vs $0.94\%$) | $p = 0.0402$ | $p = 0.0405$ |
| **7** | **Syntax Entropy Rate** | Bigram Syntax | $H(S_{t+1} \mid S_t)$ | $-\sum_{i,j} P(s_i, s_j) \log_2 P(s_j \mid s_i)$ | **$+2.5\%$ higher transition entropy in PFF** ($2.624$ vs $2.560\text{ bits}$) | $p = 0.0294$ | $p = 0.0342$ |

---

## 1. Mean Locomotor Speed ($\langle v \rangle$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> Measure the animal's speed at every single frame (80 times per second), add all those speeds up, and divide by the total number of frames in the session. This gives the average exploration speed of the animal.

### Mathematical Formulation
Let $\mathbf{x}(t) = (x(t), y(t))^\top$ be the 2D centroid position at frame $t \in \{1, \dots, T\}$ at sample rate $f_s = 80\text{ Hz}$. The instantaneous velocity vector and scalar speed are:
$$\mathbf{v}(t) = f_s \cdot (\mathbf{x}(t) - \mathbf{x}(t-1)), \qquad v(t) = \|\mathbf{v}(t)\|_2 = \sqrt{\dot{x}(t)^2 + \dot{y}(t)^2}$$
The session mean speed is:
$$\langle v \rangle = \frac{1}{T} \sum_{t=1}^T v(t)$$

### Observed Data Difference
* **Monomer (Control):** $74.77 \pm 38.05\text{ px/s}$
* **PFF:** $61.16 \pm 14.99\text{ px/s}$
* **Finding:** PFF mice move **$18.2\%$ slower on average** than Monomer controls ($p = 0.0310$, empirical permutation $p_{\text{perm}} = 0.0110$).

---

## 2. Peak Burst Velocity ($v_{95}$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> Sort all instantaneous speed values across the entire session from slowest to fastest, and take the speed value at the **95th percentile mark** ($v_{95}$). 
> 
> Taking the 95th percentile rather than the absolute maximum ($v_{\max}$) measures how fast the animal moves during its top 5% fastest sprints while ignoring occasional single-frame tracking artifacts.

### Mathematical Formulation
Given the empirical cumulative distribution function ($\text{ECDF}$) of speed $F_V(v) = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(v(t) \le v)$, $v_{95}$ is:
$$v_{95} = F_V^{-1}(0.95) = \inf \{ v \in \mathbb{R} : F_V(v) \ge 0.95 \}$$

### Observed Data Difference
* **Monomer (Control):** $232.47 \pm 155.90\text{ px/s}$
* **PFF:** $175.16 \pm 80.48\text{ px/s}$
* **Finding:** The top-end sprint speeds in PFF mice are **$24.7\%$ lower** than in Monomer controls ($p = 0.0320$, $p_{\text{perm}} = 0.0150$).

---

## 3. Normalized Movement Jerk ($\mathcal{J}_{\text{norm}}$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> In physics, **jerk** is the rate of change of acceleration (how abruptly an animal speeds up or slows down). 
> 
> We compute the frame-by-frame jerk, take the average magnitude, and divide it by the animal's average speed so that faster animals aren't penalized. A higher normalized jerk value means the animal's trajectory contains more abrupt, shaky, and stuttering velocity fluctuations rather than smooth, continuous acceleration curves.

### Mathematical Formulation
Acceleration $\mathbf{a}(t)$ and jerk $\mathbf{j}(t)$ are the discrete second and third time-derivatives of position:
$$\mathbf{a}(t) = f_s \cdot (\mathbf{v}(t) - \mathbf{v}(t-1)), \qquad \mathbf{j}(t) = f_s \cdot (\mathbf{a}(t) - \mathbf{a}(t-1)) = \frac{d^3 \mathbf{x}}{dt^3}$$
The **normalized jerk** scales mean absolute jerk by mean speed:
$$\mathcal{J}_{\text{norm}} = \frac{\frac{1}{T}\sum_{t=1}^T \|\mathbf{j}(t)\|_2}{\langle v \rangle + \epsilon}$$

### Observed Data Difference
* **Monomer (Control):** $2366.9 \pm 503.2$
* **PFF:** $2627.8 \pm 628.7$
* **Finding:** PFF mice have **$+11.0\%$ higher normalized jerk** than Monomer controls ($p = 0.0341$, $p_{\text{perm}} = 0.0264$). Their movement exhibits more erratic and fragmented acceleration changes.

---

## 4. Macro Locomotor Bout Length ($\bar{\tau}_{\text{macro\_bout}}$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> Start a timer whenever the animal begins moving faster than $5\text{ px/s}$, and stop the timer the moment speed drops below $5\text{ px/s}$. 
> 
> Averaging the length of all these uninterrupted movement episodes measures the duration of continuous exploration runs across the arena (which typically chain together several different syllables over a few seconds).

### Mathematical Formulation
Threshold velocity into binary active states $A(t) = \mathbb{I}(v(t) \ge 5.0\text{ px/s})$. An active bout $b$ is a contiguous interval $[t_{\text{start}}^{(b)}, t_{\text{stop}}^{(b)}]$ where $A(t) = 1$. The mean macro bout duration is:
$$\tau_b = \frac{t_{\text{stop}}^{(b)} - t_{\text{start}}^{(b)}}{f_s} \quad (\text{seconds}), \qquad \bar{\tau}_{\text{macro\_bout}} = \frac{1}{B}\sum_{b=1}^B \tau_b$$

### Observed Data Difference
* **Monomer (Control):** $2.97 \pm 0.82\text{ s}$
* **PFF:** $2.61 \pm 0.81\text{ s}$
* **Finding:** Continuous movement episodes in PFF mice are **$12.0\%$ shorter** than in Monomers ($p = 0.0287$, $p_{\text{perm}} = 0.0287$).

---

## 5. Micro Syllable Dwell Time ($\bar{\tau}_{\text{syll\_dwell}}$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> Look at the discrete syllable labels ($s_t \in \{0, \dots, 99\}$). Whenever the animal enters a syllable state, count how many consecutive frames it stays in that **exact same syllable** before switching to another. Divide by $80\text{ fps}$ to convert frames to seconds.
> 
> This measures the sub-second duration of an individual discrete posture / movement primitive.

### Mathematical Formulation
Given syllable sequence $S = (s_1, \dots, s_T)$, a contiguous run $m$ of the identical state $k$ is:
$$\{s_t = k \quad \text{for } t \in [t_{\text{start}}^{(m)}, t_{\text{end}}^{(m)}] \mid s_{t_{\text{start}}-1} \ne k, \, s_{t_{\text{end}}+1} \ne k\}$$
With frame length $L_m = t_{\text{end}}^{(m)} - t_{\text{start}}^{(m)} + 1$, the mean syllable dwell time is:
$$\bar{\tau}_{\text{syll\_dwell}} = \frac{1}{M} \sum_{m=1}^M \frac{L_m}{f_s} \quad (\text{seconds})$$

### Observed Data Difference
* **Monomer (Control):** $0.495 \pm 0.131\text{ s}$
* **PFF:** $0.569 \pm 0.255\text{ s}$
* **Finding:** Individual syllable states persist **$+14.9\%$ longer in PFF mice** before switching to the next syllable ($p = 0.1066$, $p_{\text{perm}} = 0.1628$).

---

## 🔍 Macro vs. Micro Bouts: The Difference in Timescales

```
========================================================================================
                      MACRO BOUT vs. MICRO SYLLABLE DWELL DYNAMICS
========================================================================================

1. Macro Locomotor Bout (~2.5 - 4.0 s): Continuous exploration run (Multi-Syllable)
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ [Syll 14: Init] ──► [Syll 43: Bound] ──► [Syll 5: Stride] ──► [Syll 22: Decel]│
   └───────────────────────────────────────────────────────────────────────────┘
   • Filter: Continuous velocity v(t) >= 5 px/s
   • Timescale: Multi-second continuous movement across the arena
   • Data Result: PFF runs are 12.0% shorter (2.61s vs 2.97s, p = 0.0287)

2. Micro Syllable Dwell Time (~250 - 500 ms): Single discrete state repetition
   ┌───────────────────┐
   │ [14, 14, 14, 14]  │  (Uninterrupted run of the exact same syllable state)
   └───────────────────┘
   • Filter: Identical syllable state s_t == k
   • Timescale: Sub-second state duration
   • Data Result: PFF syllables persist slightly longer (0.569s vs 0.495s, p = 0.1066)
========================================================================================
```

---

## 6. Stationary Immobility / Freezing ($\Phi_{\text{freeze}}$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> Count the total number of frames where the animal's speed is below $5\text{ px/s}$ (essentially stationary) and divide by the total number of frames in the session.
> 
> This measures the percentage of total recording time the animal spends completely still.

### Mathematical Formulation
The percentage of frames below $v_{\text{thresh}} = 5.0\text{ px/s}$:
$$\Phi_{\text{freeze}} = \left( \frac{1}{T} \sum_{t=1}^T \mathbb{I}(v(t) < 5.0\text{ px/s}) \right) \times 100\%$$

### Observed Data Difference
* **Monomer (Control):** $0.94\% \pm 0.48\%$
* **PFF:** $1.05\% \pm 0.44\%$
* **Finding:** PFF mice spend **$+11.1\%$ more time completely stationary** than Monomer controls ($p = 0.0402$, $p_{\text{perm}} = 0.0405$).

---

## 7. Behavioral Syntax Transition Entropy Rate ($H(S_{t+1} \mid S_t)$)

> [!TIP]
> **💡 Pedagogical Explanation:**
> Look at the transitions between consecutive syllables ($s_t \to s_{t+1}$ with self-transitions removed). 
> 
> * **Low Entropy (High Predictability / Organization):** Given the current syllable, the next syllable is predictable (e.g., *Syllable 14* is almost always followed by *Syllable 43*).
> * **High Entropy (Higher Disorder / Randomness):** Given the current syllable, the next syllable is uncertain and can jump into many different states with equal probability.

### Mathematical Formulation
Let $S = (s_1, s_2, \dots, s_M)$ be the sequence of discrete Keypoint-MoSeq syllables with self-transitions collapsed. Let $P(s_i, s_j)$ be the joint transition probability and $P(s_j \mid s_i)$ the conditional probability.

The **First-Order Conditional Shannon Entropy Rate** is:
$$H(S_{t+1} \mid S_t) = -\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} P(s_i, s_j) \log_2 P(s_j \mid s_i) \quad (\text{bits/transition})$$

### Observed Data Difference
* **Monomer (Control):** $2.560 \pm 0.163\text{ bits/step}$
* **PFF:** $2.624 \pm 0.135\text{ bits/step}$
* **Finding:** PFF transition sequences have **$+2.5\%$ higher conditional entropy** than Monomers ($p = 0.0294$, $p_{\text{perm}} = 0.0342$). Given the current syllable, the next behavior in PFF mice is statistically less predictable and more disordered than in Monomer controls.

---

## Related Files & Analysis Scripts
- 📊 **Master Figure**: [`karen_unified_6panel_master.png`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/07_longitudinal_cohorts/karen_unified_6panel_master.png)
- 🐍 **Figure Generator**: [`generate_unified_executive_presentation_plot.py`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/generate_unified_executive_presentation_plot.py)
- 📄 **Presentation Overview**: [`KAREN_BEHAVIORAL_PRESENTATION.md`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/KAREN_BEHAVIORAL_PRESENTATION.md)
- 📄 **10,000 Permutations CSV**: [`label_permutation_test_results.csv`](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/ulrik_kinematics/output/karen_genotype_comparison/permutation_tests/label_permutation_test_results.csv)
