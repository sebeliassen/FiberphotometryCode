# Guide to Reading the N-1 Confound Control Plots

This guide explains how to interpret the **$N-1$ Syllable Decay Confound Control** plots generated during the sliding-window planning horizon analysis. 

---

## 1. The Core Scientific Problem: Decay vs. Planning
When decoding an upcoming syllable $N$ using dopamine signals prior to its onset (from $-300\text{ ms}$ to $0\text{ ms}$), we face a potential confound:
* **The Confound (Slow Decay):** The slow decay kinetics of the fluorescent dopamine sensor (e.g., dLight1.1) or calcium indicators could cause dopamine signals from the *preceding* syllable ($N-1$) to leak into the planning window of the *current* syllable ($N$). This could artificially inflate decoding accuracy, reflecting past history rather than future planning.
* **The Control (Holding $N-1$ Constant):** To isolate true prospective planning, we select only those transition events where the animal starts from the **exact same preceding syllable** (e.g., holding $N-1$ constant as Syllable `62`, `6`, or `36`).
* **The Planning Signal:** If the classifier can still decode which syllable $N$ is about to occur when starting from the identical preceding syllable, we prove that the dopamine signal contains true, prospective planning information.

---

## 2. Visual Anatomy of the Plots

Each plot presents the decoding trajectory leading up to the transition.

### Axes & Reference Lines
* **X-Axis (Pre-Onset Time Relative to Syllable N):** Spans from $-300\text{ ms}$ to $0\text{ ms}$ prior to the movement onset of Syllable $N$. It is divided into 6 bins of $50\text{ ms}$ each.
* **Y-Axis (Decoding Accuracy - Fold over Shuffle):** Quantifies how much better the Random Forest classifier performs compared to a shuffled control. 
  * A value of **`1.0`** (marked by the dotted black line) represents **Chance Baseline**.
  * A value of **`1.2`** means the classifier is 20% more accurate than chance.

### Curve Styling & Legend
The curves are color-coded by **Genotype** and styled by **Feature Set**:

| Genotype | Color | Code | Description |
| :--- | :--- | :--- | :--- |
| **Wild Type (WT)** | Green | `#2ecc71` | Control animals |
| **Knock-In (KI)** | Red | `#e74c3c` | Parkinson's/Dopamine dysfunction model |

| Signal Type | Line Style | Marker | Description |
| :--- | :--- | :--- | :--- |
| **Raw DA** | Solid Line | Circle (`o`) | Total dopamine signal (includes vigor and behavioral intent) |
| **Residual DA** | Dashed Line | Cross (`x`) | Kinematics-free dopamine signal (vigor regressed out; pure abstract intent) |

* **Shaded Regions:** Represent the **95% Confidence Intervals** ($1.96 \times \text{SEM}$) across cross-validation folds. If the shaded region does not overlap with the dotted line (`1.0`), the decoding is statistically significant.

---

## 3. Key Interpretations: What to Look For

### 1. Does the Planning Horizon Exist?
Look at the **$-200\text{ ms}$ to $-100\text{ ms}$** window:
* If the WT curves (green) are significantly above `1.0`, it confirms a **prospective planning horizon**—dopamine encodes the upcoming choice before the animal moves, even when starting from the same preceding state.

### 2. Is Planning Free of Vigor/Kinematics?
Compare **Raw DA (Solid)** and **Residual DA (Dashed)**:
* If the dashed curve remains high and significant, it shows that the planning signal is **not** just a reflection of planned movement speed or acceleration. Dopamine is planning the *identity* (posture/shape) of the syllable.

### 3. WT vs. KI Deficits
Compare the **WT (Green)** and **KI (Red)** curves:
* A lower or delayed rise in the red curves indicates a deficit in prospective planning or behavioral representation in the dopamine-deficient model.

---

## 4. File Naming Convention
The generated plots are saved in `output/temporal_saliency/` with the following structure:
`amph_confound_control_prev_syll_[N-1]_[drug]_[phase].png`

* **`[N-1]`**: The syllable held constant (`62`, `6`, or `36`).
* **`[drug]`**: `saline` or `amphetamine`.
* **`[phase]`**: `pre` (baseline) or `post` (after drug injection).
