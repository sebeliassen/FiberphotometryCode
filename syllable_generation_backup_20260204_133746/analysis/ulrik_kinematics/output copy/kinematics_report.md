# Kinematics Analysis Report
**Date**: 2026-01-22
**Datasets**: IFDN (WT vs KI), AMPH (WT/KI x Saline/Amphetamine)

This report details the computational ethology analysis performed on the Fiberphotometry dataset. It breaks down behavior into discrete syllables and analyzes their kinematic properties, sequencing, and "texture".

---

## 1. Global Summary Panels
**What is this?**
A concise overview of all core kinematic statistics in a single view.
*   **Metrics**: Velocity, Acceleration, Transition Entropy, Stereotypy, Duration Entropy.
*   **Significance**: Stars indicate statistical differences (* p<0.05, ** p<0.01, *** p<0.001).

````carousel
![AMPH Summary Panel](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_summary_panel.png)
<!-- slide -->
![IFDN Summary Panel](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_summary_panel.png)
````

---

## 2. Quantitative Matrix (Turn x Vigor)
**What is this?**
A 3x3 grid classifying all behavior into 9 fundamental categories based on **Turn Velocity** (Left, Straight, Right) and **Movement Vigor** (Low, Medium, High). This abstracts away abstract syllable IDs into understandable kinematic modes.

**How to read it:**
*   **Rows**: Vigor (Top=High, Bottom=Low).
*   **Columns**: Turn (Left, Middle, Right).
*   **Values**:
    *   **Time_Pct**: Percentage of session time spent in that state.
    *   **Onset_Count**: Number of times the animal entered that state (Bout frequency).

**Interpretation:**
*   **Hyperactivity** is seen as a shift from Bottom-Rows (Low Vigor) to Top-Rows (High Vigor).
*   **Rotational Bias** would show asymmetry between Left and Right columns.

````carousel
![IFDN Time Matrix](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_quantitative_matrix_time.png)
<!-- slide -->
![IFDN Onsets Matrix](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_quantitative_matrix_onsets.png)
<!-- slide -->
![AMPH Time Matrix](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_quantitative_matrix_time.png)
<!-- slide -->
![AMPH Onsets Matrix](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_quantitative_matrix_onsets.png)
<!-- slide -->
![AMPH Delta (Amph - Saline)](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_quantitative_delta_time.png)
````
> **Note on Delta**: The Delta plot shows (Amphetamine - Saline). Red squares mean "Increased by Drug", Blue squares mean "Decreased by Drug".

---

## 3. Advanced Kinematic Stats (The "Texture" of Movement)
These plots measure *how* the movement is performed, independent of *what* syllable is chosen. Outliers >3.0x IQR have been excluded for clarity.

### A. Stereotypy (Duration CV)
**Metric**: Coefficient of Variation (StdDev / Mean) of syllable duration.
**Meaning**: How "robotic" is the movement? A low CV means the animal performs the action exactly the same length every time.
**Result**: Contrary to the "robotic drug state" hypothesis, Amphetamine showed **no significant decrease** (ns) in variation compared to Saline (and even trended higher/variable in some cases). The behavior remains kinetically variable.

````carousel
![IFDN Duration CV](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_duration_cv.png)
<!-- slide -->
![AMPH Duration CV](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_duration_cv.png)
````

### B. Explosiveness (P95 Acceleration)
**Metric**: The 95th percentile of acceleration magnitude.
**Meaning**: Peak force generation. Does the animal launch into movements?
**Result**: KI shows higher explosiveness. Amph-WT does **not** show a significant increase compared to Saline (ns).

````carousel
![IFDN Explosiveness](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_mean_explosiveness.png)
<!-- slide -->
![AMPH Explosiveness](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_mean_explosiveness.png)
````

### C. Tortuosity (Path Wiggle)
**Metric**: Path Length / Displacement.
**Meaning**: A value of 1.0 is a straight line. Higher values mean "wiggly" or meandering paths.
**Result**: No significant reduction in tortuosity was observed with Amphetamine. The movements are **not** straighter, suggesting the drug increases vigor without necessarily linearizing the path.

````carousel
![IFDN Tortuosity](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_mean_tortuosity.png)
<!-- slide -->
![AMPH Tortuosity](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_mean_tortuosity.png)
````

### D. Velocity & Duration (Control Stats)
**Metric**: Mean Velocity (px/s) and Mean Duration (s) of sequences.
**Meaning**: Basic kinematic properties of the syllables.

````carousel
![IFDN Velocity](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_mean_velocity.png)
<!-- slide -->
![AMPH Velocity](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_mean_velocity.png)
<!-- slide -->
![IFDN Duration](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_mean_duration.png)
<!-- slide -->
![AMPH Duration](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_mean_duration.png)
````

---

## 4. Kinematic Correlations (Scatter Plots)
These plots verify physical relationships between variables for each group.
**Stats**: Dashed lines = Linear Regression. `r` = Pearson correlation.

### Ballistic Check (Velocity vs Duration)
**Question**: Do faster movements take less time?
**Interpretation of `r`**:
*   **Positive (r > 0)**: "Runners". Faster movements are sustained for longer (Accleleration).
*   **Negative (r < 0)**: "Strikers". Faster movements are brief/ballistic.
*   **NS (p > 0.05)**: Velocity and Duration are decoupled controls.

````carousel
![IFDN Vel vs Dur](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_mean_velocity_vs_mean_duration.png)
<!-- slide -->
![AMPH Vel vs Dur](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_mean_velocity_vs_mean_duration.png)
````

### Vigor vs Straightness
**Question**: Does moving faster make you move straighter?
**Result**: A strong negative correlation (higher velocity = lower tortuosity) indicates "Ballistic" movement strategies (point-to-point shooting), common in hyperdopaminergic states.

````carousel
![IFDN Vigor vs Straight](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_mean_velocity_vs_mean_tortuosity.png)
<!-- slide -->
![AMPH Vigor vs Straight](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_mean_velocity_vs_mean_tortuosity.png)
````

---

## 5. Movement Signature (Summary)
**What is this?**
A multi-dimensional summary of the animal's "Kinematic State". We normalize 5 key dimensions (Velocity, Acceleration, Straightness, Duration, CV) to a 0-1 scale and plot them as a heatmap.

**Interpretation**:
*   This visualizes the multivariate phenotype.
*   **Significance**: Refer to the Violin Plots above for the statistical significance (*, **, ***) of each individual dimension (e.g., Explosiveness or CV). The heatmap is the summary combination of those significant features.

````carousel
![IFDN Signature](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/ifdn_movement_signature.png)
<!-- slide -->
![AMPH Signature](/home/fsp585/.gemini/antigravity/brain/1418f717-39f8-4235-ada8-2ed55ea96967/amph_movement_signature.png)
````
