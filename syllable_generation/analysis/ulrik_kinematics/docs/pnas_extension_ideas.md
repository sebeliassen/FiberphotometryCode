# Striatal Dopamine Dynamics: Encoding Behavior vs. Kinematics

This document outlines a refined conceptual roadmap extending the PNAS paper (**Jørgensen et al., 2023**). Our focus is narrowed to the **encoding properties of dopamine (DA) relative to ensuing behavior**, specifically leveraging our work on **kinematic grid analysis, Random Forest Classifiers (RFC), and temporal saliency**.

---

## The Core Narrative: Motor Physics vs. Behavioral Intent

The original paper established that Dorsal Striatal (DS) dopamine peaks at syllable transitions and can be used to decode syllable identity. However, a major question remains: **What exactly is dopamine encoding?** 
* Does it represent the low-level, continuous **physics of the movement** (velocity, acceleration, height, posture)?
* Or does it encode high-level, discrete **behavioral intent** (syllable identity, action categories)?

By analyzing the temporal alignment and content of dopamine signals relative to behavioral execution, we aim to map the precise nature of this encoding.

---

## 1. The Kinematic Confounder & The Unique Information Pipeline

### 🧠 The Story (Core Concept)
Because different behavioral syllables are inherently characterized by different physical movements (e.g., a "rear" has high vertical velocity; a "left turn" has high angular acceleration), any signal that tracks kinematics will automatically appear to track syllables. To prove DA encodes behavioral syllables *themselves*, we must separate the continuous kinematics from the discrete behavior and demonstrate that DA contains unique information about behavioral intent.

### 📚 What it Builds on in the Paper
* **Finding**: Jørgensen et al. demonstrate that DA signals correlate with movement velocity and that syllables can be decoded from DA traces.
* **Limitation**: The paper does not systematically isolate continuous kinematics from discrete syllable categories, leaving open the possibility that syllable decodability is a downstream byproduct of simple kinematic tracking.

### 🚀 How We Extend It
We implement a **Kinematic Grid & Unique Information Pipeline**:
1. **Kinematic Grid Analysis**: We map dopamine release across a multi-dimensional grid of continuous kinematic parameters (e.g., speed vs. height). This visualizes whether DA fluctuations are strictly bound to motor execution space.
2. **Unique Information Isolation**: We strip continuous kinematic variables from the raw DA traces using predictive models, isolating the "behavioral residual." We then test whether this residual DA signal can still be classified into discrete syllables, proving whether DA carries information about action identity above and beyond raw movement physics.

### 🛠️ Potential Upgrades & Methodological Rigor
> [!IMPORTANT]
> **Non-Linear Kinematic Stripping**
> The mapping between mouse body kinematics and striatal dopamine is complex and highly non-linear. Simple linear subtraction (e.g., linear regression) will leave behind substantial non-linear kinematic signatures, leading to false positives in subsequent syllable decoding.
* **The Upgrade**: We propose using non-linear models (e.g., **XGBoost** or a feedforward neural network) trained to predict DA traces purely from continuous kinematic features (centroid velocity, height, roll, pitch, yaw, and their temporal derivatives). The prediction residuals of this model represent the truly kinematic-free DA signal. Running the RFC on these residuals provides a bulletproof test for abstract behavioral encoding.

---

## 2. Temporal Saliency & The Pre-Onset Planning Window

### 🧠 The Story (Core Concept)
If dopamine is actively involved in the **planning and initiation** of an action, its encoding of syllable identity must begin *before* the physical movement starts. Conversely, if it acts purely as sensory or motor feedback, its syllable-specific encoding will emerge only *after* physical onset. By evaluating decodability across a sliding temporal window, we can identify the exact "planning horizon" of striatal dopamine.

### 📚 What it Builds on in the Paper
* **Finding**: The paper aligns average DA traces to syllable onsets, showing sharp peaks at transition points.
* **Limitation**: The alignment is static, and the predictive capability of the signal in the moments immediately preceding physical movement is not systematically quantified.

### 🚀 How We Extend It
We introduce **Time-Resolved Decoding & Temporal Saliency**:
1. **Sliding-Window RFC**: Instead of decoding from the entire syllable trace, we train and test our Random Forest Classifiers on individual, narrow time bins (e.g., 20 ms bins) relative to syllable onset (from -500 ms to +500 ms).
2. **Temporal Saliency Curve**: We plot decoding accuracy over time. The point at which decoding accuracy rises significantly above chance represents the **temporal saliency threshold**—identifying exactly when the striatum begins to encode the identity of the upcoming syllable.

### 🛠️ Potential Upgrades & Methodological Rigor
> [!WARNING]
> **The $N-1$ Syllable Decay Confound**
> In actively behaving mice, syllables are executed rapidly (every 100–300 ms). Because fluorescent sensors (like dLight or GCaMP) have slow decay half-lives (300–600 ms), the DA signal during the "pre-onset planning phase" of syllable $N$ is contaminated by the decaying peak of syllable $N-1$.
* **The Upgrade**: To ensure our "pre-onset planning" signal is not merely a mathematical artifact of the previous syllable's tail, we must implement a deconvolution analysis or run a conditional decoding control. Specifically, we can train the sliding-window RFC exclusively on transitions where the preceding syllable $N-1$ is held constant, or apply a deconvolution filter to estimate underlying DA release events prior to decoding.

---

## 3. Refining the RFC Decoding Pipeline

### 🧠 The Story (Core Concept)
To draw strong conclusions about behavioral encoding, the machine learning tools we use must be free of bias and overfitting. By refining the cross-validation and shuffle protocols of our Random Forest Classifier (RFC), we establish a highly reliable benchmark for behavioral decoding that can be trusted across different experimental sessions and animals.

### 📚 What it Builds on in the Paper
* **Finding**: The paper decodes syllables from DA signals, reporting classification success rates around $1.5\times$ over shuffle control.
* **Limitation**: Cross-validation details and protection against temporal autocorrelation (which can cause severe overfitting in time-series data) are not fully detailed.

### 🚀 How We Extend It
We build a highly standardized **RFC Decoding Pipeline**:
1. **Local Baseline Subtraction**: We pre-process DA signals using localized baseline correction around each syllable onset to eliminate slow drift and isolate event-related transients.
2. **Rigorous Shuffle Controls**: We run two distinct control models: a complete randomized shuffle (to break all relationships) and a **block-wise shift shuffle** (which preserves the temporal autocorrelation of the DA signal while breaking its alignment with behavioral events). This ensures that our decoding metrics are not inflated by low-frequency noise.

### 🛠️ Potential Upgrades & Methodological Rigor
> [!TIP]
> **Group-Level Cross-Validation & Generalization**
> Standard K-fold cross-validation can overfit if frames from the same continuous behavioral bout are split between training and testing sets.
* **The Upgrade**: We implement a **GroupKFold** cross-validation strategy, where entire behavioral sessions or continuous active bouts are kept whole and assigned either fully to the training set or fully to the test set. This ensures the classifier learns true, generalizable behavioral encodings rather than memorizing the temporal noise patterns of specific recording blocks.
