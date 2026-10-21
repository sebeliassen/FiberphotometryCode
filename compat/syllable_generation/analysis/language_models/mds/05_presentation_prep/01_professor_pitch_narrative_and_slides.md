# Presentation Blueprint: Narrative Structure & Slide-by-Slide Guide

---

## Executive Overview

> **"MoSeq provides an interpretable alphabet of sub-second action syllables, and recent work shows these syllables organize into multi-second task states. Our project asks: Can we compress extended behavioral histories (~1 minute) into a compact, interpretable higher-order representation that preserves useful behavioral information, maps onto striatal dopamine dynamics, and improves experimental phenotyping?"**

---

## The Core Scientific Logic

$$
\underbrace{\text{What information matters?}}_{\text{Slides 4–5}}
\longrightarrow
\underbrace{\text{How to digest the benchmark}}_{\text{Slide 6}}
\longrightarrow
\underbrace{\text{Can we compress repertoire?}}_{\text{Slide 7}}
\longrightarrow
\underbrace{\text{Retain continuous temporal evolution}}_{\text{Slide 8}}
\longrightarrow
\underbrace{\text{Does dopamine reflect this structure?}}_{\text{Slide 9}}
\longrightarrow
\underbrace{\text{Where could this become useful?}}_{\text{Slide 10}}
$$

---

## Slide-by-Slide Architecture (10 Slides)

| Slide | Title | Category | Core Purpose / Takeaway |
| :---: | :--- | :---: | :--- |
| **1** | **The Granularity Dilemma** | MOTIVATION | 10,000 syllables per session overwhelm manual review; need higher-order summaries. |
| **2** | **The Rising Hierarchy in Computational Ethology** | CONTEXT | Natural progression: Keypoints $\to$ Syllables (MoSeq) $\to$ Task States (shMoSeq) $\to$ Extended Episodes. |
| **3** | **The Representation Problem** | FRAMEWORK | Can we build a reusable, low-dimensional representation of extended behavioral histories? |
| **4** | **Behavioral Information Integration Timescale** | OBSERVED | Predictive information about future behavior rises over tens of seconds and plateaus around ~1 minute. |
| **5** | **What Actually Adds Predictive Information?** | OBSERVED | The information ladder: Repertoire carries the largest gain; dwell is redundant; temporal order adds a small residual. |
| **6** | **What Does This Suggest for a Higher-Order Representation?** | SUB-CONCLUSION | Repertoire dominance motivates compression; goal is the simplest description preserving useful signal. |
| **7** | **Can Recent Behavior Be Compressed Into Interpretable Components?** | PRELIMINARY | Testing NMF: 100 syllable frequencies $\to$ 4 candidate components with loadings, maps, and video exemplars. |
| **8** | **A Higher-Order State Does Not Have to Be a Hard Label** | FRAMEWORK | Contrasting hard states vs. continuous mixtures ($z_t$); behavior as continuous flow through a low-dimensional space. |
| **9** | **Does Dopamine Reflect This Higher-Order Behavioral Structure?** | PROPOSED | Testing DA correlations with current context ($z_t$), transitions ($\Delta z_t$), and matching timescales. |
| **10** | **Where Could This Become Useful?** | PROPOSED | Applications to phenotyping (organization over averages), experimental comparisons, and cross-dataset representations. |

---

## Detailed Slide Content & Talk Track

---

### Slide 1: The Granularity Dilemma (Motivation)
* **Visual:** A 30-minute ethogram of 10,000 colored syllable tick marks vs. the high-level ethological question (*"What broader behavioral regimes is this animal moving between?"*).
* **Key Points:**
  * MoSeq successfully segmented continuous movement into an interpretable vocabulary of sub-second syllables (~400 ms).
  * In a standard 30-minute session, an animal produces ~5,000–10,000 syllable bouts.
  * Looking at raw syllable sequences or session-wide averages makes it difficult to summarize how behavior is organized across minutes.
* **Talk Track:**
  > *"Keypoint-MoSeq gave us a principled vocabulary of sub-second behavioral syllables. But when a session yields 10,000 syllables, raw event logs overwhelm manual inspection. If we want to understand how behavior is organized over minutes, we need a way to summarize extended histories into higher-order descriptions."*

---

### Slide 2: The Rising Hierarchy in Computational Ethology (Context)
* **Visual:** 4-tier hierarchy diagram:
  1. *Continuous Keypoints (DLC / SLEAP)*: Frame-by-frame coordinate tracking.
  2. *Action Syllables (MoSeq, 2015 / 2024)*: Sub-second motor primitives (~400 ms).
  3. *Task-Like States (Weinreb et al., Neuron 2026)*: Multi-second behavioral chunks (~2–10 s) represented in PFC.
  4. *Extended Behavioral Organization (Our Focus)*: Macro-scale behavioral patterns (~1 minute).
* **Key Points:**
  * Computational ethology has naturally progressed from tracking to syllables to multi-second states.
  * Recent landmark work (Weinreb et al., Neuron 2026) showed syllables organize into 2–10 s task-like states.
  * A natural next question is how behavior is structured across even longer windows (tens of seconds to minutes).
* **Talk Track:**
  > *"Ethology has naturally moved upward in abstraction: from continuous keypoints, to sub-second syllables, to multi-second task states shown by Weinreb et al. We are asking a complementary question: How is behavior organized across extended windows of tens of seconds to minutes?"*

---

### Slide 3: The Representation Problem (Framework)
* **Visual:** Conceptual schematic comparing a 1-minute sequence of discrete syllables to a compact representation vector $\mathbf{z}_t$.
* **Key Points:**
  * How should we represent a 1-minute window of behavior?
  * The deeper scientific question: **Finding the Analytical Ceiling** (*How far can we compress behavioral history before we start losing information that matters?*).
  * Desirable properties:
    1. *Low-dimensional & Compressed:* Turn thousands of raw syllable bouts into 4–6 manageable quantities without sacrificing predictive power.
    2. *Interpretable to Behaviorologists:* Grounded in observable actions, inspectable across animals, and verifiable through video exemplars.
    3. *Continuous & Dynamic:* Preserve smooth transitions and trajectory flows across extended timescales.
* **Talk Track:**
  > *"Rather than proposing an arbitrary algorithm, we formulate this as a representation problem: Can we build a compact summary of a 1-minute behavioral history that is low-dimensional, preserves predictive information, and remains directly interpretable to an ethologist? We are asking an empirical question: How far can we compress behavior before further complexity stops buying us anything useful?"*

---

### Slide 4: Behavioral Information Integration Timescale (Observed)
* **Visual:** Memory curve ($W = 2\text{s} \to 360\text{s}$) showing predictive gain rising steeply from 5s to 30s and plateauing around ~50s–90s.
* **Key Points:**
  * Tested how far back in time past behavioral history provides information about future behavior (predicting behavior over the next 120 s).
  * Information gain rises steeply over the first 30 seconds and broadly plateaus around ~1 minute.
  * Justifies using ~1 minute (68.3 s) as an empirical operational window for analyzing extended behavioral history.
* **Talk Track:**
  > *"Before choosing a representation, we measured how much history actually matters. Across 30 open-field sessions, predictive information about future behavior rises steeply over the first 30 seconds and plateaus around roughly one minute. This gives us an empirical timescale to ground our analysis."*

---

### Slide 5: What Actually Adds Predictive Information? (Observed)
* **Visual:** The Information Ladder:

| Level | Representation | Feature Dims | Gain (% $\Delta D_{\text{JS}}$) | Increment |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Spatial Context** (Position, velocity, arena geometry) | $14\text{-D}$ | **13.3%** | $+13.3\text{ pp}$ |
| **2** | **Current Syllable** (Instantaneous MoSeq action) | $114\text{-D}$ | **20.9%** | $+7.6\text{ pp}$ |
| **3** | **Recent Repertoire** (Syllable proportions over 1 min) | $214\text{-D}$ | **31.2%** | **$+10.3\text{ pp}$** |
| **4** | **Duration / Pacing** (Time spent dwelled in each action) | $314\text{-D}$ | **31.4%** | $+0.2\text{ pp}$ |
| **5** | **Context without True Order** (Shuffled Transformer) | Seq | **34.6%** | $+3.2\text{ pp}$ |
| **6** | **Context with True Order** (Ordered Transformer) | Seq | **35.8%** | $+1.2\text{ pp}$ |

* **Key Points:**
  * Repertoire composition over the preceding minute provides the largest single jump (+10.3 pp).
  * Explicit dwell time adds virtually nothing (+0.2 pp) beyond initiation counts.
  * Temporal context adds a smaller residual (+3.2 pp), with true chronological order contributing +1.2 pp.
* **Talk Track:**
  > *"When we decompose what information drives future prediction: knowing recent syllable composition provides the largest jump (+10.3 pp). Adding explicit dwell times adds essentially zero (+0.2 pp). Richer temporal context provides a smaller residual (+3.2 pp), of which exact chronological order accounts for +1.2 pp."*

---

### Slide 6: What Does This Suggest for a Higher-Order Representation? (Sub-Conclusion)
* **Visual:** 3 structured summary cards:
  1. **Recent repertoire carries substantial information:**  
     The largest gain (+10.3 pp) came from knowing which syllables occurred over the preceding minute. Not every added detail matters equally—explicit dwell time is largely redundant (+0.2 pp), and exact order adds a smaller residual (+1.2–3.2 pp).
  2. **Substantial compression may be possible (Compression as Interpretability):**  
     A small number of higher-order measures may capture much of the useful information in extended behavior.  
     > *"The value of compression is not just computational efficiency — it is **interpretability**. If a few variables explain most of the useful structure, thousands of syllables turn into a manageable set of interpretable quantities that behaviorologists can inspect, compare across animals, and validate with representative video exemplars."*
  3. **Establishing the Analytical Ceiling:**  
     The deeper question is not simply proposing another arbitrary layer of abstraction, but asking:  
     > **"How far can we compress behavior before we start losing information that matters?"**  
     Empirically testing when a representation becomes sufficient enough that further complexity stops buying anything useful.
* **Bottom Line:**
  > **"The goal is the simplest higher-order description that preserves useful behavioral information."**
* **Talk Track:**
  > *"This benchmark gives us three concrete design takeaways: First, recent repertoire carries the vast majority of predictive signal, while dwell time and exact order add minor residuals. Second, this means substantial compression is possible—and the primary value of compression is interpretability. If a few variables explain most of the useful structure, behavioral analysis becomes much easier to interpret and compare across animals. Third, this establishes our analytical ceiling: we ask how far we can compress behavior before we start losing information that matters."*

---

### Slide 7: Can Recent Behavior Be Compressed Into Interpretable Components? (Preliminary)
* **Visual:** Non-Negative Matrix Factorization schematic & Component Retention Curve:
  * **Figure:** [fig1b_nmf_predictive_utility_curve.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/presentation/figures/fig1b_nmf_predictive_utility_curve.png)
  $$\text{100 Syllable Frequencies} \quad\longrightarrow\quad \text{4–6 Candidate Components } \mathbf{z}_t = [0.60, 0.25, 0.10, 0.05]$$
  * Multi-panel card for Components 1–4:
    * Top syllable loadings (fold-enrichment).
    * Spatial occupancy heatmaps in the arena.
    * Timestamps linking to representative video clips.
  * **Component Retention Benchmark (Strict Nested CV — NMF Fit on Train Only):**

| Components ($K$) | Syllable Variance Explained | Pure NMF Predictive Gain | Information Retention (vs. Pure 100-D BoW) | Cumulative Gain ([Curr Syll + NMF]) |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 2$** | $42.1\%$ | $+6.42\%$ | $22.1\%$ | $20.49\%$ |
| **$K = 3$** | $47.4\%$ | $+15.28\%$ | $52.6\%$ | $24.13\%$ |
| **$K = 4$** | **$51.4\%$** | **$+17.55\%$** | **$60.4\%$** | **$25.38\%$** |
| **$K = 6$** | **$58.6\%$** | **$+19.37\%$** | **$66.7\%$** | **$26.29\%$** |
| **$K = 8$** | **$65.3\%$** | **$+21.97\%$** | **$75.6\%$** | **$27.77\%$** |
| **$K = 10$** | **$70.9\%$** | **$+24.54\%$** | **$84.4\%$** | **$29.28\%$** |
| **$K = 15$** | **$80.5\%$** | **$+29.02\%$** | **$99.8\%$** | **$32.25\%$** |
| **$K = 20$** | $86.2\%$ | $+32.12\%$ | $110.5\%$ *(Denoising)* | $34.45\%$ |
| **Full 100-D** | $100.0\%$ | $+29.06\% \pm 3.01\%$ | **$100.0\%$ (Reference)** | **$+32.60\% \pm 3.22\%$** |

* **Key Points:**
  * **The Compression Spectrum:** Rather than an arbitrary single choice, NMF reveals a principled tradeoff:
    * **$K = 4$:** Maximum interpretability (4 inspectable video sets) while preserving $>60\%$ of future predictive signal.
    * **$K = 8\text{--}10$:** Retains $\approx 76\%\text{--}84\%$ of predictive signal.
    * **$K = 15$:** Reaches $99.8\%$ of full 100-D performance (an $85\%$ compression with zero loss).
  * **Human-in-the-Loop Principle:** The algorithm proposes candidate components; video exemplars allow a domain behaviorologist to determine whether they have an ethologically coherent interpretation.
* **Talk Track:**
  > *"To test whether behavioral composition can be compressed, we fit NMF strictly within training splits under cross-validation. We observe a clear spectrum: just 4 components retain 60% of predictive power for maximum interpretability, 8–10 components retain over 80%, and by 15 components we reach 100% of the full repertoire benchmark. Rather than prematurely fixing a single state count, our pipeline provides representative video clips so a behaviorologist can inspect and validate the biological meaning across this spectrum."*

---

### Slide 8: A Higher-Order State Does Not Have to Be a Hard Label (Framework)
* **Visual:** Comparison diagram:
  * *Hard-State View:* $\text{Window}_t \in \{\text{State 1, State 2, State 3}\}$ (discrete categorical jumps).
  * *Mixture / Continuous View:* $\mathbf{z}_t = [0.60, 0.30, 0.10, 0.00] \longrightarrow \mathbf{z}_{t+1} \longrightarrow \mathbf{z}_{t+2}$ (smooth trajectory on a simplex).
* **Key Points:**
  * Extended behavior may be better represented as continuous movement through a low-dimensional behavioral space than as a sequence of rigid, discrete labels.
  * An animal can smoothly transition between regimes (e.g., gradually shifting from active exploration to quiet investigation).
  * The temporal evolution of $\mathbf{z}_t$ over time preserves the sequence dynamics identified in our benchmark.
* **Talk Track:**
  > *"A higher-order behavioral state does not have to be a hard categorical label. An animal may be better described as occupying a mixture of behavioral tendencies that drifts continuously through time. This continuous view allows us to track smooth trajectories through behavioral space while retaining temporal dynamics."*

---

### Slide 9: Does Dopamine Reflect This Higher-Order Behavioral Structure? (Proposed)
* **Visual:** Schematic aligning continuous behavioral state $\mathbf{z}_t$ with real-time striatal fiber photometry (dLight1 dopamine traces).
* **Three Concrete Experimental Questions for the Lab:**
  1. **Current Behavioral Context ($DA_t \leftrightarrow \mathbf{z}_t$):**  
     *Does baseline dopamine tone differ depending on the animal's broader behavioral mixture?*
  2. **State Transitions ($DA_t \leftrightarrow \Delta \mathbf{z}_t$):**  
     *Are dopamine transients associated with shifts between different behavioral modes on the manifold?*
  3. **Timescale Alignment:**  
     *Do faster and slower temporal components of the dopamine signal relate to different levels of behavioral organization (sub-second actions vs. minute-scale context)?*
* **Talk Track:**
  > *"This framework directly connects to our striatal fiber photometry data. We can ask three concrete questions: Does baseline dopamine tone reflect the animal's broader behavioral context? Do dopamine transients align with transitions between behavioral modes? And do slow versus fast components of the dopamine signal map onto different levels of the behavioral hierarchy?"*

---

### Slide 10: Where Could This Become Useful? (Proposed)
* **Visual:** 3 applied domain cards:
  1. **Phenotyping Beyond Gross Metrics:**  
     Two experimental groups may show identical total distance traveled or mean syllable frequencies, but differ in how their behavior is structured over minutes.
  2. **Experimental Comparisons:**  
     Ask whether genotype, pharmacological interventions, disease models, or session phases shift:
     * Component occupancy distributions,
     * Transition frequencies between modes,
     * Persistence vs. fragmentation of behavioral regimes.
  3. **Cross-Dataset Behavioral Representations:**  
     Longer-term: Learn common behavioral coordinate systems across large archives to test whether representations generalize across experiments.
* **Talk Track:**
  > *"Where does this become useful for the lab? First, in phenotyping: two cohorts might have identical overall movement averages but organize their extended behavior completely differently. Second, in experimental comparisons: testing whether a drug, genotype, or disease model alters regime occupancy, persistence, or transitions. And third, establishing shared coordinate representations across experimental datasets."*

---

## Summary of Anticipated Q&A

### Q1: *"Weinreb et al. already showed syllables form task states (shMoSeq). Why do we need this?"*
* **Answer:** *"Weinreb et al. (Neuron 2026) made a major breakthrough showing syllables organize into 2–10 second task states in PFC. Our work investigates extended behavioral organization across longer windows (tens of seconds to minutes) and asks how to compress those histories into compact representations for photometry analysis and phenotyping. The framework is tokenizer-agnostic and can take either MoSeq syllables or shMoSeq states as input."*

### Q2: *"Why use NMF instead of training a deep end-to-end neural network?"*
* **Answer:** *"Our predictive benchmark demonstrated that simple syllable repertoire composition captures the vast majority of predictable behavioral signal (+31.2%). Therefore, linear matrix factorization (NMF) provides a transparent, additive representation that a domain behaviorologist can directly inspect, while deep sequence models can be retained to model the temporal evolution of those representations."*

### Q3: *"Have you proven that the 4 NMF components correspond to real biological states?"*
* **Answer:** *"No, and that is an intentional methodological boundary. The algorithm proposes candidate components and extracts representative video exemplars. Domain ethologists must inspect those videos to determine whether the components represent coherent biological regimes."*

### Q4: *"What is the broader conceptual goal if not simply adding another arbitrary layer above shMoSeq?"*
* **Answer:** *"The deeper question is discovering the **analytical ceiling** of behavioral representation: How far can we compress behavioral history before we start losing information that matters? The value of compression is interpretability—if 4–6 higher-order dimensions explain most of the useful structure in thousands of syllables, behaviorologists gain a manageable, inspectable vocabulary to compare across animals. We are empirically testing when a representation becomes sufficient enough that further architectural complexity stops buying useful signal."*
