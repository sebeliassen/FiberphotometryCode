# Gether Lab Monday Meeting Presentation Strategy & Slide Blueprint

**Meeting Context**: Gether Lab (Neuropharmacology & In Vivo Neuroscience, UCPH)  
**Date**: Monday 14th  
**Format**: 30–40 minutes uninterrupted presentation, ~20 slides, English  
**Audience**: 15–25 colleagues (in vivo researchers, molecular biologists, pharmacologists)  
**Core Goal**: Establish yourself as the data science champion who builds reusable, high-impact behavioral pipelines, while presenting a compelling biological story about dopamine transporter (DAT) dynamics and behavioral organization.

---

## 1. Tactical Alignment with Your Lab's Culture

### What Wins the Room
* **The "Scientific Journey" Narrative**: Colleagues don't need exaggerated claims. They respect an honest, clear progression: *"Here was the problem with our existing tracking -> here is what we tried -> here is what worked -> here is how you can use this in your experiments."*
* **Tangible Video Clips Over Abstract Math**: In vivo researchers want to see the mouse moving. A 10-second video grid showing 24 instances of a recognized syllable or state is worth 50 equations.
* **Direct Relevance to the Lab’s Core Mission**: Everything ties back to **DAT genotypes (DAT-KI)**, **psychostimulant pharmacology (Amphetamine / Cocaine)**, and **striatal dopamine signaling (DMS, VS/NAc, DLS, PFC)**.

### The Pitfall to Avoid
* **"Is this too abstract?"**: Never present math or information theory as an end in itself. Frame every metric as an answer to a biological question:
  * *Lagged Mutual Information* $\to$ *"How long does a mouse remember what it was doing?"*
  * *HMM Transition Matrix* $\to$ *"Does amphetamine make behavior more rigid or more random?"*
  * *Macro-States* $\to$ *"Why does rearing lead to forward running in healthy mice, but head-bobbing in DAT mutants?"*

---

## 2. Deconstructing Questions 7 & 8: What Videos & Verification Give You

### Question 7: What Video Inspection Gives Your Presentation
Depending on what you observe when watching the grid clips this weekend, here is how you use it on slides:

| Observation in Video Clips | Slide Presentation Strategy |
| :--- | :--- |
| **Clear, recognizable postures**<br>(e.g. clean rearing, head-sniffing, grooming) | **The "Instant Credibility" Slide**: Show a 4x6 grid clip. Say: *"The unsupervised algorithm parsed 5.5 million frames without human labels, yet it automatically isolated this exact unsupported rearing behavior with sub-second precision."* |
| **Compound / Contextual actions**<br>(e.g. rearing in the arena center vs. at the wall) | **The "Why Syllables Need Context" Slide**: Show that the physical posture is identical, but the animal's subsequent action is completely different depending on location or state. |
| **Aberrant / Repetitive looping**<br>(e.g. tight circling, alternating head movements) | **The "Phenotypic Discovery" Slide**: Show that standard open-field software just calls this "high velocity locomotion," but Keypoint-MoSeq proves the mouse is trapped in a pathological stereotypic loop. |

### Question 8: How to Present "Ready Results" vs. "Hypotheses"
Colleagues love honesty about what is verified vs. what is a working hypothesis:

* **What is Verified (Present as Hard Findings)**:
  * Skeletal tracking (27 keypoints) and MoSeq extraction across the 30-session cohort.
  * Real mouse behavior possesses excess multi-second memory peaking at **~6.3 seconds** (proving behavior is organized into ~6s bouts, not memoryless 300 ms twitches).
  * 40 Hz MoSeq and 80 Hz photometry timelines are strictly synchronized with verified handling boundaries.
* **What is a Working Hypothesis (Present as Exciting Open Questions)**:
  * *Exact biological names of macro-states*: *"We currently label State 6 as exploratory locomotion and State 1 as stereotypic pacing based on preliminary syllable loadings; full ethological validation is ongoing."*
  * *Dopamine's causal role*: *"Does dopamine release in the Ventral Striatum trigger the switch into a new macro-state, or does it reflect ongoing vigor? Here is how our pipeline will test this."*

---

## 3. The 20-Slide Master Architecture (30–40 Minutes)

### Section 1: The Problem with Traditional Behavioral Tracking (Slides 1–3)
* **Slide 1: Title & The Mission**
  * *Title*: Decoding Behavioral Grammar & Dopaminergic Organization in Freely Moving Mice.
  * *Message*: Bridging sub-second keypoint kinematics, multi-second behavioral states, and striatal dopamine photometry into a reusable lab pipeline.
* **Slide 2: The Ethologist’s Dilemma**
  * *Visual*: Centroid tracking trajectory heatmaps.
  * *Point*: Two mice can have the exact same distance traveled and velocity, but one is performing rich exploratory rearing while the other is compulsively running in circles. Summary metrics hide the biology.
* **Slide 3: The Technological Progression: Trajectories $\to$ Pose $\to$ Syllables**
  * *Visual*: DeepLabCut skeletal tracking $\to$ Keypoint-MoSeq segmentation.
  * *Point*: Keypoint-MoSeq turns continuous motion into an unsupervised "alphabet" of ~300–400 ms behavioral syllables.

---

### Section 2: Beyond 1-Step Transitions: The Need for Behavioral Memory (Slides 4–7)
* **Slide 4: Are Syllables Enough? The Hidden Flaw in 1st-Order Markov Models**
  * *Diagram*: $S_t \to S_{t+1}$.
  * *Point*: Standard syntax assumes behavior is memoryless every 300 ms. But an animal does not reset its brain every third of a second.
* **Slide 5: Proving Behavioral Memory: The Weinreb Lagged MI Benchmark**
  * *Visual*: Real vs. 1st-Order Markov Lagged Mutual Information curve.
  * *Point*: 1st-order Markov decays to zero in $<15$ seconds. Real mouse behavior maintains excess memory peaking at **$\tau \approx 6.3\text{ seconds}$**.
* **Slide 6: What Does the ~6.3s Peak Mean? The Behavioral Bout**
  * *Point*: 6 seconds represents the natural lifespan of a behavioral bout (a cluster of purposeful actions). Syllables are constrained by underlying internal states.
* **Slide 7: Ruling Out the Clock Confound**
  * *Visual*: State vs. Session Time decomposition plot.
  * *Point*: Multi-second memory is not just arena habituation over 60 minutes; inferred states provide $+0.18$ bits of independent predictive power.

---

### Section 3: Applying shMoSeq to Our Cohort: What States Look Like (Slides 8–11)
* **Slide 8: Introducing shMoSeq: Grouping Syllables into Macro-States**
  * *Diagram*: Hierarchical Semi-Hidden Markov Model (Syllables nested under Macro-States).
  * *Point*: How macro-states act as the "grammar" or "task context" governing which syllables can be performed.
* **Slide 9: Video Ground-Truth: Visualizing Macro-State 6 (Exploration)**
  * *Visual*: Video grid showing mice engaged in rearing and forward locomotion.
  * *Point*: Real ethological grounding of unsupervised clusters.
* **Slide 10: Video Ground-Truth: Visualizing Macro-State 1 (Repetitive / Grooming)**
  * *Visual*: Video grid showing stationary behaviors, head-grooming, and paw movements.
* **Slide 11: Syllable Repertoire Fingerprints Across States**
  * *Visual*: Stacked bar charts showing syllable probability distributions for each macro-state.

---

### Section 4: Making It Biological: DAT Genotypes & Amphetamine Challenge (Slides 12–15)
* **Slide 12: Our Experimental Cohort: Genotypes & Pharmacology**
  * *Setup*: WT vs. DAT-KI (dopamine transporter mutant), Saline vs. Amphetamine ($2\times 2$ factorial design, $n=30$ sessions).
* **Slide 13: Baseline Deficit: The 11-Fold Collapse of State 6 in DAT-KI**
  * *Visual*: State occupancy bar graphs (WT vs. DAT-KI Saline).
  * *Finding*: Normative exploration (State 6) occupies $48.3\%$ in WT, but collapses to $4.4\%$ in DAT-KI mutants ($p = 0.015$). Mutants are trapped in repetitive regimes even before drug challenge.
* **Slide 14: Amphetamine Response: Pathological Persistence vs. Loop Fragmentation**
  * *Visual*: Group excess memory curves ($\Delta I$) and transition graphs.
  * *Finding*: In WT mice, amphetamine inflates persistence (prolonged stereotypic locomotive loops lasting $>60$s). In DAT-KI mutants, it collapses natural bouts into hyper-rigid 2-syllable alternating loops ($A \leftrightarrow B$).
* **Slide 15: What Added Resolution Buys Us: The Capacity Ladder ($K=4 \to 8 \to 12$)**
  * *Visual*: State splitting tree diagram.
  * *Point*: At $K=4$, broad categories (locomotion vs. stationary). At $K=8$, locomotion cleanly splits into directed exploration vs. perimeter pacing. How colleagues can choose the right granularity for their specific mouse line.
* **Slide 15b (The Critical Pivot): The Reality Check — Why Macro-States Fail & The Need for Bottom-Up Syllable Chains**
  * *Visuals*:
    * **Old Long-Duration Model (17KP $\kappa=500,000$):** [pres_reality_check_composite_slide_old_model.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/presentations/pres_reality_check_composite_slide_old_model.png) (shows multi-minute session locks, $p = 0.7542$).
    * **Calibrated ~12s Model (8KP $\kappa=50,000$, Clean Data):** [pres_reality_check_composite_slide_12s_model.png](file:///home/fsp585/Desktop/FiberphotometryCode/syllable_generation/presentations/pres_reality_check_composite_slide_12s_model.png) (shows fluid multi-second bout switching, median 9.25s, mean 16.16s, ~200 switches/session, $p = 0.3329$).
  * *Point*: Whether top-down stickiness is locked into multi-minute regimes ($p = 0.7542$) or calibrated to ethological bouts matching our lagged MI timescale ($p = 0.3329$), top-down macro-states fail to reach statistical significance across animal permutations, whereas raw KPMS syllables cleanly discriminate genotypes ($p = 0.0356$). Top-down clustering skipped the critical intermediate scale of behavior: we must build upward from syllables to tangible, verifiable **Syllable Chains & Episodes**.

---

### Section 5: Ground-Truth Validation & Multimodal Photometry Integration (Slides 16–18)
* **Slide 16: Technical Rigor: Ground-Truthing Injection Handling via DLC Morphology**
  * *Visual*: Tracking confidence drop during physical mouse removal.
  * *Point*: Pre-injection baseline is verified to $20.00 \pm 0.40$ min, handling to $61.7$s, and post-injection response to $59.3$ min.
* **Slide 17: Multimodal Validation: Optical Cable Physics Confirms Epochs**
  * *Visual*: Fiber photometry motion artifacts during animal handling ($1.77\times$ variance surge).
  * *Point*: Optical physics provides independent verification of physical animal removal boundaries.
* **Slide 18: The Big Frontier: Connecting States to Striatal Dopamine Dynamics**
  * *Diagram*: Dual-color photometry (dLight/GRAB-DA in DMS, DLS, VS/NAc, PFC; NE in LC/PFC) aligned to macro-state transitions.
  * *The Question*: Does dopamine release in DMS/VS *cause* the transition into exploratory state 6, or does it reflect ongoing vigor?

---

### Section 6: Summary, Reusable Tools & Discussion (Slides 19–20)
* **Slide 19: The Reusable Pipeline for Gether Lab Colleagues**
  * *Summary of Tools Built*:
    1. Automated Keypoint-MoSeq batch training and fast checkpointing.
    2. Synchronized 80 Hz / 40 Hz master session pickles.
    3. Grid movie generator for visual scoring.
    4. Expected Marginal Likelihood (EML) automated model selection.
  * *Call to Action*: *"If you have DLC-tracked open field, elevated plus maze, or operant videos, this pipeline is ready to run on your mice."*
* **Slide 20: Summary & Discussion Questions**
  * *Takeaways*:
    1. Mouse behavior is non-Markovian with a ~6.3s bout timescale.
    2. DAT disruption specifically dismantles behavioral grammar and exploratory bout persistence.
    3. Next step: Coupling to striatal dopamine photometry transients.
  * *Open Question for the Room*: *"In your experiments, do you see DAT mutants fail to initiate new behaviors, or fail to terminate ongoing loops?"*

---

## 4. Deep Literature Research Tool Prompt (Tailored for Gether Lab Context)

Use this prompt in your research tool to extract literature that directly addresses your colleagues' questions:

```markdown
You are an expert computational neuroethologist and systems neuropharmacologist preparing a high-impact research briefing for a dopamine transporter (DAT) pharmacology laboratory (Gether Lab).

Please perform an exhaustive literature search and provide detailed, cited answers to the following 4 core questions:

### 1. Identify the Recent (2024–2025) Drosophila / Fruit Fly Behavioral Hierarchy Paper
- What recent (2024–2025) papers from Gordon Berman, Mala Murthy, Jonathan Pillow, or Talmo Pereira investigate multi-timescale behavioral hierarchy, internal states, or sequence modeling in *Drosophila*?
- Provide title, authors, journal, year, and a 3-sentence summary of how fruit fly behavioral states compare to mammalian pose syllables (Keypoint-MoSeq / shMoSeq).

### 2. Dopamine Dynamics Across Striatal Subregions (DMS, DLS, VS/NAc) During Naturalistic Bouts
- In freely moving rodents, how do dopamine transients in the Dorsomedial Striatum (DMS), Dorsolateral Striatum (DLS), and Ventral Striatum / Nucleus Accumbens (VS/NAc) correlate with:
  a) The initiation vs. termination of behavioral bouts (e.g. rearing, grooming, locomotion).
  b) Movement vigor / velocity versus cognitive task/internal state switching.
- Is dopamine signaling considered causal for switching behavioral states, or a permissive correlate of ongoing state execution? Cite key papers (e.g. Costa, Uchida, Mohebi, Berke, Datta).

### 3. Phenotypic Grammar of DAT-KI and Psychostimulant-Induced Stereotypy
- How do dopamine transporter disruptions (DAT knock-in mutants, DAT knockdowns) and psychostimulants (amphetamine, cocaine) alter behavioral syntax?
- What are the documented mechanisms behind:
  a) Perseverative locomotion loops (prolonged multi-minute pacing).
  b) High-frequency alternating stereotypic loops (rapid 2-syllable reciprocal repetition).
- Cite relevant behavioral pharmacology literature on DAT mutant mouse phenotyping.

### 4. Methodological Standards for Information-Theoretic Ethology
- In Caleb Weinreb & Sandeep Robert Datta's 2026 Neuron paper ("Spontaneous behavior is a succession of self-directed tasks") and Gordon Berman's 2016 PNAS paper:
  a) How is Lagged Mutual Information computed and corrected for finite-sample bias?
  b) What controls rule out non-stationarity (e.g. arena habituation over 60 minutes)?

### Deliverable:
Provide a structured synthesis with direct paper citations (Title, Author, Journal, Year) and 3 key takeaways formatted as talking points for a lab meeting presentation.
```
