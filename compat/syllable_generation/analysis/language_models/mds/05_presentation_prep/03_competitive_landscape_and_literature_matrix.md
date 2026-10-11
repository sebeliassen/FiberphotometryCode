# Competitive Landscape & Literature Matrix: The 5 Landmarks

---

## Executive Overview

To present our project convincingly to a neuroscience professor, we position our work at the convergence of **five landmark papers** spanning computational ethology, information theory, representation learning, and systems neuroscience:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 5 LANDMARK PILLARS MATRIX                                 │
├─────────────────────┬───────────────────┬─────────────────────────────────┬─────────────────────┤
│ Landmark Paper      │ Key Question      │ What They Established           │ What We Add / Gap   │
├─────────────────────┼───────────────────┼─────────────────────────────────┼─────────────────────┤
│ 1. Berman et al.    │ Is behavior       │ Proved multi-scale hierarchy &  │ Mammalian 3D MoSeq  │
│    (PNAS, 2016)     │ predictively      │ long-range non-Markovian memory │ syllables + nested  │
│                     │ hierarchical?     │ in Drosophila (Information Btl).│ P_n decomposition.  │
├─────────────────────┼───────────────────┼─────────────────────────────────┼─────────────────────┤
│ 2. uBAM             │ Can unsupervised  │ Learned latent posture          │ Discrete symbolic   │
│    (Nature MI, 2021)│ embeddings enable │ representations to detect subtle│ grammar + multi-    │
│                     │ phenotyping?      │ mutant & drug phenotypes.       │ horizon forecasting.│
├─────────────────────┼───────────────────┼─────────────────────────────────┼─────────────────────┤
│ 3. CEBRA            │ Can latent spaces │ Joint contrastive embedding of  │ Discrete symbolic   │
│    (Nature, 2023)   │ link behavior &   │ continuous behavior and neural  │ language models +   │
│                     │ neural dynamics?  │ timeseries across animals.      │ macro-topic cards.  │
├─────────────────────┼───────────────────┼─────────────────────────────────┼─────────────────────┤
│ 4. Weinreb et al.   │ Are mouse actions │ Discovered seconds-long task-   │ Macro-scale episode │
│    (shMoSeq, 2026)  │ organized into    │ like states in mice encoded in  │ semantics (68s) +   │
│                     │ task programs?    │ Prefrontal Cortex (PFC).        │ dopamine steering.  │
├─────────────────────┼───────────────────┼─────────────────────────────────┼─────────────────────┤
│ 5. BehaVERT         │ Can motion be     │ Applied BERT (masked cloze) to  │ Discrete syllables  │
│    (IJCV, 2026)     │ treated as a      │ raw skeletal keypoint frames for│ (words) vs raw pose │
│                     │ language?         │ autism model phenotyping.       │ jitter (pixels).    │
└─────────────────────┴───────────────────┴─────────────────────────────────┴─────────────────────┘
```

---

## 1. Deep Dive into the 5 Landmarks

### 1. Berman, Bialek & Shaevitz (*PNAS*, 2016)
* **Title:** *"Predictability and hierarchy in Drosophila behavior"*
* **Core Contribution:** First paper to use information theory (Deterministic Information Bottleneck) and Markov operator eigenspectra to prove that animal behavior possesses a **hierarchical tree of timescales** extending far beyond memoryless transitions.
* **Our Take & Positioning:** We extend predictive hierarchy analysis from 2D fruit fly wavelets to **3D mammalian rodent syllables**, introducing an explicit **$\mathcal{P}_0 \to \mathcal{P}_5$ information decomposition** that isolates the predictive contribution of session clock, arena geometry, motor grammar, action budgets, and sequential syntax.

---

### 2. Brattoli et al. (*Nature Machine Intelligence*, 2021) — uBAM
* **Title:** *"Unsupervised behaviour analysis and magnification (uBAM) using deep learning"*
* **Core Contribution:** Demonstrated that unsupervised latent representations of animal motion can discover subtle neurological disease phenotypes and optogenetic effects that are invisible to human observers.
* **Our Take & Positioning:** uBAM operates on continuous, dense visual frames. Our framework operates on **discrete, ethologically validated syllables** ($K \approx 50\text{–}100$), enabling formal discrete sequence algebra, Skip-Gram action semantics (Action2Vec), and human-interpretable topic cards.

---

### 3. Schneider, Lee & Mathis (*Nature*, 2023) — CEBRA
* **Title:** *"Learnable latent embeddings for joint behavioural and neural analysis with CEBRA"*
* **Core Contribution:** Established that self-supervised contrastive learning can map multi-animal behavioral and neural recordings into invariant, consistent latent spaces.
* **Our Take & Positioning:** CEBRA uses continuous contrastive objectives. We build a **discrete symbolic language model** that explicitly models the chronological grammar of behavioral tokens, and we couple those macro-trajectories ($\Delta \mathbf{z}_t$) to real-time subcortical dopamine dynamics.

---

### 4. Weinreb et al. (*Neuron*, 2026) — shMoSeq
* **Title:** *"Spontaneous behavior is a succession of self-directed tasks"*
* **Core Contribution:** Proved that freely moving mice organize sub-second MoSeq syllables into **multi-second task-like programs (~2–10 s)** encoded by ensemble dynamics in the Prefrontal Cortex (PFC).
* **Our Take & Positioning:** Weinreb et al. established the biological reality of the **meso-scale (Level 2: 2–10 s)**. Our framework addresses the **macro-scale (Level 3: 30–120 s)**—how an animal organizes sequences of actions across minutes into sustained behavioral regimes, creating a compact topic representation for phenotyping and striatal dopamine modulation.

---

### 5. Shin, Kim et al. (*IJCV*, 2026) — BehaVERT
* **Title:** *"BehaVERT: A Transformer-Based Motion Language Model for Decoding Behavioral Semantics in Mice"*
* **Core Contribution:** Framed mouse skeletal movements as words in a language model, applying BERT cloze-masking to raw keypoints for social phenotyping and temporal-order sensitivity testing.
* **Our Take & Positioning:** BehaVERT's input tokens are raw continuous coordinate frames ($T_t \in \mathbb{R}^{768}$). Because video is acquired at 80 Hz, adjacent frames are $>95\%$ identical due to video oversampling. Our model operates downstream of Keypoint-MoSeq, starting with **compressed, reusable behavioral words** (~400 ms primitives), eliminating video jitter and making the language analogy computationally and conceptually clean.

---

## 2. The Grand Summary: Our Unique Synthesis

$$\boxed{\text{Compressed Syllables (MoSeq)}} + \boxed{\text{Multi-Scale Predictive Hierarchy (Berman)}} + \boxed{\text{Motion Language Modeling (BehaVERT)}} + \boxed{\text{Phenotyping (uBAM)}} + \boxed{\text{Photometry (DA/ACh)}}$$

We are not claiming to have invented hierarchy, embeddings, or language models for behavior. **Our distinct contribution is connecting them into an end-to-end, interpretable framework for extended behavioral episodes and neuromodulatory steering.**
