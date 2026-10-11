# Generalization Paradigms: From Exact $n$-Grams to Continuous Semantic Geometry

---

## Executive Overview: The Generalization Spectrum

A foundational challenge in computational ethology is **combinatorial sequence sparsity**: as the length of an action sequence grows, the probability of observing the *exact same string of sub-second syllables* drops exponentially to zero.

Different modeling paradigms resolve this sparsity through distinct forms of generalization:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE 4 TIERS OF BEHAVIORAL GENERALIZATION                          │
├───────┬─────────────────────────────┬─────────────────────────────────┬─────────────────────────┤
│ Tier  │ Paradigm                    │ Mathematical Mechanism          │ What Makes Sequences    │
│       │                             │                                 │ "Similar"?              │
├───────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────┤
│ 1     │ Exact $n$-Gram Matching     │ Literal string matching         │ Exact character match:  │
│       │                             │ $s_1 \to s_2 \to \dots \to s_n$ │ $A \to B \to C$ only.   │
├───────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────┤
│ 2     │ Transition Regimes          │ Hidden Markov Model with state- │ Shared transition bias  │
│       │ (shMoSeq / Weinreb 2026)    │ specific transition matrices    │ over orthogonal tokens  │
│       │                             │ $\mathbf{T}^{(k)} \in \mathbb{R}^{K \times K}$          │ ($A \to B \to C \sim A \to C \to B$).│
├───────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────┤
│ 3     │ Distributional Action       │ Static vector embeddings        │ Synonymous substitution │
│       │ Semantics (Action2Vec/SGNS) │ $\mathbf{e}_i \in \mathbb{R}^d$ via Skip-Gram   │ ($\mathbf{e}_{12} \approx \mathbf{e}_{38}$ for     │
│       │                             │ context co-occurrence           │ related rearing motifs).│
├───────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────┤
│ 4     │ Contextual Geometry &       │ Dynamic self-attention latents  │ Graded episode distance │
│       │ Manifolds (Transformer/Mamba)│ $\mathbf{z}_t = f(H_t) \in \mathbb{R}^d$         │ $\mathbf{z}_A \approx \mathbf{z}_B$ across entire  │
│       │                             │ with contextual polysemy        │ 68s behavioral windows. │
└───────┴─────────────────────────────┴─────────────────────────────────┴─────────────────────────┘
```

---

## 1. Why Exact $n$-Gram Matching Fails in Ethology

Searching for exact recurring sub-sequences:
$$\text{Rear} \longrightarrow \text{Sniff} \longrightarrow \text{Locomote} \longrightarrow \text{Rear} \longrightarrow \text{Turn}$$

becomes hopelessly sparse because:
1. **Biological Stochasticity:** Animals do not execute deterministic clockwork scripts; motor execution varies naturally across trials.
2. **Syllable Fragmentation:** If an unsupervised algorithm (like MoSeq) splits "rearing" into three subtle variants ($s_{12}, s_{38}, s_{71}$), a literal sequence match fails whenever the animal executes variant $s_{38}$ instead of $s_{12}$, despite identical ethological intent.

---

## 2. How shMoSeq Relaxes Exact Matching (The Transition Regime)

In **shMoSeq** (*Weinreb et al., Neuron 2026*), higher-order behavioral states are **not** defined as single rigid sequences. Instead:
* There is a global baseline transition matrix shared across the session.
* Each shMoSeq state possesses **state-specific transition biases** that increase the probability of certain syllable-to-syllable transitions and syllable occupancies.

Thus, sequences with varying order:
$$A \longrightarrow B \longrightarrow C \longrightarrow A \quad \text{and} \quad A \longrightarrow C \longrightarrow B \longrightarrow A \quad \text{and} \quad B \longrightarrow A \longrightarrow C \longrightarrow A$$
can all belong to the **same discrete state** if they are statistically compatible with that state's transition matrix.

### The Limitation of Categorical State-Spaces:
While shMoSeq elegantly softens exact sequence matching, it still treats all $K$ syllables as **orthogonal, discrete categories**:
* Syllable $s_{12}$ and Syllable $s_{38}$ have mathematical distance $\delta_{12, 38} = 0$ (orthogonal one-hot vectors).
* The model cannot recognize that $s_{12}$ and $s_{38}$ are interchangeable "synonyms" unless manually coarse-grained post-hoc.

---

## 3. Action2Vec: Distributional Semantic Embeddings

In Natural Language Processing, **Word2Vec** solved vocabulary fragmentation by establishing that *"words that appear in similar contexts share similar meanings"*.

Applying **Skip-Gram with Negative Sampling (SGNS)** to behavioral syllables creates **Action2Vec**:
$$\max_{\mathbf{W}, \mathbf{C}} \sum_{t=1}^T \sum_{-w \le j \le w, j \neq 0} \left[ \log \sigma(\mathbf{c}_{s_{t+j}}^\top \mathbf{e}_{s_t}) + \sum_{k=1}^K \mathbb{E}_{s_k \sim P_n} \log \sigma(-\mathbf{c}_{s_k}^\top \mathbf{e}_{s_t}) \right]$$

### The Result:
* If Syllable $12$ (Supported Wall Rear) and Syllable $38$ (Free Unsupported Rear) consistently precede locomotor bouts and follow wall sniffs, the model learns:
  $$\|\mathbf{e}_{12} - \mathbf{e}_{38}\|_2 \approx 0$$
* The representation **automatically generalizes across synonymous action motifs**, replacing discrete orthogonal tokens with a smooth metric space.

---

## 4. Contextual Embeddings & Manifolds: Resolving Behavioral Polysemy

Static embeddings (Action2Vec) assign each syllable a single fixed vector $\mathbf{e}_i$. However, natural behavior exhibits **behavioral polysemy** (contextual ambiguity):

```
                       BEHAVIORAL POLYSEMY: THE SAME ACTION IN DIFFERENT CONTEXTS
                       
  Context A (Active Exploration):
  [Locomote] ──► [Wall Sniff] ──► [ REAR ] ──► [Locomote] ──► [Center Cross]
                                     │
                                     ▼
                     e_Rear(Exploration): Encodes high exploratory drive & spatial mapping.

  Context B (Stereotyped Grooming Chain):
  [Face Wash] ──► [Body Lick] ──► [ REAR / HUNCH ] ──► [Tail Groom] ──► [Scratch]
                                     │
                                     ▼
                     e_Rear(Grooming): Encodes low locomotor intent & grooming syntax.
```

### Contextual Self-Attention (Causal Transformer):
A deep sequence model maps the same syllable token $s_t$ to a dynamic contextual vector that depends on the surrounding behavioral history:
$$\mathbf{h}_t = \text{Transformer}(s_t \mid s_{t-1}, s_{t-2}, \dots, s_{t-W})$$

When pooled over an extended window ($W_t = 68.3\text{s}$), the model produces a continuous episode coordinate:
$$\mathbf{z}_t \in \mathbb{R}^d$$

---

## 5. Why Continuous Episode Geometry Transforms Phenotyping

Moving from hard categorical states to a continuous semantic space ($\mathbf{z}_t \in \mathbb{R}^d$) provides a critical advantage for disease phenotyping:

```
  DISCRETE STATE-SPACE (Hard Labels):
  Control Mouse (WT):   [ State 3 ] ──► (Assigned nominal label "State 3")
  Parkinsonian (PFF):   [ State 3 ] ──► (Assigned nominal label "State 3")
  Result: WT == PFF in State Occupancy (Deficit is masked!)

  CONTINUOUS MANIFOLD GEOMETRY (Our Approach):
  Control Mouse (WT):   z_WT  = (+1.8, -0.4, +2.1)
  Parkinsonian (PFF):   z_PFF = (+0.6, -0.4, +0.9)
  Result: ||z_WT - z_PFF|| > 0 (Exposes subtle kinematic & contextual shift inside the state!)
```

* **Graded Phenotypes:** An animal model of neurological disease might enter the exact same broad behavioral programs as wild-type controls, but execute them with altered vigor, restricted transitions, or fragmented syntax.
* A hard categorical classifier forces both animals into the same bin ("State 3"), whereas **continuous episode geometry preserves the metric displacement**.

---

## 6. How to Present This Distinction to Your Professor

If your professor asks: *"How is your sequence representation different from what shMoSeq does?"*

### Your 30-Second Answer:
> *"shMoSeq relaxes exact sequence matching by grouping syllables into recurring transition regimes. That is a major advance. 
>
> What language-inspired models add is **continuous semantic geometry**:
> 1. **Action Synonyms:** Actions that play similar contextual roles share close embedding vectors ($\mathbf{e}_{12} \approx \mathbf{e}_{38}$), preventing vocabulary fragmentation.
> 2. **Context-Dependence:** The same action (like rearing) is embedded differently depending on whether it occurs during spatial exploration or repetitive grooming.
> 3. **Graded Phenotyping:** Instead of forcing behavior into rigid all-or-nothing bins, episodes map to continuous coordinates ($\mathbf{z}_t \in \mathbb{R}^d$), allowing us to detect subtle disease shifts even within the same broad behavioral state."*
