# SKILL DOC: Sequence Motif Analysis Pipeline

This document explains the architecture and mathematics behind the Sequence Motif module (`lib/motifs.py`) in the `ulrik_kinematics` pipeline. It is designed to act as an AI reference guide when modifying or extending this behavior sequence analyzer.

## 1. Goal & Execution
The goal of this module is to graduate from global session averages (e.g., "how fast is the mouse?") to complex intentional strategies (e.g., "does the mouse routinely execute a `rear -> turn -> dash` behavioral chunk?").

### Endpoints
The pipeline is invoked via the `run_analysis.py` orchestrator:
- `python run_analysis.py --verify-motifs` (Executes Double-Negative integrity test).
- `python run_analysis.py --run-motifs-discovery` (Executes the full Two-Step biological mapping).

## 2. Core Mining Logic (`extract_ngrams`)

When mining motifs (n-grams) from an array of MoSeq syllables, we must avoid "structural freezing." A mouse standing still for 5 seconds generates identical frames (`A A A A A A...`). A naive n-gram miner will misinterpret this as a complex `A -> A -> A` motif.

**The Solution:**
1. **Semantic Translation First**: We first optionally translate the numerical syllable IDs (e.g. `12`) to a human label (e.g., `walk`) using `get_label_mapping()`. We support a cleaned `standard` mapping or an unbiased `alphabet` mapping.
2. **Semantic Run-Length Encoding**: We collapse adjacent identical frames (e.g., `walk -> walk -> walk -> rear`) into a single state boundary (`walk -> rear`). This ensures the sequence miner extracts true behavioral state transitions, not just the duration of a single behavior.

## 3. The Two-Step Biological Discovery (`run_discovery`)

Biological systems generate a massive amount of "incidental" transitions. We use a rigid Two-Step mathematical approach to separate true biological strategy from noise.

### Step 1: The Markov Null Filter (PMI Ratio)
If `walk` naturally comprises 50% of an animal's time, then `walk -> pause -> walk` will occur at a staggeringly high frequency just by mathematical coincidence.
To filter out coincidence, we create a 1st-Order Markov Random Walk expectation:
- **P_Expected = P_Base(A) × P_Transition(B|A) × P_Transition(C|B)**
- **Ratio = P_Observed / P_Expected**
The script mathematically calculates this ratio for every trigram globally. **Only motifs with a Ratio ≥ 1.5 are preserved.** Everything else (random walk artifacts) is dropped.

### Step 2: Cross-Phenotype Significance
For the surviving "intentional" motifs, we query their probability on a session-by-session basis. This gives us population distributions (e.g., an array of 12 probabilities for `WT Amphetamine`, and 12 probabilities for `KI Amphetamine`).
We use the **两-sided Mann-Whitney U Test** (`scipy.stats.mannwhitneyu`) to identify which structured motifs are significantly altered by the Genotype or Drug condition.

## 4. Double-Negative Verification (`verify_motifs_module`)

To guarantee the module isn't hallucinating structure, it must successfully fail.
1. The script first runs `get_top_motifs` on real data and plots the true Sequence Entropy and Top Motif support.
2. The script clones the dataset and wildly **shuffles the syllable array**, completely destroying temporal order while perfectly preserving the literal absolute frequency of behaviors.
3. The script asserts that the Sequence Entropy skyrockets to near-maximum capacity, and the actual Top Motif frequency drops precipitously to ~0.001. If the scripts mathematically verify this crash occurs, the test passes.
