# Panel 1z Replication: Fingerprinting & Stereotypy

## Overview
This module replicates the "Neural Fingerprinting" and "Stereotypy" analysis from the Datta 2023 paper. It aims to quantify how consistent neural activity is within animals vs. between animals (Fingerprinting) and how repetitive the behavioral sequence is using information-theoretic metrics (Stereotypy).

## Components

### 1. Neural Fingerprinting (`modules/fingerprint.py`)
- **Goal**: Assess if an animal's neural response to syllables is idiosyncratic (unique to the individual).
- **Method**:
    - Calculates the mean Z-scored response for each syllable.
    - Constructs a vector (fingerprint) for each animal x genotype.
    - Computes pairwise Euclidean distances between these vectors.
    - Comparing "Within-Group" (same genotype) vs. "Between-Group" (different genotype) distances.
- **Current Limitation**: 
    - **Drug Conflation**: For the AMPH dataset, the current logic averages Saline and Amphetamine sessions together for a given animal/genotype. It does *not* treat the drug state as a separate grouping variable. This means the "fingerprint" is a hybrid of baseline and drug states.

### 2. Stereotypy (`modules/stereotypy.py`)
- **Goal**: Measure the variability of syllable usage.
- **Metric**: **Usage Entropy**.
    - $H = -\sum p(x) \log p(x)$
    - Higher entropy = More uniform/random usage.
    - Lower entropy = More repetitive/stereotyped usage.
- **Output**: `stereotypy_stats.csv`

### 3. Transition Matrices (`modules/bigram.py`)
- **Goal**: Analyze the probability of Syllable B following Syllable A.
- **Method**:
    - Computes row-normalized transition matrices.
    - Compares transition probabilities between groups.

## Execution
Run the full panel via the runner script:
```bash
python analysis/replications/panel_1z/run_panel_1z.py --project amph --type all
```

## Directory Structure
- `modules/`: Contains calculation logic.
- `output_amph/`:
    - `fingerprint/`: Plots of pairwise distances.
    - `stereotypy/`: CSV statistics.
    - `bigram/`: Transition matrix plots.
