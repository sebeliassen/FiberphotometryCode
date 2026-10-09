# Syllable Language Modeling & Generative Sequence Prediction

**Location**: `studies/behavior/language_models`  
**Status**: `ACTIVE`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Transformer, LSTM, and Markov models of syllable syntax; action prediction error (APE) benchmarks and hyperparameter sweeps.

## 2. Inputs
Tokenized syllable sequences and continuous kinematic embeddings

## 3. Outputs
Perplexity metrics, action prediction error curves, generation checkpoints

## 4. How to Run
```bash
python run_extended_sweep.py
```
