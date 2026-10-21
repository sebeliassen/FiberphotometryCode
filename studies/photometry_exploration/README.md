# Exploratory Photometry & Syllable Analysis Suite

**Location**: `studies/photometry/exploratory`  
**Status**: `FROZEN`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Exploratory scripts, session state analyzers, signal-action prediction models, and legacy plotting utilities.

## 2. Inputs
Processed session pickle files (*.pkl) and photometry HDF5 / CSV streams

## 3. Outputs
Exploratory figures, session states, and t-SNE projections

## 4. How to Run
```bash
python kpms/load_and_analyze_session_state.py
```
