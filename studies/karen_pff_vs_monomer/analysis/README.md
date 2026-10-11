# Karen PFF vs Monomer Longitudinal Behavioral Study

**Location**: `studies/behavior/karen_studies`  
**Status**: `ACTIVE`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Longitudinal behavioral phenotyping comparing alpha-synuclein PFF vs Monomer models across multi-week timepoints.

## 2. Inputs
BotCam ventral pose estimations, TopCam dorsal tracks, and MoSeq syllable annotations

## 3. Outputs
Longitudinal progression curves, motor impairment metrics, transition graphs

## 4. How to Run
```bash
python run_karen_analysis.py
```
