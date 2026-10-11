# SHMoSeq Hierarchical Sequence Analysis Study

**Location**: `studies/behavior/shmoseq`  
**Status**: `ACTIVE`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Semi-Hidden Markov Model behavioral sequencing, sub-state transition dynamics, and hierarchical behavioral parsing.

## 2. Inputs
Animal keypoint trajectories and model checkpoint states

## 3. Outputs
Hierarchical state transition matrices and sequence decompositions

## 4. How to Run
```bash
python run_shmoseq.py
```
