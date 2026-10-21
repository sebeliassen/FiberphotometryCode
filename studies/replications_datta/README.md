# Datta Lab Syllable Dynamics Replication Study

**Location**: `studies/replications_datta`
**Status**: `FROZEN`  
**Standard**: Conforms to [`docs/architecture/UNIT_CONTRACT.md`](../../docs/architecture/UNIT_CONTRACT.md)  

## 1. Purpose
Replication and validation of Markowitz et al. (2023) / Datta lab panels (1f, 1h, 1z) and baseline comparison pipelines.

## 2. Inputs
MoSeq syllable annotations and synchronized photometry signals

## 3. Outputs
Replication figures, transition entropy curves, and usage rankings

## 4. How to Run
```bash
python panel_1f/generate_panel_1f_publication.py
```

## Panel 1z session input boundary

`panel_1z.lib.loader.load_sessions(project, pkl_path=None)` accepts a selected
trusted pickle, or discovers under `resolve_data_path("data")`. `FP_DATA_ROOT`
selects a mirror root containing `data/`, independently of the code checkout.
The project pattern, newest-file priority, historical generic fallback, valid
session filter and region/hemisphere standardization are preserved. This path
repair does not establish support for every historical serialized object.
