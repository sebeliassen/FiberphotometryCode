import os
import sys
import pickle
import numpy as np
import pandas as pd

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
LM_DIR = os.path.join(ROOT_DIR, "syllable_generation", "analysis", "language_models")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, LM_DIR)

pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
if not os.path.exists(pkl_path):
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")

with open(pkl_path, "rb") as f:
    sessions = pickle.load(f)

print(f"Loaded {len(sessions)} raw sessions.")

session_info = []
for i, s in enumerate(sessions):
    meta = getattr(s, "metadata", {})
    gt = meta.get("genotype", "WT")
    drug = meta.get("drug", "Saline")
    inj_f = meta.get("injection_frame", None)
    entry_f = meta.get("entry_frame", None)
    sid = getattr(s, "session_id", f"sess_{i}")
    session_info.append({
        "index": i,
        "session_id": sid,
        "genotype": gt,
        "drug": drug,
        "cohort": f"{gt}_{drug}",
        "inj_frame": inj_f,
        "entry_frame": entry_f
    })

df_info = pd.DataFrame(session_info)
print("\nCohort distribution:")
print(df_info["cohort"].value_counts())
print("\nSaline / WT sessions:")
print(df_info[df_info["drug"].str.lower().str.contains("saline") | (df_info["cohort"] == "WT_Saline")])
