import sys
from pathlib import Path
project_root = Path("../../").resolve()
sys.path.append(str(project_root))
import pickle
import pandas as pd
with open('/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/data/sessions_jonathan_juvenile_intruder_dirty.pkl', 'rb') as f: sessions = pickle.load(f)

unique_sessions = []
seen_csvs = set()
for s in sessions:
    csv_path = s.metadata.get('photometry_csv_path')
    if csv_path and csv_path not in seen_csvs:
        seen_csvs.add(csv_path)
        unique_sessions.append(s)

print('Unique sessions:', len(unique_sessions))

valid_wt = 0
valid_th = 0
for s in unique_sessions:
    if '470' not in s.photometry: continue
    sm = s.metadata.get("signal_meta", {})
    region_map = {k: v.get("region") for k, v in sm.items() if v.get("region")}
    regs = list(region_map.values())
    if "lNAc_5HT" in regs and "rNAc_DA" in regs:
        if s.metadata.get('genotype') == 'WT': valid_wt += 1
        elif s.metadata.get('genotype') == 'TH-Cre': valid_th += 1

print(f"Sessions with both lNAc_5HT and rNAc_DA: WT={valid_wt}, TH-Cre={valid_th}")

