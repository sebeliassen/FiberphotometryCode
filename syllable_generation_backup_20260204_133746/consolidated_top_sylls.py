
import sys
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")

import pandas as pd
import numpy as np
from syllable_generation.datamodel import SessionRecord

# SETTINGS
POST_INJECTION_DELAY_MIN = 5
FPS = 80 

PROJECTS = {
    'AMPH': "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl",
    'IFDN': "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_ifdn_sessions_w_sylls.pkl"
}

def get_post_injection_data(session):
    inj_frame = session.metadata.get('injection_frame')
    if inj_frame is None:
        # IFDN might not have injection_frame if it's not a drug study?
        # Let's check metadata in a bit or assume 0 for IFDN.
        return session.syllables
    
    start_frame = int(inj_frame + POST_INJECTION_DELAY_MIN * 60 * FPS)
    sylls = session.syllables
    if sylls is None or start_frame >= len(sylls):
        return None
    return sylls[start_frame:]

def analyze_project(name, path):
    print(f"\n--- ANALYZING PROJECT: {name} ---")
    sessions = pd.read_pickle(path)
    print(f"Loaded {len(sessions)} sessions.")
    
    group_stats = {}
    for s in sessions:
        g = s.metadata.get('genotype', 'UNK')
        d = s.metadata.get('drug', 'None')
        group_key = f"{g}_{d}" if d != 'None' else g
        
        if group_key not in group_stats: group_stats[group_key] = {}
        
        syll_data = get_post_injection_data(s)
        if syll_data is None: continue
        
        arr = np.array(syll_data)
        if arr.ndim > 1: arr = arr.flatten()
        series = pd.Series(arr).dropna().astype(int)
        if len(series) == 0: continue
        
        mask = series != series.shift(1)
        onsets = series[mask]
        counts = onsets.value_counts()
        
        for syll, count in counts.items():
            if syll not in group_stats[group_key]:
                group_stats[group_key][syll] = {'onsets': 0, 'sessions': set()}
            group_stats[group_key][syll]['onsets'] += count
            group_stats[group_key][syll]['sessions'].add(s.session_id)

    # Optimal T (Min Trials)
    max_trials = 0
    for g in group_stats.values():
        for d in g.values():
            if isinstance(d, dict) and 'sessions' in d:
                max_trials = max(max_trials, len(d['sessions']))
                
    optimal_T = 0
    for T in range(max_trials, 0, -1):
        sat = True
        for gkey, smap in group_stats.items():
            valid = [s for s, d in smap.items() if isinstance(d, dict) and 'sessions' in d and len(d['sessions']) >= T]
            if len(valid) < 20: 
                sat = False; break
        if sat:
            optimal_T = T; break
    
    print(f"Optimal T: {optimal_T}")
    
    # Rank-Sum
    all_sylls = set()
    for gkey in group_stats.keys():
        smap = group_stats[gkey]
        cands = []
        for s, d in smap.items():
            if isinstance(d, dict) and 'sessions' in d and len(d['sessions']) >= optimal_T:
                cands.append((s, d['onsets']))
                all_sylls.add(s)
        cands.sort(key=lambda x: x[1], reverse=True)
        group_stats[gkey]['ranks'] = {s: i+1 for i, (s, _) in enumerate(cands)}
        group_stats[gkey]['max_r'] = len(cands)

    scores = []
    for s in all_sylls:
        # Skip artifactual negative IDs
        if not isinstance(s, (int, np.integer)) or s < 0: continue
        
        rs = 0
        for gkey in group_stats.keys():
            r = group_stats[gkey]['ranks'].get(s, group_stats[gkey]['max_r'] + 5)
            rs += r
        scores.append({'id': s, 'score': rs})
    
    scores.sort(key=lambda x: x['score'])
    top_20 = [x['id'] for x in scores[:20]]
    return top_20

results = {}
for name, path in PROJECTS.items():
    results[name] = analyze_project(name, path)

print("\n" + "="*40)
print("FINAL FAIR TOP SYLLABLES (SHARED CORE)")
for name, ids in results.items():
    print(f"{name}: {ids}")
print("="*40)
