
import sys
# Ensure we can load custom classes
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")

import pandas as pd
import numpy as np
from syllable_generation.datamodel import SessionRecord

# import processing  # REMOVED

# SETTINGS
POST_INJECTION_DELAY_MIN = 5
FPS = 80 # Video FPS is 80Hz.
SYLLABLE_KEY = 'syllable' # Key in results if we had them, but here we look at session.syllables


# pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_sanitized.pkl"
pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl"

def get_post_injection_data(session):
    # Locate injection frame
    inj_frame = session.metadata.get('injection_frame')
    if inj_frame is None:
        return None
    
    # Calculate Start Frame
    start_frame = int(inj_frame + POST_INJECTION_DELAY_MIN * 60 * FPS)
    
    # Get Syllables Dataframe (if present)
    # The session object might not have 'results' loaded directly?
    # Usually session.syllables is a Series or array aligned to timeline
    # INSPECT SHOWS: 'syllables' attribute
    
    sylls = session.syllables
    if sylls is None:
        return None
    
    # Check length
    if start_frame >= len(sylls):
        return None # Session ended before window
        
    # Slice
    # Note: Syllables might be NaN or ints.
    subset = sylls[start_frame:]
    return subset

print("Loading Sessions...")
sessions = pd.read_pickle(pkl_path)
print(f"Loaded {len(sessions)} sessions.")

# 1. Aggregate Usage Stats per Group
# Groups: Genotype x Drug
# Structure: group_stats[group_key][syllable] = { 'onsets': X, 'trials': {session_id, ...} }

group_stats = {}

# Define Groups
for s in sessions:
    g = s.metadata.get('genotype', 'UNK')
    d = s.metadata.get('drug', 'UNK')
    
    # Normalize Drug names if needed (Amph vs Amphetamine?)
    if d.lower().startswith('amph'): d = 'Amph'
    elif d.lower().startswith('sal'): d = 'Saline'
    
    group_key = f"{g}_{d}"
    
    if group_key not in group_stats:
        group_stats[group_key] = {}
        
    # Get Data
    syll_data = get_post_injection_data(s)
    if syll_data is None: continue
    
    # Convert to array/series
    # Find onsets: change != previous
    # Handle NaNs: drop
    # Series is easiest
    
    # Flatten if 2D (kpms returns (T,1) sometimes)
    arr = np.array(syll_data)
    if arr.ndim > 1:
        arr = arr.flatten()
        
    series = pd.Series(arr).dropna().astype(int)
    if len(series) == 0: continue
    
    # Onsets: value != shift(1)
    # Or just run-length encoding.
    # Simple Onset Count:
    # A bit slow if loop?
    # Pandas shift:
    mask = series != series.shift(1)
    onsets = series[mask]
    
    # Count onsets per syllable
    counts = onsets.value_counts()
    
    # Aggregation
    for syll, count in counts.items():
        if syll not in group_stats[group_key]:
            group_stats[group_key][syll] = {'onsets': 0, 'sessions': set()}
        
        group_stats[group_key][syll]['onsets'] += count
        group_stats[group_key][syll]['sessions'].add(s.session_id)

print(f"Groups Found: {list(group_stats.keys())}")

# 2. Iterative Filter optimization
# Find max T (min_trials) such that ALL 4 groups have >= 20 syllables.

# Determine max possible trials
all_counts = []
for g in group_stats.values():
    for data in g.values():
        all_counts.append(len(data['sessions']))
        
max_trials = max(all_counts) if all_counts else 0
print(f"Max Sessions for any syllable: {max_trials}")

optimal_T = 0
valid_sylls_per_group = {}

# Search downwards
for T in range(max_trials, 0, -1):
    # Check condition
    satisfied = True
    current_sets = {}
    
    for gkey, syll_map in group_stats.items():
        # Find sylls with num_sessions >= T
        valid = [s for s, d in syll_map.items() if len(d['sessions']) >= T]
        current_sets[gkey] = valid
        
        if len(valid) < 20: 
            satisfied = False
            break
    
    if satisfied:
        optimal_T = T
        valid_sylls_per_group = current_sets
        break

print("-" * 40)
print(f"OPTIMAL MIN_TRIALS (T): {optimal_T}")
print("-" * 40)

if optimal_T == 0:
    print("WARNING: Could not find any T where all groups have 20 syllables. Even at T=1?")
    # Fallback to T=1 stats
else:
    print(f"At T={optimal_T}, syllable counts per group:")
    for gkey, sylls in valid_sylls_per_group.items():
        print(f"  {gkey}: {len(sylls)} syllables")

# 3. FAIR SORTING: Rank-Sum (Borda Count)
# Goal: Find syllables that are consistently high-ranked across groups, 
# preventing one group's massive counts from dominating the list.

print("\nRunning Rank-Sum (Borda Count) Analysis...")

# A. Compute Ranks within each group
# syll_ranks[syll] = [rank_g1, rank_g2, ...]
syll_ranks_map = {}
all_sylls = set()

for gkey in group_stats.keys():
    # Get all sylls for this group
    syll_map = group_stats[gkey]
    
    # Filter by T constraint? 
    # User asked for "top 20", but maybe we relax T for the ranking pool 
    # and only filter T for the final list?
    # Let's keep the T filter to ensure robustness (must be present in X sessions).
    # If a syllable is not present in >= T sessions, does it get a "penalty rank"?
    # Yes, effectively Rank = NumSylls + 1 (or infinity).
    
    candidates = []
    for s, data in syll_map.items():
        if len(data['sessions']) >= optimal_T:
            candidates.append((s, data['onsets']))
            all_sylls.add(s)
            
    # Sort by onsets desc (Rank 1 is best)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Map back to rank
    rank_lookup = {syll: i+1 for i, (syll, _) in enumerate(candidates)}
    
    # Store
    # We need to handle sylls that are missing/filtered in this group later.
    group_stats[gkey]['rank_lookup'] = rank_lookup
    group_stats[gkey]['max_rank'] = len(candidates)

# B. Compute Global Rank-Sum
global_scores = []

for s in all_sylls:
    rank_sum = 0
    penalty_count = 0
    details = []
    
    for gkey in group_stats.keys():
        lookup = group_stats[gkey].get('rank_lookup', {})
        max_r = group_stats[gkey].get('max_rank', 0)
        
        if s in lookup:
            r = lookup[s]
            rank_sum += r
            details.append(f"{gkey}:#{r}")
        else:
            # Penalize missing/filtered syllables
            # Penalty = MaxRank + 1? Or a fixed large number?
            # MaxRank + 1 is reasonable for Borda.
            penalty_r = max_r + 5 # Slight extra penalty?
            rank_sum += penalty_r
            penalty_count += 1
            details.append(f"{gkey}:X({penalty_r})")
            
    # Normalize? No, sum is fine since same number of groups.
    global_scores.append({
        'syllable': s,
        'rank_sum': rank_sum,
        'penalties': penalty_count,
        'details': ", ".join(details)
    })

# C. Sort by Rank Sum (Ascending is better)
global_scores.sort(key=lambda x: x['rank_sum'])

# D. Select Final Sets
# 1. Consensus/Shared: Top 20 by Rank-Sum
consensus_limit = 20
consensus_sylls = [x['syllable'] for x in global_scores[:consensus_limit]]

# 2. Specific Stars: Syllables that are NOT in Consensus, but were Top 10 in at least one group.
specific_sylls = set()

print(f"\nSpecific Group Stars (Not in Consensus Top {consensus_limit}):")
for gkey in group_stats.keys():
    # Get top 15 for this group
    syll_map = group_stats[gkey]
    candidates = []
    for s, data in syll_map.items():
        if isinstance(s, (int, np.integer)):
            if len(data['sessions']) >= optimal_T:
                candidates.append((s, data['onsets']))
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    top_local = candidates[:15]
    for i, (s, ons) in enumerate(top_local):
        if s not in consensus_sylls and isinstance(s, (int, np.integer)) and s >= 0:
            print(f"  {gkey} #{i+1}: Syll {s} ({ons} onsets)")
            specific_sylls.add(s)

# Combine
final_ids_shared = sorted([int(s) for s in consensus_sylls if int(s) >= 0])
final_ids_specific = sorted([int(s) for s in specific_sylls if int(s) >= 0])

print("\n" + "="*40)
print(f"REPORT:")
print(f"1. Shared Core (Top {len(final_ids_shared)}): {final_ids_shared}")
print(f"2. Specific Stars ({len(final_ids_specific)}): {final_ids_specific}")
print(f"Total Unique Targets: {len(set(final_ids_shared + final_ids_specific))}")
