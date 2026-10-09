import pickle
import sys
from pathlib import Path

sys.path.append('/home/fsp585/Desktop/FiberphotometryCode')

SESSIONS_PKL = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl'
with open(SESSIONS_PKL, 'rb') as f:
    sessions = pickle.load(f)

stem_to_aid = {}
for s in sessions:
    original_path = s.video_refs_df.iloc[0]['path']
    stem = Path(original_path).stem
    aid = None
    if hasattr(s, 'metadata') and 'animal_id' in s.metadata:
        aid = s.metadata['animal_id']
    elif hasattr(s, 'fiber_refs_df') and 'animal_id' in s.fiber_refs_df.columns:
        aid = s.fiber_refs_df.iloc[0]['animal_id']
    stem_to_aid[stem] = str(aid)

from syllable_generation.syllabizers.keypoint_moseq_syll.utils.optimal_flip_list import OPTIMAL_FLIP_IDS

OPTIMAL_FLIP_STRS = [str(x) for x in OPTIMAL_FLIP_IDS]

import glob

results_path = glob.glob('/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project/*/results.h5')
if results_path:
    import keypoint_moseq as kpms
    results = kpms.load_results('/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project', 'optimized_flip_activity_k100k')

    video_paths_keys = list(results.keys())

    kept = []
    dropped = []
    for key in video_paths_keys:
        found_stem = next((s for s in stem_to_aid if s in key), None)
        if found_stem:
            aid = stem_to_aid[found_stem]
            if aid in OPTIMAL_FLIP_STRS:
                dropped.append((key, aid))
            else:
                kept.append((key, aid))
        else:
            kept.append((key, "UNKNOWN"))

    print(f"Dropped {len(dropped)}: {[a[1] for a in dropped]}")
    print(f"\nKept {len(kept)}:")
    for k, a in kept: print(f"  {k} (AID: {a})")
