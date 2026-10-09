import pickle
from pathlib import Path

# Load Sessions & Map Videos
SESSIONS_PKL = '/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_dirty.pkl'
with open(SESSIONS_PKL, 'rb') as f:
    sessions = pickle.load(f)

stem_to_aid = {}
for s in sessions:
    if hasattr(s, 'video_refs_df') and not s.video_refs_df.empty:
        stem = Path(s.video_refs_df.iloc[0]['path']).stem 
        if hasattr(s, 'metadata') and 'is_flipped' in s.metadata:
            # Maybe 'is_flipped' exists? Let's print out what metadata there is
            stem_to_aid[stem] = s.metadata
        elif hasattr(s, 'metadata') and 'animal_id' in s.metadata:
            stem_to_aid[stem] = s.metadata['animal_id']

print("Metadata examples:")
for k, v in list(stem_to_aid.items())[:3]:
    print(k, v)
