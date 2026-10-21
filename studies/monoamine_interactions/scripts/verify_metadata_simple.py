import os
import pickle
import sys
import time

sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")

pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/data/sessions_dirty.pkl"

if not os.path.exists(pkl_path):
    print("Pickle file does not exist!")
    sys.exit(1)

mtime = os.path.getmtime(pkl_path)
print(f"Pickle modified time: {time.ctime(mtime)}")

with open(pkl_path, "rb") as f:
    sessions = pickle.load(f)

print(f"Loaded {len(sessions)} sessions.")

for i, s in enumerate(sessions):
    print(f"\nSession {i}: {s.session_id}")
    print(f"Animal ID: {s.metadata.get('animal_id')}")
    print(f"Genotype Keys: {[k for k in s.metadata.keys() if 'geno' in k.lower()]}")
    print(f"Genotype Value: {s.metadata.get('genotype')}")
    print(f"Signal 0 Value: {s.metadata.get('signal_0')}")
    if s.metadata.get('animal_id'):
        print(f"FULL METADATA: {s.metadata}")
