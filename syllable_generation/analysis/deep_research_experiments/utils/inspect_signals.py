
import pandas as pd

paths = [
    "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_ifdn_sessions_w_sylls.pkl",
    "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl"
]

for p in paths:
    print(f"--- Loading {p} ---")
    try:
        data = pd.read_pickle(p)
        if isinstance(data, list) and len(data) > 0:
            s = data[0]
            print(f"Session: {s.session_id}")
            print(f"Metadata keys: {list(s.metadata.keys())}")

            if 'signal_keys' in s.metadata:
                print(f"signal_keys: {s.metadata['signal_keys']}")

            if 'signal_meta' in s.metadata:
                print(f"signal_meta: {s.metadata['signal_meta']}")

            # Check for other potential mapping keys
            for k in s.metadata:
                if 'signal' in k or 'channel' in k or 'fiber' in k:
                    print(f"  {k}: {s.metadata[k]}")

    except Exception as e:
        print(f"Error: {e}")
