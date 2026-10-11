
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

            syll_df = s.syllables
            photo_df = None

            if isinstance(s.photometry, dict) and 'combined_filtered' in s.photometry:
                photo_df = s.photometry['combined_filtered']

            print(f"Session: {s.session_id}")

            if isinstance(syll_df, pd.DataFrame):
                print(f"Syllables Len: {len(syll_df)}")
            else:
                print(f"Syllables content: {type(syll_df)}")

            if isinstance(photo_df, pd.DataFrame):
                print(f"Photometry Len: {len(photo_df)}")
                print(f"Photometry Cols: {list(photo_df.columns)}")

            if isinstance(syll_df, pd.DataFrame) and isinstance(photo_df, pd.DataFrame):
                print(f"Lengths Match? {len(syll_df) == len(photo_df)}")

    except Exception as e:
        print(f"Error: {e}")
