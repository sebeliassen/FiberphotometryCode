import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- SETUP ---
this_file = Path(__file__).resolve()
repo_root = this_file.parents[3]
syll_gen_root = repo_root / "syllable_generation"

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(syll_gen_root) not in sys.path:
    sys.path.insert(0, str(syll_gen_root))

from analysis.replications.syllable_traces.lib import data_loader

# --- CONFIGURATION ---
DATA_DIR = repo_root / "syllable_generation" / "data"
OUTPUT_DIR = repo_root / "syllable_generation" / "output" / "lateralized_da"

INPUT_PKL = DATA_DIR / "jonathan_amph_sessions_double_w_sylls.pkl"
OUTPUT_PKL = OUTPUT_DIR / "PETH_lateralized_da.pkl"

TARGET_SYLLABLES = {
    13: "right turn (45)",
    18: "left turn",
    20: "look right",
    28: "stand down / left",
    36: "look left",
    41: "groom right",
    42: "groom left"
}

PRE_ONSET_SEC = 2.0
POST_ONSET_SEC = 5.0

def extract_peths():
    if not INPUT_PKL.exists():
        print(f"[ERROR] Input file not found: {INPUT_PKL}")
        return

    print(f"Loading {INPUT_PKL.name}...")
    with open(INPUT_PKL, 'rb') as f:
        sessions = pickle.load(f)

    all_peths = []

    print(f"Extracting PETHs for syllables: {list(TARGET_SYLLABLES.keys())}")

    for s_idx, session in enumerate(sessions):
        if not hasattr(session, 'all_data_df') or session.all_data_df is None:
            continue

        df = session.all_data_df.copy()
        session_meta = session.metadata or {}

        # Determine sampling rate for this session
        if 'timestamp' in df.columns:
            dt = np.median(np.diff(df['timestamp'].values))
            fps = 1.0 / dt
        else:
            print(f"Warning: No timestamp in session {s_idx}, assuming 40 FPS")
            fps = 40.0

        pre_frames = int(PRE_ONSET_SEC * fps)
        post_frames = int(POST_ONSET_SEC * fps)

        # Base metadata
        genotype = session_meta.get("genotype", "UNK").upper()
        if genotype == "UNK": genotype = getattr(session, 'genotype', 'UNK').upper()

        drug = session_meta.get("drug", getattr(session, 'drug', 'saline'))
        inj_frame = session_meta.get("injection_frame")

        # Get signal metadata to map channels to brain regions
        valid_signals = []
        try:
            sig_meta = data_loader.extract_signal_metadata(df, session_meta)

            # Find the green signal that matches Datta Z-score if possible
            valid_signals = [m for m in sig_meta if m["orig_letter"] == "G" and "zscore_datta" in m["signal"]]
            if not valid_signals:
                valid_signals = [m for m in sig_meta if m["orig_letter"] == "G" and "zscore" in m["signal"]]

        except Exception as e:
            print(f"Failed to load sig meta: {e}")

        if not valid_signals or genotype == "UNK":
            print(f"[DEBUG] Skipping Session {s_idx}: No valid green signals or Genotype=UNK.")
            continue

        for signal_dict in valid_signals:
            target_signal_col = signal_dict["signal"]
            brain_reg = signal_dict.get("brain_reg", "UNK")
            hemisphere = signal_dict.get("hemisphere", getattr(session, 'hemisphere', 'UNK'))

            if brain_reg == "UNK":
                print(f"[DEBUG] Skipping Signal {target_signal_col} in Session {s_idx}: Region=UNK.")
                continue


            animal_id = str(getattr(session, 'animal_id', session_meta.get('animal_id', 'UNK')))

            meta_base = {
                'session_id': getattr(session, 'session_id', f"sess_{s_idx}"),
                'animal_id': animal_id,
                'genotype': genotype,
                'brain_reg': brain_reg,
                'hemisphere': hemisphere,
                'drug': drug
            }

            # We look at both 'syllable' (Normal) and 'syllable_flipped' (Flipped)
            for target_syll in TARGET_SYLLABLES.keys():

                # --- NORMAL ORIENTATION ---
                if 'syllable' in df.columns:
                    is_target = (df['syllable'] == target_syll).fillna(False).to_numpy(dtype=bool)
                    shifted = np.concatenate(([False], is_target[:-1]))
                    onsets_normal = np.where(is_target & ~shifted)[0]

                    valid_count = 0
                    for idx in onsets_normal:
                        if idx - pre_frames < 0 or idx + post_frames >= len(df): continue

                        window = df[target_signal_col].values[idx - pre_frames : idx + post_frames]
                        if np.isnan(window).any(): continue

                        baseline_mean = np.nanmean(df[target_signal_col].values[idx - pre_frames : idx])
                        if np.isnan(baseline_mean): continue

                        trace = window - baseline_mean
                        if np.isnan(trace).any(): continue

                        # Calculate Amph Condition
                        condition = f"{drug.capitalize()}-{genotype}"
                        if drug.lower() == "amphetamine" and inj_frame is not None:
                            try:
                                loc = df.index.get_loc(idx)
                                if isinstance(loc, slice): loc = loc.start
                                elif hasattr(loc, "__len__"): loc = loc[0]
                                curr_frame = df["frame_index"].iloc[loc]
                                condition = f"Amph-Pre-{genotype}" if curr_frame < inj_frame else f"Amph-Post-{genotype}"
                            except:
                                pass

                        all_peths.append({
                            **meta_base,
                            'condition': condition,
                            'syllable': target_syll,
                            'is_flipped': False,
                            'true_direction': 'Original',
                            'trace': trace,
                            'time_axis': np.linspace(-PRE_ONSET_SEC, POST_ONSET_SEC, len(window))
                        })
                        valid_count += 1

                    if valid_count > 0:
                        print(f"[DEBUG] Session {s_idx} | Syllable {target_syll} (Normal): Extracted {valid_count} via {brain_reg} ({target_signal_col})")

                # --- FLIPPED ORIENTATION ---
                if 'syllable_flipped' in df.columns:
                    is_target_f = (df['syllable_flipped'] == target_syll).fillna(False).to_numpy(dtype=bool)
                    shifted_f = np.concatenate(([False], is_target_f[:-1]))
                    onsets_flipped = np.where(is_target_f & ~shifted_f)[0]

                    valid_count_f = 0
                    for idx in onsets_flipped:
                        if idx - pre_frames < 0 or idx + post_frames >= len(df): continue

                        window = df[target_signal_col].values[idx - pre_frames : idx + post_frames]
                        if np.isnan(window).any(): continue

                        baseline_mean = np.nanmean(df[target_signal_col].values[idx - pre_frames : idx])
                        if np.isnan(baseline_mean): continue

                        trace = window - baseline_mean
                        if np.isnan(trace).any(): continue

                        # Calculate Amph Condition
                        condition = f"{drug.capitalize()}-{genotype}"
                        if drug.lower() == "amphetamine" and inj_frame is not None:
                            try:
                                loc = df.index.get_loc(idx)
                                if isinstance(loc, slice): loc = loc.start
                                elif hasattr(loc, "__len__"): loc = loc[0]
                                curr_frame = df["frame_index"].iloc[loc]
                                condition = f"Amph-Pre-{genotype}" if curr_frame < inj_frame else f"Amph-Post-{genotype}"
                            except:
                                pass

                        all_peths.append({
                            **meta_base,
                            'condition': condition,
                            'syllable': target_syll,
                            'is_flipped': True,
                            'true_direction': 'Flipped',
                            'trace': trace,
                            'time_axis': np.linspace(-PRE_ONSET_SEC, POST_ONSET_SEC, len(window))
                        })
                        valid_count_f += 1

                    if valid_count_f > 0:
                        print(f"[DEBUG] Session {s_idx} | Syllable {target_syll} (Flipped): Extracted {valid_count_f} via {brain_reg} ({target_signal_col})")

    peth_df = pd.DataFrame(all_peths)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    peth_df.to_pickle(OUTPUT_PKL)
    print(f"\\nSaved {len(peth_df)} traces to {OUTPUT_PKL}")
    print(f"Regions saved: {peth_df['brain_reg'].value_counts().to_dict()}")

if __name__ == "__main__":
    extract_peths()
