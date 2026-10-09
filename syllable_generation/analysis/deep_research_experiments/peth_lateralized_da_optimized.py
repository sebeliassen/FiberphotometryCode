import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy.stats import pearsonr

# --- SETUP ---
this_file = Path(__file__).resolve()
repo_root = this_file.parents[3] 


# --- CONFIGURATION ---
INPUT_PKL = repo_root / "syllable_generation" / "data" / "jonathan_amph_sessions_optimized_w_sylls.pkl"
BASE_OUTPUT_DIR = repo_root / "syllable_generation" / "output" / "lateralized_da_optimized"
PETH_CACHE = BASE_OUTPUT_DIR / "PETH_lateralized_da_optimized.pkl"

TARGET_SYLLABLES_OPT = {
    # Left physical movements
    2: "Left Turn",
    5: "Turn Left",
    18: "Left Turn",
    28: "Turn Left",
    # Right physical movements
    12: "Right Turn",
    30: "Turn Right",
    46: "Look Right",
    59: "Lean Right"
}

# Based on optimized_flipping_config.txt
FLIPPED_MICE = ["728", "740", "1664", "1666", "1767", "1773", "1779", "1781"]

PRE_ONSET_SEC = -2.0
POST_ONSET_SEC = 5.0
FPS_PHOTOMETRY = 80.0
PRE_FRAMES = int(abs(PRE_ONSET_SEC) * FPS_PHOTOMETRY)
POST_FRAMES = int(POST_ONSET_SEC * FPS_PHOTOMETRY)

def _get_is_flipped(session_id: str) -> bool:
    # animal ID is typically the 3rd part of session id 's_xx_AID_...' or similar,
    # let's just do a robust substring match for the 3/4 digit ID
    for aid in FLIPPED_MICE:
        if aid in session_id:
            return True
    return False

def extract_peths():
    if PETH_CACHE.exists():
        print(f"Loading cached PETH data from {PETH_CACHE.name}...")
        return pd.read_pickle(PETH_CACHE)

    print(f"Loading {INPUT_PKL.name}...")
    try:
        sessions = pd.read_pickle(INPUT_PKL)
    except Exception as e:
        print(f"Error loading {INPUT_PKL}: {e}")
        return pd.DataFrame()

    results = []
    
    unique_labeled_flipped = set()
    unique_labeled_non = set()

    for s in tqdm(sessions, desc="Extracting PETHs from sessions"):
        if not hasattr(s, 'all_data_df') or s.all_data_df.empty:
            continue
            
        df = s.all_data_df.copy()
        
        # Find all valid signal columns (0, 1, 2, 3...)
        sig_cols = [c for c in df.columns if c.startswith("signal_") and c.endswith("_zscore_datta")]
        
        if not sig_cols:
            print(f"[DEBUG] Skipping {s.session_id} - no signal_X_zscore_datta columns found")
            continue
            
        animal_id = s.metadata.get('animal_id', 'UNK')
        is_flipped = _get_is_flipped(animal_id)
        
        label = "Flipped Session" if is_flipped else "Non-Flipped Session"
        if is_flipped: unique_labeled_flipped.add(animal_id)
        else: unique_labeled_non.add(animal_id)

        print(f"[DEBUG] Session {s.session_id} | Animal {animal_id} | Cols: {sig_cols} | Label: {label}")
        
        # We process each signal column separately
        for sig_col in sig_cols:
            sig_key = sig_col.replace('_zscore_datta', '')
            try:
                sig_meta = s.metadata['signal_meta'][sig_key]
                reg = sig_meta.get('brain_reg', 'UNK')
            except (KeyError, TypeError):
                reg = "UNK"
                
            if reg == 'UNK': 
                print(f"[DEBUG] Skipping {s.session_id} - UNK Region")
                continue
            if reg == 'VS': continue # User asked to ignore VS for this parity check

            # Standardized drug mapping
            cond_raw = s.metadata.get('drug', 'UNK')
            if 'Saline' in cond_raw: cond = 'Saline'
            elif 'Amph-Pre' in cond_raw: cond = 'Amph-Pre'
            elif 'Amph-Post' in cond_raw: cond = 'Amph-Post'
            elif 'Amphetamine' in cond_raw:
                # If it's just 'Amphetamine', we assume it's the post-dose recording
                cond = 'Amph-Post' if 'Post' in cond_raw else ('Amph-Pre' if 'Pre' in cond_raw else 'Amph-Post')
            else: cond = cond_raw

            genotype = s.metadata.get('genotype', 'UNK')
            if genotype == 'UNK': 
                print(f"[DEBUG] Skipping Session {s.session_id} - UNK Genotype")
                continue

            # Find onsets for all target syllables
            for syll_val in TARGET_SYLLABLES_OPT.keys():
                syll_mask = (df['syllable'] == syll_val)
                # Find exact turn-ons
                onsets = syll_mask & (~syll_mask.shift(1).fillna(False))
                onset_indices = df.index[onsets].tolist()
                
                for idx in onset_indices:
                    start_idx = idx - PRE_FRAMES
                    end_idx = idx + POST_FRAMES
                    
                    if start_idx < 0 or end_idx >= len(df): continue
                    
                    # Check for NaNs
                    chunk = df.iloc[start_idx:end_idx]
                    if chunk[sig_col].isna().any():
                        continue
                        
                    raw_trace = chunk[sig_col].values
                    # Baseline subtraction [first PRE_FRAMES]
                    baseline = np.mean(raw_trace[:PRE_FRAMES])
                    norm_trace = raw_trace - baseline
                    
                    time_axis = np.linspace(PRE_ONSET_SEC, POST_ONSET_SEC, len(norm_trace))
                    
                    results.append({
                        "session_id": s.session_id,
                        "genotype": genotype,
                        "drug": cond,
                        "brain_reg": reg,
                        "syllable": syll_val,
                        "is_flipped": label,
                        "trace": norm_trace,
                        "time_axis": time_axis
                    })
                
    out_df = pd.DataFrame(results)
    if not out_df.empty:
        print("\n[DEBUG] Extraction Summary:")
        print(out_df['is_flipped'].value_counts())
        
        BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_df.to_pickle(PETH_CACHE)
        print(f"Saved {len(out_df)} instances to {PETH_CACHE}")
    return out_df

def process_and_plot(peth_meta_df, average_by, share_y_axis):
    out_dir_name = f"plots_{average_by}_{'global' if share_y_axis else 'local'}_ylim"
    out_dir = BASE_OUTPUT_DIR / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating {out_dir_name} ---")
    
    if average_by == "session":
        group_cols = ['session_id', 'genotype', 'drug', 'brain_reg', 'syllable', 'is_flipped']
        agg_df = peth_meta_df.groupby(group_cols).agg({'trace': lambda x: np.nanmean(np.vstack(x), axis=0), 'time_axis': 'first'}).reset_index()
    else:
        agg_df = peth_meta_df.copy()

    print(f"[DEBUG] Plotting Data Summary ({average_by}):")
    print(agg_df['is_flipped'].value_counts())

    final_long_records = []
    
    for _, row in agg_df.iterrows():
        n_points = len(row['trace'])
        times = row['time_axis']
        traces = row['trace']
        
        base_dict = {
            'genotype': row['genotype'],
            'drug': row['drug'],
            'brain_reg': row['brain_reg'],
            'syllable': row['syllable'],
            'is_flipped': row['is_flipped'],
            'session_id': row.get('session_id', 'UNK')
        }
        
        for i in range(n_points):
            rec = base_dict.copy()
            rec['time'] = times[i]
            rec['zdFF'] = traces[i]
            final_long_records.append(rec)

    plot_df = pd.DataFrame(final_long_records)

    # Set up styling
    sns.set(style="whitegrid", context="talk", font_scale=0.8)
    flip_palette = {"Flipped Session": "blue", "Non-Flipped Session": "green"}
    
    regions = plot_df['brain_reg'].unique()
    for reg in regions:
        if reg == 'UNK': continue
        
        # NOTE: VS data is mixed parity. Flipped vs Non-Flipped representation is not cleanly biological.
        # However, for pure visualization against DLS/DMS, we will plot it anyway to see the washout.
        
        reg_dir = out_dir / f"Region_{reg}"
        reg_dir.mkdir(parents=True, exist_ok=True)
        
        reg_df = plot_df[plot_df['brain_reg'] == reg]
        unique_sylls = reg_df['syllable'].unique()
        
        for syll in tqdm(unique_sylls, desc=f"Syllables in {reg} ({out_dir_name})"):
            syll_df = reg_df[reg_df['syllable'] == syll]
            syll_name = TARGET_SYLLABLES_OPT[syll]
            
            row_order = ["WT", "KI"]
            col_order = ["Saline", "Amph-Pre", "Amph-Post"]
            
            g = sns.relplot(
                data=syll_df,
                x="time", y="zdFF", hue="is_flipped", style="is_flipped",
                row="genotype", col="drug", row_order=row_order, col_order=col_order,
                kind="line", errorbar="se",
                height=3.5, aspect=1.2, palette=flip_palette,
                facet_kws={'sharey': share_y_axis}
            )
            
            avg_type = "Sessions Averaged" if average_by == "session" else "All Events Averaged"
            y_axis_type = "Shared Y-Axis" if share_y_axis else "Local Y-Axis"
            g.fig.suptitle(f"Parity Confirmation ({avg_type} | {y_axis_type})\nSyllable {syll}: {syll_name} | {reg}", y=1.08)
            g.set_axis_labels("Time from Onset (s)", "Z-Scored dF/F (Baseline Subtracted)")
            
            for ax in g.axes.flat:
                title_text = ax.get_title()
                # relplot titles are like 'genotype = WT | drug = Saline'
                geno_val = "WT" if "genotype = WT" in title_text else ("KI" if "genotype = KI" in title_text else "UNK")
                drug_val = "Saline" if "drug = Saline" in title_text else ("Amph-Pre" if "drug = Amph-Pre" in title_text else ("Amph-Post" if "drug = Amph-Post" in title_text else "UNK"))
                
                # Filter agg_df for counts
                sub_agg = agg_df[
                    (agg_df['syllable'] == syll) & 
                    (agg_df['genotype'] == geno_val) & 
                    (agg_df['drug'] == drug_val) & 
                    (agg_df['brain_reg'] == reg)
                ]
                
                n_flip = len(sub_agg[sub_agg['is_flipped'] == 'Flipped Session'])
                n_non = len(sub_agg[sub_agg['is_flipped'] == 'Non-Flipped Session'])
                
                ax.set_title(f"{geno_val} | {drug_val}\n(Flipped n={n_flip} | Non-n={n_non})")
                
                sub_df = syll_df[(syll_df['genotype'] == geno_val) & (syll_df['drug'] == drug_val)]
                if not sub_df.empty:
                    try:
                        mean_traces = sub_df.groupby(['is_flipped', 'time'])['zdFF'].mean().unstack(level=0)
                        if 'Flipped Session' in mean_traces.columns and 'Non-Flipped Session' in mean_traces.columns:
                            f_trace = mean_traces['Flipped Session'].values
                            nf_trace = mean_traces['Non-Flipped Session'].values
                            valid = ~(np.isnan(f_trace) | np.isnan(nf_trace))
                            if np.sum(valid) > 2:
                                r, p_val = pearsonr(f_trace[valid], nf_trace[valid])
                                textstr = f"r = {r:.2f}\np = {p_val:.3g}"
                                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                                        verticalalignment='top', bbox=props)
                    except Exception as e:
                        print(f"Error calculating correlation: {e}")
                
                ax.axvline(0, color='gray', linestyle='--', zorder=0)
                ax.axhline(0, color='gray', linewidth=0.8, zorder=0)
                
            out_path = reg_dir / f"Syllable_{syll}_optimized_parity.png"
            g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(g.fig)

def main():
    peth_df = extract_peths()
    if peth_df.empty:
        return
        
    for avg_by in ["event", "session"]:
        for share_y in [True, False]:
            process_and_plot(peth_df, avg_by, share_y)
            
    print("\nAll plot variations generated successfully.")

if __name__ == "__main__":
    main()
