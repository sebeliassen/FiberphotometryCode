import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr

# --- SETUP ---
this_file = Path(__file__).resolve()
repo_root = this_file.parents[3] 

# --- CONFIGURATION ---
DATA_DIR = repo_root / "syllable_generation" / "data"
BASE_OUTPUT_DIR = repo_root / "syllable_generation" / "output" / "lateralized_da"

INPUT_PKL = repo_root / "syllable_generation" / "output" / "lateralized_da" / "PETH_lateralized_da.pkl"

TARGET_SYLLABLES = {
    13: "Right Turn (45)",
    18: "Left Turn",
    20: "Look Right",
    28: "Stand Down / Look Left",
    36: "Look Left",
    41: "Groom Right",
    42: "Groom Left"
}

# Mapping syllables to their physical turn/lean direction in original video orientation
SYLLABLE_DIRECTIONS = {
    13: 'Right',
    18: 'Left',
    20: 'Right',
    28: 'Left',
    36: 'Left',
    41: 'Right',
    42: 'Left'
}

sns.set(style="whitegrid", context="talk", font_scale=0.8)

def get_base_records(peth_df):
    print("Melt to long format...")
    long_records = []
    
    for idx, row in peth_df.iterrows():
        n_points = len(row['trace'])
        syll_physical_dir = SYLLABLE_DIRECTIONS.get(row['syllable'], 'UNK')
        hemi = row['hemisphere'] # 'L' or 'R'
        
        if row['true_direction'] == 'Original':
            current_turn_dir = syll_physical_dir
        else:
            current_turn_dir = 'Right' if syll_physical_dir == 'Left' else 'Left'
            
        if current_turn_dir == 'UNK' or hemi == 'UNK':
            bio_direction = 'Unknown'
        elif current_turn_dir[0] == hemi: 
            bio_direction = 'Ipsilateral'
        else:
            bio_direction = 'Contralateral'

        raw_cond = row.get('condition', row['genotype'])
        drug_cond = raw_cond.rsplit('-', 1)[0] if '-' in raw_cond else raw_cond
        
        base_dict = {
            'session_id': row['session_id'],
            'genotype': row['genotype'],
            'drug_cond': drug_cond,
            'brain_reg': row['brain_reg'],
            'hemisphere': row['hemisphere'],
            'syllable': row['syllable'],
            'true_direction': row['true_direction'],
            'bio_direction': bio_direction,
            'syllable_name': f"Syllable {row['syllable']}: {TARGET_SYLLABLES.get(row['syllable'], '')}",
            'trace_data': row['trace'],
            'time_axis': row['time_axis']
        }
        long_records.append(base_dict)
            
    return pd.DataFrame(long_records)

def process_and_plot(peth_meta_df, average_by, share_y_axis):
    out_dir_name = f"plots_{average_by}_{'global' if share_y_axis else 'local'}_ylim"
    out_dir = BASE_OUTPUT_DIR / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating {out_dir_name} ---")
    
    if average_by == "session":
        print("Averaging traces by session...")
        group_cols = ['session_id', 'genotype', 'drug_cond', 'brain_reg', 'hemisphere', 'syllable', 'bio_direction', 'syllable_name']
        agg_df = peth_meta_df.groupby(group_cols).agg({'trace_data': lambda x: np.nanmean(np.vstack(x), axis=0), 'time_axis': 'first'}).reset_index()
    else:
        agg_df = peth_meta_df.copy()

    print(f"Melt to long format (n={len(agg_df)} base traces)...")
    final_long_records = []
    
    # Fast unpacking
    for idx, row in agg_df.iterrows():
        n_points = len(row['trace_data'])
        times = row['time_axis']
        traces = row['trace_data']
        
        base_dict = {
            'session_id': row['session_id'],
            'genotype': row['genotype'],
            'drug_cond': row['drug_cond'],
            'brain_reg': row['brain_reg'],
            'syllable': row['syllable'],
            'bio_direction': row['bio_direction'],
            'syllable_name': row['syllable_name']
        }
        
        for i in range(n_points):
            rec = base_dict.copy()
            rec['time'] = times[i]
            rec['zdFF'] = traces[i]
            final_long_records.append(rec)

    plot_df = pd.DataFrame(final_long_records)
    print(f"Expanded to {len(plot_df)} individual time bins for plotting.")

    direction_palette = {"Contralateral": "red", "Ipsilateral": "blue"}
    regions = plot_df['brain_reg'].unique()
    
    for reg in regions:
        if reg == 'UNK': continue
        reg_dir = out_dir / f"Region_{reg}"
        reg_dir.mkdir(parents=True, exist_ok=True)
        
        reg_df = plot_df[(plot_df['brain_reg'] == reg) & (plot_df['bio_direction'] != 'Unknown')]
        unique_sylls = reg_df['syllable'].unique()
        
        for syll in tqdm(unique_sylls, desc=f"Syllables in {reg} ({out_dir_name})"):
            syll_df = reg_df[reg_df['syllable'] == syll]
            syll_name = syll_df['syllable_name'].iloc[0]
            
            row_order = ["WT", "KI"]
            col_order = ["Saline", "Amph-Pre", "Amph-Post"]
            
            g = sns.relplot(
                data=syll_df,
                x="time", y="zdFF", hue="bio_direction", style="bio_direction",
                row="genotype", col="drug_cond", row_order=row_order, col_order=col_order,
                kind="line", errorbar="se",
                height=3.5, aspect=1.2, palette=direction_palette,
                facet_kws={'sharey': share_y_axis}
            )
            
            avg_type = "Sessions Averaged" if average_by == "session" else "All Events Averaged"
            y_axis_type = "Shared Y-Axis" if share_y_axis else "Local Y-Axis"
            g.fig.suptitle(f"Lateralized DA ({avg_type} | {y_axis_type}) - {syll_name} | {reg}", y=1.05)
            g.set_axis_labels("Time from Onset (s)", "Z-Scored dF/F (Baseline Subtracted)")
            
            for ax in g.axes.flat:
                title = ax.get_title()
                parts = title.split('|')
                geno_val = "WT"
                drug_val = "Saline"
                
                for p in parts:
                    if "genotype" in p: geno_val = p.split('=')[-1].strip()
                    elif "drug_cond" in p: drug_val = p.split('=')[-1].strip()
                        
                n_contra = len(agg_df[(agg_df['syllable']==syll) & (agg_df['genotype']==geno_val) & (agg_df['drug_cond']==drug_val) & (agg_df['bio_direction']=='Contralateral') & (agg_df['brain_reg']==reg)])
                n_ipsi = len(agg_df[(agg_df['syllable']==syll) & (agg_df['genotype']==geno_val) & (agg_df['drug_cond']==drug_val) & (agg_df['bio_direction']=='Ipsilateral') & (agg_df['brain_reg']==reg)])
                
                ax.set_title(f"{geno_val} | {drug_val}\n(Contra n={n_contra} | Ipsi n={n_ipsi})")
                
                sub_df = syll_df[(syll_df['genotype']==geno_val) & (syll_df['drug_cond']==drug_val)]
                if not sub_df.empty:
                    try:
                        mean_traces = sub_df.groupby(['bio_direction', 'time'])['zdFF'].mean().unstack(level=0)
                        if 'Contralateral' in mean_traces.columns and 'Ipsilateral' in mean_traces.columns:
                            c_trace = mean_traces['Contralateral'].values
                            i_trace = mean_traces['Ipsilateral'].values
                            valid = ~(np.isnan(c_trace) | np.isnan(i_trace))
                            if np.sum(valid) > 2:
                                r, p_val = pearsonr(c_trace[valid], i_trace[valid])
                                textstr = f"r = {r:.2f}\np = {p_val:.3g}"
                                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                                        verticalalignment='top', bbox=props)
                    except Exception as e:
                        print(f"Error calculating correlation: {e}")
                
                ax.axvline(0, color='gray', linestyle='--', zorder=0)
                ax.axhline(0, color='gray', linewidth=0.8, zorder=0)
                
            out_path = reg_dir / f"Syllable_{syll}_lateralized_da.png"
            g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(g.fig)

def plot_peths():
    if not INPUT_PKL.exists():
        print(f"[ERROR] Input file not found: {INPUT_PKL}")
        return

    print(f"Loading {INPUT_PKL.name}...")
    peth_df = pd.read_pickle(INPUT_PKL)
    print(f"Loaded {len(peth_df)} traces.")

    if peth_df.empty:
        print("[WARNING] The PETH dataframe is empty.")
        return

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse into base long-records ONCE
    peth_meta_df = get_base_records(peth_df)
    
    # 2. Iterate through all 4 variations
    for avg_by in ["event", "session"]:
        for share_y in [True, False]:
            process_and_plot(peth_meta_df, avg_by, share_y)

    print("\nAll plot variations generated successfully.")



if __name__ == "__main__":
    plot_peths()
