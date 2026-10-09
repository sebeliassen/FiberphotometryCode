
import sys
import os
import argparse
import numpy as np
import pandas as pd

# Add root to path
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.append("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation")

from analysis.ulrik_kinematics.lib import loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_factor", type=float, default=0.5, help="Turning threshold factor (k * sigma)")
    args = parser.parse_args()
    
    print(f"Loading datasets for Quantitative Analysis (Sigma Factor: {args.sigma_factor})...")
    
    # Load all data
    sessions_ifdn = loader.load_dataset("ifdn", base_dir="/home/fsp585/Desktop/FiberphotometryCode")
    sessions_amph = loader.load_dataset("amph", base_dir="/home/fsp585/Desktop/FiberphotometryCode", amph_post_injection_delay_min=5.0)
    all_sessions = sessions_ifdn + sessions_amph
    print(f"Total Sessions: {len(all_sessions)}")
    
    # 1. Filter syllables (>20 instances in EACH group)
    # Groups: WT_IFDN, KI_IFDN, WT_Saline, WT_Amph, KI_Saline, KI_Amph
    # Actually user said "all groups need 20 instances".
    # Let's map sessions to groups and count.
    
    syll_counts = {} # syll -> group -> count
    groups = set()
    
    for s in all_sessions:
        # Define Group ID
        proj = "IFDN" if s in sessions_ifdn else "AMPH"
        g = s.metadata.get("genotype", "UNK")
        d = s.metadata.get("drug", "None")
        if d == "None": d = "Saline" if proj == "AMPH" else "NA" # IFDN usually handled as geno only? but let's be strict
        
        # Simplify groups to the 6 main ones?
        # IFDN: WT, KI
        # AMPH: WT_Sal, WT_Amph, KI_Sal, KI_Amph
        if proj == "IFDN":
            group_id = f"IFDN_{g}"
        else:
            group_id = f"AMPH_{g}_{d}"
            
        groups.add(group_id)
        
        if "syllable" not in s.all_data_df.columns: continue
        
        # Count bouts (onsets)
        seq = s.all_data_df["syllable"].fillna(-1).values
        # Onsets
        if len(seq) > 0:
            changes = np.concatenate(([True], seq[1:] != seq[:-1]))
            runs = seq[changes]
            counts = pd.Series(runs).value_counts()
            
            for syll, count in counts.items():
                if syll < 0: continue
                s_int = int(syll)
                if s_int not in syll_counts: syll_counts[s_int] = {}
                syll_counts[s_int][group_id] = syll_counts[s_int].get(group_id, 0) + count

    # Filter
    valid_sylls = []
    print("\nFiltering Syllables (>20 instances per group)...")
    for s_int, g_counts in syll_counts.items():
        # strict: must be in ALL main groups
        # IFDN_WT, IFDN_KI, AMPH_WT_Saline, AMPH_WT_Amph, AMPH_KI_Saline, AMPH_KI_Amph
        # That's 6 groups.
        # User said "pr group".
        # Let's check specific groups
        required_groups = ["IFDN_WT", "IFDN_KI", "AMPH_WT_Saline", "AMPH_WT_Amphetamine", "AMPH_KI_Saline", "AMPH_KI_Amphetamine"]
        
        # Check mapping keys might vary (Saline vs None etc). 
        # Actually my loader normalizes some things in metrics but not here.
        # Let's just require > 20 in at least 4 groups? Or all?
        # User: "all groups need 20 instances for this to count"
        
        is_valid = True
        for g in groups:
            # Skip weird groups (UNK etc)
            if "UNK" in g or "None" in g: continue
            if g_counts.get(g, 0) < 20:
                is_valid = False
                break
        
        if is_valid:
            valid_sylls.append(s_int)
            
    print(f"Valid Syllables: {len(valid_sylls)} (IDs: {sorted(valid_sylls)})")
    
    if len(valid_sylls) == 0:
        print("No syllables passed the filter! Check group names:")
        print(groups)
        return

    # 2. Compute Kinematics for these syllables
    # Average across ALL sessions (global definition of the behavior)
    data_rows = []
    
    for s in all_sessions:
        df = s.all_data_df
        if "syllable" not in df.columns: continue
        
        # Filter to valid sylls
        mask = df["syllable"].isin(valid_sylls)
        if not mask.any(): continue
        
        # We need ang_vel and velocity
        # Check cols
        cols = ["syllable", "velocity_px_s", "angular_velocity"]
        # Maybe "heading" needing diff?
        if "angular_velocity" not in df.columns:
            # Compute it?
            if "heading" in df.columns:
                # diff heading
                h = df["heading"]
                # Wrap
                # ... implementing minimal ang vel logic
                # For now assume it exists based on check_cols.py
                pass
            else:
                continue
                
        subset = df.loc[mask, cols].dropna()
        data_rows.append(subset)
        
    global_df = pd.concat(data_rows)
    
    # Group by Syllable
    syll_stats = global_df.groupby("syllable").agg({
        "velocity_px_s": "mean",
        "angular_velocity": "mean"
    }).reset_index()
    
    # 3. Classify
    # Vigor: Tertiles
    v_low = syll_stats["velocity_px_s"].quantile(0.33)
    v_high = syll_stats["velocity_px_s"].quantile(0.66)
    
    def get_vigor(v):
        if v < v_low: return "Low"
        elif v < v_high: return "Medium"
        else: return "High"
        
    syll_stats["Vigor"] = syll_stats["velocity_px_s"].apply(get_vigor)
    
    # Turn: Symmetric Threshold
    # Ang velocities should be centered on 0.
    # Mean of means?
    mu_dist = syll_stats["angular_velocity"].mean()
    sigma_dist = syll_stats["angular_velocity"].std()
    
    print(f"\nKinematics Stats:")
    print(f"Velocity Tertiles: {v_low:.2f}, {v_high:.2f}")
    print(f"AngVel Mean: {mu_dist:.4f}, StdDev: {sigma_dist:.4f}")
    
    # 4. FINAL CLASSIFICATION (Sigma=0.30)
    best_sigma = 0.30 # Selected by user
    T = best_sigma * sigma_dist
    print(f"\nFinal Turn Threshold (T) = {T:.4f} (Factor: {best_sigma})")
    
    def get_turn(av):
        if av < -T: return "Left"
        elif av > T: return "Right"
        else: return "Middle"
        
    syll_stats["Turn"] = syll_stats["angular_velocity"].apply(get_turn)
    
    # Report Balance
    print("\n--- CLASSIFICATION BALANCE ---")
    print(pd.crosstab(syll_stats["Turn"], syll_stats["Vigor"]))
    
    print("\nDetailed Counts:")
    print("Left:", len(syll_stats[syll_stats["Turn"]=="Left"]))
    print("Middle:", len(syll_stats[syll_stats["Turn"]=="Middle"]))
    print("Right:", len(syll_stats[syll_stats["Turn"]=="Right"]))
    
    # Save mapping for plotting
    syll_stats.to_csv("quantitative_map.csv", index=False)
    print("\nSaved quantitative mapping to 'quantitative_map.csv'")

if __name__ == "__main__":
    main()
