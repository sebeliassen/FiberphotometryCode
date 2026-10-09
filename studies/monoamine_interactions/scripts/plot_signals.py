
import os
import pickle
import sys

import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = "/home/fsp585/Desktop/FiberphotometryCode"
if project_root not in sys.path:
    sys.path.append(project_root)

# Paths
pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/data/sessions_dirty.pkl"
output_dir = "/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/outputs"

def load_sessions(p):
    print(f"Loading sessions from {p}...")
    with open(p, "rb") as f:
        return pickle.load(f)

def plot_session_signals(s, out_dir):
    sid = s.session_id
    print(f"Processing session: {sid}")

    df = s.photometry.get("combined_filtered")
    if df is None:
        print(f"  Skipping {sid}: No combined_filtered photometry data.")
        return

    # Ignore unknown genotypes as per LLM rules
    geno = str(s.metadata.get("genotype", "")).strip().lower()
    if not geno or geno in ["unk", "unknown", "nan", "none"]:
        print(f"  Skipping {sid}: Unknown genotype ('{geno}').")
        return

    # Check for signal columns
    signals = [c for c in df.columns if c.startswith("signal_")]
    if not signals:
        print(f"  Skipping {sid}: No signal columns found.")
        return

    # Slice data 50k to 250k
    start_idx = 50000
    end_idx = 250000

    # Ensure we don't go out of bounds (though slicing handles this gracefully)
    df_slice = df.iloc[start_idx:end_idx]

    if df_slice.empty:
        print(f"  Skipping {sid}: Data slice {start_idx}-{end_idx} is empty (Total len: {len(df)}).")
        return

    plt.figure(figsize=(12, 8))

    # Plot each signal
    for sig in signals:
        # Prefer standard Z-score as requested
        if f"{sig}_zscore" in df_slice.columns:
            y_data = df_slice[f"{sig}_zscore"]
            label = f"{sig} (Z-Score)"
        elif f"{sig}_zscore_datta" in df_slice.columns:
            y_data = df_slice[f"{sig}_zscore_datta"]
            label = f"{sig} (Z-Datta)"
        else:
            y_data = df_slice[sig]
            label = f"{sig} (Raw)"

        plt.plot(df_slice.index, y_data, label=label, alpha=0.7)

    plt.title(f"Photometry Z-Scores ({start_idx}-{end_idx}) - {sid}\nAnimal: {s.metadata.get('animal_id', 'Unknown')} | Geno: {s.metadata.get('genotype', 'Unknown')}")
    plt.xlabel("Frame Index")
    plt.ylabel("Z-Score")
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{sid}_signals.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved plot to {out_path}")

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sessions = load_sessions(pkl_path)
    print(f"Loaded {len(sessions)} sessions.")

    # Limit to first 5 for now
    for i, s in enumerate(sessions[:5]):
        plot_session_signals(s, output_dir)

if __name__ == "__main__":
    main()
