import os
import glob
import pandas as pd
import numpy as np

def analyze_csv_average_score(csv_path, num_bins=5):
    # Load the CSV file.
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        print(f"File {csv_path} missing 'frame' column; skipping.")
        return None

    # Identify all score columns (columns ending with '_score')
    score_cols = [col for col in df.columns if col.endswith('_score')]
    if not score_cols:
        print(f"No score columns found in {csv_path}")
        return None

    # Compute the average score per frame (averaging over all bodyparts)
    df['avg_score'] = df[score_cols].mean(axis=1)

    # Create time bins based on the frame column.
    min_frame, max_frame = df['frame'].min(), df['frame'].max()
    bins = np.linspace(min_frame, max_frame, num_bins + 1)
    labels = [f"Bin{i+1}" for i in range(num_bins)]
    df['time_bin'] = pd.cut(df['frame'], bins=bins, labels=labels, include_lowest=True)

    # Compute average score in each time bin.
    binned_avg = df.groupby('time_bin', observed=True)['avg_score'].mean()
    return binned_avg

def main():
    folder_path = "/home/fsp585/Desktop/jonathan_ifdn_3d_poses/anipose_csv_kpms"  # Update to your CSV folder path.
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    for csv_file in csv_files:
        print(f"\nProcessing file: {os.path.basename(csv_file)}")
        binned_scores = analyze_csv_average_score(csv_file, num_bins=5)
        if binned_scores is not None:
            print(binned_scores)

if __name__ == "__main__":
    main()
