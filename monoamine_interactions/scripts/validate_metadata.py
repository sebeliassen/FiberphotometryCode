import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def validate():
    print("=== Metadata Validation ===")

    # Paths
    proj_dir = Path(__file__).resolve().parents[1] / "jonathan_oft_baseline"
    data_dir = Path("/home/fsp585/Desktop/FiberphotometryCode/src_mono_inter/jonathan_oft_baseline/Jonathan_IFxDN_csvs")
    animals_csv = proj_dir / "overrides" / "animals.csv"

    if not animals_csv.exists():
        print(f"ERROR: animals.csv not found at {animals_csv}")
        return

    # 1. Load animals
    df = pd.read_csv(animals_csv, dtype=str).fillna("")
    assigned_ids = set(df['ID'].unique())
    print(f"Loaded {len(assigned_ids)} animal IDs from animals.csv")

    # 2. Discover actual sessions from CSVs
    # Format: BotCam1_AllTimestampsCombined2026-01-20T15_07_45.csv
    # The session ID logic usually extracts the timestamp
    csvs = list(data_dir.glob("photometry_data_combined_*.csv"))
    print(f"Found {len(csvs)} photometry CSVs in data folder.")

    # 3. Check for 'ManualTimestamp_F{ID}' files to see which animals were used
    manual_ts_files = list(data_dir.glob("ManualTimestamp_F*.csv"))
    found_ids = set()
    for f in manual_ts_files:
        # Extract ID from ManualTimestamp_F102026-01-20T15_07_45.csv
        # This is a bit tricky due to timestamp concatenation, but we just need to see if it's in our list
        name = f.stem.replace("ManualTimestamp_F", "")
        # Try to match against assigned IDs
        for aid in assigned_ids:
            if name.startswith(aid):
                found_ids.add(aid)

    print(f"Detected {len(found_ids)} unique Animal IDs in session files.")

    missing = assigned_ids - found_ids
    extra = found_ids - assigned_ids

    if missing:
        print(f"WARNING: IDs in animals.csv but NOT found in data: {missing}")
    if extra:
        print(f"WARNING: IDs in data but NOT found in animals.csv: {extra}")

    if not missing and not extra:
        print("SUCCESS: Metadata matches detected data IDs.")

if __name__ == "__main__":
    validate()
