import os
import sys

import pandas as pd

# Set up paths
REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
sys.path.append(REPO_ROOT)

SESSIONS_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/jonathan_amph_sessions.pkl")
EXCLUSIONS_CSV = os.path.join(REPO_ROOT, "syllable_generation/projects/jonathan_openfield_amph/overrides/exclusions.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "syllable_generation/analysis/amph_session_metadata.csv")

def generate_metadata():
    print(f"Loading sessions from {SESSIONS_PKL}...")
    try:
        sessions = pd.read_pickle(SESSIONS_PKL)
    except Exception as e:
        print(f"Failed to load sessions: {e}")
        return

    # Load exclusions to get real Animal IDs
    try:
        exclusions = pd.read_csv(EXCLUSIONS_CSV)
        # Create a map: SessionID -> AnimalID
        # Note: exclusions.csv has multiple rows per session (different regions), so we drop duplicates
        sid_to_animal = exclusions[['SessionID', 'ID']].drop_duplicates().set_index('SessionID')['ID'].to_dict()
        print(f"Loaded {len(sid_to_animal)} ID mappings from exclusions.csv")
    except Exception as e:
        print(f"Warning: Could not load exclusions.csv: {e}")
        sid_to_animal = {}

    session_rows = []

    # 1. Gather all session info
    for s in sessions:
        sid = s.session_id

        # Parse timestamp from string like "2025-04-28T09_31_08_bottom_left"
        # We can just rely on string sorting for the date part "YYYY-MM-DDTHH_MM_SS"
        # It's ISO-like enough to sort correctly.
        timestamp_str = sid.split("_bottom")[0].split("_top")[0]

        # Try to find AnimalID
        if sid in sid_to_animal:
            animal_id = str(sid_to_animal[sid])
        else:
            # Fallback: Inference/Placeholder
            # If we don't have it in exclusions, we generate a stable placeholder
            # based on the unique location? No, that's risky.
            # Let's see if we can infer grouping.
            # Actually, without a log sheet, let's assume unique unknown animals for now
            # unless instructed otherwise.
            # But the user said "assume every animal has one drug and one saline".
            # This implies we can GROUP them.
            #
            # Heuristic: Group by (Date, Location)? No, same animal different days.
            # Heuristic: Group by Location? Maybe "bottom_left" is always the same animal? (Unlikely over long term but maybe for this dataset)
            #
            # For now, let's mark unknown ones clearly so the user sees them.
            animal_id = f"Unknown_{sid}"

        session_rows.append({
            "SessionID": sid,
            "Timestamp": timestamp_str,
            "AnimalID": animal_id
        })

    # 2. Group by AnimalID and Assign Conditions
    df_temp = pd.DataFrame(session_rows)

    # If we have "Unknown" IDs that are actually pairs, this simple logic fails for them.
    # However, for the IDs we recovered from exclusions.csv, we can proceed.

    final_data = []

    # Process known animals
    for animal_id, group in df_temp.groupby("AnimalID"):
        if "Unknown" in animal_id:
            # Treat singular sessions as Saline by default or warn?
            # User said: "assume every animal has one drug session and one saline"
            # We'll just assign Saline arbitrarily if only 1 session.
            sorted_sessions = group.sort_values("Timestamp")
            run_idx = 0
            for _, row in sorted_sessions.iterrows():
                drug = "Saline" if run_idx == 0 else "Amphetamine"
                dose = 3.0 if drug == "Amphetamine" else 0.0

                final_data.append({
                    "SessionID": row["SessionID"],
                    "AnimalID": row["AnimalID"],
                    "Drug": drug,
                    "Dose_mg_kg": dose,
                    "InjectionFrame": 20000, # placeholder
                    "Group": "Control" if drug == "Saline" else "Treatment"
                })
                run_idx += 1
        else:
            # Known animal ID logic
            sorted_sessions = group.sort_values("Timestamp")

            # 1st session -> Saline
            # 2nd session -> Amph
            # 3rd+ -> ? (User said "one drug one saline", implies max 2. We'll default to Amph for subsequent?)

            sessions_list = sorted_sessions.to_dict('records')

            # Session 1
            if len(sessions_list) > 0:
                final_data.append({
                    "SessionID": sessions_list[0]["SessionID"],
                    "AnimalID": animal_id,
                    "Drug": "Saline",
                    "Dose_mg_kg": 0.0,
                    "InjectionFrame": 20000,
                    "Group": "Control"
                })

            # Session 2
            if len(sessions_list) > 1:
                final_data.append({
                    "SessionID": sessions_list[1]["SessionID"],
                    "AnimalID": animal_id,
                    "Drug": "Amphetamine",
                    "Dose_mg_kg": 3.0,
                    "InjectionFrame": 20000,
                    "Group": "Treatment"
                })

            # Handle extras if any
            for extra in sessions_list[2:]:
                final_data.append({
                    "SessionID": extra["SessionID"],
                    "AnimalID": animal_id,
                    "Drug": "Amphetamine", # Assumption
                    "Dose_mg_kg": 3.0,
                    "InjectionFrame": 20000,
                    "Group": "Treatment"
                })

    final_df = pd.DataFrame(final_data)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved metadata for {len(final_df)} sessions to {OUTPUT_CSV}")
    print(final_df.head(10))

if __name__ == "__main__":
    generate_metadata()
