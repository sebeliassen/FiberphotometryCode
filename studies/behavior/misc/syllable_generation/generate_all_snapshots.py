import os
import pickle
import subprocess
import sys
from pathlib import Path

# Add current directory to path so pickle can find syllable_generation module
sys.path.append(os.getcwd())

def main():
    pkl_path = "data/jonathan_amph_sessions_optimized_w_sylls.pkl"
    output_dir = "data/optimized_snapshots_output"

    # Needs to run in kpms_gpu environment
    env_name = "kpms_gpu"

    print(f"Loading sessions from {pkl_path}")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    num_sessions = len(sessions)
    print(f"Total sessions to process: {num_sessions}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_sessions):
        print(f"\n[{i+1}/{num_sessions}] Generating snapshots for session {i}...")
        cmd = [
            "conda", "run", "-n", env_name,
            "python", "videogen/main.py",
            "--pkl", pkl_path,
            "--output", str(out_path),
            "--session", str(i),
            "--kinematic-proof",
            "--snapshots-only",
            "--test-syllable-mode",
            "--pdf-dir", "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/optimized_flip_project/optimized_flip_activity_k100k/trajectory_plots"
        ]

        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error processing session {i}:")
            print(result.stderr)
        else:
            print(f"Session {i} successful.")
            # Print a few lines of stdout just to confirm
            for line in result.stdout.split('\n')[-5:]:
                if line.strip():
                    print(f"  {line.strip()}")

    print("\nSnapshots generation complete!")

if __name__ == "__main__":
    main()
