import sys

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from syllable_generation.analysis.deep_research_experiments.utils.loaders import load_sessions

for proj_name in ['ifdn', 'amph']:
    print(f"\n--- {proj_name.upper()} Dataset Check ---")
    try:
        sessions = load_sessions(proj_name, apply_filter=False)
        if sessions:
            s = sessions[0]
            print(f"session_id: {s.session_id}")
            print(f"project attr: '{s.project}'")
            print(f"drug metadata: '{s.metadata.get('drug')}'")
    except Exception as e:
        print(f"Error: {e}")
