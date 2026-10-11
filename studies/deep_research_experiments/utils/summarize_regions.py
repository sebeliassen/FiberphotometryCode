import sys
from collections import Counter

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from syllable_generation.analysis.deep_research_experiments.utils.loaders import load_sessions

for proj in ['ifdn', 'amph']:
    print(f"\n--- {proj.upper()} Region Summary ---")
    try:
        sessions = load_sessions(proj, apply_filter=False)
        regions = []
        for s in sessions:
            meta = s.metadata.get('signal_meta', {})
            session_regions = set()
            for k, info in meta.items():
                reg = info.get('brain_reg', 'UNK').upper()
                session_regions.add(reg)
            regions.append(tuple(sorted(list(session_regions))))

        counts = Counter(regions)
        for regs, count in counts.items():
            print(f"  {regs}: {count} sessions")

    except Exception as e:
        print(f"Error loading {proj}: {e}")
