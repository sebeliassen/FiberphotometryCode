import re
import sys
from pathlib import Path

# Add path to config and project root
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append("/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/jonathan_oft_baseline")
from config import JCFG

print(f"Loaded JCFG regex: {JCFG.video_filename_regex}")

sample_filename = "T1_4.8.e.e_Arena1and2_LeftTopCamVid_2026-01-20T15_07_45.avi"
print(f"\nTesting against: {sample_filename}")

pattern = re.compile(JCFG.video_filename_regex)
match = pattern.match(sample_filename)

if match:
    print("MATCH SUCCESS!")
    print(match.groupdict())
else:
    print("MATCH FAILED")

# Debug: list actual files in directory to ensure we cover them
video_dir = Path("/home/fsp585/Desktop/FiberphotometryCode/src_mono_inter/jonathan_oft_baseline/video_files")
if video_dir.exists():
    print("\nReal file check:")
    for f in list(video_dir.glob("*.avi"))[:3]:
        m = pattern.match(f.name)
        status = "MATCH" if m else "FAIL"
        print(f"{status}: {f.name}")
else:
    print("Video dir not found (for listing)")
