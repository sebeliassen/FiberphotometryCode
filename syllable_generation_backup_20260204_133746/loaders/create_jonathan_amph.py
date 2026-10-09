"""
Build sessions for Jonathan AMPH dataset using CUSTOM enrichment.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])) 
import pickle

# --- IMPORTS FOR CUSTOM BUILD PIPELINE ---
from jonathan.steps import tracking, video, fiberphotometry
from syllable_generation.projects.jonathan_openfield_amph import enrich as amph_enrich
from jonathan.utils.photometry import list_combined_csv_entries # Import for debugging
from jonathan.config import JCFG

# ========= constants =========
# Workspace for H5/Video (Flat structure)
WORKSPACE_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode/src_oft/jonathan_amph_workspace")
H5_DIR    = WORKSPACE_ROOT / "h5_files"
VIDEO_DIR = WORKSPACE_ROOT / "video_files"

# [FIX] Source for CSVs (Recursive structure, bypassing symlink issues)
CSV_ROOT  = Path("/mnt/hdd/jonathan_amph")

# If True, skips exclusions (manual/auto masks) and saves as '..._dirty.pkl'
FORCE_DIRTY = False
SUFFIX = "_v2"

OUTPUT_SESSIONS_PKL = Path(__file__).resolve().parent.parent / "data" / f"jonathan_amph_sessions{SUFFIX}_{'dirty' if FORCE_DIRTY else 'sanitized'}.pkl"
PROJECT_NAME = "jonathan_amph"
OVERWRITE = True

JCFG.tolerance_seconds = 10.0
# ==========================================

def _custom_enrich_pre(ctx, sessions):
    overrides_dir = Path(amph_enrich.__file__).parent / "overrides"
    ret = amph_enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=FORCE_DIRTY)
    return ret if ret is not None else sessions

def _custom_enrich_post(ctx, sessions):
    overrides_dir = Path(amph_enrich.__file__).parent / "overrides"
    ret = amph_enrich.enrich(ctx, sessions, overrides_dir=overrides_dir, skip_exclusions=FORCE_DIRTY)
    return ret if ret is not None else sessions

CUSTOM_STEPS = [
    tracking.discover_sessions,
    video.attach_video_refs,
    fiberphotometry.attach_photometry_refs,
    _custom_enrich_pre,
    fiberphotometry.realize_photometry_columns,
    _custom_enrich_post,
    fiberphotometry.split_photometry,
    fiberphotometry.compute_dff_zscore,
]

def build_sessions():
    from jonathan.context import BuildContext
    from jonathan.build import build
    
    # --- Pre-flight Check ---
    print(f"[Debug] Checking CSV discovery in: {CSV_ROOT}")
    entries = list_combined_csv_entries(CSV_ROOT, JCFG)
    print(f"[Debug] Found {len(entries)} photometry CSVs.")
    if len(entries) == 0:
        raise RuntimeError("No CSVs found! Check the path.")
    # ------------------------

    print(f"[sessions] Building sessions for project: {PROJECT_NAME}")
    
    ctx = BuildContext(
        root=WORKSPACE_ROOT, # This is a fallback root
        h5_dir=H5_DIR,
        video_dir=VIDEO_DIR,
        csv_root=CSV_ROOT,   # Using the HDD path
        project=PROJECT_NAME,
    )

    sessions = build(ctx, steps=CUSTOM_STEPS)

    with OUTPUT_SESSIONS_PKL.open("wb") as f:
        pickle.dump(sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"[sessions] Saved {len(sessions)} sessions -> {OUTPUT_SESSIONS_PKL.resolve()}")
    
    if sessions:
        s = sessions[0]
        print(f"\n--- Session 0 Check ---")
        print(f"ID: {s.session_id}")
        # If this prints an ID (e.g., '1694'), success!
        print(f"Animal: {s.metadata.get('animal_id', 'N/A')}")
        print(f"Genotype: {s.metadata.get('genotype', 'N/A')}")

if __name__ == "__main__":
    build_sessions()