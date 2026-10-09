import pickle
import sys
import re
from pathlib import Path
import logging

logger = logging.getLogger("Panel1z.Loader")

def load_sessions(project="ifdn", pkl_path=None):
    """
    Load sessions from the most recent .pkl file for the given project.
    Applies metadata filtering and region name standardization.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    loaders_dir = repo_root / "data"
    
    # 1. Locate Pickle
    if pkl_path:
        pkl_file = Path(pkl_path)
    else:
        # Auto-discovery logic matching existing patterns
        pattern = f"*jonathan_{project}_sessions*w_sylls*.pkl"
        candidates = list(loaders_dir.glob(pattern))
        
        if not candidates:
            # Fallback to generic
            candidates = list(loaders_dir.glob("*sessions*all_data*.pkl"))
            
        if not candidates:
             raise FileNotFoundError(f"No pickle files found in {loaders_dir} matching {pattern}")
             
        # Pick most recent
        pkl_file = max(candidates, key=lambda p: p.stat().st_mtime)
        
    logger.info(f"📂 Loading sessions from: {pkl_file.name}")
    
    if not pkl_file.exists():
        raise FileNotFoundError(f"Pickle file does not exist: {pkl_file}")
        
    # 2. Load
    with pkl_file.open("rb") as f:
        sessions = pickle.load(f)
        
    logger.info(f"   Raw Load: {len(sessions)} sessions")
    
    # 3. Filter (Must have all_data_df)
    valid_sessions = []
    for s in sessions:
        if hasattr(s, "all_data_df") and not s.all_data_df.empty:
            valid_sessions.append(s)
            
    logger.info(f"   Valid Sessions: {len(valid_sessions)}")
    
    # 4. Patch Metadata (Region Names)
    # Copied from messy_north_star.py logic
    patch_count = 0
    for s in valid_sessions:
        sig_meta = s.metadata.get("signal_meta", {})
        if not sig_meta: continue
        
        updated = False
        for key, info in sig_meta.items():
            raw_reg = str(info.get("brain_reg", "UNK"))
            hemi = str(info.get("hemisphere", "UNK")).strip().upper()
            
            # 4a. Clean Base Name (Remove existing L/R prefixes)
            base_reg = re.sub(r"^[LRlr][\s\-_]+", "", raw_reg).strip().upper()
            
            if base_reg in ["UNK", "", "NONE", "NAN"]:
                info["brain_reg"] = "UNK"
                continue
                
            # 4b. Bake Hemisphere into Region Name
            if hemi in ["L", "R", "LEFT", "RIGHT"]:
                short_hemi = "L" if hemi.startswith("L") else "R"
                new_reg_name = f"{short_hemi}-{base_reg}"
                
                info["brain_reg"] = new_reg_name
                info["hemisphere"] = short_hemi
                updated = True
            else:
                info["brain_reg"] = base_reg
                
        if updated: patch_count += 1
        
    logger.info(f"   Metadata Patch: Region names standardized for {patch_count} sessions.")
            
    return valid_sessions
