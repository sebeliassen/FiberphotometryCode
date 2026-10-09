import pickle
import re
import pandas as pd
from pathlib import Path
from . import config

def extract_signal_metadata(df: pd.DataFrame, session_metadata: dict) -> list[dict]:
    """
    Parses signal columns and metadata to return a list of defined signals.
    Adapted from analysis/replications/datta/lib/features.py to maintain independence.
    """
    sig_annot_rows = []
    
    # Identify Signal Columns
    all_sig_cols = sorted([c for c in df.columns if re.match(config.SIGNAL_COL_REGEX, c)])
    
    smeta = session_metadata.get("signal_meta", {}) or {}
    
    for c in all_sig_cols:
        match = re.match(config.SIGNAL_COL_REGEX, c)
        if not match:
            continue
            
        parts = c.split("_")
        # Format: signal_{id}_{suffix...}
        try:
            k = parts[1]
            sig_key = f"signal_{k}"
            # Suffix is everything after signal_{id}_
            # easier to grab from regex group 1 if strict, 
            # or just join the rest for safety
            suffix = "_".join(parts[2:]) 
        except IndexError:
            continue
            
        m = smeta.get(sig_key, {})
        
        brain_reg = m.get("brain_reg", "UNK")
        
        if brain_reg == "UNK":
            continue
            
        sig_annot_rows.append({
            "signal": c, 
            "signal_type": suffix,  # e.g. 'zscore' or 'zscore_datta'
            "base_signal": sig_key,
            "brain_reg": brain_reg, 
            "hemisphere": m.get("hemisphere", "UNK"),
            "fiber_num": m.get("fiber_num", "UNK"),
            "orig_letter": m.get("orig_letter", "UNK")
        })
        
    return sig_annot_rows

def load_project_sessions(project_name: str) -> list:
    """
    Loads the most recent w_sylls pickle for the specified project.
    """
    if project_name not in ["ifdn", "amph"]:
        raise ValueError(f"Unknown project: {project_name}")
        
    print(f"Searching for {project_name} sessions...")
    
    # Pattern matching based on project name
    # IFDN: jonathan_ifdn_sessions_w_sylls*.pkl
    # AMPH: jonathan_amph_sessions_w_sylls*.pkl
    pattern = f"*jonathan_{project_name}_sessions*w_sylls*.pkl"
    
    candidates = list(config.DATA_DIR.glob(pattern))
    
    if not candidates:
        raise FileNotFoundError(f"Could not find sessions pickle for {project_name} in {config.DATA_DIR}")
        
    # Get newest
    pkl_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading sessions from: {pkl_path}")
    
    with pkl_path.open("rb") as f:
        sessions = pickle.load(f)
        
    # Validation
    valid = [
        s for s in sessions
        if hasattr(s, "all_data_df") and isinstance(s.all_data_df, pd.DataFrame) and not s.all_data_df.empty
    ]
    print(f"Loaded {len(valid)} usable sessions.")
    return valid, pkl_path
