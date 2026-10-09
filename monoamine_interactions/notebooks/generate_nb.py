

import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cells
cells = []

# Cell 1: Imports and Path Setup
cells.append(nbf.v4.new_markdown_cell("# Session Inspection Notebook\nThis notebook loads the `sessions_dirty.pkl` file and performs basic integrity checks."))

cells.append(nbf.v4.new_code_cell("""import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
project_root = "/home/fsp585/Desktop/FiberphotometryCode"
if project_root not in sys.path:
    sys.path.append(project_root)

pkl_path = "/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/data/sessions_dirty.pkl"
"""))

# Cell 2: Load Sessions
cells.append(nbf.v4.new_markdown_cell("## Load Sessions"))
cells.append(nbf.v4.new_code_cell("""with open(pkl_path, "rb") as f:
    sessions = pickle.load(f)

print(f"Loaded {len(sessions)} sessions.")
"""))

# Cell 3: Summary Table
cells.append(nbf.v4.new_markdown_cell("## Session Summary Table"))
cells.append(nbf.v4.new_code_cell("""rows = []
for s in sessions:
    md = s.metadata
    rows.append({
        "session_id": s.session_id,
        "animal_id": md.get("animal_id"),
        "genotype": md.get("genotype"),
        "chamber": md.get("chamber"),
        "has_photometry": "470" in getattr(s, 'photometry', {}),
        "n_signals": len(md.get("signal_meta", {}))
    })

df_summary = pd.DataFrame(rows)
df_summary
"""))

# Cell 4: Deep Dive into a specific session
cells.append(nbf.v4.new_markdown_cell("## Deep Dive: First Valid Session"))
cells.append(nbf.v4.new_code_cell("""# Find first session with signals
s = next((sess for sess in sessions if sess.metadata.get("signal_meta")), None)

if s:
    print(f"Inspecting Session: {s.session_id}")
    print(f"Animal ID: {s.metadata.get('animal_id')}")
    print(f"Signal Meta: {s.metadata.get('signal_meta')}")
    
    if hasattr(s, 'photometry'):
        if '470' in s.photometry:
            print("\\n470 Stream Columns:", s.photometry['470'].columns.tolist())
            print("470 Stream Shape:", s.photometry['470'].shape)
            display(s.photometry['470'].head())
        
        if '415' in s.photometry:
            print("\\n415 Stream Shape:", s.photometry['415'].shape)
else:
    print("No sessions with signals found.")
"""))

# Cell 5: Plotting signals
cells.append(nbf.v4.new_markdown_cell("## Quick Plot"))
cells.append(nbf.v4.new_code_cell("""if s and hasattr(s, 'photometry') and '470' in s.photometry:
    fig, ax = plt.subplots(figsize=(12, 4))
    # Plot first two signals if they exist
    cols = [c for c in s.photometry['470'].columns if 'signal' in c and 'zscore' not in c and 'dff' not in c]
    for c in cols[:2]:
        ax.plot(s.photometry['470'][c].values[:5000], label=c, alpha=0.7)
    ax.set_title(f"Raw 470 Signals (first 5000 frames) - {s.session_id}")
    ax.legend()
    plt.show()
"""))

nb['cells'] = cells

with open('/home/fsp585/Desktop/FiberphotometryCode/monoamine_interactions/notebooks/01_inspect_sessions.ipynb', 'w') as f:
    nbf.write(nb, f)
