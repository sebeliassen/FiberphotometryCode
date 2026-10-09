# %% [markdown]
# # Signal Alignment & Epoch Audit
# This notebook provides an interactive Plotly-based view of every session to verify that our behavioral timestamps and analysis epochs accurately capture biological signals without including handling or movement artifacts.

# %%
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path("../../").resolve()
    
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from monoamine_interactions.notebooks.utils.photometry import motion_correct_biexponential

# %%
# 1. Load Data
pkl_path = project_root / "monoamine_interactions" / "data" / "sessions_jonathan_juvenile_intruder_dirty.pkl"
print(f"Loading {pkl_path}...")
with open(pkl_path, "rb") as f:
    sessions = pickle.load(f)
print(f"Loaded {len(sessions)} sessions.")

# Define epochs for shading markers
EPOCHS_CONFIG = [
    ("Exploration", "inarena_min", "home_cage_min", "rgba(100, 100, 100, 0.1)"),
    ("Anticipation", "home_cage_min", "juvenile_first_sniff_min", "rgba(255, 165, 0, 0.2)"),
    ("Interaction", "juvenile_first_sniff_min", "juvenile_removed_min", "rgba(255, 0, 0, 0.2)"),
    ("Recovery", "juvenile_removed_min", "homecage_removed_min", "rgba(0, 0, 255, 0.1)"),
]

# %%
def plot_session_audit(s):
    """Generates an interactive Plotly figure for a single session."""
    sm = s.metadata.get("signal_meta", {})
    tm = s.metadata.get("timing_meta", {})
    region_map = {k: v.get("region") for k, v in sm.items() if v.get("region")}
    regions = list(region_map.values())
    
    if '470' not in s.photometry: 
        print(f"Skipping {s.session_id}: No photometry data.")
        return None
        
    df_470 = s.photometry['470']
    df_415 = s.photometry['415']
    
    # Ensure channels are the same length (handles dropped frames at end of session)
    min_len = min(len(df_470), len(df_415))
    df_470 = df_470.iloc[:min_len]
    df_415 = df_415.iloc[:min_len]
    
    ts_array = (df_470['SystemTimestamp'] - df_470['SystemTimestamp'].iloc[0]).values / 60.0 # Minutes from start
    
    # Create subplots: One for Raw, One for Corrected dFF
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=(f"Raw Signals (470/415) - {s.session_id}", "Motion-Corrected dFF (Bi-Exponential)"))
    
    # Calculate global range for Y-axis scaling (only within phases)
    valid_mins = [tm.get(e[1]) for e in EPOCHS_CONFIG if pd.notna(tm.get(e[1]))]
    valid_maxs = [tm.get(e[2]) for e in EPOCHS_CONFIG if pd.notna(tm.get(e[2]))]
    
    if valid_mins and valid_maxs:
        t_global_start = min(valid_mins)
        t_global_end = max(valid_maxs)
        mask = (ts_array >= t_global_start) & (ts_array <= t_global_end)
    else:
        mask = slice(None) # Use all if no epochs found
    
    # Track min/max for axes
    raw_vals = []
    dff_vals = []
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    for i, (col, region) in enumerate(region_map.items()):
        color = colors[i % len(colors)]
        
        # Panel 1: Raw
        fig.add_trace(go.Scatter(x=ts_array, y=df_470[col], name=f"{region} 470", line=dict(color=color), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts_array, y=df_415[col], name=f"{region} 415 (iso)", line=dict(color=color, dash='dot'), opacity=0.4), row=1, col=1)
        raw_vals.extend(df_470[col].values[mask])
        raw_vals.extend(df_415[col].values[mask])
        
        # Panel 2: Corrected dFF
        try:
            dff = motion_correct_biexponential(df_470[col].values, df_415[col].values)
            fig.add_trace(go.Scatter(x=ts_array, y=dff, name=f"{region} dFF", line=dict(color=color)), row=2, col=1)
            dff_vals.extend(dff[mask])
        except:
            pass
            
    # Add vertical lines for ALL timestamps
    for key, val in tm.items():
        if pd.isna(val): continue
        fig.add_vline(x=val, line_dash="dash", line_color="black", opacity=0.5, annotation_text=f" {key}", annotation_position="top right")
        
    # Add Shaded Rectangles for Epochs
    for ep_name, start_key, end_key, color in EPOCHS_CONFIG:
        t0 = tm.get(start_key)
        t1 = tm.get(end_key)
        if pd.notna(t0) and pd.notna(t1):
            fig.add_vrect(x0=t0, x1=t1, fillcolor=color, layer="below", line_width=0, annotation_text=ep_name)
    
    # Set initial view focused on the valid mask data (while keeping all data available)
    if raw_vals and len(raw_vals) > 0:
        ymin, ymax = np.nanmin(raw_vals), np.nanmax(raw_vals)
        pad = (ymax - ymin) * 0.1
        fig.update_yaxes(range=[ymin - pad, ymax + pad], autorange=False, row=1, col=1)
        
    if dff_vals and len(dff_vals) > 0:
        ymin, ymax = np.nanmin(dff_vals), np.nanmax(dff_vals)
        pad = (ymax - ymin) * 0.1
        fig.update_yaxes(range=[ymin - pad, ymax + pad], autorange=False, row=2, col=1)
            
    fig.update_layout(height=800, template="plotly_white", showlegend=True, 
                      hovermode="x unified", xaxis2_title="Time (Minutes)")
    return fig

# %% [markdown]
# ## 2. Visualize Sessions
# Use the dropdown or scroll to audit each of the 12 sessions.

# %%
for s in sessions:
    fig = plot_session_audit(s)
    if fig:
        fig.show()
