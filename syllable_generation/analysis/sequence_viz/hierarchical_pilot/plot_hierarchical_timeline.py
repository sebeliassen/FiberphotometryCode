#!/usr/bin/env python3
"""
plot_hierarchical_timeline.py
=============================================================================
Hierarchical Timeline Visualizer: Three Aligned Horizontal Tracks per Session.
Features:
- TOP: Upper Macro-State (U1, U2, U3) in subdued palette
- MIDDLE: Routine Occurrences (R1..R6) with thin dividing boundaries at EVERY
  completed occurrence (including consecutive repetitions of the same routine ID)
  and hatch markings on boundary-truncated occurrences
- BOTTOM: Observed KPMS Syllable Bouts (syll_000..syll_099)
- Physical elapsed seconds on the x-axis (preserving actual bout durations)
- Outputs:
  1. Whole-session overview figure (fig_hierarchical_timeline_overview.png)
  2. 60-second readable detail figure (fig_hierarchical_timeline_detail_60s.png)
  3. Interactive HTML timeline with tooltips (hierarchical_timeline_interactive.html)
=============================================================================
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.hierarchical_pilot.run_hierarchical_feasibility_experiment import (
    load_cohort_sessions, segment_session_organization
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
BUNDLE_PKL = OUTPUT_DIR / "hierarchical_model_results_bundle_repaired.pkl"

# Color Palettes
ROUTINE_COLORS = {
    0: "#E41A1C", # R1: Vivid Crimson
    1: "#377EB8", # R2: Sapphire Blue
    2: "#4DAF4A", # R3: Emerald Green
    3: "#984EA3", # R4: Royal Amethyst
    4: "#FF7F00", # R5: Amber Orange
    5: "#00A896"  # R6: Deep Teal
}

ROUTINE_LABELS = {
    0: "R1 (Dwell: 6.6b / 3.7s)",
    1: "R2 (Dwell: 3.8b / 1.5s)",
    2: "R3 (Dwell: 17.9b / 7.5s)",
    3: "R4 (Dwell: 4.3b / 2.5s)",
    4: "R5 (Dwell: 3.2b / 1.6s)",
    5: "R6 (Dwell: 11.3b / 4.7s)"
}

UPPER_COLORS = {
    0: "#6C8EBF", # U1: Soft Slate Blue (Active)
    1: "#82B366", # U2: Soft Muted Sage (Investigative)
    2: "#D6B656"  # U3: Soft Warm Ochre (Quiescent)
}

UPPER_LABELS = {
    0: "U1 (Active Locomotion Context)",
    1: "U2 (Local Investigation Context)",
    2: "U3 (Quiescent / Slow Dwell Context)"
}

# 100-syllable consistent palette
rng = np.random.default_rng(12345)
# Generate 100 pleasing, distinct colors using HSV with high saturation/value
hues = np.linspace(0, 1, 100, endpoint=False)
rng.shuffle(hues)
SYLLABLE_COLORS = {}
for s in range(100):
    rgb = mcolors.hsv_to_rgb([hues[s], 0.75, 0.85])
    SYLLABLE_COLORS[s] = mcolors.to_hex(rgb)


def extract_session_tracks(session_info, model):
    """
    Extracts second-accurate time intervals for all 3 tracks in a session:
    - Upper state visits
    - Routine occurrences (complete & truncated)
    - Routine internal stages (s=0, 1, 2)
    - Syllable bouts
    """
    sylls = session_info["syllables"]
    durs = session_info["durations"]
    aid = session_info["animal_id"]
    gt = session_info["genotype"]
    s_idx = session_info["session_idx"]
    drug = session_info["drug"]

    # Elapsed times
    T = len(sylls)
    offsets = np.cumsum(durs)
    onsets = np.r_[0.0, offsets[:-1]]

    # Decode
    m_seq, r_seq, st_seq = model.decode_session_viterbi(sylls)
    df_rout, df_rep, df_macro, df_gaps = segment_session_organization(
        m_seq, r_seq, st_seq, durs, aid, gt, s_idx
    )

    # 1. Syllable bouts track
    syll_blocks = []
    for t in range(T):
        syll_blocks.append({
            "syllable": int(sylls[t]),
            "start_sec": float(onsets[t]),
            "end_sec": float(offsets[t]),
            "duration_sec": float(durs[t]),
            "stage": int(st_seq[t]),
            "routine": int(r_seq[t]),
            "macro": int(m_seq[t])
        })

    # 2. Routine occurrences track
    rout_blocks = []
    for _, row in df_rout.iterrows():
        b_st = int(row["start_t"])
        b_en = int(row["end_t"])
        rout_blocks.append({
            "routine_id": int(row["routine_id"]),
            "macro_state": int(row["macro_state"]),
            "start_sec": float(onsets[b_st]),
            "end_sec": float(offsets[b_en - 1]),
            "bouts": int(row["bouts"]),
            "duration_sec": float(row["duration_sec"]),
            "truncated": bool(row["truncated"]),
            "start_bout": b_st,
            "end_bout": b_en
        })

    # 3. Upper state visits track
    macro_blocks = []
    for _, row in df_macro.iterrows():
        b_st = int(row["start_t"])
        b_en = int(row["end_t"])
        macro_blocks.append({
            "macro_state": int(row["macro_state"]),
            "start_sec": float(onsets[b_st]),
            "end_sec": float(offsets[b_en - 1]),
            "bouts": int(row["bouts"]),
            "duration_sec": float(row["duration_sec"]),
            "routine_occurrences": int(row["routine_occurrences"]),
            "truncated": bool(row["truncated"])
        })

    return {
        "session_idx": s_idx,
        "animal_id": aid,
        "genotype": gt,
        "drug": drug,
        "total_sec": float(offsets[-1]),
        "total_bouts": T,
        "syll_blocks": syll_blocks,
        "rout_blocks": rout_blocks,
        "macro_blocks": macro_blocks
    }


def draw_hierarchical_tracks(
    ax,
    session_data,
    time_window=None,
    show_routine_labels=True,
    show_stages=False,
    y_offset=0.0
):
    """
    Draws the three aligned horizontal tracks on ax:
    TOP (y in [2.2, 3.0]): Upper State
    MIDDLE (y in [1.1, 1.9]): Routine Occurrences
    BOTTOM (y in [0.0, 0.8]): Syllable Bouts
    """
    t_min = time_window[0] if time_window else 0.0
    t_max = time_window[1] if time_window else session_data["total_sec"]

    # 1. TOP TRACK: Upper State Visits
    for block in session_data["macro_blocks"]:
        st = block["start_sec"]
        en = block["end_sec"]
        if en < t_min or st > t_max:
            continue
        c_st = max(st, t_min)
        c_en = min(en, t_max)
        width = c_en - c_st
        m_id = block["macro_state"]
        color = UPPER_COLORS.get(m_id, "#AAAAAA")

        # Rectangle
        rect = patches.Rectangle(
            (c_st, y_offset + 2.15), width, 0.85,
            facecolor=color, edgecolor="black", linewidth=0.75,
            hatch="///" if block["truncated"] else None
        )
        ax.add_patch(rect)

        # Label if wide enough
        if width > (t_max - t_min) * 0.05:
            ax.text(
                c_st + width / 2.0, y_offset + 2.57, f"U{m_id+1}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="#111111"
            )

    # 2. MIDDLE TRACK: Routine Occurrences
    for block in session_data["rout_blocks"]:
        st = block["start_sec"]
        en = block["end_sec"]
        if en < t_min or st > t_max:
            continue
        c_st = max(st, t_min)
        c_en = min(en, t_max)
        width = c_en - c_st
        r_id = block["routine_id"]
        color = ROUTINE_COLORS.get(r_id, "#777777")

        # Routine rectangle with thin black boundary
        rect = patches.Rectangle(
            (c_st, y_offset + 1.05), width, 0.90,
            facecolor=color, edgecolor="black", linewidth=1.0,
            hatch="///" if block["truncated"] else None
        )
        ax.add_patch(rect)

        # Routine occurrence label
        if show_routine_labels and width > (t_max - t_min) * 0.025:
            ax.text(
                c_st + width / 2.0, y_offset + 1.50, f"R{r_id+1}",
                ha="center", va="center", fontsize=8, fontweight="bold", color="white"
            )

    # 3. BOTTOM TRACK: KPMS Syllable Bouts
    for block in session_data["syll_blocks"]:
        st = block["start_sec"]
        en = block["end_sec"]
        if en < t_min or st > t_max:
            continue
        c_st = max(st, t_min)
        c_en = min(en, t_max)
        width = c_en - c_st
        s_id = block["syllable"]
        color = SYLLABLE_COLORS.get(s_id, "#CCCCCC")

        rect = patches.Rectangle(
            (c_st, y_offset + 0.0), width, 0.85,
            facecolor=color, edgecolor=(0.0, 0.0, 0.0, 0.3), linewidth=0.3
        )
        ax.add_patch(rect)

        # Label in detail view if wide enough
        if (t_max - t_min) <= 120 and width > (t_max - t_min) * 0.02:
            ax.text(
                c_st + width / 2.0, y_offset + 0.42, f"{s_id}",
                ha="center", va="center", fontsize=7, color="#222222"
            )


def plot_overview_comparison(sessions_data, time_window=(0, 600)):
    """
    Generates whole-session overview figure comparing 4 exemplar sessions
    (2 WT Saline, 2 KI Saline) across the first 10 minutes (600 seconds).
    """
    n_sess = len(sessions_data)
    fig, axes = plt.subplots(n_sess, 1, figsize=(20, 3.5 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    t_min, t_max = time_window

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_hierarchical_tracks(ax, s_data, time_window=time_window, show_routine_labels=False)

        # Labels and margin metadata
        aid = s_data["animal_id"]
        gt = s_data["genotype"]
        drug = s_data["drug"]
        s_idx = s_data["session_idx"]

        margin_text = f"Session {s_idx}\nAnimal {aid}\n({gt}, {drug})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=75, va="center")

        # Track category ticks on y
        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables", "Routines", "Upper State"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(t_min, t_max)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")

        if i == 0:
            ax.set_title("A. Whole-Session Overview (Three Aligned Horizontal Tracks per Recording)\nElapsed Physical Time (0 to 600 Seconds)", fontsize=12, fontweight="bold", pad=15)

    axes[-1].set_xlabel("Elapsed Time (Seconds)", fontsize=11, fontweight="bold")

    # Legends outside the plot
    # Upper state legend
    upper_patches = [patches.Patch(facecolor=UPPER_COLORS[u], edgecolor="black", label=UPPER_LABELS[u]) for u in range(3)]
    # Routine legend
    routine_patches = [patches.Patch(facecolor=ROUTINE_COLORS[r], edgecolor="black", label=f"R{r+1}: {ROUTINE_LABELS[r]}") for r in range(6)]
    # Hatch patch for truncation
    trunc_patch = patches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Boundary Truncated")

    leg1 = fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.30, 0.98), ncol=3, frameon=True, fontsize=9, title="Upper Macro-States")
    leg2 = fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=3, frameon=True, fontsize=9, title="Shared Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.05, 0.03, 0.98, 0.93])
    out_path = OUTPUT_DIR / "fig_hierarchical_timeline_overview.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved whole-session overview timeline to: {out_path.name}")


def plot_detail_60s_window(sessions_data, time_window=(100, 160)):
    """
    Generates 60-second readable detail figure with individual syllables,
    routine text labels, and vertical boundaries between consecutive occurrences.
    """
    n_sess = len(sessions_data)
    fig, axes = plt.subplots(n_sess, 1, figsize=(20, 3.8 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    t_min, t_max = time_window

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_hierarchical_tracks(ax, s_data, time_window=time_window, show_routine_labels=True)

        aid = s_data["animal_id"]
        gt = s_data["genotype"]
        drug = s_data["drug"]
        s_idx = s_data["session_idx"]

        margin_text = f"Session {s_idx}\nAnimal {aid}\n({gt}, {drug})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=75, va="center")

        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables\n(KPMS Bouts)", "Shared\nRoutines", "Upper Context\n(Macro-State)"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(t_min, t_max)
        ax.grid(True, axis="x", alpha=0.4, linestyle="--")

        if i == 0:
            ax.set_title(f"B. Selectable 60-Second Detail View ({t_min:.0f}s to {t_max:.0f}s)\nAligned Boundaries: Syllable Constituent Bouts -> Routine Occurrences -> Upper Regimes", fontsize=12, fontweight="bold", pad=15)

    axes[-1].set_xlabel("Elapsed Time in Recording (Seconds)", fontsize=11, fontweight="bold")

    upper_patches = [patches.Patch(facecolor=UPPER_COLORS[u], edgecolor="black", label=UPPER_LABELS[u]) for u in range(3)]
    routine_patches = [patches.Patch(facecolor=ROUTINE_COLORS[r], edgecolor="black", label=f"R{r+1}: {ROUTINE_LABELS[r]}") for r in range(6)]
    trunc_patch = patches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Boundary Truncated")

    fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.30, 0.98), ncol=3, frameon=True, fontsize=9, title="Upper Macro-States")
    fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=3, frameon=True, fontsize=9, title="Shared Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.05, 0.03, 0.98, 0.93])
    out_path = OUTPUT_DIR / "fig_hierarchical_timeline_detail_60s.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved 60-second detail timeline to: {out_path.name}")


def generate_interactive_html(sessions_data):
    """
    Generates a standalone, rich interactive HTML5 visualization
    with canvas rendering, zooming, panning, time slider, preset buttons
    (60s, 120s, 300s, Full), and interactive hover tooltips showing
    Track Name, State ID, Onset (s), End (s), Duration (s), and Bouts.
    """
    # Prepare serializable JSON payload for top 4 exemplar sessions
    json_sessions = []
    for s_data in sessions_data:
        json_sessions.append({
            "session_idx": s_data["session_idx"],
            "animal_id": s_data["animal_id"],
            "genotype": s_data["genotype"],
            "drug": s_data["drug"],
            "total_sec": round(s_data["total_sec"], 2),
            "total_bouts": s_data["total_bouts"],
            "macro_blocks": s_data["macro_blocks"],
            "rout_blocks": s_data["rout_blocks"],
            "syll_blocks": s_data["syll_blocks"]
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hierarchical Timeline Visualizer: Three Aligned Tracks</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background: #0f172a;
    color: #f8fafc;
  }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #38bdf8; }}
  p.subtitle {{ margin: 0 0 20px 0; font-size: 14px; color: #94a3b8; }}
  .toolbar {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: #1e293b;
    padding: 12px 18px;
    border-radius: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }}
  .btn {{
    background: #3b82f6;
    color: white;
    border: none;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }}
  .btn:hover {{ background: #2563eb; }}
  .btn-secondary {{ background: #475569; }}
  .btn-secondary:hover {{ background: #334155; }}
  .legend-container {{
    display: flex;
    gap: 24px;
    background: #1e293b;
    padding: 12px 18px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 12px;
    flex-wrap: wrap;
  }}
  .legend-group {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .legend-title {{ font-weight: bold; color: #cbd5e1; margin-right: 4px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-color {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.5); }}
  #canvas-container {{
    position: relative;
    background: #1e293b;
    border-radius: 8px;
    padding: 15px;
    overflow-x: auto;
  }}
  canvas {{ display: block; }}
  #tooltip {{
    position: absolute;
    display: none;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #38bdf8;
    color: #f8fafc;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    z-index: 100;
    line-height: 1.5;
  }}
</style>
</head>
<body>

<h1>Hierarchical Timeline Visualizer: Three Aligned Tracks</h1>
<p class="subtitle">Upper Macro-State &rarr; Shared Behavioral Subroutine &rarr; KPMS Syllable Bouts on Physical Elapsed Seconds</p>

<div class="toolbar">
  <span style="font-weight:600; font-size:13px; color:#cbd5e1;">Time Window Preset:</span>
  <button class="btn" onclick="setWindow(0, 60)">0s – 60s (Detail)</button>
  <button class="btn" onclick="setWindow(60, 120)">60s – 120s</button>
  <button class="btn" onclick="setWindow(100, 160)">100s – 160s (Recommended Exemplar)</button>
  <button class="btn" onclick="setWindow(0, 300)">0s – 300s (5 Min)</button>
  <button class="btn btn-secondary" onclick="setWindow(0, maxTime)">Full Overview (600s)</button>
  <div style="flex:1;"></div>
  <button class="btn btn-secondary" onclick="zoomIn()">Zoom In (+)</button>
  <button class="btn btn-secondary" onclick="zoomOut()">Zoom Out (&minus;)</button>
  <button class="btn btn-secondary" onclick="panLeft()">&larr; Pan</button>
  <button class="btn btn-secondary" onclick="panRight()">Pan &rarr;</button>
</div>

<div class="legend-container">
  <div class="legend-group">
    <span class="legend-title">Upper States:</span>
    <div class="legend-item"><div class="legend-color" style="background:#6C8EBF;"></div> U1: Active Locomotion</div>
    <div class="legend-item"><div class="legend-color" style="background:#82B366;"></div> U2: Local Investigation</div>
    <div class="legend-item"><div class="legend-color" style="background:#D6B656;"></div> U3: Quiescent / Rest</div>
  </div>
  <div class="legend-group">
    <span class="legend-title">Shared Routines:</span>
    <div class="legend-item"><div class="legend-color" style="background:#E41A1C;"></div> R1 (6.6b)</div>
    <div class="legend-item"><div class="legend-color" style="background:#377EB8;"></div> R2 (3.8b)</div>
    <div class="legend-item"><div class="legend-color" style="background:#4DAF4A;"></div> R3 (17.9b)</div>
    <div class="legend-item"><div class="legend-color" style="background:#984EA3;"></div> R4 (4.3b)</div>
    <div class="legend-item"><div class="legend-color" style="background:#FF7F00;"></div> R5 (3.2b)</div>
    <div class="legend-item"><div class="legend-color" style="background:#00A896;"></div> R6 (11.3b)</div>
  </div>
</div>

<div id="canvas-container">
  <canvas id="timelineCanvas" width="1600" height="950"></canvas>
  <div id="tooltip"></div>
</div>

<script>
const SESSIONS = {json.dumps(json_sessions)};
const ROUTINE_COLORS = {json.dumps(ROUTINE_COLORS)};
const UPPER_COLORS = {json.dumps(UPPER_COLORS)};
const SYLLABLE_COLORS = {json.dumps(SYLLABLE_COLORS)};

let viewMin = 100.0;
let viewMax = 160.0;
const maxTime = 600.0;

const canvas = document.getElementById("timelineCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

const MARGIN_LEFT = 180;
const MARGIN_RIGHT = 40;
const MARGIN_TOP = 40;
const TRACK_GROUP_HEIGHT = 200;
const TRACK_H = 45;
const TRACK_GAP = 12;

function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const plotW = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;

  function timeToX(t) {{
    return MARGIN_LEFT + ((t - viewMin) / (viewMax - viewMin)) * plotW;
  }}

  // Draw time grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";

  const tSpan = viewMax - viewMin;
  const step = tSpan <= 70 ? 5 : (tSpan <= 150 ? 10 : (tSpan <= 350 ? 30 : 60));
  const firstTick = Math.ceil(viewMin / step) * step;

  for (let t = firstTick; t <= viewMax; t += step) {{
    const x = timeToX(t);
    ctx.beginPath();
    ctx.moveTo(x, MARGIN_TOP - 15);
    ctx.lineTo(x, canvas.height - 30);
    ctx.stroke();
    ctx.fillText(t + "s", x, canvas.height - 15);
  }}

  // Draw each session group
  SESSIONS.forEach((sess, sIdx) => {{
    const groupY = MARGIN_TOP + sIdx * TRACK_GROUP_HEIGHT;

    // Session left margin
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 13px -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`Session ${{sess.session_idx}}`, MARGIN_LEFT - 15, groupY + 50);
    ctx.font = "12px -apple-system, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`Mouse ${{sess.animal_id}} (${{sess.genotype}})`, MARGIN_LEFT - 15, groupY + 70);
    ctx.fillText(`${{sess.drug}}`, MARGIN_LEFT - 15, groupY + 88);

    // Track labels
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.fillText("Upper State", MARGIN_LEFT - 15, groupY + 28);
    ctx.fillText("Routine", MARGIN_LEFT - 15, groupY + 28 + TRACK_H + TRACK_GAP);
    ctx.fillText("Syllables", MARGIN_LEFT - 15, groupY + 28 + (TRACK_H + TRACK_GAP) * 2);

    // Track 1: Upper State
    const yMacro = groupY + 10;
    sess.macro_blocks.forEach(b => {{
      if (b.end_sec < viewMin || b.start_sec > viewMax) return;
      const x1 = Math.max(timeToX(b.start_sec), MARGIN_LEFT);
      const x2 = Math.min(timeToX(b.end_sec), canvas.width - MARGIN_RIGHT);
      const w = Math.max(1, x2 - x1);

      ctx.fillStyle = UPPER_COLORS[b.macro_state] || "#888";
      ctx.fillRect(x1, yMacro, w, TRACK_H);
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, yMacro, w, TRACK_H);

      if (w > 35) {{
        ctx.fillStyle = "#111";
        ctx.font = "bold 11px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`U${{b.macro_state + 1}}`, x1 + w / 2, yMacro + 27);
      }}
    }});

    // Track 2: Routines
    const yRout = yMacro + TRACK_H + TRACK_GAP;
    sess.rout_blocks.forEach(b => {{
      if (b.end_sec < viewMin || b.start_sec > viewMax) return;
      const x1 = Math.max(timeToX(b.start_sec), MARGIN_LEFT);
      const x2 = Math.min(timeToX(b.end_sec), canvas.width - MARGIN_RIGHT);
      const w = Math.max(1, x2 - x1);

      ctx.fillStyle = ROUTINE_COLORS[b.routine_id] || "#888";
      ctx.fillRect(x1, yRout, w, TRACK_H);
      // Thin black boundary for every routine completion (even consecutive repetitions!)
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(x1, yRout, w, TRACK_H);

      if (w > 20) {{
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`R${{b.routine_id + 1}}`, x1 + w / 2, yRout + 27);
      }}
    }});

    // Track 3: Syllables
    const ySyll = yRout + TRACK_H + TRACK_GAP;
    sess.syll_blocks.forEach(b => {{
      if (b.end_sec < viewMin || b.start_sec > viewMax) return;
      const x1 = Math.max(timeToX(b.start_sec), MARGIN_LEFT);
      const x2 = Math.min(timeToX(b.end_sec), canvas.width - MARGIN_RIGHT);
      const w = Math.max(0.5, x2 - x1);

      ctx.fillStyle = SYLLABLE_COLORS[b.syllable] || "#ccc";
      ctx.fillRect(x1, ySyll, w, TRACK_H);
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x1, ySyll, w, TRACK_H);

      if (viewMax - viewMin <= 100 && w > 16) {{
        ctx.fillStyle = "#111";
        ctx.font = "9px -apple-system, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(b.syllable, x1 + w / 2, ySyll + 26);
      }}
    }});
  }});
}}

function setWindow(t0, t1) {{
  viewMin = Math.max(0, t0);
  viewMax = Math.min(maxTime, t1);
  render();
}}

function zoomIn() {{
  const span = (viewMax - viewMin) * 0.75;
  const mid = (viewMin + viewMax) / 2;
  viewMin = Math.max(0, mid - span / 2);
  viewMax = Math.min(maxTime, mid + span / 2);
  render();
}}

function zoomOut() {{
  const span = (viewMax - viewMin) * 1.33;
  const mid = (viewMin + viewMax) / 2;
  viewMin = Math.max(0, mid - span / 2);
  viewMax = Math.min(maxTime, mid + span / 2);
  render();
}}

function panLeft() {{
  const shift = (viewMax - viewMin) * 0.25;
  if (viewMin - shift >= 0) {{
    viewMin -= shift;
    viewMax -= shift;
    render();
  }}
}}

function panRight() {{
  const shift = (viewMax - viewMin) * 0.25;
  if (viewMax + shift <= maxTime) {{
    viewMin += shift;
    viewMax += shift;
    render();
  }}
}}

// Interactive Hover Tooltip
canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const plotW = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;
  if (mouseX < MARGIN_LEFT || mouseX > canvas.width - MARGIN_RIGHT) {{
    tooltip.style.display = "none";
    return;
  }}

  const hoverTime = viewMin + ((mouseX - MARGIN_LEFT) / plotW) * (viewMax - viewMin);
  let hit = null;

  SESSIONS.forEach((sess, sIdx) => {{
    const groupY = MARGIN_TOP + sIdx * TRACK_GROUP_HEIGHT;
    const yMacro = groupY + 10;
    const yRout = yMacro + TRACK_H + TRACK_GAP;
    const ySyll = yRout + TRACK_H + TRACK_GAP;

    // Check Macro
    if (mouseY >= yMacro && mouseY <= yMacro + TRACK_H) {{
      const b = sess.macro_blocks.find(m => hoverTime >= m.start_sec && hoverTime <= m.end_sec);
      if (b) hit = {{
        track: "Upper Macro-State",
        id: `U${{b.macro_state + 1}}`,
        onset: b.start_sec.toFixed(2),
        end: b.end_sec.toFixed(2),
        dur: b.duration_sec.toFixed(2),
        bouts: b.bouts,
        extra: `Constituent Routines: ${{b.routine_occurrences}}`
      }};
    }}
    // Check Routine
    else if (mouseY >= yRout && mouseY <= yRout + TRACK_H) {{
      const b = sess.rout_blocks.find(r => hoverTime >= r.start_sec && hoverTime <= r.end_sec);
      if (b) hit = {{
        track: "Shared Behavioral Routine",
        id: `R${{b.routine_id + 1}}`,
        onset: b.start_sec.toFixed(2),
        end: b.end_sec.toFixed(2),
        dur: b.duration_sec.toFixed(2),
        bouts: b.bouts,
        extra: `Macro Context: U${{b.macro_state + 1}}${{b.truncated ? ' (Boundary Truncated)' : ''}}`
      }};
    }}
    // Check Syllable
    else if (mouseY >= ySyll && mouseY <= ySyll + TRACK_H) {{
      const b = sess.syll_blocks.find(s => hoverTime >= s.start_sec && hoverTime <= s.end_sec);
      if (b) hit = {{
        track: "KPMS Syllable Bout",
        id: `syll_${{String(b.syllable).padStart(3, '0')}}`,
        onset: b.start_sec.toFixed(2),
        end: b.end_sec.toFixed(2),
        dur: b.duration_sec.toFixed(2),
        bouts: 1,
        extra: `Stage: ${{b.stage + 1}} of Routine R${{b.routine + 1}}`
      }};
    }}
  }});

  if (hit) {{
    tooltip.style.display = "block";
    tooltip.style.left = (e.pageX + 15) + "px";
    tooltip.style.top = (e.pageY + 15) + "px";
    tooltip.innerHTML = `
      <div style="font-weight:bold; color:#38bdf8; margin-bottom:3px;">${{hit.track}}: ${{hit.id}}</div>
      <div><strong>Onset:</strong> ${{hit.onset}}s | <strong>End:</strong> ${{hit.end}}s</div>
      <div><strong>Duration:</strong> ${{hit.dur}}s (${{hit.bouts}} bout${{hit.bouts > 1 ? 's' : ''}})</div>
      <div style="color:#94a3b8; font-size:11px; margin-top:2px;">${{hit.extra}}</div>
    `;
  }} else {{
    tooltip.style.display = "none";
  }}
}});

canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

// Initial render
render();
</script>
</body>
</html>"""

    html_path = OUTPUT_DIR / "hierarchical_timeline_interactive.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"✓ Saved interactive HTML timeline to: {html_path.name}")


def main():
    print("=" * 80)
    print("GENERATING HIERARCHICAL TIMELINE VISUALIZATIONS (THREE ALIGNED TRACKS)")
    print("=" * 80)

    if not BUNDLE_PKL.exists():
        print(f"Error: results bundle not found at {BUNDLE_PKL}")
        return

    with open(BUNDLE_PKL, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model_free"]

    session_list, _ = load_cohort_sessions(max_bouts_per_session=1500)

    # Pick 4 exemplar sessions comparing WT and KI under baseline Saline
    # Session 1: Mouse 728 (WT), Session 2: Mouse 1680 (WT),
    # Session 0: Mouse 1694 (KI), Session 3: Mouse 1664 (KI)
    exemplar_indices = [1, 2, 0, 3]
    exemplar_sessions = [session_list[idx] for idx in exemplar_indices]

    print("Extracting timeline intervals for 4 exemplar recordings (2 WT, 2 KI)...")
    sessions_data = []
    for s_info in exemplar_sessions:
        s_data = extract_session_tracks(s_info, model)
        sessions_data.append(s_data)
        print(f"  • Session {s_data['session_idx']:2d} | Animal {s_data['animal_id']:>4s} ({s_data['genotype']}): {s_data['total_sec']:.1f}s, {len(s_data['rout_blocks'])} routines, {len(s_data['macro_blocks'])} macro visits.")

    # 1. Whole-Session Overview (0 to 600s)
    print("\n[1/3] Rendering Whole-Session Overview (fig_hierarchical_timeline_overview.png)...")
    plot_overview_comparison(sessions_data, time_window=(0, 600))

    # 2. Selectable 60-second Detail (100 to 160s)
    print("\n[2/3] Rendering 60-Second Detail View (fig_hierarchical_timeline_detail_60s.png)...")
    plot_detail_60s_window(sessions_data, time_window=(100, 160))

    # 3. Interactive HTML Timeline
    print("\n[3/3] Generating Interactive HTML Timeline (hierarchical_timeline_interactive.html)...")
    generate_interactive_html(sessions_data)

    print("\n" + "=" * 80)
    print("ALL THREE HIERARCHICAL TIMELINE VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
