#!/usr/bin/env python3
"""
generate_30min_hierarchical_timeline.py
=============================================================================
Hierarchical Timeline Visualizer for 30-Minute Behavioral Epochs (1,800 Seconds).
Joint Categorical Token + Continuous Kinematics Model.

Outputs:
1. fig_hierarchical_timeline_30min_overview.png (Full 0 to 1,800s / 30 min)
2. fig_hierarchical_timeline_30min_detail.png (120s detail window: 600s to 720s)
3. hierarchical_timeline_30min_interactive.html (Interactive 30-min browser visualizer)
4. summary table of 30-minute macro-state visit durations and occupancy
=============================================================================
"""

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from studies.behavior.sequence_viz.hierarchical_pilot.test_continuous_emissions import (
    ContinuousHierarchicalModel,
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
ARTIFACT_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82")
DATA_PKL = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data/unified_sequence_dataset.pkl"

# Palettes
ROUTINE_COLORS = {
    0: "#E41A1C", # R1: Vivid Crimson (Acceleration motif)
    1: "#377EB8", # R2: Sapphire Blue (Fast transition motif)
    2: "#4DAF4A", # R3: Emerald Green (Quiescent / resting mode)
    3: "#984EA3", # R4: Royal Amethyst (Intermediate motif)
    4: "#FF7F00", # R5: Amber Orange (High-speed sprint motif)
    5: "#00A896"  # R6: Deep Teal (Extended local exploration)
}

ROUTINE_LABELS = {
    0: "R1 (Acceleration Motif)",
    1: "R2 (Fast Transition)",
    2: "R3 (Quiescent / Resting)",
    3: "R4 (Intermediate Motif)",
    4: "R5 (High-Speed Sprint)",
    5: "R6 (Local Exploration)"
}

UPPER_COLORS = {
    0: "#6C8EBF", # U1: Slate Blue (Locomotion Context)
    1: "#82B366", # U2: Sage Green (Quiescent / Investigation Context)
    2: "#E6A15C"  # U3: Warm Amber (High-Cadence Acceleration Context)
}

UPPER_LABELS = {
    0: "U1 (Active Locomotion Context)",
    1: "U2 (Quiescent / Investigation Context)",
    2: "U3 (Acceleration Burst Context)"
}

# 100-syllable palette
rng = np.random.default_rng(12345)
hues = np.linspace(0, 1, 100, endpoint=False)
rng.shuffle(hues)
SYLLABLE_COLORS = {}
for s in range(100):
    rgb = mcolors.hsv_to_rgb([hues[s], 0.75, 0.85])
    SYLLABLE_COLORS[s] = mcolors.to_hex(rgb)


def load_30min_cohort(max_duration_sec=1800.0):
    """Loads up to 30 minutes (1,800 seconds) for each recording session."""
    with open(DATA_PKL, "rb") as f:
        bundle = pickle.load(f)

    sessions_raw = bundle["sessions"]
    session_list = []

    all_log_dur = []
    all_log_vel = []

    for s_idx, sess in enumerate(sessions_raw):
        aid = str(sess["animal_id"])
        gt = sess["genotype"]
        drug = sess["drug"]
        df = sess["bouts_df"]

        # Filter up to max_duration_sec
        cum_durs = df["duration_sec"].cumsum().values
        valid_mask = cum_durs <= max_duration_sec
        if not np.any(valid_mask):
            valid_mask = np.zeros(len(df), dtype=bool)
            valid_mask[0] = True
        sub_df = df[valid_mask]

        sylls = sub_df["syllable"].values.astype(int)
        durs = sub_df["duration_sec"].values.astype(float)
        vels = sub_df["mean_vel"].values.astype(float)

        log_dur = np.log(np.clip(durs, 1e-3, None))
        log_vel = np.log(np.clip(vels + 1.0, 1e-3, None))

        all_log_dur.extend(log_dur)
        all_log_vel.extend(log_vel)

        session_list.append({
            "session_idx": s_idx,
            "animal_id": aid,
            "genotype": gt,
            "drug": drug,
            "syllables": sylls,
            "durations": durs,
            "velocities": vels,
            "log_dur": log_dur,
            "log_vel": log_vel,
            "total_sec": float(np.sum(durs)),
            "total_bouts": len(sylls)
        })

    dur_mean = float(np.mean(all_log_dur))
    dur_std = float(np.std(all_log_dur))
    vel_mean = float(np.mean(all_log_vel))
    vel_std = float(np.std(all_log_vel))

    for sess in session_list:
        norm_dur = (sess["log_dur"] - dur_mean) / dur_std
        norm_vel = (sess["log_vel"] - vel_mean) / vel_std
        sess["features"] = np.column_stack([norm_dur, norm_vel])

    return session_list, (dur_mean, dur_std, vel_mean, vel_std)


def extract_session_tracks(session_info, model):
    sylls = session_info["syllables"]
    durs = session_info["durations"]
    feats = session_info["features"]
    aid = session_info["animal_id"]
    gt = session_info["genotype"]
    s_idx = session_info["session_idx"]
    drug = session_info["drug"]

    T = len(sylls)
    offsets = np.cumsum(durs)
    onsets = np.r_[0.0, offsets[:-1]]

    m_seq, r_seq, st_seq = model.decode_session_viterbi_joint(sylls, feats)

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

    # 2. Routine occurrences track (boundaries at every completion)
    rout_blocks = []
    curr_r = r_seq[0]
    curr_m = m_seq[0]
    r_start = 0
    for t in range(1, T):
        if r_seq[t] != curr_r or (st_seq[t] == 0 and st_seq[t-1] == 2):
            rout_blocks.append({
                "routine_id": int(curr_r),
                "macro_state": int(curr_m),
                "start_sec": float(onsets[r_start]),
                "end_sec": float(offsets[t - 1]),
                "duration_sec": float(np.sum(durs[r_start:t])),
                "bouts": int(t - r_start),
                "truncated": (r_start == 0)
            })
            curr_r = r_seq[t]
            curr_m = m_seq[t]
            r_start = t

    rout_blocks.append({
        "routine_id": int(curr_r),
        "macro_state": int(curr_m),
        "start_sec": float(onsets[r_start]),
        "end_sec": float(offsets[T - 1]),
        "duration_sec": float(np.sum(durs[r_start:T])),
        "bouts": int(T - r_start),
        "truncated": True
    })

    # 3. Macro state visits
    macro_blocks = []
    curr_m = m_seq[0]
    m_start = 0
    for t in range(1, T):
        if m_seq[t] != curr_m:
            macro_blocks.append({
                "macro_state": int(curr_m),
                "start_sec": float(onsets[m_start]),
                "end_sec": float(offsets[t - 1]),
                "duration_sec": float(np.sum(durs[m_start:t])),
                "bouts": int(t - m_start),
                "truncated": (m_start == 0)
            })
            curr_m = m_seq[t]
            m_start = t

    macro_blocks.append({
        "macro_state": int(curr_m),
        "start_sec": float(onsets[m_start]),
        "end_sec": float(offsets[T - 1]),
        "duration_sec": float(np.sum(durs[m_start:T])),
        "bouts": int(T - m_start),
        "truncated": True
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


def draw_tracks(ax, s_data, time_window, show_labels=True):
    t_min, t_max = time_window
    span = t_max - t_min

    # 1. TOP: Macro state
    for b in s_data["macro_blocks"]:
        if b["end_sec"] < t_min or b["start_sec"] > t_max:
            continue
        st = max(b["start_sec"], t_min)
        en = min(b["end_sec"], t_max)
        w = en - st
        m = b["macro_state"]
        rect = patches.Rectangle(
            (st, 2.15), w, 0.85,
            facecolor=UPPER_COLORS.get(m, "#AAA"), edgecolor="black", linewidth=0.8,
            hatch="///" if b["truncated"] else None
        )
        ax.add_patch(rect)
        if w > span * 0.035:
            ax.text(st + w / 2.0, 2.57, f"U{m+1}", ha="center", va="center", fontsize=9, fontweight="bold", color="#111")

    # 2. MIDDLE: Routines (thin solid black border at EVERY occurrence)
    for b in s_data["rout_blocks"]:
        if b["end_sec"] < t_min or b["start_sec"] > t_max:
            continue
        st = max(b["start_sec"], t_min)
        en = min(b["end_sec"], t_max)
        w = en - st
        r = b["routine_id"]
        rect = patches.Rectangle(
            (st, 1.05), w, 0.90,
            facecolor=ROUTINE_COLORS.get(r, "#777"), edgecolor="black", linewidth=0.9,
            hatch="///" if b["truncated"] else None
        )
        ax.add_patch(rect)
        if show_labels and w > span * 0.025:
            ax.text(st + w / 2.0, 1.50, f"R{r+1}", ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # 3. BOTTOM: Syllables
    for b in s_data["syll_blocks"]:
        if b["end_sec"] < t_min or b["start_sec"] > t_max:
            continue
        st = max(b["start_sec"], t_min)
        en = min(b["end_sec"], t_max)
        w = en - st
        s = b["syllable"]
        rect = patches.Rectangle(
            (st, 0.0), w, 0.85,
            facecolor=SYLLABLE_COLORS.get(s, "#CCC"), edgecolor=(0, 0, 0, 0.25), linewidth=0.25
        )
        ax.add_patch(rect)
        if span <= 180 and w > span * 0.02:
            ax.text(st + w / 2.0, 0.42, f"{s}", ha="center", va="center", fontsize=7, color="#222")


def render_30min_visualizations(sessions_data):
    # Overview (0 to 1,800s / 30 Minutes)
    n_sess = len(sessions_data)
    fig, axes = plt.subplots(n_sess, 1, figsize=(22, 3.6 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_tracks(ax, s_data, (0, 1800), show_labels=False)
        margin_text = f"Session {s_data['session_idx']}\nAnimal {s_data['animal_id']}\n({s_data['genotype']}, {s_data['drug']})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=80, va="center")
        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables", "Routines", "Upper State"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(0, 1800)
        # Major ticks every 5 min (300s), minor every 1 min (60s)
        ax.set_xticks(np.arange(0, 1801, 300))
        ax.set_xticks(np.arange(0, 1801, 60), minor=True)
        ax.grid(True, which="major", axis="x", alpha=0.4, linestyle="-", color="#777")
        ax.grid(True, which="minor", axis="x", alpha=0.15, linestyle=":")
        if i == 0:
            ax.set_title("30-Minute Continuous Hierarchical Timeline (1,800 Seconds / 0.5 Hour)\nAligned Tracks: Upper Macro-Context -> Subroutine Occurrences -> KPMS Syllables", fontsize=13, fontweight="bold", pad=15)

    # Format bottom x-axis with both seconds and minutes
    tick_locs = np.arange(0, 1801, 300)
    tick_labels = [f"{s}s\n({s//60}m)" for s in tick_locs]
    axes[-1].set_xticks(tick_locs)
    axes[-1].set_xticklabels(tick_labels, fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Elapsed Time in Recording: Seconds (Minutes)", fontsize=11, fontweight="bold")

    upper_patches = [patches.Patch(facecolor=UPPER_COLORS[u], edgecolor="black", label=UPPER_LABELS[u]) for u in range(3)]
    routine_patches = [patches.Patch(facecolor=ROUTINE_COLORS[r], edgecolor="black", label=f"R{r+1}: {ROUTINE_LABELS[r]}") for r in range(6)]
    trunc_patch = patches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Boundary Truncated")

    fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.28, 0.98), ncol=3, frameon=True, fontsize=9, title="Upper Macro-States (Kinematic Regimes)")
    fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=3, frameon=True, fontsize=9, title="Shared Behavioral Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    out_overview = OUTPUT_DIR / "fig_hierarchical_timeline_30min_overview.png"
    fig.savefig(out_overview, dpi=300)
    plt.close(fig)
    print(f"✓ Saved 30-Minute Overview: {out_overview.name}")

    # Detail View: 120-Second Window (600s to 720s, i.e. 10m to 12m)
    fig, axes = plt.subplots(n_sess, 1, figsize=(22, 3.8 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_tracks(ax, s_data, (600, 720), show_labels=True)
        margin_text = f"Session {s_data['session_idx']}\nAnimal {s_data['animal_id']}\n({s_data['genotype']}, {s_data['drug']})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=80, va="center")
        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables\n(KPMS Bouts)", "Shared\nRoutines", "Upper Context\n(Kinematic State)"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(600, 720)
        ax.grid(True, axis="x", alpha=0.35, linestyle="--")
        if i == 0:
            ax.set_title("120-Second Detail Window (600s to 720s / 10m to 12m into Recording)\nReadable Microstructure: Repetition Boundaries, Stage Transitions, Syllable IDs", fontsize=13, fontweight="bold", pad=15)

    axes[-1].set_xlabel("Elapsed Time in Recording (Seconds)", fontsize=11, fontweight="bold")

    fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.28, 0.98), ncol=3, frameon=True, fontsize=9, title="Upper Macro-States (Kinematic Regimes)")
    fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=3, frameon=True, fontsize=9, title="Shared Behavioral Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.06, 0.04, 0.98, 0.93])
    out_detail = OUTPUT_DIR / "fig_hierarchical_timeline_30min_detail.png"
    fig.savefig(out_detail, dpi=300)
    plt.close(fig)
    print(f"✓ Saved 120s Detail View: {out_detail.name}")

    # Copy to artifact dir for direct embedding
    import shutil
    shutil.copy(out_overview, ARTIFACT_DIR / out_overview.name)
    shutil.copy(out_detail, ARTIFACT_DIR / out_detail.name)
    print("✓ Copied both 30-minute figures to artifacts directory for inline display.")


def generate_30min_interactive_html(sessions_data):
    """Generates a standalone HTML5 visualizer supporting zoom & pan across 30 minutes."""
    json_sessions = []
    for s in sessions_data:
        json_sessions.append({
            "session_idx": s["session_idx"],
            "animal_id": s["animal_id"],
            "genotype": s["genotype"],
            "drug": s["drug"],
            "total_sec": round(s["total_sec"], 1),
            "total_bouts": s["total_bouts"],
            "macro_blocks": s["macro_blocks"],
            "rout_blocks": s["rout_blocks"],
            "syll_blocks": s["syll_blocks"]
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>30-Minute Hierarchical Timeline Visualizer</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #0f172a; color: #f8fafc; }}
  h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #38bdf8; }}
  p.subtitle {{ margin: 0 0 20px 0; font-size: 14px; color: #94a3b8; }}
  .toolbar {{ display: flex; align-items: center; gap: 10px; background: #1e293b; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
  .btn {{ background: #3b82f6; color: white; border: none; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; }}
  .btn:hover {{ background: #2563eb; }}
  .btn-secondary {{ background: #475569; }}
  .btn-secondary:hover {{ background: #334155; }}
  .legend-container {{ display: flex; gap: 24px; background: #1e293b; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 12px; flex-wrap: wrap; }}
  .legend-group {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .legend-title {{ font-weight: bold; color: #cbd5e1; margin-right: 4px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-color {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.5); }}
  #canvas-container {{ position: relative; background: #1e293b; border-radius: 8px; padding: 15px; overflow-x: auto; }}
  canvas {{ display: block; }}
  #tooltip {{ position: absolute; display: none; background: rgba(15, 23, 42, 0.95); border: 1px solid #38bdf8; color: #f8fafc; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 100; line-height: 1.5; }}
</style>
</head>
<body>
<h1>30-Minute Hierarchical Timeline Visualizer</h1>
<p class="subtitle">Upper Kinematic Regimes &rarr; Behavioral Subroutines &rarr; Syllables across 1,800 Seconds (0.5 Hour)</p>

<div class="toolbar">
  <span style="font-weight:600; font-size:13px; color:#cbd5e1;">Time Window Preset:</span>
  <button class="btn" onclick="setWindow(0, 300)">0m – 5m (Initial)</button>
  <button class="btn" onclick="setWindow(600, 900)">10m – 15m (Mid-Session)</button>
  <button class="btn" onclick="setWindow(1500, 1800)">25m – 30m (Late-Session)</button>
  <button class="btn" onclick="setWindow(600, 720)">10m – 12m (Detail View)</button>
  <button class="btn btn-secondary" onclick="setWindow(0, 1800)">Full 30-Min Overview</button>
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
    <div class="legend-item"><div class="legend-color" style="background:#82B366;"></div> U2: Quiescent / Investigation</div>
    <div class="legend-item"><div class="legend-color" style="background:#E6A15C;"></div> U3: Acceleration Burst</div>
  </div>
  <div class="legend-group">
    <span class="legend-title">Shared Routines:</span>
    <div class="legend-item"><div class="legend-color" style="background:#E41A1C;"></div> R1 (Accel)</div>
    <div class="legend-item"><div class="legend-color" style="background:#377EB8;"></div> R2 (Fast Trans)</div>
    <div class="legend-item"><div class="legend-color" style="background:#4DAF4A;"></div> R3 (Quiescent)</div>
    <div class="legend-item"><div class="legend-color" style="background:#984EA3;"></div> R4 (Interm)</div>
    <div class="legend-item"><div class="legend-color" style="background:#FF7F00;"></div> R5 (Sprint)</div>
    <div class="legend-item"><div class="legend-color" style="background:#00A896;"></div> R6 (Local Expl)</div>
  </div>
</div>

<div id="canvas-container">
  <canvas id="timelineCanvas" width="1800" height="980"></canvas>
  <div id="tooltip"></div>
</div>

<script>
const SESSIONS = {json.dumps(json_sessions)};
const ROUTINE_COLORS = {json.dumps(ROUTINE_COLORS)};
const UPPER_COLORS = {json.dumps(UPPER_COLORS)};
const SYLLABLE_COLORS = {json.dumps(SYLLABLE_COLORS)};

let viewMin = 0.0;
let viewMax = 1800.0;
const maxTime = 1800.0;

const canvas = document.getElementById("timelineCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

const MARGIN_LEFT = 180;
const MARGIN_RIGHT = 40;
const MARGIN_TOP = 40;
const TRACK_GROUP_HEIGHT = 210;
const TRACK_H = 46;
const TRACK_GAP = 12;

function render() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const plotW = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;

  function timeToX(t) {{
    return MARGIN_LEFT + ((t - viewMin) / (viewMax - viewMin)) * plotW;
  }}

  // Draw grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textAlign = "center";

  const tSpan = viewMax - viewMin;
  const step = tSpan <= 120 ? 10 : (tSpan <= 400 ? 30 : (tSpan <= 900 ? 60 : 300));
  const firstTick = Math.ceil(viewMin / step) * step;

  for (let t = firstTick; t <= viewMax; t += step) {{
    const x = timeToX(t);
    ctx.beginPath();
    ctx.moveTo(x, MARGIN_TOP - 15);
    ctx.lineTo(x, canvas.height - 30);
    ctx.stroke();
    const minStr = Math.floor(t / 60) + "m";
    ctx.fillText(`${{t}}s (${{minStr}})`, x, canvas.height - 12);
  }}

  SESSIONS.forEach((sess, sIdx) => {{
    const groupY = MARGIN_TOP + sIdx * TRACK_GROUP_HEIGHT;

    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 13px -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`Session ${{sess.session_idx}}`, MARGIN_LEFT - 15, groupY + 50);
    ctx.font = "12px -apple-system, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(`Mouse ${{sess.animal_id}} (${{sess.genotype}})`, MARGIN_LEFT - 15, groupY + 70);
    ctx.fillText(`${{sess.drug}}`, MARGIN_LEFT - 15, groupY + 88);

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
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, yRout, w, TRACK_H);
      if (w > 20) {{
        ctx.fillStyle = "#fff";
        ctx.font = "bold 10px -apple-system, sans-serif";
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
      const w = Math.max(0.4, x2 - x1);
      ctx.fillStyle = SYLLABLE_COLORS[b.syllable] || "#ccc";
      ctx.fillRect(x1, ySyll, w, TRACK_H);
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      ctx.lineWidth = 0.4;
      ctx.strokeRect(x1, ySyll, w, TRACK_H);
      if (tSpan <= 150 && w > 16) {{
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
  const span = viewMax - viewMin;
  if (span <= 30) return;
  const mid = (viewMin + viewMax) / 2;
  const newSpan = span * 0.6;
  viewMin = Math.max(0, mid - newSpan / 2);
  viewMax = Math.min(maxTime, mid + newSpan / 2);
  render();
}}

function zoomOut() {{
  const span = viewMax - viewMin;
  if (span >= maxTime) return;
  const mid = (viewMin + viewMax) / 2;
  const newSpan = Math.min(maxTime, span * 1.6);
  viewMin = Math.max(0, mid - newSpan / 2);
  viewMax = Math.min(maxTime, mid + newSpan / 2);
  render();
}}

function panLeft() {{
  const span = viewMax - viewMin;
  const shift = span * 0.25;
  if (viewMin - shift < 0) {{
    viewMax -= viewMin;
    viewMin = 0;
  }} else {{
    viewMin -= shift;
    viewMax -= shift;
  }}
  render();
}}

function panRight() {{
  const span = viewMax - viewMin;
  const shift = span * 0.25;
  if (viewMax + shift > maxTime) {{
    viewMin += (maxTime - viewMax);
    viewMax = maxTime;
  }} else {{
    viewMin += shift;
    viewMax += shift;
  }}
  render();
}}

// Hover tooltip
canvas.addEventListener("mousemove", (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const plotW = canvas.width - MARGIN_LEFT - MARGIN_RIGHT;

  if (mouseX < MARGIN_LEFT || mouseX > canvas.width - MARGIN_RIGHT) {{
    tooltip.style.display = "none";
    return;
  }}

  const hoverTime = viewMin + ((mouseX - MARGIN_LEFT) / plotW) * (viewMax - viewMin);
  const sessIdx = Math.floor((mouseY - MARGIN_TOP) / TRACK_GROUP_HEIGHT);
  if (sessIdx < 0 || sessIdx >= SESSIONS.length) {{
    tooltip.style.display = "none";
    return;
  }}

  const sess = SESSIONS[sessIdx];
  const relY = (mouseY - MARGIN_TOP) % TRACK_GROUP_HEIGHT;
  let hit = null;

  if (relY >= 10 && relY <= 10 + TRACK_H) {{
    const b = sess.macro_blocks.find(blk => hoverTime >= blk.start_sec && hoverTime <= blk.end_sec);
    if (b) {{
      hit = {{
        track: "Upper Macro-State",
        id: `U${{b.macro_state + 1}} (${{b.macro_state === 0 ? "Active Locomotion" : (b.macro_state === 1 ? "Quiescent/Investigatory" : "Acceleration Burst")}} Context)`,
        onset: b.start_sec.toFixed(1),
        end: b.end_sec.toFixed(1),
        dur: b.duration_sec.toFixed(1),
        bouts: b.bouts,
        extra: `Session ${{sess.session_idx}} (${{sess.genotype}})`
      }};
    }}
  }} else if (relY >= 10 + TRACK_H + TRACK_GAP && relY <= 10 + (TRACK_H + TRACK_GAP) + TRACK_H) {{
    const b = sess.rout_blocks.find(blk => hoverTime >= blk.start_sec && hoverTime <= blk.end_sec);
    if (b) {{
      hit = {{
        track: "Shared Subroutine",
        id: `R${{b.routine_id + 1}}`,
        onset: b.start_sec.toFixed(1),
        end: b.end_sec.toFixed(1),
        dur: b.duration_sec.toFixed(1),
        bouts: b.bouts,
        extra: `Session ${{sess.session_idx}} (${{sess.genotype}})`
      }};
    }}
  }} else if (relY >= 10 + (TRACK_H + TRACK_GAP)*2 && relY <= 10 + (TRACK_H + TRACK_GAP)*2 + TRACK_H) {{
    const b = sess.syll_blocks.find(blk => hoverTime >= blk.start_sec && hoverTime <= blk.end_sec);
    if (b) {{
      hit = {{
        track: "KPMS Syllable Bout",
        id: `Syllable ${{b.syllable}} (Stage ${{b.stage + 1}})`,
        onset: b.start_sec.toFixed(2),
        end: b.end_sec.toFixed(2),
        dur: b.duration_sec.toFixed(2),
        bouts: 1,
        extra: `Routine: R${{b.routine + 1}} | Upper: U${{b.macro + 1}}`
      }};
    }}
  }}

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

render();
</script>
</body>
</html>"""

    html_path = OUTPUT_DIR / "hierarchical_timeline_30min_interactive.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"✓ Saved 30-Minute Interactive HTML: {html_path.name}")


def main():
    print("=" * 80)
    print("FITTING AND RENDERING 30-MINUTE HIERARCHICAL TIMELINES (1,800 SECONDS)")
    print("=" * 80)

    t0 = time.time()
    sessions, params = load_30min_cohort(max_duration_sec=1800.0)
    print(f"Loaded {len(sessions)} sessions up to 1,800s. Total bouts in cohort: {sum(s['total_bouts'] for s in sessions)}")

    print("\nFitting Continuous Kinematics Model on 30-Minute Epochs (25 iters)...")
    model = ContinuousHierarchicalModel(M=3, K=6, S=3, n_features=2, random_seed=42)
    history = model.fit_multi_sequence_joint(sessions, max_iters=25)
    print(f"Fitting completed in {time.time() - t0:.1f}s. Final LL: {history[-1]:.1f}")

    # Exemplars across 30 minutes
    # Session 1: Mouse 728 (WT), Session 2: Mouse 1680 (WT),
    # Session 4: Mouse 1666 (KI), Session 0: Mouse 1694 (KI)
    exemplar_indices = [1, 2, 4, 0]
    print(f"\nExtracting tracks for 4 exemplar sessions across 1,800s: {exemplar_indices}...")
    sessions_data = []
    for idx in exemplar_indices:
        s_data = extract_session_tracks(sessions[idx], model)
        sessions_data.append(s_data)
        print(f"  • Session {s_data['session_idx']:2d} | Mouse {s_data['animal_id']} ({s_data['genotype']}): {s_data['total_sec']:.1f}s, {len(s_data['rout_blocks'])} routines, {len(s_data['macro_blocks'])} macro visits.")

    print("\n[1/3] Rendering 30-Minute Overview Figure (0 to 1,800s)...")
    render_30min_visualizations(sessions_data)

    print("\n[2/3] Generating 30-Minute Interactive HTML Visualizer...")
    generate_30min_interactive_html(sessions_data)

    print("\n" + "=" * 80)
    print("30-MINUTE HIERARCHICAL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
