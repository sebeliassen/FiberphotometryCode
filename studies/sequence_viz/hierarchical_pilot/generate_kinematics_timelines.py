#!/usr/bin/env python3
"""
generate_kinematics_timelines.py
=============================================================================
Generates hierarchical 3-track timeline visualizations using the
Joint Categorical Token + Continuous Kinematics Model
(Tokens + log-duration + log-velocity).

Tracks per session:
- TOP: Upper Macro-State (U1 Locomotion, U2 Quiescent/Investigation, U3 Burst)
- MIDDLE: Routine Occurrences (R1..R6) with boundaries at every completion
- BOTTOM: Observed KPMS Syllable Bouts on physical elapsed time

Outputs:
1. fig_hierarchical_timeline_kinematics_overview.png (0 to 600s)
2. fig_hierarchical_timeline_kinematics_detail_60s.png (100 to 160s)
3. hierarchical_timeline_kinematics_interactive.html
=============================================================================
"""

import sys
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
    load_cohort_with_kinematics,
)

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/hierarchical_pilot"
ARTIFACT_DIR = Path("/home/fsp585/.gemini/antigravity-ide/brain/b03aa115-6afd-40d7-a4d1-a025411e6b82")

# Palette definitions
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

# 100-syllable consistent palette
rng = np.random.default_rng(12345)
hues = np.linspace(0, 1, 100, endpoint=False)
rng.shuffle(hues)
SYLLABLE_COLORS = {}
for s in range(100):
    rgb = mcolors.hsv_to_rgb([hues[s], 0.75, 0.85])
    SYLLABLE_COLORS[s] = mcolors.to_hex(rgb)


def extract_kinematics_session_tracks(session_info, model):
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

    # Decode with continuous model
    m_seq, r_seq, st_seq = model.decode_session_viterbi_joint(sylls, feats)

    # 1. Syllable bouts track
    syll_blocks = []
    for t in range(T):
        syll_blocks.append({
            "syllable": int(sylls[t]),
            "start_sec": float(onsets[t]),
            "end_sec": float(offsets[t]),
            "duration_sec": float(durs[t]),
            "velocity": float(session_info["velocities"][t]),
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
        # boundary if routine changes OR completed cycle (stage 2 -> stage 0)
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

    # Final routine
    rout_blocks.append({
        "routine_id": int(curr_r),
        "macro_state": int(curr_m),
        "start_sec": float(onsets[r_start]),
        "end_sec": float(offsets[T - 1]),
        "duration_sec": float(np.sum(durs[r_start:T])),
        "bouts": int(T - r_start),
        "truncated": True
    })

    # 3. Macro state visits track
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
        if w > (t_max - t_min) * 0.04:
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
            facecolor=ROUTINE_COLORS.get(r, "#777"), edgecolor="black", linewidth=1.0,
            hatch="///" if b["truncated"] else None
        )
        ax.add_patch(rect)
        if show_labels and w > (t_max - t_min) * 0.025:
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
            facecolor=SYLLABLE_COLORS.get(s, "#CCC"), edgecolor=(0, 0, 0, 0.3), linewidth=0.3
        )
        ax.add_patch(rect)
        if (t_max - t_min) <= 120 and w > (t_max - t_min) * 0.02:
            ax.text(st + w / 2.0, 0.42, f"{s}", ha="center", va="center", fontsize=7, color="#222")


def render_figures(sessions_data):
    # Overview (0 - 600s)
    n_sess = len(sessions_data)
    fig, axes = plt.subplots(n_sess, 1, figsize=(20, 3.5 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_tracks(ax, s_data, (0, 600), show_labels=False)
        margin_text = f"Session {s_data['session_idx']}\nAnimal {s_data['animal_id']}\n({s_data['genotype']}, {s_data['drug']})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=75, va="center")
        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables", "Routines", "Upper State"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(0, 600)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")
        if i == 0:
            ax.set_title("A. Whole-Session Overview (Continuous Kinematics-Grounded Model)\nElapsed Time (0 to 600 Seconds) Across WT and DAT-Het KI", fontsize=12, fontweight="bold", pad=15)

    axes[-1].set_xlabel("Elapsed Time in Recording (Seconds)", fontsize=11, fontweight="bold")

    upper_patches = [patches.Patch(facecolor=UPPER_COLORS[u], edgecolor="black", label=UPPER_LABELS[u]) for u in range(3)]
    routine_patches = [patches.Patch(facecolor=ROUTINE_COLORS[r], edgecolor="black", label=f"R{r+1}: {ROUTINE_LABELS[r]}") for r in range(6)]
    trunc_patch = patches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Boundary Truncated")

    fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.28, 0.98), ncol=3, frameon=True, fontsize=9, title="Upper Macro-States (Kinematic Contexts)")
    fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=3, frameon=True, fontsize=9, title="Shared Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.05, 0.03, 0.98, 0.93])
    out_overview = OUTPUT_DIR / "fig_hierarchical_timeline_kinematics_overview.png"
    fig.savefig(out_overview, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_overview.name}")

    # Detail (100 - 160s)
    fig, axes = plt.subplots(n_sess, 1, figsize=(20, 3.8 * n_sess), sharex=True)
    if n_sess == 1:
        axes = [axes]

    for i, (ax, s_data) in enumerate(zip(axes, sessions_data)):
        draw_tracks(ax, s_data, (100, 160), show_labels=True)
        margin_text = f"Session {s_data['session_idx']}\nAnimal {s_data['animal_id']}\n({s_data['genotype']}, {s_data['drug']})"
        ax.set_ylabel(margin_text, fontsize=10, fontweight="bold", rotation=0, labelpad=75, va="center")
        ax.set_yticks([0.42, 1.50, 2.57])
        ax.set_yticklabels(["Syllables\n(KPMS Bouts)", "Shared\nRoutines", "Upper Context\n(Kinematic State)"], fontsize=9, fontweight="bold")
        ax.set_ylim(-0.15, 3.15)
        ax.set_xlim(100, 160)
        ax.grid(True, axis="x", alpha=0.4, linestyle="--")
        if i == 0:
            ax.set_title("B. 60-Second Detail View (Continuous Kinematics-Grounded Model: 100s to 160s)\nAligned Boundaries: Syllables -> Shared Routines -> Kinematic Regimes", fontsize=12, fontweight="bold", pad=15)

    axes[-1].set_xlabel("Elapsed Time in Recording (Seconds)", fontsize=11, fontweight="bold")

    fig.legend(handles=upper_patches, loc="upper center", bbox_to_anchor=(0.28, 0.98), ncol=3, frameon=True, fontsize=9, title="Upper Macro-States (Kinematic Contexts)")
    fig.legend(handles=routine_patches, loc="upper center", bbox_to_anchor=(0.72, 0.98), ncol=3, frameon=True, fontsize=9, title="Shared Subroutines")
    fig.legend(handles=[trunc_patch], loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.05, 0.03, 0.98, 0.93])
    out_detail = OUTPUT_DIR / "fig_hierarchical_timeline_kinematics_detail_60s.png"
    fig.savefig(out_detail, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {out_detail.name}")

    # Copy to artifact dir for direct embedding
    import shutil
    shutil.copy(out_overview, ARTIFACT_DIR / out_overview.name)
    shutil.copy(out_detail, ARTIFACT_DIR / out_detail.name)
    print("✓ Copied to artifacts directory for display.")


def main():
    print("Loading sessions with kinematics...")
    sessions, _ = load_cohort_with_kinematics(max_bouts_per_session=1500)

    print("Fitting model...")
    model = ContinuousHierarchicalModel(M=3, K=6, S=3, n_features=2, random_seed=42)
    model.fit_multi_sequence_joint(sessions, max_iters=25)

    # Pick 4 sessions showing WT vs KI and U1, U2, U3
    # Session 1: Mouse 728 (WT, Saline) - U1 & U2
    # Session 2: Mouse 1680 (WT, Saline) - U1, U2, U3
    # Session 4: Mouse 1666 (KI, Saline) - U1 & U3 bursts
    # Session 0: Mouse 1694 (KI, Saline) - U1 continuous hyper-locomotion
    exemplar_indices = [1, 2, 4, 0]
    print(f"Extracting tracks for sessions: {exemplar_indices}...")
    sessions_data = []
    for idx in exemplar_indices:
        s_data = extract_kinematics_session_tracks(sessions[idx], model)
        sessions_data.append(s_data)

    print("Rendering overview and detail timelines...")
    render_figures(sessions_data)
    print("All done!")


if __name__ == "__main__":
    main()
