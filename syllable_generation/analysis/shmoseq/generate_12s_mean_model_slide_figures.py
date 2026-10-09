#!/usr/bin/env python3
"""
generate_12s_mean_model_slide_figures.py
=============================================================================
Generates the presentation slide figures for the calibrated 11.98s mean model
(8KP shMoSeq with kappa=50,000):

1. pres_shmoseq_state_timelines_8kp_k50k_12s.png
   - Stacked vertical timeline (Saline Day 1 top, Amph Day 2 bottom)
   - Shows dynamic multi-second bout switching (mean 11.98s, median 6.93s)
   - 11 representative mice (6 WT in blue, 5 KI in red)
   - Cohort labels: WT (n=8) and DAT-KI (n=7)

2. pres_syllables_vs_states_permutation_8kp_k50k.png
   - Stacked vertical permutation test (Syllables top, Macro-states bottom)
   - Syllables: d_TV = 0.3540, p = 0.0356 (Significant)
   - Calibrated 12s States: d_TV = 0.0656, p = 0.8847 (88.5% null exceed observed)
   - Clean objective styling, no pointing arrows

3. pres_reality_check_composite_slide_8kp_k50k.png
   - Unified 16:9 UHD (3840x2160) slide with 3-column composition:
     * Left: 12s mean state timelines
     * Center: Permutation test (p=0.0356 vs p=0.8847)
     * Right: Hierarchy pyramid + Key Takeaway card
=============================================================================
"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
SHMOSEQ_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq")
OUTPUT_DIRS = [
    os.path.join(SHMOSEQ_DIR, "output"),
    os.path.join(SHMOSEQ_DIR, "output", "figures"),
    os.path.join(BASE_DIR, "syllable_generation", "presentations"),
]

MODEL_8KP_K50K = os.path.join(SHMOSEQ_DIR, "stickiness_comparison_8kp_vs_17kp", "models", "model_8kp_kappa_50000.pkl")
DATA_8KP = os.path.join(SHMOSEQ_DIR, "output_8kp_run03", "data", "shmoseq_input_data_8kp.pkl")
DATA_17KP = os.path.join(SHMOSEQ_DIR, "output", "data", "shmoseq_input_data.pkl")

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10.5,
        "figure.titlesize": 14.5,
    })

def save_fig_to_all(fig, filename):
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")

def make_timeline_12s():
    print("Generating 11.98s Mean Model State Timelines (8KP kappa=50,000)...")
    with open(MODEL_8KP_K50K, "rb") as f:
        m = pickle.load(f)
    states_arr = m["decoded_states"]

    with open(DATA_8KP, "rb") as f:
        d = pickle.load(f)
    meta = d["metadata_dict"]
    s_keys = list(meta.keys())

    exclude = {"1765", "728", "1694", "1664"}
    animals = {}

    for i, k in enumerate(s_keys):
        row_m = meta[k]
        aid = str(row_m["animal_id"])
        if aid in exclude:
            continue
        geno = row_m["genotype"]
        drug_raw = row_m["drug"]
        drug_key = "saline" if "sal" in drug_raw.lower() else "amphetamine"
        if aid not in animals:
            animals[aid] = {"geno": geno, "saline": None, "amphetamine": None}
        animals[aid][drug_key] = states_arr[i]

    sorted_aids = sorted(list(animals.keys()), key=lambda a: (0 if "WT" in animals[a]["geno"] else 1, a))
    n_mice = len(sorted_aids)
    wt_count = sum(1 for a in sorted_aids if "WT" in animals[a]["geno"])

    palette = [
        "#1f77b4",  # Z_0: Blue
        "#ff7f0e",  # Z_1: Orange
        "#2ca02c",  # Z_2: Green
        "#9467bd",  # Z_3: Purple
        "#8c564b",  # Z_4: Brown
        "#7f7f7f",  # Z_5: Gray
        "#bcbd22",  # Z_6: Olive/Yellow
        "#17becf",  # Z_7: Cyan
    ]

    fig, (ax_sal, ax_amp) = plt.subplots(2, 1, figsize=(10.5, 12.8), sharex=True, dpi=300)
    fps = 40.0125
    y_pos = np.arange(n_mice)

    for i, aid in enumerate(sorted_aids):
        sal = animals[aid]["saline"]
        amp = animals[aid]["amphetamine"]

        # Saline
        if sal is not None:
            change_pts = np.where(np.diff(sal) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [len(sal)]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = sal[s_idx]
                ax_sal.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.76, color=palette[st_val], edgecolor="none")

        # Amphetamine
        if amp is not None:
            change_pts = np.where(np.diff(amp) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [len(amp)]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = amp[s_idx]
                ax_amp.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.76, color=palette[st_val], edgecolor="none")

    y_labels = [f"{animals[a]['geno']} #{a}" for a in sorted_aids]

    for ax, title in [(ax_sal, "A. Saline Baseline Sessions (Day 1) — Calibrated κ=50,000 (Mean Bout: 12.0s)"),
                      (ax_amp, "B. Amphetamine Challenge Sessions (Day 2) — Calibrated κ=50,000")]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")
        ax.invert_yaxis()

        for tick_label in ax.get_yticklabels():
            txt = tick_label.get_text()
            if "WT" in txt:
                tick_label.set_color("#0D47A1")
            else:
                tick_label.set_color("#B71C1C")

        ax.axhline(wt_count - 0.5, color="#212121", lw=2.2, linestyle="--")
        ax.set_xlim(0, 60)
        ax.set_title(title, fontweight="bold", fontsize=12.5, pad=10, loc="left")
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)

        ax.text(60.8, (wt_count - 1) / 2.0, "WT\n(n=8)", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#0D47A1")
        ax.text(60.8, wt_count + (n_mice - wt_count - 1) / 2.0, "DAT-KI\n(n=7)", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#B71C1C")

    ax_amp.set_xlabel("Time in Arena Post-Injection (minutes)", fontweight="bold", fontsize=12.5)

    legend_patches = [patches.Patch(color=palette[k], label=f"Z_{k}") for k in range(8)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.015),
               frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=11.5)

    fig.suptitle("Calibrated shMoSeq (Mean Bout 12.0s): Dynamic Multi-Second Bout Switching Across Mice",
                 fontweight="bold", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, top=0.95, right=0.90)

    save_fig_to_all(fig, "pres_shmoseq_state_timelines_8kp_k50k_12s.png")
    # Also save as default stacked timeline so user can preview directly
    save_fig_to_all(fig, "pres_shmoseq_state_timelines_stacked.png")
    plt.close(fig)

def make_permutation_12s():
    print("Computing Permutation Test for 11.98s Model vs. Raw Syllables...")
    # 1. Raw KPMS Syllables
    with open(DATA_17KP, "rb") as f:
        bd = pickle.load(f)
    sylls = bd["syllables_dict"]
    meta = bd["metadata_dict"]

    animals_syll = {}
    for k, s in sylls.items():
        m = meta[k]
        if m["drug"] == "Saline":
            aid = str(m["animal_id"])
            geno = m["genotype"]
            counts_s = np.bincount(s, minlength=100)
            animals_syll[aid] = {"geno": geno, "props": counts_s / counts_s.sum()}

    aids = sorted(list(animals_syll.keys()))
    n_total = len(aids)
    wt_aids = [a for a in aids if animals_syll[a]["geno"] == "WT"]
    n_wt = len(wt_aids)
    combos = list(itertools.combinations(range(n_total), n_wt))

    syll_mat = np.array([animals_syll[a]["props"] for a in aids])
    wt_mask = np.array([animals_syll[a]["geno"] == "WT" for a in aids])

    obs_tv_syll = 0.5 * np.sum(np.abs(syll_mat[wt_mask].mean(axis=0) - syll_mat[~wt_mask].mean(axis=0)))
    null_tv_syll = []
    for c in combos:
        m = np.zeros(n_total, dtype=bool)
        m[list(c)] = True
        d_s = 0.5 * np.sum(np.abs(syll_mat[m].mean(axis=0) - syll_mat[~m].mean(axis=0)))
        null_tv_syll.append(d_s)
    null_tv_syll = np.array(null_tv_syll)
    p_syll = float((null_tv_syll >= obs_tv_syll).mean())
    q95_syll = float(np.percentile(null_tv_syll, 95))

    # 2. 8KP kappa=50,000 (11.98s mean)
    with open(MODEL_8KP_K50K, "rb") as f:
        m_bundle = pickle.load(f)
    occs = m_bundle["occupancies_by_animal"]

    with open(DATA_8KP, "rb") as f:
        d_8kp = pickle.load(f)
    meta_8kp = d_8kp["metadata_dict"]
    aid_geno_8kp = {str(row_m["animal_id"]): row_m["genotype"] for row_m in meta_8kp.values()}

    aids_8kp = sorted(list(occs.keys()))
    wt_mask_8kp = np.array(["WT" in aid_geno_8kp[a] for a in aids_8kp])
    state_mat_8kp = np.array([occs[a]["Saline"] for a in aids_8kp])

    obs_tv_state = 0.5 * np.sum(np.abs(state_mat_8kp[wt_mask_8kp].mean(axis=0) - state_mat_8kp[~wt_mask_8kp].mean(axis=0)))
    null_tv_state = []
    for c in combos:
        m = np.zeros(n_total, dtype=bool)
        m[list(c)] = True
        d_z = 0.5 * np.sum(np.abs(state_mat_8kp[m].mean(axis=0) - state_mat_8kp[~m].mean(axis=0)))
        null_tv_state.append(d_z)
    null_tv_state = np.array(null_tv_state)
    p_state = float((null_tv_state >= obs_tv_state).mean())
    q95_state = float(np.percentile(null_tv_state, 95))

    print(f"KPMS Syllables: obs={obs_tv_syll:.4f}, p={p_syll:.4f}")
    print(f"8KP Calibrated 12s States: obs={obs_tv_state:.4f}, 95th={q95_state:.4f}, p={p_state:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 11.2), dpi=300)

    # Panel A: Syllables
    counts1, edges1, patches1 = ax1.hist(null_tv_syll, bins=35, color="#CFD8DC", edgecolor="white", lw=0.8, alpha=0.9)
    for patch, edge in zip(patches1, edges1[:-1]):
        if edge >= obs_tv_syll:
            patch.set_facecolor("#2E7D32")
    ax1.axvline(obs_tv_syll, color="#1B5E20", lw=2.8,
                label=f"Observed Distance ($d_{{TV}} = {obs_tv_syll:.3f}, p = {p_syll:.4f}$)")
    ax1.axvline(q95_syll, color="#37474F", lw=1.8, linestyle="--",
                label=f"95th Percentile Null ({q95_syll:.3f})")
    ax1.set_xlabel("Total Variation Distance ($d_{TV}$)", fontweight="bold", fontsize=11.5)
    ax1.set_ylabel("Permutations (n=6,435)", fontweight="bold", fontsize=11.5)
    ax1.set_title("A. KPMS Syllable Repertoire (100 Syllables)", fontweight="bold", pad=10, fontsize=13, loc="left")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10.5)
    ax1.set_ylim(0, max(counts1) * 1.15)

    # Panel B: Calibrated 12s States
    counts2, edges2, patches2 = ax2.hist(null_tv_state, bins=35, color="#CFD8DC", edgecolor="white", lw=0.8, alpha=0.9)
    for patch, edge in zip(patches2, edges2[:-1]):
        if edge >= obs_tv_state:
            patch.set_facecolor("#C62828")
    ax2.axvline(obs_tv_state, color="#B71C1C", lw=2.8,
                label=f"Observed Distance ($d_{{TV}} = {obs_tv_state:.3f}, p = {p_state:.4f}$)")
    ax2.axvline(q95_state, color="#37474F", lw=1.8, linestyle="--",
                label=f"95th Percentile Null ({q95_state:.3f})")
    ax2.set_xlabel("Total Variation Distance ($d_{TV}$)", fontweight="bold", fontsize=11.5)
    ax2.set_ylabel("Permutations (n=6,435)", fontweight="bold", fontsize=11.5)
    ax2.set_title("B. Calibrated shMoSeq Macro-States (κ=50,000, Mean Bout: 12.0s)", fontweight="bold", pad=10, fontsize=13, loc="left")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10.5)
    ax2.set_ylim(0, max(counts2) * 1.15)

    fig.suptitle("Exact Animal Permutation Test: WT vs. DAT-KI Baseline (n=6,435 Partitions)",
                 fontweight="bold", fontsize=13.5, y=0.995)
    plt.tight_layout()

    save_fig_to_all(fig, "pres_syllables_vs_states_permutation_8kp_k50k.png")
    # Also save as default permutation figure
    save_fig_to_all(fig, "pres_syllables_vs_states_genotype_permutation.png")
    plt.close(fig)

def make_composite_12s():
    print("Assembling Unified Composite Slide for 11.98s Model (16:9)...")
    time_img_path = os.path.join(OUTPUT_DIRS[0], "pres_shmoseq_state_timelines_8kp_k50k_12s.png")
    perm_img_path = os.path.join(OUTPUT_DIRS[0], "pres_syllables_vs_states_permutation_8kp_k50k.png")
    pyramid_img_path = os.path.join(OUTPUT_DIRS[0], "pres_behavioral_hierarchy_pyramid.png")

    img_time = Image.open(time_img_path)
    img_perm = Image.open(perm_img_path)
    img_pyramid = Image.open(pyramid_img_path)

    W, H = 3840, 2160
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    content_y = 150
    content_h = 1960

    # Col 1: Timeline
    col1_w = 1360
    time_h = content_h
    time_w = int(img_time.width * (time_h / img_time.height))
    if time_w > col1_w:
        time_w = col1_w
        time_h = int(img_time.height * (time_w / img_time.width))
    img_time_resized = img_time.resize((time_w, time_h), Image.Resampling.LANCZOS)
    x1 = 60 + (col1_w - time_w) // 2
    y1 = content_y + (content_h - time_h) // 2
    canvas.paste(img_time_resized, (x1, y1))

    # Col 2: Permutation Test
    col2_w = 1120
    perm_h = content_h - 40
    perm_w = int(img_perm.width * (perm_h / img_perm.height))
    if perm_w > col2_w:
        perm_w = col2_w
        perm_h = int(img_perm.height * (perm_w / img_perm.width))
    img_perm_resized = img_perm.resize((perm_w, perm_h), Image.Resampling.LANCZOS)
    x2 = 1450 + (col2_w - perm_w) // 2
    y2 = content_y + (content_h - perm_h) // 2
    canvas.paste(img_perm_resized, (x2, y2))

    # Col 3: Pyramid + Card
    col3_w = 1200
    pyr_w = col3_w
    pyr_h = int(img_pyramid.height * (pyr_w / img_pyramid.width))
    if pyr_h > 1050:
        pyr_h = 1050
        pyr_w = int(img_pyramid.width * (pyr_h / img_pyramid.height))
    img_pyr_resized = img_pyramid.resize((pyr_w, pyr_h), Image.Resampling.LANCZOS)
    x3 = 2600 + (col3_w - pyr_w) // 2
    y3 = content_y + 40
    canvas.paste(img_pyr_resized, (x3, y3))

    # Summary Box
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        font_card_head = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_card_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_head = ImageFont.load_default()
        font_card_body = ImageFont.load_default()

    card_x0 = 2600
    card_y0 = y3 + pyr_h + 50
    card_x1 = 2600 + col3_w
    card_y1 = content_y + content_h - 20

    draw.rounded_rectangle([card_x0, card_y0, card_x1, card_y1], radius=18,
                           fill="#F8F9FA", outline="#CFD8DC", width=2)

    draw.text((card_x0 + 35, card_y0 + 35), "Why Calibrated Bout Lengths Still Fail",
              fill="#212121", font=font_card_head)

    bullets = [
        "• Bout Calibration Achieved: Tuning stickiness to κ = 50,000 produces",
        "  an ethological mean bout duration of 12.0s (median 6.9s), eliminating",
        "  monolithic session locking and generating ~250 transitions/session.",
        "",
        "• Genotype Signal Annihilated: Despite realistic bout durations, the",
        "  state occupancies between WT and DAT-KI become identical (p = 0.8847).",
        "  88.5% of random animal shuffles produce greater separation!",
        "",
        "• The Root Dilemma: Top-down shMoSeq compresses syllables into global",
        "  mixture bags, ignoring sequential transitions where the phenotype lives.",
        "",
        "• The Bottom-Up Path: Syllable Chains & Episodes directly preserve the",
        "  multi-second grammar without collapsing genotype discrimination."
    ]

    cur_y = card_y0 + 85
    for line in bullets:
        if line == "":
            cur_y += 8
        else:
            draw.text((card_x0 + 35, cur_y), line, fill="#37474F", font=font_card_body)
            cur_y += 32

    draw.text((80, 40), "Calibrated shMoSeq (12s Mean Bout): The Reality Check & The Bottom-Up Pivot",
              fill="#212121", font=font_title)
    draw.text((80, 98), "Multi-Second Bout Switching (Left)  •  Genotype Signal Erased (Center, p = 0.8847)  •  The Missing Behavioral Rung (Right)",
              fill="#546E7A", font=font_sub)

    for d in OUTPUT_DIRS:
        p_named = os.path.join(d, "pres_reality_check_composite_slide_8kp_k50k.png")
        canvas.save(p_named, quality=95)
        print(f"Saved: {p_named}")
        p_def = os.path.join(d, "pres_reality_check_composite_slide.png")
        canvas.save(p_def, quality=95)

def main():
    set_style()
    make_timeline_12s()
    make_permutation_12s()
    make_composite_12s()
    print("\nAll 11.98s Model Slide Assets Generated Successfully!")

if __name__ == "__main__":
    main()
