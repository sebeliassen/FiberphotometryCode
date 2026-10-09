#!/usr/bin/env python3
"""
generate_both_stacked_models_slides.py
=============================================================================
Generates side-by-side presentation-ready stacked slides for BOTH:
1. The Old Long-Duration Model (17KP kappa=500,000, multi-minute / session locked)
2. The Calibrated ~12s Model (8KP kappa=50,000, median 9.25s, mean 16.16s, clean data)

Layout Specifications (User Preferences):
- Vertically stacked: Panel A (Saline Day 1) on top, Panel B (Amphetamine Day 2) on bottom.
- 11 representative mice displayed (mice 1765, 728, 1694, 1664 omitted to lessen vertical space).
- Group totals accurately preserved: WT (n=8) and DAT-KI (n=7).
- Permutation test vertically stacked and objective (no pointing arrows, no sensationalist wording).
- Unified 16:9 UHD (3840x2160) slide composites for both models.
=============================================================================
"""

import itertools
import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
SHMOSEQ_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq"
OUT_DIRS = [
    SHMOSEQ_DIR / "output",
    SHMOSEQ_DIR / "output" / "figures",
    REPO_ROOT / "syllable_generation" / "presentations",
]

MODEL_OLD_PATH = SHMOSEQ_DIR / "output" / "models" / "selected_exploratory_model.pkl"
DATA_OLD_PATH = SHMOSEQ_DIR / "output" / "data" / "shmoseq_input_data.pkl"

MODEL_12S_PATH = SHMOSEQ_DIR / "stickiness_comparison_8kp_vs_17kp" / "models" / "model_8kp_kappa_50000.pkl"
DATA_12S_PATH = SHMOSEQ_DIR / "output_8kp_run03" / "data" / "shmoseq_input_data_8kp.pkl"

PALETTE = [
    "#1f77b4",  # Z_0: Blue
    "#ff7f0e",  # Z_1: Orange
    "#2ca02c",  # Z_2: Green
    "#9467bd",  # Z_3: Purple
    "#8c564b",  # Z_4: Brown
    "#7f7f7f",  # Z_5: Gray
    "#bcbd22",  # Z_6: Olive/Yellow
    "#17becf",  # Z_7: Cyan
]

EXCLUDE_MICE = {"1765", "728", "1694", "1664"}


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
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")


def tv_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def compute_exact_permutation_distribution(occupancies_by_animal, animals_meta, n_states):
    unique_animals = sorted(list(animals_meta.keys()))
    wt_animals = [a for a in unique_animals if "WT" in animals_meta[a]["genotype"]]
    ki_animals = [a for a in unique_animals if "KI" in animals_meta[a]["genotype"] or "Het" in animals_meta[a]["genotype"]]

    n_total = len(unique_animals)
    n_wt = len(wt_animals)

    sal_wt_obs = np.array([occupancies_by_animal[a]["Saline"] for a in wt_animals])
    sal_ki_obs = np.array([occupancies_by_animal[a]["Saline"] for a in ki_animals])
    obs_tv = tv_distance(sal_wt_obs.mean(axis=0), sal_ki_obs.mean(axis=0))

    all_sal = np.array([occupancies_by_animal[a]["Saline"] for a in unique_animals])
    all_indices = set(range(n_total))
    combos = list(itertools.combinations(range(n_total), n_wt))

    null_dists = []
    for c in combos:
        c_set = set(c)
        c_comp = list(all_indices - c_set)
        m_wt = all_sal[list(c)].mean(axis=0)
        m_ki = all_sal[c_comp].mean(axis=0)
        null_dists.append(tv_distance(m_wt, m_ki))

    null_dists = np.array(null_dists)
    p_val = float(np.mean(null_dists >= obs_tv - 1e-9))
    return obs_tv, null_dists, p_val


def make_stacked_timelines(states_dict, meta_dict, title_prefix, sub_subtitle, out_filename):
    """
    Renders stacked timelines for Saline (top) and Amphetamine (bottom)
    across the 11 representative mice.
    """
    animals = {}
    for k, st in states_dict.items():
        m = meta_dict[k]
        aid = str(m["animal_id"])
        if aid in EXCLUDE_MICE:
            continue
        geno = m["genotype"]
        drug_raw = m["drug"]
        drug_key = "saline" if "sal" in drug_raw.lower() else "amphetamine"
        if aid not in animals:
            animals[aid] = {"geno": geno, "saline": None, "amphetamine": None}
        animals[aid][drug_key] = st

    sorted_aids = sorted(list(animals.keys()), key=lambda a: (0 if "WT" in animals[a]["geno"] else 1, a))
    n_mice = len(sorted_aids)
    wt_count = sum(1 for a in sorted_aids if "WT" in animals[a]["geno"])

    fig, (ax_sal, ax_amp) = plt.subplots(2, 1, figsize=(10.5, 12.8), sharex=True, dpi=300)
    fps = 40.0125
    y_pos = np.arange(n_mice)

    for i, aid in enumerate(sorted_aids):
        sal = animals[aid]["saline"]
        amp = animals[aid]["amphetamine"]

        # Plot Saline
        if sal is not None:
            T = len(sal)
            change_pts = np.where(np.diff(sal) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [T]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = sal[s_idx]
                ax_sal.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.74, color=PALETTE[st_val % len(PALETTE)], edgecolor="none")

        # Plot Amphetamine
        if amp is not None:
            T = len(amp)
            change_pts = np.where(np.diff(amp) != 0)[0]
            starts = np.concatenate(([0], change_pts + 1))
            ends = np.concatenate((change_pts + 1, [T]))
            for s_idx, e_idx in zip(starts, ends):
                st_val = amp[s_idx]
                ax_amp.barh(i, (e_idx - s_idx)/(fps * 60.0), left=s_idx/(fps * 60.0),
                            height=0.74, color=PALETTE[st_val % len(PALETTE)], edgecolor="none")

    # Y-axis labels
    y_labels = [f"{animals[a]['geno']} #{a}" for a in sorted_aids]
    for ax in (ax_sal, ax_amp):
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=11.5, fontweight="bold")
        ax.invert_yaxis()
        for tick_label in ax.get_yticklabels():
            txt = tick_label.get_text()
            if "WT" in txt:
                tick_label.set_color("#0D47A1")
            else:
                tick_label.set_color("#B71C1C")
        ax.axhline(wt_count - 0.5, color="#212121", lw=2.2, linestyle="--")
        ax.set_xlim(0, 60)
        ax.grid(axis="x", linestyle=":", alpha=0.45)

    # Right side group annotations (True Cohort Size: WT n=8, DAT-KI n=7)
    mid_wt = (wt_count - 1) / 2.0
    mid_ki = wt_count + (n_mice - wt_count - 1) / 2.0

    for ax in (ax_sal, ax_amp):
        ax.text(61.2, mid_wt, "WT\n(n=8)", fontsize=12, fontweight="bold", color="#0D47A1", va="center")
        ax.text(61.2, mid_ki, "DAT-KI\n(n=7)", fontsize=12, fontweight="bold", color="#B71C1C", va="center")

    ax_sal.set_title(f"A. Saline Baseline Sessions (Day 1) — {sub_subtitle}", fontweight="bold", fontsize=13, pad=10)
    ax_amp.set_title(f"B. Amphetamine Challenge Sessions (Day 2) — {sub_subtitle}", fontweight="bold", fontsize=13, pad=10)
    ax_amp.set_xlabel("Time in Arena Post-Injection (minutes)", fontweight="bold", fontsize=12.5)

    legend_patches = [patches.Patch(color=PALETTE[k], label=f"Z_{k}") for k in range(8)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.02),
               frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=11.5)

    fig.suptitle(f"{title_prefix}", fontweight="bold", fontsize=14, y=0.985)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.94)

    save_fig_to_all(fig, out_filename)
    plt.close(fig)


def make_permutation_plot(syll_obs, syll_null, syll_p, state_obs, state_null, state_p, state_label, out_filename):
    """
    Renders stacked objective permutation test:
    Top: KPMS Syllables (p=0.0356, sig)
    Bottom: Macro-States (objective p-value)
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 11.5), dpi=300)

    # Top: Syllables
    p95_syll = np.percentile(syll_null, 95)
    ax_top.hist(syll_null, bins=35, color="#CFD8DC", edgecolor="white", alpha=0.95)
    ax_top.hist(syll_null[syll_null >= p95_syll], bins=15, color="#43A047", edgecolor="white", alpha=0.85)
    ax_top.axvline(syll_obs, color="#1B5E20", lw=3.0, label=f"Observed Distance ($d_{{TV}} = {syll_obs:.3f}, p = {syll_p:.4f}$)")
    ax_top.axvline(p95_syll, color="#37474F", lw=2.0, linestyle="--", label=f"95th Percentile Null ({p95_syll:.3f})")
    ax_top.set_title("A. KPMS Syllable Repertoire (100 Syllables)", fontweight="bold", fontsize=13.5, pad=10, loc="left")
    ax_top.set_xlabel("Total Variation Distance ($d_{TV}$)", fontweight="bold", fontsize=12)
    ax_top.set_ylabel("Permutations (n=6,435)", fontweight="bold", fontsize=12)
    ax_top.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax_top.grid(True, linestyle="--", alpha=0.35)

    # Bottom: States
    p95_state = np.percentile(state_null, 95)
    ax_bot.hist(state_null, bins=35, color="#CFD8DC", edgecolor="white", alpha=0.95)
    ax_bot.hist(state_null[state_null >= p95_state], bins=15, color="#C62828", edgecolor="white", alpha=0.85)
    ax_bot.axvline(state_obs, color="#B71C1C", lw=3.0, label=f"Observed Distance ($d_{{TV}} = {state_obs:.3f}, p = {state_p:.4f}$)")
    ax_bot.axvline(p95_state, color="#37474F", lw=2.0, linestyle="--", label=f"95th Percentile Null ({p95_state:.3f})")
    ax_bot.set_title(f"B. {state_label}", fontweight="bold", fontsize=13.5, pad=10, loc="left")
    ax_bot.set_xlabel("Total Variation Distance ($d_{TV}$)", fontweight="bold", fontsize=12)
    ax_bot.set_ylabel("Permutations (n=6,435)", fontweight="bold", fontsize=12)
    ax_bot.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax_bot.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Exact Animal Permutation Test: WT vs. DAT-KI Baseline (n=6,435 Partitions)", fontweight="bold", fontsize=14.5, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    save_fig_to_all(fig, out_filename)
    plt.close(fig)


def compose_slide(timeline_img_path, perm_img_path, pyramid_img_path, title, subtitle, bullets, out_filename):
    """
    Composes a 16:9 UHD (3840x2160) slide with 3 columns:
    Left: State timelines
    Center: Permutation test
    Right: Pyramid + Bullet card
    """
    slide_w, slide_h = 3840, 2160
    slide = Image.new("RGB", (slide_w, slide_h), "#F8F9FA")
    draw = ImageDraw.Draw(slide)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        font_card_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_card_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_card_h = ImageFont.load_default()
        font_card_body = ImageFont.load_default()

    # Title Banner
    draw.text((80, 50), title, fill="#1A202C", font=font_title)
    draw.text((84, 115), subtitle, fill="#4A5568", font=font_sub)

    # Column Layout
    top_y = 175
    bot_y = 2080
    avail_h = bot_y - top_y

    col1_w = 1250
    col2_w = 1150
    col3_w = 1180

    col1_x = 80
    col2_x = col1_x + col1_w + 40
    col3_x = col2_x + col2_w + 40

    # 1. Left: Timeline
    if os.path.exists(timeline_img_path):
        img_tl = Image.open(timeline_img_path)
        scale_tl = min(col1_w / img_tl.width, avail_h / img_tl.height)
        new_w, new_h = int(img_tl.width * scale_tl), int(img_tl.height * scale_tl)
        img_tl_resized = img_tl.resize((new_w, new_h), Image.Resampling.LANCZOS)
        slide.paste(img_tl_resized, (col1_x + (col1_w - new_w)//2, top_y + (avail_h - new_h)//2))

    # 2. Center: Permutation
    if os.path.exists(perm_img_path):
        img_perm = Image.open(perm_img_path)
        scale_perm = min(col2_w / img_perm.width, avail_h / img_perm.height)
        new_w, new_h = int(img_perm.width * scale_perm), int(img_perm.height * scale_perm)
        img_perm_resized = img_perm.resize((new_w, new_h), Image.Resampling.LANCZOS)
        slide.paste(img_perm_resized, (col2_x + (col2_w - new_w)//2, top_y + (avail_h - new_h)//2))

    # 3. Right: Pyramid on top, Bullets card on bottom
    if os.path.exists(pyramid_img_path):
        img_pyr = Image.open(pyramid_img_path)
        pyr_max_h = 800
        scale_pyr = min(col3_w / img_pyr.width, pyr_max_h / img_pyr.height)
        new_w, new_h = int(img_pyr.width * scale_pyr), int(img_pyr.height * scale_pyr)
        img_pyr_resized = img_pyr.resize((new_w, new_h), Image.Resampling.LANCZOS)
        slide.paste(img_pyr_resized, (col3_x + (col3_w - new_w)//2, top_y + 30))

    # Bullets Card
    card_y = top_y + 880
    card_h = avail_h - 910
    card_rect = [(col3_x, card_y), (col3_x + col3_w, card_y + card_h)]
    draw.rounded_rectangle(card_rect, radius=20, fill="#FFFFFF", outline="#D1D5DB", width=2)

    draw.text((col3_x + 35, card_y + 35), "Why Bottom-Up Sequences Are Required", fill="#111827", font=font_card_h)

    curr_y = card_y + 100
    for header, body in bullets:
        draw.text((col3_x + 35, curr_y), f"• {header}:", fill="#1F2937", font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22) if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") else font_card_body)
        curr_y += 34
        # Wrap body
        words = body.split()
        line = "  "
        for w in words:
            if len(line + " " + w) > 65:
                draw.text((col3_x + 35, curr_y), line, fill="#4B5563", font=font_card_body)
                curr_y += 30
                line = "  " + w
            else:
                line += " " + w if line != "  " else w
        if line.strip():
            draw.text((col3_x + 35, curr_y), line, fill="#4B5563", font=font_card_body)
            curr_y += 44

    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / out_filename
        slide.save(path, quality=95)
        print(f"Saved: {path}")


def main():
    set_style()
    print("=" * 80)
    print("GENERATING PRESENTATION SLIDES FOR BOTH SHMOSEQ MODELS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 0. Load KPMS Syllables for Null Distribution
    # -------------------------------------------------------------------------
    with open(DATA_OLD_PATH, "rb") as f:
        data_old = pickle.load(f)
    sylls_dict = data_old["syllables_dict"]
    meta_old = data_old["metadata_dict"]

    # Compute Syllable Occupancies by Animal
    animals_meta = {}
    syll_occ_by_animal = {}
    n_sylls = 100
    for k, s_seq in sylls_dict.items():
        m = meta_old[k]
        aid = str(m["animal_id"])
        drug = str(m["drug"])
        geno = str(m["genotype"])
        animals_meta[aid] = {"genotype": geno}
        counts = np.bincount(s_seq, minlength=n_sylls)
        occ = counts / (counts.sum() + 1e-9)
        if aid not in syll_occ_by_animal:
            syll_occ_by_animal[aid] = {}
        syll_occ_by_animal[aid][drug] = occ

    syll_obs, syll_null, syll_p = compute_exact_permutation_distribution(syll_occ_by_animal, animals_meta, n_sylls)
    print(f"KPMS Syllable Repertoire: Obs TV={syll_obs:.4f}, p={syll_p:.4f}")

    # =========================================================================
    # 1. MODEL 1: Old Long-Duration Model (selected_exploratory_model.pkl)
    # =========================================================================
    print("\nProcessing Model 1: Old Long-Duration Model (17KP kappa=500,000)...")
    with open(MODEL_OLD_PATH, "rb") as f:
        m_old = pickle.load(f)

    states_old = m_old["states_dict"]
    meta_old = m_old["metadata_dict"]

    # Calculate State Occupancies
    state_occ_old = {}
    for k, st in states_old.items():
        m = meta_old[k]
        aid = str(m["animal_id"])
        drug = str(m["drug"])
        counts = np.bincount(st, minlength=8)
        occ = counts / (counts.sum() + 1e-9)
        if aid not in state_occ_old:
            state_occ_old[aid] = {}
        state_occ_old[aid][drug] = occ

    old_state_obs, old_state_null, old_state_p = compute_exact_permutation_distribution(state_occ_old, animals_meta, 8)
    print(f"Old Model Macro-States: Obs TV={old_state_obs:.4f}, p={old_state_p:.4f}")

    # 1A. State Timeline
    tl_old_png = "pres_shmoseq_state_timelines_old_model_stacked.png"
    make_stacked_timelines(
        states_old, meta_old,
        title_prefix="Old shMoSeq Model (kappa=500,000): Severe Animal-Level State Entrapment",
        sub_subtitle="Sticky Prior kappa=500,000 (Monolithic 60-Min Session Locks)",
        out_filename=tl_old_png
    )

    # 1B. Permutation Plot
    perm_old_png = "pres_syllables_vs_states_permutation_old_model_stacked.png"
    make_permutation_plot(
        syll_obs, syll_null, syll_p,
        old_state_obs, old_state_null, old_state_p,
        state_label="Old shMoSeq Macro-States (kappa=500,000)",
        out_filename=perm_old_png
    )

    # 1C. 16:9 Composite Slide
    pyramid_png = OUT_DIRS[0] / "pres_behavioral_hierarchy_pyramid.png"
    bullets_old = [
        ("Monolithic Session Entrapment", "High stickiness (kappa=500,000) traps individual mice in idiosyncratic states for entire 60-minute sessions."),
        ("Genotype Signal Erased", f"While raw syllable repertoires significantly separate genotypes (p = {syll_p:.4f}), macro-states completely erase discrimination (p = {old_state_p:.4f})."),
        ("The Missing Rung", "Top-down HMMs over-abstract without sequence syntax. We must bridge syllables to macro-states via tangible Syllable Chains & Episodes."),
    ]
    compose_slide(
        OUT_DIRS[0] / tl_old_png,
        OUT_DIRS[0] / perm_old_png,
        pyramid_png,
        title="Old shMoSeq Model: Session Locking & Genotype Signal Loss",
        subtitle="Monolithic Session Entrapment (Left) • Genotype Discrimination Lost (Center, p = 0.754) • The Missing Behavioral Rung (Right)",
        bullets=bullets_old,
        out_filename="pres_reality_check_composite_slide_old_model.png"
    )

    # =========================================================================
    # 2. MODEL 2: Calibrated ~12s Model (model_8kp_kappa_50000.pkl - Clean Data)
    # =========================================================================
    print("\nProcessing Model 2: Calibrated ~12s Model (8KP kappa=50,000 - Clean Data)...")
    with open(MODEL_12S_PATH, "rb") as f:
        m_12s = pickle.load(f)

    with open(DATA_12S_PATH, "rb") as f:
        d_12s = pickle.load(f)

    states_12s = m_12s["decoded_states_dict"]
    meta_12s = d_12s["metadata_dict"]

    # Calculate State Occupancies
    state_occ_12s = {}
    for k, st in states_12s.items():
        m = meta_12s[k]
        aid = str(m["animal_id"])
        drug = str(m["drug"])
        counts = np.bincount(st, minlength=8)
        occ = counts / (counts.sum() + 1e-9)
        if aid not in state_occ_12s:
            state_occ_12s[aid] = {}
        state_occ_12s[aid][drug] = occ

    s12_obs, s12_null, s12_p = compute_exact_permutation_distribution(state_occ_12s, animals_meta, 8)
    print(f"Clean 12s Model Macro-States: Obs TV={s12_obs:.4f}, p={s12_p:.4f}")

    # 2A. State Timeline
    tl_12s_png = "pres_shmoseq_state_timelines_12s_model_stacked.png"
    make_stacked_timelines(
        states_12s, meta_12s,
        title_prefix="Calibrated shMoSeq Model (Bout ~12s): Dynamic Multi-Second Switching",
        sub_subtitle="Calibrated kappa=50,000 (Median Bout: 9.3s, Mean: 16.2s)",
        out_filename=tl_12s_png
    )

    # 2B. Permutation Plot
    perm_12s_png = "pres_syllables_vs_states_permutation_12s_model_stacked.png"
    make_permutation_plot(
        syll_obs, syll_null, syll_p,
        s12_obs, s12_null, s12_p,
        state_label="Calibrated shMoSeq Macro-States (kappa=50,000, ~12s Bouts)",
        out_filename=perm_12s_png
    )

    # 2C. 16:9 Composite Slide
    bullets_12s = [
        ("Ethological Timescale Achieved", "Calibrating stickiness (kappa=50,000) restores natural multi-second bouts (median 9.3s, mean 16.2s), eliminating session locking."),
        ("Genotype Separation Still Fails", f"Even with realistic bout dynamics, macro-states fail to reach significance under exact animal permutations (p = {s12_p:.4f})."),
        ("Architectural Limitation", "Top-down clustering compresses syllables into global mixture bags. Sequential syntax and true genotype differences are preserved in Syllable Chains & Episodes."),
    ]
    compose_slide(
        OUT_DIRS[0] / tl_12s_png,
        OUT_DIRS[0] / perm_12s_png,
        pyramid_png,
        title="Calibrated shMoSeq Model (~12s Bouts): Dynamic Switching & The Bottom-Up Pivot",
        subtitle="Multi-Second Dynamic Bout Switching (Left) • Genotype Signal Still Unresolved (Center, p = 0.333) • The Missing Behavioral Rung (Right)",
        bullets=bullets_12s,
        out_filename="pres_reality_check_composite_slide_12s_model.png"
    )

    print("\n" + "=" * 80)
    print("ALL SLIDE ASSETS GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
