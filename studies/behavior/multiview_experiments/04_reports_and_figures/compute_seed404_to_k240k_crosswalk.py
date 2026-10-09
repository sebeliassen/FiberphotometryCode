#!/usr/bin/env python3
"""
compute_seed404_to_k240k_crosswalk.py
=====================================
Computes empirical frame-by-frame and bout-level co-occurrence between
Champion Seed 404 (Tandem 23kp) and the user's previously labeled k240k (TopCam 17kp).
Generates:
  1. SEED404_TO_K240K_PRELABELING_CROSSWALK.md
  2. seed404_prelabeled_annotations.json (importable directly into the gallery!)
  3. seed404_prelabeled_annotations.csv
"""

import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
EXP_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments"
REPORTS_DIR = EXP_DIR / "04_reports_and_figures"
VIZ_DIR = REPORTS_DIR / "visualizations_seed404"

K240K_RES = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k/results.h5"
K240K_VFI = REPO_ROOT / "syllable_generation/syllabizers/keypoint_moseq_syll/karen_project_2/final_karen_model_k240k/video_frame_indexes.npz"

SEED404_DIR = EXP_DIR / "kpms_tandem_cohort_continuous_calibrated"
MODEL_NAME = "tandem_calibrated_50ar_200slds_seed404_20261002_012237_940073"
SEED404_RES = SEED404_DIR / MODEL_NAME / "results.h5"
SEED404_VFI = SEED404_DIR / "video_frame_indexes.npz"

OUT_MD = VIZ_DIR / "SEED404_TO_K240K_PRELABELING_CROSSWALK.md"
OUT_JSON = VIZ_DIR / "seed404_prelabeled_annotations.json"
OUT_CSV = VIZ_DIR / "seed404_prelabeled_annotations.csv"

# User's manual annotations for k240k
USER_LABELS = {
    0: {"name": "run right", "notes": "typically from a lean/rear", "meta": ""},
    1: {"name": "walk right", "notes": "more in place, less 'runny'", "meta": ""},
    2: {"name": "turn right", "notes": "mostly head turn right", "meta": "0,1,2 can be treated as turn right under relaxed definitions"},
    3: {"name": "walk left", "notes": "", "meta": ""},
    4: {"name": "back down + turn left", "notes": "really good example for presentation", "meta": "can also be treated as turn left"},
    5: {"name": "look right", "notes": "", "meta": "look < turn < walk < run"},
    6: {"name": "turn left", "notes": "", "meta": ""},
    7: {"name": "back down + turn right", "notes": "", "meta": ""},
    8: {"name": "sniff", "notes": "slight left", "meta": ""},
    9: {"name": "sniff", "notes": "slight right", "meta": ""},
    10: {"name": "sniff", "notes": "more of an elongated middle sniff, almost like its looking ahead", "meta": ""},
    11: {"name": "walk right", "notes": "", "meta": ""},
    12: {"name": "hunched rear", "notes": "small chance of scrunch, example of the problems with 2d", "meta": ""},
    13: {"name": "dash", "notes": "", "meta": "again strethcing can cheat kpms in few cases"},
    14: {"name": "rear/lean", "notes": "exempler is rear but looks rear-lean-bimodal w/out density", "meta": ""},
    15: {"name": "sniff upwards", "notes": "many shorter 'bunk' instances", "meta": ""},
    16: {"name": "lean left", "notes": "exempler is lean left, non exempler is mostly also leans", "meta": ""},
    17: {"name": "run left", "notes": "", "meta": ""},
    18: {"name": "sniff", "notes": "seems like there are many outliers but main behaviour is sniff", "meta": ""},
    19: {"name": "walk", "notes": "", "meta": ""},
    20: {"name": "sniff", "notes": "slight right head-turn", "meta": ""},
    21: {"name": "dash", "notes": "mostly short dash", "meta": ""},
    22: {"name": "look right", "notes": "often very right-looking; often followed by groom sine it ends in looking at its hind", "meta": ""},
    23: {"name": "walk left", "notes": "", "meta": ""},
    24: {"name": "scrunched sniffig", "notes": "very low energy syll, sometimes stationary", "meta": ""},
    25: {"name": "sniff", "notes": "", "meta": ""},
    26: {"name": "dash left", "notes": "only slight left", "meta": ""},
    27: {"name": "groom", "notes": "", "meta": ""},
    28: {"name": "look left", "notes": "", "meta": ""},
    29: {"name": "walk left", "notes": "sharp left", "meta": ""},
    30: {"name": "walk right", "notes": "sharp right", "meta": ""},
    31: {"name": "rear/lean", "notes": "", "meta": ""},
    32: {"name": "stationary", "notes": "", "meta": ""},
    33: {"name": "dash", "notes": "sometimes really fast", "meta": ""},
    34: {"name": "look left (from right)", "notes": "very subtle", "meta": ""},
    35: {"name": "hand in frame", "notes": "", "meta": "low conf syll, look if properly coupled to low conf in kp data"},
    36: {"name": "sniff", "notes": "mostly left-looking, but stationary", "meta": ""},
    37: {"name": "stand down", "notes": "", "meta": ""},
    38: {"name": "walk", "notes": "can be many directions, mostly straight", "meta": ""},
    39: {"name": "sniff", "notes": "look right ish also; stationary", "meta": ""},
    40: {"name": "look right", "notes": "", "meta": "mostly; stuff is getting very misc'y down here"},
    41: {"name": "run + turn right", "notes": "", "meta": ""},
    42: {"name": "look right", "notes": "slight sniff", "meta": ""},
    43: {"name": "stationary", "notes": "", "meta": ""},
    44: {"name": "groom", "notes": "", "meta": ""},
    49: {"name": "stationary", "notes": "sometimes very slight sniff or groom", "meta": ""},
    52: {"name": "groom", "notes": "", "meta": ""},
    55: {"name": "walk left", "notes": "exemplar has stand down turn left but not general; mostly a slight left look while walking", "meta": ""},
    58: {"name": "groom", "notes": "", "meta": ""},
    60: {"name": "groom", "notes": "", "meta": ""},
    61: {"name": "groom", "notes": "", "meta": ""},
    63: {"name": "groom", "notes": "", "meta": ""},
    65: {"name": "groom", "notes": "", "meta": ""},
    66: {"name": "groom", "notes": "", "meta": "some hands, ok-ish compared to the other grooms"},
}

def main():
    t0 = time.time()
    print("=" * 80)
    print("🧠 SEED 404 TO K240K PRE-LABELING CROSSWALK ENGINE")
    print("=" * 80)

    pat = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})_([a-z]+_[a-z]+)")

    print("[1/5] Indexing session keys across models...", flush=True)
    with h5py.File(K240K_RES, "r") as fk, h5py.File(SEED404_RES, "r") as fs:
        map_k = {}
        for k in fk.keys():
            m = pat.search(k)
            if m:
                sid = f"{m.group(1)}_{m.group(2)}"
                map_k[sid] = k

        map_s = {}
        for k in fs.keys():
            m = pat.search(k)
            if m:
                sid = f"{m.group(1)}_{m.group(2)}"
                if sid not in map_s:
                    map_s[sid] = []
                map_s[sid].append(k)

    common_sids = sorted(list(set(map_k.keys()) & set(map_s.keys())))
    print(f"  Mapped {len(common_sids)} shared sessions.")

    print("\n[2/5] Loading VFI tables...", flush=True)
    with np.load(K240K_VFI) as zk:
        vfi_k_dict = {k: zk[k] for k in zk.files}
    with np.load(SEED404_VFI) as zs:
        vfi_s_dict = {k: zs[k] for k in zs.files}

    # Initialize contingency: counts[seed404_id, k240k_id]
    N_S = 100
    N_K = 100
    C = np.zeros((N_S, N_K), dtype=np.int64)

    # Kinematics accumulators for Seed 404
    syll_speeds = {s: [] for s in range(N_S)}
    syll_ang_vels = {s: [] for s in range(N_S)}
    syll_durs = {s: [] for s in range(N_S)}

    print("\n[3/5] Accumulating frame co-occurrence & kinematics across cohort...", flush=True)
    total_aligned_frames = 0

    with h5py.File(K240K_RES, "r") as fk, h5py.File(SEED404_RES, "r") as fs:
        for idx, sid in enumerate(common_sids):
            k_key = map_k[sid]
            if k_key not in vfi_k_dict or k_key not in fk:
                continue

            vfi_k = vfi_k_dict[k_key]
            sylls_k = fk[k_key]["syllable"][:]

            # Map raw frame -> k240k syllable
            # Pre-allocate lookup array or dict
            raw_to_k = {int(f): int(s) for f, s in zip(vfi_k, sylls_k)}

            # Process all Seed 404 segments for this session
            for sk in sorted(map_s[sid]):
                if sk not in vfi_s_dict or sk not in fs:
                    continue

                vfi_s = vfi_s_dict[sk]
                sylls_s = fs[sk]["syllable"][:]
                centroid_s = fs[sk]["centroid"][:]
                heading_s = fs[sk]["heading"][:]

                # Compute kinematics
                dx = np.diff(centroid_s[:, 0])
                dy = np.diff(centroid_s[:, 1])
                speed = np.pad(np.sqrt(dx**2 + dy**2) * 60.0, (0, 1), mode="edge")
                dh = np.diff(np.unwrap(heading_s))
                ang_vel = np.pad(dh * 60.0, (0, 1), mode="edge")

                # Bout durations
                diffs = np.diff(sylls_s)
                bounds = np.where(diffs != 0)[0] + 1
                starts = np.r_[0, bounds]
                ends = np.r_[bounds, len(sylls_s)]
                for st, en in zip(starts, ends):
                    s_id = int(sylls_s[st])
                    if 0 <= s_id < N_S:
                        syll_durs[s_id].append(en - st)

                # Frame co-occurrence matching
                for f_s, s_val, sp, av in zip(vfi_s, sylls_s, speed, ang_vel):
                    f_int = int(f_s)
                    s_int = int(s_val)
                    if 0 <= s_int < N_S:
                        syll_speeds[s_int].append(sp)
                        syll_ang_vels[s_int].append(av)

                        if f_int in raw_to_k:
                            k_val = raw_to_k[f_int]
                            if 0 <= k_val < N_K:
                                C[s_int, k_val] += 1
                                total_aligned_frames += 1

            if (idx + 1) % 25 == 0 or idx == len(common_sids) - 1:
                print(f"  Processed {idx+1}/{len(common_sids)} sessions ({total_aligned_frames:,} frames aligned)...", flush=True)

    print("\n[4/5] Computing conditional overlap probabilities & kinematic metrics...", flush=True)
    row_sums = C.sum(axis=1) # Total Seed 404 frames per syllable
    col_sums = C.sum(axis=0) # Total k240k frames per syllable

    prelabel_results = []
    annotations_export = []

    for s in range(N_S):
        total_s = row_sums[s]
        med_spd = float(np.median(syll_speeds[s])) if syll_speeds[s] else 0.0
        mean_ang = float(np.mean(syll_ang_vels[s])) if syll_ang_vels[s] else 0.0
        med_dur = float(np.median(syll_durs[s])) if syll_durs[s] else 0.0

        if total_s == 0:
            top_matches = []
        else:
            probs = C[s] / float(total_s)
            best_k_indices = np.argsort(probs)[::-1]
            top_matches = [(k, probs[k], C[s, k]) for k in best_k_indices[:3] if probs[k] > 0.05]

        # Determine Primary and Secondary Matches
        if top_matches:
            k1, p1, c1 = top_matches[0]
            k1_info = USER_LABELS.get(k1, {"name": f"unlabeled_{k1}", "notes": "", "meta": ""})
            k1_name = k1_info["name"]
            k1_notes = k1_info["notes"]
            k1_meta = k1_info["meta"]
            p1_pct = p1 * 100.0

            if len(top_matches) > 1:
                k2, p2, c2 = top_matches[1]
                k2_info = USER_LABELS.get(k2, {"name": f"unlabeled_{k2}", "notes": "", "meta": ""})
                k2_str = f"k240k Syll {k2:02d} ({k2_info['name']}, {p2*100:.1f}%)"
            else:
                k2_str = "None"

            # Determine confidence & category
            if p1 >= 0.40:
                conf = "HIGH"
                cat = "DIRECT MATCH"
                suggested_name = k1_name
            elif p1 >= 0.20:
                conf = "MEDIUM"
                cat = "SPLIT / MULTI-VIEW SUBSTATE"
                suggested_name = f"{k1_name} (split)"
            else:
                conf = "LOW"
                cat = "DIFFUSE / NOVEL"
                suggested_name = f"{k1_name}?"
        else:
            k1, p1_pct = -1, 0.0
            k1_name, k1_notes, k1_meta = "novel/unaligned", "", ""
            k2_str = "None"
            conf, cat = "NONE", "NOVEL STATE"
            suggested_name = "novel state"

        # Kinematic descriptors
        turn_dir = "right turn" if mean_ang > 0.3 else ("left turn" if mean_ang < -0.3 else "straight/neutral")
        speed_desc = "stationary" if med_spd < 15.0 else ("slow" if med_spd < 60.0 else ("moderate" if med_spd < 130.0 else "fast/dash"))

        # Formulate synthesized notes
        syn_notes = []
        if k1 >= 0:
            syn_notes.append(f"TopCam match: k240k Syll {k1:02d} ({p1_pct:.1f}% overlap).")
            if k1_notes:
                syn_notes.append(f"Original notes: '{k1_notes}'.")
            if k1_meta:
                syn_notes.append(f"Metanotes: '{k1_meta}'.")
        if k2_str != "None":
            syn_notes.append(f"Secondary: {k2_str}.")
        syn_notes.append(f"Kinematics: {speed_desc} ({med_spd:.1f} px/s), {turn_dir} ({mean_ang:.2f} rad/s), median dur: {med_dur:.0f} frames.")
        full_notes = " ".join(syn_notes)

        prelabel_results.append({
            "seed404_id": s,
            "suggested_name": suggested_name,
            "category": cat,
            "confidence": conf,
            "primary_k240k_id": k1,
            "primary_k240k_name": k1_name,
            "primary_overlap_pct": p1_pct,
            "secondary_match": k2_str,
            "median_speed_px_s": med_spd,
            "mean_ang_vel_rad_s": mean_ang,
            "median_duration_frames": med_dur,
            "notes": full_notes,
        })

        annotations_export.append({
            "syllable_id": s,
            "taxonomy_name": f"[PRE-LABEL] {suggested_name}",
            "kinematic_notes": full_notes,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    print("\n[5/5] Writing reports and export files...", flush=True)
    df_prelabel = pd.DataFrame(prelabel_results)
    df_prelabel.to_csv(OUT_CSV, index=False)
    print(f"  ✅ Saved Prelabel CSV: {OUT_CSV}")

    # Build gallery importable JSON
    json_payload = {
        "model_name": MODEL_NAME,
        "eml_rank": 1,
        "eml_score": 225001430.22,
        "iterations": 250,
        "export_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_syllables": N_S,
        "prelabel_engine": "k240k_empirical_cooccurrence_v1",
        "annotations": annotations_export,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"  ✅ Saved Gallery Importable JSON: {OUT_JSON}")

    # Build Comprehensive Markdown Crosswalk
    md_lines = [
        "# Seed 404 to k240k Pre-Labeling Crosswalk & Behavioral Mapping",
        "",
        f"**Model:** `{MODEL_NAME}` (EML Champion Seed 404, Rank 1)  ",
        "**Reference Model:** `final_karen_model_k240k` (TopCam 17-keypoint baseline)  ",
        f"**Cohort Co-occurrence Coverage:** {total_aligned_frames:,} precisely aligned frames across 110 shared sessions.  ",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Confidence Breakdown",
        "",
        f"- **Total Syllables:** {N_S}",
        f"- **High Confidence Direct Matches (>=40% overlap):** {(df_prelabel['confidence'] == 'HIGH').sum()}",
        f"- **Medium Confidence Sub-States / Splits (20-40% overlap):** {(df_prelabel['confidence'] == 'MEDIUM').sum()}",
        f"- **Diffuse / Novel Multi-View States (<20% overlap):** {(df_prelabel['confidence'] == 'LOW').sum() + (df_prelabel['confidence'] == 'NONE').sum()}",
        "",
        "---",
        "",
        "## 2. Complete 100-Syllable Pre-Labeling Table",
        "",
        "| New Syll (Seed 404) | Suggested Taxonomy Name | Match Cat | Conf | Old Syll (k240k) | Overlap % | Secondary Overlap | Speed (px/s) | Ang Vel (rad/s) | Original Notes |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |",
    ]

    for _, r in df_prelabel.iterrows():
        k_str = f"Syll {r['primary_k240k_id']:02d} ({r['primary_k240k_name']})" if r['primary_k240k_id'] >= 0 else "None"
        md_lines.append(
            f"| **{r['seed404_id']:02d}** | **{r['suggested_name']}** | {r['category']} | {r['confidence']} | {k_str} | {r['primary_overlap_pct']:.1f}% | {r['secondary_match']} | {r['median_speed_px_s']:.1f} | {r['mean_ang_vel_rad_s']:.2f} | {r['notes'][:60]}... |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. How to Use in the Interactive Gallery",
        "",
        "1. Open [`syllable_labeling_gallery_seed404.html`](file://" + str(VIZ_DIR / "syllable_labeling_gallery_seed404.html") + ") in your browser.",
        "2. Click **'📂 Import Annotations'** in the top toolbar.",
        "3. Select [`seed404_prelabeled_annotations.json`](file://" + str(OUT_JSON) + ").",
        "4. All 100 syllable cards will instantly populate with `[PRE-LABEL] <name>` and synthesized kinematic/provenance notes.",
        "5. Review each card against the synchronized Top/Bot video and fine-tune!",
    ])

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  ✅ Saved Markdown Crosswalk: {OUT_MD}")
    print(f"\n🎉 CROSSWALK COMPUTATION COMPLETE in {time.time() - t0:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
