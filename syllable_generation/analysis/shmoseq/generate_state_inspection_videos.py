#!/usr/bin/env python3
"""
generate_state_inspection_videos.py
=============================================================================
Extracts and renders informative behavioral video clips from raw session recordings
to inspect:
  1. Variation WITHIN prolonged macro-state episodes:
     - Beginning, Middle, and End of grey state (Z_5) in KI #1771
     - Beginning, Middle, and End of yellow state (Z_6) in WT #1765 and #1773
  2. Cross-animal comparison of the SAME state:
     - Orange state (Z_1) under amphetamine in WT #1767 vs. KI #1694

Overlays on each frame:
  - Animal ID | Genotype | Drug | Macro-State (Z_k)
  - Epoch (Beginning / Middle / End) | Elapsed Session Time (MM:SS)
  - Current Keypoint-MoSeq Syllable ID
  - Instantaneous Velocity (px/s)
  - Tracking Centroid Marker

Outputs:
  - Video clips in syllable_generation/analysis/shmoseq/output/videos/
  - Video audit summary CSV table
=============================================================================
"""

import os
import pickle
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output")
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
ARTIFACTS_DIR = "/home/fsp585/.gemini/antigravity-ide/brain/65380ca4-d043-4134-88d6-bf1bc271e579"
MODEL_FILE = os.path.join(OUTPUT_DIR, "models", "selected_exploratory_model.pkl")
RAW_DATA_PATH = os.path.join(BASE_DIR, "syllable_generation", "data", "jonathan_amph_sessions_v2_w_sylls.pkl")

# Target Clip Specs (Duration: 6.0s = 240 40Hz steps / 480 80Hz frames)
CLIP_DURATION_SEC = 6.0
FPS_40HZ = 40.0125
FPS_80HZ = 80.025
N_FRAMES_40HZ = int(CLIP_DURATION_SEC * FPS_40HZ)

# State Colors (BGR for OpenCV drawing)
STATE_COLORS_BGR = {
    0: (220, 100, 30),    # Blue / Dark Orange
    1: (0, 140, 255),    # Orange
    2: (60, 180, 75),    # Green
    3: (230, 50, 50),    # Red
    4: (180, 50, 180),   # Purple
    5: (128, 128, 128),  # Grey
    6: (0, 215, 255),    # Yellow
    7: (200, 130, 0),    # Cyan/Teal
}

TARGET_SPECS = [
    # 1. KI #1771 Saline (Z_5 Grey)
    {
        "session_idx": 9,
        "session_key": "1771_Saline",
        "animal_id": "1771",
        "genotype": "KI",
        "drug": "Saline",
        "target_state": 5,
        "state_name": "Z_5 (Grey)",
        "epoch": "beginning",
        "center_40hz_idx": 85000,
        "clip_name": "KI_1771_Saline_Z5_beginning.mp4",
        "description": "KI #1771 Saline - Z5 Beginning (Run start ~35m)"
    },
    {
        "session_idx": 9,
        "session_key": "1771_Saline",
        "animal_id": "1771",
        "genotype": "KI",
        "drug": "Saline",
        "target_state": 5,
        "state_name": "Z_5 (Grey)",
        "epoch": "middle",
        "center_40hz_idx": 110000,
        "clip_name": "KI_1771_Saline_Z5_middle.mp4",
        "description": "KI #1771 Saline - Z5 Middle (Run midpoint ~45m)"
    },
    {
        "session_idx": 9,
        "session_key": "1771_Saline",
        "animal_id": "1771",
        "genotype": "KI",
        "drug": "Saline",
        "target_state": 5,
        "state_name": "Z_5 (Grey)",
        "epoch": "end",
        "center_40hz_idx": 135000,
        "clip_name": "KI_1771_Saline_Z5_end.mp4",
        "description": "KI #1771 Saline - Z5 End (Run conclusion ~56m)"
    },

    # 2. KI #1771 Amphetamine (Z_5 Grey - Long 46.8m run)
    {
        "session_idx": 24,
        "session_key": "1771_Amphetamine",
        "animal_id": "1771",
        "genotype": "KI",
        "drug": "Amphetamine",
        "target_state": 5,
        "state_name": "Z_5 (Grey)",
        "epoch": "beginning",
        "center_40hz_idx": 15000,
        "clip_name": "KI_1771_Amph_Z5_beginning.mp4",
        "description": "KI #1771 Amph - Z5 Beginning (~6m post-return)"
    },
    {
        "session_idx": 24,
        "session_key": "1771_Amphetamine",
        "animal_id": "1771",
        "genotype": "KI",
        "drug": "Amphetamine",
        "target_state": 5,
        "state_name": "Z_5 (Grey)",
        "epoch": "middle",
        "center_40hz_idx": 60000,
        "clip_name": "KI_1771_Amph_Z5_middle.mp4",
        "description": "KI #1771 Amph - Z5 Middle (~25m post-return)"
    },
    {
        "session_idx": 24,
        "session_key": "1771_Amphetamine",
        "animal_id": "1771",
        "genotype": "KI",
        "drug": "Amphetamine",
        "target_state": 5,
        "state_name": "Z_5 (Grey)",
        "epoch": "end",
        "center_40hz_idx": 115000,
        "clip_name": "KI_1771_Amph_Z5_end.mp4",
        "description": "KI #1771 Amph - Z5 End (~48m post-return)"
    },

    # 3. WT #1765 Saline (Z_6 Yellow - 51.5m run)
    {
        "session_idx": 10,
        "session_key": "1765_Saline",
        "animal_id": "1765",
        "genotype": "WT",
        "drug": "Saline",
        "target_state": 6,
        "state_name": "Z_6 (Yellow)",
        "epoch": "beginning",
        "center_40hz_idx": 25000,
        "clip_name": "WT_1765_Saline_Z6_beginning.mp4",
        "description": "WT #1765 Saline - Z6 Beginning (~10m post-return)"
    },
    {
        "session_idx": 10,
        "session_key": "1765_Saline",
        "animal_id": "1765",
        "genotype": "WT",
        "drug": "Saline",
        "target_state": 6,
        "state_name": "Z_6 (Yellow)",
        "epoch": "middle",
        "center_40hz_idx": 80000,
        "clip_name": "WT_1765_Saline_Z6_middle.mp4",
        "description": "WT #1765 Saline - Z6 Middle (~33m post-return)"
    },
    {
        "session_idx": 10,
        "session_key": "1765_Saline",
        "animal_id": "1765",
        "genotype": "WT",
        "drug": "Saline",
        "target_state": 6,
        "state_name": "Z_6 (Yellow)",
        "epoch": "end",
        "center_40hz_idx": 135000,
        "clip_name": "WT_1765_Saline_Z6_end.mp4",
        "description": "WT #1765 Saline - Z6 End (~56m post-return)"
    },

    # 4. WT #1773 Saline (Z_6 Yellow - 28.3m run)
    {
        "session_idx": 8,
        "session_key": "1773_Saline",
        "animal_id": "1773",
        "genotype": "WT",
        "drug": "Saline",
        "target_state": 6,
        "state_name": "Z_6 (Yellow)",
        "epoch": "beginning",
        "center_40hz_idx": 78000,
        "clip_name": "WT_1773_Saline_Z6_beginning.mp4",
        "description": "WT #1773 Saline - Z6 Beginning (~32m post-return)"
    },
    {
        "session_idx": 8,
        "session_key": "1773_Saline",
        "animal_id": "1773",
        "genotype": "WT",
        "drug": "Saline",
        "target_state": 6,
        "state_name": "Z_6 (Yellow)",
        "epoch": "middle",
        "center_40hz_idx": 108000,
        "clip_name": "WT_1773_Saline_Z6_middle.mp4",
        "description": "WT #1773 Saline - Z6 Middle (~45m post-return)"
    },
    {
        "session_idx": 8,
        "session_key": "1773_Saline",
        "animal_id": "1773",
        "genotype": "WT",
        "drug": "Saline",
        "target_state": 6,
        "state_name": "Z_6 (Yellow)",
        "epoch": "end",
        "center_40hz_idx": 138000,
        "clip_name": "WT_1773_Saline_Z6_end.mp4",
        "description": "WT #1773 Saline - Z6 End (~57m post-return)"
    },

    # 5. Cross-Animal Orange State (Z_1) under Amphetamine: WT #1767 vs. KI #1694
    {
        "session_idx": 27,
        "session_key": "1767_Amphetamine",
        "animal_id": "1767",
        "genotype": "WT",
        "drug": "Amphetamine",
        "target_state": 1,
        "state_name": "Z_1 (Orange)",
        "epoch": "representative",
        "center_40hz_idx": 68000,
        "clip_name": "WT_1767_Amph_Z1_representative.mp4",
        "description": "WT #1767 Amph - Z1 (~28m post-return)"
    },
    {
        "session_idx": 15,
        "session_key": "1694_Amphetamine",
        "animal_id": "1694",
        "genotype": "KI",
        "drug": "Amphetamine",
        "target_state": 1,
        "state_name": "Z_1 (Orange)",
        "epoch": "representative",
        "center_40hz_idx": 112000,
        "clip_name": "KI_1694_Amph_Z1_representative.mp4",
        "description": "KI #1694 Amph - Z1 (~46m post-return)"
    }
]

def find_avi_path(sess):
    ch = sess.metadata.get("chamber")
    phot_path = sess.metadata.get("photometry_csv_path")
    day_dir = os.path.dirname(phot_path)
    candidates = [os.path.join(day_dir, f) for f in os.listdir(day_dir) if f.endswith(".avi") and ch in f]
    if candidates:
        return candidates[0]
    return None

def compute_entropy(labels):
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def render_clip(spec, sess, states_seq, avi_path):
    clip_filename = spec["clip_name"]
    raw_mp4_path = os.path.join(VIDEOS_DIR, f"temp_{clip_filename}")
    final_mp4_path = os.path.join(VIDEOS_DIR, clip_filename)

    c_idx = spec["center_40hz_idx"]
    start_40hz = max(0, c_idx - N_FRAMES_40HZ // 2)
    end_40hz = min(len(states_seq), start_40hz + N_FRAMES_40HZ)

    df_syll = sess.syllables
    # Sliced post-return valid frames
    det_ret_f = spec.get("det_ret_f", 0)
    post_df = df_syll.iloc[det_ret_f:] if det_ret_f < len(df_syll) else df_syll
    valid_post = post_df[post_df["syllable"].notna()]

    clip_valid_df = valid_post.iloc[start_40hz:end_40hz]
    clip_80hz_indices = clip_valid_df.index.values
    clip_sylls = clip_valid_df["syllable"].values.astype(int)
    clip_states = states_seq[start_40hz:end_40hz]

    # Kinematics
    c0 = clip_valid_df["centroid_0"].values if "centroid_0" in clip_valid_df.columns else np.zeros(len(clip_valid_df))
    c1 = clip_valid_df["centroid_1"].values if "centroid_1" in clip_valid_df.columns else np.zeros(len(clip_valid_df))
    dx = np.diff(c0, prepend=c0[0])
    dy = np.diff(c1, prepend=c1[0])
    velocities = np.hypot(dx, dy) * FPS_40HZ  # px / sec

    # Behavioral statistics
    mean_speed = float(np.mean(velocities))
    median_speed = float(np.median(velocities))
    entropy = compute_entropy(clip_sylls)
    counts = pd.Series(clip_sylls).value_counts()
    top_syll = int(counts.index[0])
    top_syll_pct = float(counts.iloc[0] / len(clip_sylls) * 100.0)
    top3_str = ", ".join([f"S{s} ({cnt/len(clip_sylls)*100:.0f}%)" for s, cnt in counts.head(3).items()])

    # Open Video Reader
    cap = cv2.VideoCapture(avi_path)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Canvas with top banner
    banner_h = 70
    out_w = in_w
    out_h = in_h + banner_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_mp4_path, fourcc, 40.0, (out_w, out_h))

    for frame_i, f_80hz in enumerate(clip_80hz_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_80hz))
        ret, frame = cap.read()
        if not ret:
            # Create black frame fallback
            frame = np.zeros((in_h, in_w, 3), dtype=np.uint8)

        # Draw centroid marker
        cx = int(c0[frame_i])
        cy = int(c1[frame_i])
        if 0 <= cx < in_w and 0 <= cy < in_h:
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)

        # Create output canvas
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[banner_h:, :] = frame

        # Banner background
        cv2.rectangle(canvas, (0, 0), (out_w, banner_h), (25, 25, 25), -1)
        state_col = STATE_COLORS_BGR.get(spec["target_state"], (200, 200, 200))
        cv2.rectangle(canvas, (0, banner_h - 4), (out_w, banner_h), state_col, -1)

        # Text line 1: Mouse & State
        line1 = f"{spec['animal_id']} ({spec['genotype']} {spec['drug']}) | State {spec['state_name']}"
        cv2.putText(canvas, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Text line 2: Epoch & Time & Syllable & Speed
        time_min = (start_40hz + frame_i) / FPS_40HZ / 60.0
        syll_cur = clip_sylls[frame_i]
        speed_cur = velocities[frame_i]
        line2 = f"Epoch: {spec['epoch'].title()} ({time_min:.1f}m) | Syll: #{syll_cur} | Spd: {speed_cur:.0f} px/s"
        cv2.putText(canvas, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 230, 255), 1, cv2.LINE_AA)

        out.write(canvas)

    cap.release()
    out.release()

    # Convert to web-standard H.264 using ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", raw_mp4_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
        final_mp4_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(raw_mp4_path):
        os.remove(raw_mp4_path)

    # Also copy to artifacts directory
    art_path = os.path.join(ARTIFACTS_DIR, clip_filename)
    subprocess.run(["cp", final_mp4_path, art_path], check=True)

    stat = {
        "clip_filename": clip_filename,
        "animal_id": spec["animal_id"],
        "genotype": spec["genotype"],
        "drug": spec["drug"],
        "state_id": spec["target_state"],
        "state_name": spec["state_name"],
        "epoch": spec["epoch"],
        "session_time_min": float(np.mean([(start_40hz)/FPS_40HZ/60.0, (end_40hz)/FPS_40HZ/60.0])),
        "mean_speed_px_s": mean_speed,
        "median_speed_px_s": median_speed,
        "dominant_syllable": top_syll,
        "dominant_syllable_pct": top_syll_pct,
        "syllable_entropy_bits": entropy,
        "top3_syllables": top3_str,
        "avi_source": os.path.basename(avi_path)
    }
    return stat

def main():
    print("=" * 80)
    print("GENERATING SHMOSEQ STATE EPISODE INSPECTION VIDEOS & BEHAVIORAL PROFILING")
    print("=" * 80)

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    with open(RAW_DATA_PATH, "rb") as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} raw sessions.")

    with open(MODEL_FILE, "rb") as f:
        model_bundle = pickle.load(f)
    states_dict = model_bundle["states_dict"]
    meta_dict = model_bundle["metadata_dict"]

    audit_records = []

    for i, spec in enumerate(TARGET_SPECS):
        s_idx = spec["session_idx"]
        sess = sessions[s_idx]
        s_key = spec["session_key"]
        states_seq = states_dict[s_key]
        spec["det_ret_f"] = meta_dict[s_key]["detected_return_frame"]

        avi_path = find_avi_path(sess)
        if avi_path is None or not os.path.exists(avi_path):
            print(f"Error: AVI video not found for {spec['clip_name']}")
            continue

        print(f"\n[{i+1}/{len(TARGET_SPECS)}] Rendering: {spec['clip_name']}")
        print(f"  Animal: {spec['animal_id']} ({spec['genotype']} {spec['drug']}) | State: {spec['state_name']} | Epoch: {spec['epoch']}")
        print(f"  Source AVI: {avi_path}")

        stat = render_clip(spec, sess, states_seq, avi_path)
        audit_records.append(stat)

        print(f"  -> Generated: {spec['clip_name']} | Speed: {stat['mean_speed_px_s']:.1f} px/s | "
              f"Top Syllable: #{stat['dominant_syllable']} ({stat['dominant_syllable_pct']:.1f}%) | "
              f"Entropy: {stat['syllable_entropy_bits']:.2f} bits | Top 3: {stat['top3_syllables']}")

    df_audit = pd.DataFrame(audit_records)
    csv_path = os.path.join(OUTPUT_DIR, "tables", "state_episode_video_audit_table.csv")
    df_audit.to_csv(csv_path, index=False)
    print("\n" + "=" * 80)
    print(f"Saved complete video audit summary to: {csv_path}")
    print(f"All {len(audit_records)} clips generated and copied to artifacts directory.")
    print("=" * 80)

if __name__ == "__main__":
    main()
