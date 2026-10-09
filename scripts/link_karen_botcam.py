#!/usr/bin/env python3
"""
Symlink Bottom Camera Videos (BotCamVid) from /mnt/hdd/karen_project
into /home/fsp585/Desktop/FiberphotometryCode/karen_project/botcam_video_files/

Matches each BotCam video against the corresponding TopCam in karen_project/video_files
to automatically determine if an animal was present in that chamber.
"""

import os
import sys
import argparse
import re
import glob
from pathlib import Path

HDD_BASE = Path("/mnt/hdd/karen_project")
TARGET_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/botcam_video_files")
TOPCAM_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/karen_project/video_files")

ARENA_TO_QUADRANTS = {
    1: ["top_left", "upper"],
    2: ["bottom_left", "lower"],
    3: ["bottom_right"],
    4: ["top_right"],
}

def get_arena_number(filename):
    match = re.search(r"Arena(\d)_BotCamVid", filename)
    if match:
        return int(match.group(1))
    return None

def extract_timestamp(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})", filename)
    if match:
        return match.group(1)
    return None

def find_all_botcam_videos():
    vids = []
    # 1. 16w root folders
    for entry in os.scandir(HDD_BASE):
        if entry.is_dir() and entry.name.startswith("T") and not entry.name.endswith("_split"):
            for f in os.scandir(entry.path):
                if f.name.endswith(".avi") and "botcamvid" in f.name.lower():
                    vids.append(("16w_PI", Path(f.path)))
    # 2. subfolders
    for wave in ["3_week_PI", "8_weeks_PI", "24_weeks_PI", "L_dopa"]:
        wave_dir = HDD_BASE / wave
        if not wave_dir.exists():
            continue
        for root, _, files in os.walk(wave_dir):
            for fname in files:
                if fname.endswith(".avi") and "botcamvid" in fname.lower():
                    vids.append((wave, Path(root) / fname))
    return vids

def create_symlinks(dry_run=True, filter_empty=False):
    mode_str = "DRY RUN (no files linked)" if dry_run else "REAL RUN (creating symlinks)"
    print(f"==================================================")
    print(f" BotCam Symlinking: {mode_str}")
    print(f" Target Directory: {TARGET_DIR}")
    print(f" Filter Empty Chambers via TopCam match: {filter_empty}")
    print(f"==================================================\n")

    if not dry_run:
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all topcam filenames for fast matching
    topcam_files = set(os.listdir(TOPCAM_DIR))
    print(f"Loaded {len(topcam_files)} verified TopCam videos for chamber verification.\n")

    videos = find_all_botcam_videos()
    print(f"Found {len(videos)} total BotCam videos on HDD.\n")

    total_linked = 0
    total_skipped = 0

    for wave, vid_path in sorted(videos, key=lambda x: str(x[1])):
        filename = vid_path.name
        arena_num = get_arena_number(filename)
        timestamp = extract_timestamp(filename)

        # Check if matching TopCam exists in video_files
        is_mouse_present = False
        matching_topcam = None
        if arena_num and timestamp and arena_num in ARENA_TO_QUADRANTS:
            for quad in ARENA_TO_QUADRANTS[arena_num]:
                # Look for matching file in topcam_files containing timestamp and quadrant
                for tf in topcam_files:
                    if timestamp in tf and f"_{quad}" in tf:
                        is_mouse_present = True
                        matching_topcam = tf
                        break
                if is_mouse_present:
                    break

        if filter_empty and not is_mouse_present:
            total_skipped += 1
            continue

        parent_folder = vid_path.parent.name
        if re.match(r"^T\d+$", parent_folder) and vid_path.parent.parent != HDD_BASE:
            parent_folder = vid_path.parent.parent.name

        status_tag = "active" if is_mouse_present else "empty_chamber"
        link_name = f"{wave}_{parent_folder}_{status_tag}_{filename}"
        symlink_dest = TARGET_DIR / link_name

        total_linked += 1
        if dry_run:
            if total_linked <= 8 or total_linked % 25 == 0:
                print(f"  [LINK] {link_name} (mouse: {is_mouse_present})")
        else:
            if symlink_dest.is_symlink() or symlink_dest.exists():
                symlink_dest.unlink()
            symlink_dest.symlink_to(vid_path)

    print(f"\n==================================================")
    print(f" Finished {mode_str}!")
    print(f" Total BotCam on HDD: {len(videos)}")
    print(f" Total Linked: {total_linked}")
    if filter_empty:
        print(f" Total Skipped (Empty Chambers): {total_skipped}")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symlink Karen Project BotCam Videos")
    parser.add_argument("--run", action="store_true", help="Execute real symlinking (default is dry-run)")
    parser.add_argument("--filter-empty", action="store_true", help="Only link chambers with verified mice")
    args = parser.parse_args()

    create_symlinks(dry_run=not args.run, filter_empty=args.filter_empty)
