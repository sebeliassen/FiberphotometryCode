import argparse
import glob
import os


def relink_videos(project_name, dry_run=True, no_dummies=False):
    # Detect the correct HDD project path
    if os.path.exists(f"/mnt/hdd/{project_name}"):
        HDD_BASE = f"/mnt/hdd/{project_name}"
    else:
        HDD_BASE = f"/mnt/hdd/mono_inter/{project_name}"

    TARGET_BASE = f"/home/fsp585/Desktop/FiberphotometryCode/studies/mono_inter_juvenile_intruder/{project_name}"

    SPLIT_DIR = os.path.join(HDD_BASE, "all_videos_split")
    VIDEO_DIR = os.path.join(TARGET_BASE, "video_files")
    H5_DIR = os.path.join(TARGET_BASE, "h5_files")

    print(f"{'--- DRY RUN ---' if dry_run else '--- RUNNING ---'} on {project_name}")

    if not os.path.exists(SPLIT_DIR):
        print(f"ERROR: Split dir {SPLIT_DIR} does not exist.")
        return

    # Automatically create missing folders
    for d in [TARGET_BASE, VIDEO_DIR, H5_DIR]:
        if not os.path.exists(d):
            print(f"  [MKDIR] Creating directory: {d}")
            if not dry_run:
                os.makedirs(d, exist_ok=True)

    print("\n1. Deleting legacy files...")
    old_videos = glob.glob(os.path.join(VIDEO_DIR, "*.avi"))
    old_h5s = glob.glob(os.path.join(H5_DIR, "*_dummyDLC_resnet50.h5"))

    for v in old_videos:
        print(f"  [DELETE] {v}")
        if not dry_run:
            os.remove(v)

    for h in old_h5s:
        print(f"  [DELETE] {h}")
        if not dry_run:
            os.remove(h)

    print("\n2. Recreating symlinks and dummy h5s from all_videos_split...")
    # Find active sessions by looking for T* folders in HDD_BASE to gracefully strip session ID
    sessions = sorted([s for s in os.listdir(HDD_BASE) if s.startswith("T") and os.path.isdir(os.path.join(HDD_BASE, s))])

    split_videos = sorted([v for v in os.listdir(SPLIT_DIR) if v.endswith(".avi")])
    for video in split_videos:
        video_src = os.path.join(SPLIT_DIR, video)
        symlink_target = os.path.join(VIDEO_DIR, video)

        print(f"\nVideo: {video}")
        print(f"  [LNK] ln -s {video_src}\n        -> {symlink_target}")
        if not dry_run:
            if os.path.exists(symlink_target):
                os.remove(symlink_target)
            os.symlink(video_src, symlink_target)

        matching_session = "Unknown"
        # Match from longest session names to avoid matching prefixes incorrectly
        for s in sorted(sessions, key=len, reverse=True):
            if video.startswith(f"{s}_"):
                matching_session = s
                break

        video_base = os.path.splitext(video)[0]
        if matching_session != "Unknown":
            h5_base = video_base[len(matching_session)+1:]
        else:
            h5_base = video_base

        if not no_dummies:
            h5_name = f"{h5_base}_dummyDLC_resnet50.h5"
            h5_target = os.path.join(H5_DIR, h5_name)
            print(f"  [H5 ] touch {h5_target}")
            if not dry_run:
                with open(h5_target, 'a'):
                    os.utime(h5_target, None)

    print("\n--- COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and recreate split video symlinks for a photometry project.")
    parser.add_argument("project_name", help="Name of the project (e.g. jonathan_oft_baseline)")
    parser.add_argument("--run", action="store_true", help="Execute the destructive operations")
    parser.add_argument("--no-dummies", action="store_true", help="Skip creating dummy DLC h5 placeholders")
    args = parser.parse_args()

    relink_videos(args.project_name, dry_run=not args.run, no_dummies=args.no_dummies)
