#!/usr/bin/env python3
"""
Rsync Bottom Camera Videos (BotCamVid) from N-drive to Local HDD for Karen Project.

Target Folders:
  - 3 week PI -> /mnt/hdd/karen_project/3_week_PI/
  - 8 weeks PI -> /mnt/hdd/karen_project/8_weeks_PI/
  - 24 weeks PI -> /mnt/hdd/karen_project/24_weeks_PI/
  - L-dopa+Benserazide -fiber photometry -> /mnt/hdd/karen_project/L_dopa/
"""

import argparse
import os
import subprocess

SRC_BASE = "/mnt/neuropharm/Common/Lab member folder/Jamila Lilja/Alpha-synuclein PFF mice/Behavior cohort 1/a-syn_Fiber photo"
DST_BASE = "/mnt/hdd/karen_project"

TIMEPOINTS = [
    ("3 week PI", "3_week_PI"),
    ("8 weeks PI", "8_weeks_PI"),
    ("24 weeks PI", "24_weeks_PI"),
    ("L-dopa+Benserazide -fiber photometry", "L_dopa"),
]

def run_rsync(dry_run=True, log_file=None):
    mode_str = "DRY RUN (no files will be transferred)" if dry_run else "REAL RUN (transferring files)"
    print("==================================================")
    print(f" Starting BotCam Rsync: {mode_str}")
    print(f" Source: {SRC_BASE}")
    print(f" Destination: {DST_BASE}")
    if log_file:
        print(f" Logging to: {log_file}")
    print("==================================================\n")

    total_files = 0
    total_bytes = 0

    log_handle = open(log_file, "a") if log_file else None

    for src_rel, dst_rel in TIMEPOINTS:
        src_path = os.path.join(SRC_BASE, src_rel) + "/"
        dst_path = os.path.join(DST_BASE, dst_rel) + "/"

        print(f"--> Processing [{src_rel}] -> [{dst_rel}]...")

        if not os.path.exists(src_path):
            print(f"    [ERROR] Source folder missing: {src_path}")
            continue

        os.makedirs(dst_path, exist_ok=True)

        cmd = [
            "rsync",
            "-avh",
            "--partial",
            "--ignore-existing",
            "--include=*/",
            "--include=*BotCamVid*.avi",
            "--exclude=*",
        ]

        if dry_run:
            cmd.append("--dry-run")
        else:
            cmd.append("--progress")

        cmd.extend([src_path, dst_path])

        if log_handle:
            log_handle.write(f"\n--- Rsync [{src_rel}] -> [{dst_rel}] ---\n")
            log_handle.write("CMD: " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
            log_handle.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        file_count = 0
        for line in process.stdout:
            if log_handle:
                log_handle.write(line)
                log_handle.flush()
            if line.strip().endswith(".avi") and "BotCamVid" in line:
                file_count += 1
                print(f"    [{'WOULD TRANSFER' if dry_run else 'TRANSFERRING'}] {line.strip()}")
            elif "sent " in line or "total size is" in line:
                print(f"    {line.strip()}")

        process.wait()
        total_files += file_count
        print(f"    Completed: {file_count} BotCam videos identified in {dst_rel}.\n")

    if log_handle:
        log_handle.close()

    print("==================================================")
    print(f" Rsync {mode_str} Finished!")
    print(f" Total BotCam Videos Processed across 4 Timepoints: {total_files}")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rsync Karen Project BotCam Videos")
    parser.add_argument("--run", action="store_true", help="Execute real rsync transfer (default is dry-run)")
    parser.add_argument("--log", type=str, default="/mnt/hdd/karen_project/rsync_botcam.log", help="Path to log file")
    args = parser.parse_args()

    run_rsync(dry_run=not args.run, log_file=args.log)
