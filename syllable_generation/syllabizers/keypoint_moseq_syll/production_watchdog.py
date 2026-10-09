#!/usr/bin/env python3
"""
production_watchdog.py

Active resource & OOM watchdog for the Keypoint-MoSeq production batch queue.
Monitors:
  - System Host RAM (Available < 4GB trigger, or Process RSS > 45GB)
  - GPU VRAM (RTX 4090 memory usage > 23.5GB trigger)
  - Disk Space (Free on / < 40GB trigger)
Sends real-time alerts to ntfy.sh/woody_videos_seb.
Terminates gracefully when target parent queue PID exits.
"""

import os
import sys
import time
import argparse
import subprocess
import urllib.request
import psutil

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(title: str, msg: str, priority: str = "default", tags: str = "robot"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode("utf-8"),
            headers={
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": priority,
                "Tags": tags,
            }
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[watchdog] Notification error: {e}", flush=True)

def get_gpu_memory_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out.split("\n")[0])
    except Exception:
        return 0.0

def get_disk_free_gb(path="/"):
    try:
        usage = psutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return 999.0

def main():
    parser = argparse.ArgumentParser(description="KPMS Production Resource & OOM Watchdog")
    parser.add_argument("--parent-pid", type=int, default=None, help="PID of the queue script to monitor")
    parser.add_argument("--check-interval", type=float, default=5.0, help="Check interval in seconds")
    parser.add_argument("--log-heartbeat-every", type=int, default=12, help="Log status every N checks (e.g. 12 * 5s = 60s)")
    args = parser.parse_args()

    parent_pid = args.parent_pid
    print("=" * 70, flush=True)
    print("  KPMS PRODUCTION RESOURCE & OOM WATCHDOG STARTED", flush=True)
    print(f"  Parent Queue PID:    {parent_pid if parent_pid else 'System-wide'}", flush=True)
    print(f"  Check Interval:      {args.check_interval}s", flush=True)
    print(f"  Host RAM Emergency:  Avail < 4.0 GB or Proc RSS > 45.0 GB", flush=True)
    print(f"  GPU VRAM Emergency:  > 23,500 MB (RTX 4090)", flush=True)
    print(f"  Disk Free Emergency: < 40.0 GB", flush=True)
    print("=" * 70, flush=True)

    check_count = 0
    warned_ram = False
    warned_vram = False
    warned_disk = False

    while True:
        try:
            # 1. Check parent liveness
            if parent_pid is not None and not psutil.pid_exists(parent_pid):
                print(f"[watchdog] Parent PID {parent_pid} has terminated. Watchdog exiting cleanly.", flush=True)
                break

            # 2. System Memory Checks
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)

            # Process RSS check for children of parent_pid or kpms python processes
            max_proc_rss_gb = 0.0
            culprit_pid = None
            for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    cmd = " ".join(p.info['cmdline'] or [])
                    if "run_amph_production" in cmd or "kpms" in cmd:
                        rss_gb = p.info['memory_info'].rss / (1024 ** 3)
                        if rss_gb > max_proc_rss_gb:
                            max_proc_rss_gb = rss_gb
                            culprit_pid = p.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # 3. GPU VRAM Check
            gpu_used_mb = get_gpu_memory_mb()

            # 4. Disk Space Check
            disk_free_gb = get_disk_free_gb("/")

            # Heartbeat logging every ~60s
            check_count += 1
            if check_count % args.log_heartbeat_every == 0:
                print(
                    f"[heartbeat {time.strftime('%H:%M:%S')}] "
                    f"RAM: {used_gb:.1f}/{total_gb:.1f} GB (Avail: {avail_gb:.1f} GB, Top Proc RSS: {max_proc_rss_gb:.1f} GB) | "
                    f"GPU VRAM: {gpu_used_mb:.0f} MB / 24564 MB | "
                    f"Disk Free: {disk_free_gb:.1f} GB",
                    flush=True
                )

            # EMERGENCY CONDITION: RAM Critical
            if avail_gb < 4.0 or max_proc_rss_gb > 45.0:
                msg = (
                    f"🚨 CRITICAL HOST RAM TRIGGERED: Avail RAM: {avail_gb:.2f} GB (Total: {total_gb:.1f} GB), "
                    f"Top Process RSS: {max_proc_rss_gb:.2f} GB (PID {culprit_pid}). Terminating runaway process!"
                )
                print(f"[watchdog] {msg}", flush=True)
                send_notification("🚨 OOM WATCHDOG KILL (HOST RAM)", msg, priority="urgent", tags="rotating_light,skull")
                if culprit_pid:
                    try:
                        os.kill(culprit_pid, 9)
                    except Exception:
                        pass
                subprocess.run("pkill -9 -f 'run_amph_production'", shell=True)
                time.sleep(10)

            # EMERGENCY CONDITION: GPU VRAM Critical (> 23.5 GB)
            if gpu_used_mb > 23500:
                msg = f"🚨 CRITICAL GPU VRAM TRIGGERED: {gpu_used_mb:.0f} MB used on RTX 4090! Approaching CUDA OOM."
                print(f"[watchdog] {msg}", flush=True)
                send_notification("🚨 OOM WATCHDOG KILL (GPU VRAM)", msg, priority="urgent", tags="rotating_light,fire")
                if culprit_pid:
                    try:
                        os.kill(culprit_pid, 9)
                    except Exception:
                        pass
                subprocess.run("pkill -9 -f 'run_amph_production'", shell=True)
                time.sleep(10)

            # EMERGENCY CONDITION: Disk Space Critical (< 40 GB)
            if disk_free_gb < 40.0:
                msg = f"🚨 CRITICAL DISK SPACE TRIGGERED: Only {disk_free_gb:.1f} GB free on /! Halting queue."
                print(f"[watchdog] {msg}", flush=True)
                send_notification("🚨 DISK SAFEGUARD HALT", msg, priority="urgent", tags="warning,floppy_disk")
                if parent_pid:
                    try:
                        os.kill(parent_pid, 15)
                    except Exception:
                        pass
                subprocess.run("pkill -9 -f 'run_amph_production'", shell=True)
                time.sleep(10)

        except Exception as ex:
            print(f"[watchdog] Exception in loop: {ex}", flush=True)

        time.sleep(args.check_interval)

    print("[watchdog] Exiting cleanly.", flush=True)

if __name__ == "__main__":
    main()
