#!/usr/bin/env python3
"""
nvidia_idle_watchdog.py
=============================================================================
Continuous GPU & Python Process Idle Watchdog for KPMS Production.

Monitors active NVIDIA GPU compute applications and running KPMS Python processes.
If NO Python compute process is active for 30 consecutive minutes:
Sends an alert to ntfy.sh/woody_videos_seb.
=============================================================================
"""

import argparse
import os
import subprocess
import time
from datetime import datetime

import psutil

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_ntfy(title: str, msg: str, priority: str = "high", tags: str = "warning"):
    # Silenced per user request: idle GPU watchdog notifications disabled
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Watchdog Log (ntfy disabled)] {title}: {msg}", flush=True)
    return

def is_python_gpu_active() -> bool:
    # 1. Check nvidia-smi compute apps
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        if out:
            for line in out.splitlines():
                if "python" in line.lower():
                    return True
    except Exception:
        pass

    # 2. Check psutil for active training scripts
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            if p.info['status'] == psutil.STATUS_ZOMBIE:
                continue
            cmd = " ".join(p.info['cmdline'] or [])
            if "run_tandem_production" in cmd or "test_2ar_2slds" in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return False

def main():
    parser = argparse.ArgumentParser(description="NVIDIA / Python GPU Idle Watchdog")
    parser.add_argument("--timeout-mins", type=float, default=30.0, help="Idle timeout in minutes before alert (default: 30)")
    parser.add_argument("--check-interval", type=float, default=30.0, help="Check interval in seconds (default: 30)")
    parser.add_argument("--alert-cooldown-mins", type=float, default=30.0, help="Cooldown between alerts in minutes")
    args = parser.parse_args()

    timeout_sec = args.timeout_mins * 60.0
    cooldown_sec = args.alert_cooldown_mins * 60.0

    print("=" * 70, flush=True)
    print("  NVIDIA / PYTHON GPU IDLE WATCHDOG ARMED", flush=True)
    print(f"  PID:                 {os.getpid()}", flush=True)
    print(f"  Idle Threshold:      {args.timeout_mins} minutes ({timeout_sec:.0f}s)", flush=True)
    print(f"  Check Interval:      {args.check_interval}s", flush=True)
    print(f"  Alert Cooldown:      {args.alert_cooldown_mins} minutes", flush=True)
    print(f"  Destination:         {NTFY_URL}", flush=True)
    print("=" * 70, flush=True)

    send_ntfy(
        "KPMS Watchdog Armed 🛡️",
        f"NVIDIA / Python Idle Watchdog active on {os.uname().nodename}.\nWill alert if GPU is idle for {args.timeout_mins:.0f} minutes.",
        priority="low",
        tags="shield"
    )

    idle_start_time = None
    last_alert_time = 0.0

    while True:
        try:
            active = is_python_gpu_active()
            now = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if active:
                if idle_start_time is not None:
                    idle_duration = (now - idle_start_time) / 60.0
                    print(f"[{now_str}] Python GPU process detected! Resetting idle timer (was idle for {idle_duration:.1f} min).", flush=True)
                idle_start_time = None
            else:
                if idle_start_time is None:
                    idle_start_time = now
                    print(f"[{now_str}] No active Python GPU process detected. Idle timer started.", flush=True)
                else:
                    elapsed = now - idle_start_time
                    elapsed_min = elapsed / 60.0
                    if elapsed >= timeout_sec and (now - last_alert_time) >= cooldown_sec:
                        msg = (
                            f"⚠️ No active Python/NVIDIA GPU process detected for {elapsed_min:.1f} minutes!\n"
                            f"Idle since: {datetime.fromtimestamp(idle_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Host: {os.uname().nodename}\n"
                            f"Please check if KPMS training has completed, crashed, or needs manual restart."
                        )
                        send_ntfy("⚠️ GPU Idle Alert: No Python/NVIDIA process for 30m!", msg, priority="urgent", tags="warning,skull")
                        last_alert_time = now

            time.sleep(args.check_interval)

        except KeyboardInterrupt:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Watchdog exiting cleanly via SIGINT.", flush=True)
            break
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Watchdog loop exception: {e}", flush=True)
            time.sleep(args.check_interval)

if __name__ == "__main__":
    main()
