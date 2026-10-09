#!/usr/bin/env python3
import os
import sys
import time
import psutil
import urllib.request
import subprocess

NTFY_URL = "https://ntfy.sh/woody_videos_seb"
RAM_THRESHOLD_PERCENT = 90.0  # 90% RAM utilization trigger (~113 GB of 125 GB)
CHECK_INTERVAL_SEC = 2  # Check every 2 seconds

def send_notification(title: str, msg: str, priority: str = "high"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": "warning,rotating_light"
            }
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[watchdog] Failed to send ntfy notification: {e}", flush=True)

def main():
    print(f"[watchdog] RAM Watchdog initialized. Monitoring system RAM every {CHECK_INTERVAL_SEC}s...", flush=True)
    print(f"[watchdog] Trigger threshold: >= {RAM_THRESHOLD_PERCENT}% RAM utilization.", flush=True)
    
    while True:
        try:
            mem = psutil.virtual_memory()
            used_percent = mem.percent
            
            if used_percent >= RAM_THRESHOLD_PERCENT:
                avail_gb = mem.available / (1024 ** 3)
                total_gb = mem.total / (1024 ** 3)
                used_gb = mem.used / (1024 ** 3)
                
                msg = (
                    f"CRITICAL EMERGENCY: System RAM utilization reached {used_percent:.1f}% "
                    f"({used_gb:.2f} GB used / {avail_gb:.2f} GB free of {total_gb:.2f} GB).\n"
                    f"Initiating pkill on Python processes immediately to prevent CPU OOM server crash!"
                )
                print(f"[watchdog] {msg}", flush=True)
                send_notification("RAM WATCHDOG EMERGENCY KILL", msg, priority="high")
                
                # Target python data processing and model scripts first
                subprocess.run("pkill -9 -f 'python.*(run_pipeline|run_limited_sweep|run_extended_sweep|keypoint_moseq|syllable_generation|lib)'", shell=True)
                time.sleep(2)
                # Fallback pkill for python processes (excluding watchdog itself)
                my_pid = os.getpid()
                subprocess.run(f"ps aux | grep python | grep -v 'ram_watchdog' | grep -v {my_pid} | awk '{{print $2}}' | xargs -r kill -9", shell=True)
                
                print("[watchdog] Killed Python processes. Waiting 10s for memory to clear before auto-resuming KPMS pipeline...", flush=True)
                time.sleep(10)
                
                # Relaunch ONLY the Keypoint MoSeq pipeline script
                kpms_launcher = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/launch_final_karen_model.sh"
                subprocess.Popen(f"nohup bash {kpms_launcher} > /dev/null 2>&1 &", shell=True)
                
                resume_msg = "OOM cleared. Automatically relaunched KPMS pipeline (launch_final_karen_model.sh) safely!"
                print(f"[watchdog] {resume_msg}", flush=True)
                send_notification("KPMS PIPELINE AUTO-RESUMED", resume_msg, priority="default")
                
                # Cooldown period before resuming monitoring loop
                time.sleep(30)
        except Exception as err:
            print(f"[watchdog] Error in monitoring loop: {err}", flush=True)
            
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
