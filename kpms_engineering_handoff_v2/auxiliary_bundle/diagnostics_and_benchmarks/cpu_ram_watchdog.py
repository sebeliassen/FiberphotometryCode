import os
import sys
import time
import psutil
import urllib.request
import signal
import subprocess

NTFY_URL = "https://ntfy.sh/woody_videos_seb"
RAM_THRESHOLD_PERCENT = 90.0

def send_ntfy(message: str, title: str = "RAM Watchdog"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=message.encode('utf-8'),
            headers={"Title": title.encode('utf-8')}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

def main():
    target_pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"[Watchdog] Active. Monitoring Available RAM (< 1.5GB trigger) and Process RSS (> 50GB trigger). Target PID: {target_pid}", flush=True)
    
    while True:
        try:
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024 ** 3)
            
            # Check target process RSS
            proc_rss_gb = 0.0
            if target_pid and psutil.pid_exists(target_pid):
                try:
                    proc_rss_gb = psutil.Process(target_pid).memory_info().rss / (1024 ** 3)
                except Exception:
                    pass
            
            # Trigger if truly < 1.5 GB available RAM on machine OR process exceeds 50 GB
            if avail_gb < 1.5 or proc_rss_gb > 50.0:
                msg = f"🚨 EMERGENCY: High memory detected! Avail RAM: {avail_gb:.1f} GB, Process RSS: {proc_rss_gb:.1f} GB! Terminating training process."
                print(msg, flush=True)
                send_ntfy(msg, title="🚨 RAM WATCHDOG TRIGGERED")
                
                if target_pid:
                    try:
                        os.kill(target_pid, signal.SIGKILL)
                    except Exception:
                        pass
                
                subprocess.run(["pkill", "-9", "-f", "run_full_karen_single_iter.py"])
                sys.exit(1)
            
            # Check if target process has finished
            if target_pid and not psutil.pid_exists(target_pid):
                print(f"[Watchdog] Target PID {target_pid} has exited. Watchdog shutting down cleanly.", flush=True)
                break
                
            time.sleep(1.0)
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(1.0)

if __name__ == "__main__":
    main()
