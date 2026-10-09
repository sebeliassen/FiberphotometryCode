import os
import subprocess
import time
import urllib.request

from fpcode.common.notify import send_ntfy
from fpcode.common.paths import REPO_ROOT


def send_notification(msg: str):
    send_ntfy(msg)


def is_limited_sweep_finished() -> bool:
    log_path = REPO_ROOT / "syllable_generation/analysis/language_models/limited_sweep.log"
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                content = f.read()
                if "Targeted Hyperparameter Sweep Completed" in content:
                    return True
        except Exception:
            pass
    return False

def is_limited_sweep_running() -> bool:
    try:
        res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in res.stdout.split("\n"):
            if "run_limited_sweep.py" in line and "grep" not in line and "launch_extended" not in line:
                return True
    except Exception as e:
        print(f"Error checking ps aux: {e}")
    return False

def main():
    print("[auto-launcher] Monitoring run_limited_sweep.py. Will launch run_extended_sweep.py upon completion...")

    while True:
        if is_limited_sweep_finished():
            print("[auto-launcher] Confirmed: run_limited_sweep.py output completed! Launching run_extended_sweep.py now...")
            break
        time.sleep(30)

    print("[auto-launcher] run_limited_sweep.py has finished! Launching run_extended_sweep.py now...")
    send_notification("Limited sweep finished! Automatically launching Extended Strategy Sweep now.")

    python_bin = "/home/fsp585/miniforge3/envs/kpms_gpu_cuml/bin/python"
    script_path = "syllable_generation/analysis/language_models/run_extended_sweep.py"
    log_path = "syllable_generation/analysis/language_models/extended_sweep.log"

    cmd = f"{python_bin} -u {script_path} --workers 1 > {log_path} 2>&1"
    subprocess.run(cmd, shell=True)

    print("[auto-launcher] run_extended_sweep.py completed successfully!")
    send_notification("Extended Strategy Sweep has completed successfully!")

if __name__ == "__main__":
    main()
