import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from fpcode.common.notify import send_ntfy
from fpcode.common.paths import REPO_ROOT


def send_notification(msg: str):
    send_ntfy(msg)


def is_limited_sweep_finished(log_path) -> bool:
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
    parser = argparse.ArgumentParser(description="Launch an explicitly selected sweep after its predecessor finishes.")
    parser.add_argument("--limited-log", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if not args.script.is_file():
        parser.error(f"Sweep script does not exist: {args.script}")
    print("[auto-launcher] Monitoring run_limited_sweep.py. Will launch run_extended_sweep.py upon completion...")

    while True:
        if is_limited_sweep_finished(args.limited_log):
            print("[auto-launcher] Confirmed: run_limited_sweep.py output completed! Launching run_extended_sweep.py now...")
            break
        time.sleep(30)

    print("[auto-launcher] run_limited_sweep.py has finished! Launching run_extended_sweep.py now...")
    send_notification("Limited sweep finished! Automatically launching Extended Strategy Sweep now.")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        subprocess.run([args.python, "-u", str(args.script.resolve()), "--workers", "1"],
                       cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    print("[auto-launcher] run_extended_sweep.py completed successfully!")
    send_notification("Extended Strategy Sweep has completed successfully!")

if __name__ == "__main__":
    main()
