import os
import re
import time

from fpcode.common.notify import send_ntfy
from fpcode.common.paths import REPO_ROOT

LOG_FILE = REPO_ROOT / "vendor/keypoint_moseq_syll/final_karen_model.log"

TARGET_ITERS = {57, 58, 59, 60}
notified_iters = set()


def send_notification(title: str, msg: str):
    send_ntfy(title, msg, priority="high", tags="bell,tada,brain")
    print(f"[iter-notifier] Sent ntfy alert: {msg}", flush=True)

def parse_log_for_iters():
    found_iters = set()
    if not os.path.exists(LOG_FILE):
        return found_iters

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()

        for line in lines[-300:]:
            # Match SLDS chunk start lines like "Starting SLDS fit from 50 to 60"
            m = re.search(r"Starting SLDS fit from (\d+) to (\d+)", line)
            if m:
                start_i = int(m.group(1))
                end_i = int(m.group(2))
                for target in TARGET_ITERS:
                    if start_i <= target <= end_i:
                        found_iters.add(target)

            # Match iteration progress log markers
            m_iter = re.search(r"Iteration (\d+)", line, re.IGNORECASE)
            if m_iter:
                val = int(m_iter.group(1))
                if val in TARGET_ITERS:
                    found_iters.add(val)
    except Exception as e:
        print(f"[iter-notifier] Log reading error: {e}", flush=True)

    return found_iters

def main():
    print(f"[iter-notifier] Monitoring final_karen_model.log strictly for target iterations: {sorted(list(TARGET_ITERS))}...", flush=True)

    while True:
        try:
            current_found = parse_log_for_iters()
            for iter_num in current_found:
                if iter_num not in notified_iters:
                    send_notification(
                        f"KPMS Iteration {iter_num} Reached!",
                        f"KPMS model final_karen_model is currently processing/reached Iteration {iter_num}."
                    )
                    notified_iters.add(iter_num)

            if TARGET_ITERS.issubset(notified_iters):
                print("[iter-notifier] All target iteration notifications sent! Exiting listener.", flush=True)
                break
        except Exception as err:
            print(f"[iter-notifier] Error: {err}", flush=True)

        time.sleep(30)

if __name__ == "__main__":
    main()
