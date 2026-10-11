#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time

from fpcode.common.notify import send_ntfy as _unified_send_ntfy


def parse_time_str(t_str):
    if not t_str:
        return 0
    parts = list(map(int, t_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def format_seconds(secs):
    if secs < 60:
        return f"{int(secs)}s"
    elif secs < 3600:
        m = int(secs // 60)
        s = int(secs % 60)
        return f"{m}m {s}s"
    else:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        return f"{h}h {m}m"



def send_ntfy(topic, message, title=None, tags=None, priority=3):
    return _unified_send_ntfy(
        title=title or "MoSeq Progress",
        msg=message,
        priority=str(priority),
        tags=tags or "robot",
        topic=topic,
    )

def main():
    parser = argparse.ArgumentParser(description="Monitor a log file and send updates via ntfy.sh")
    parser.add_argument("-l", "--log", required=True, help="Path to the log file to monitor")
    parser.add_argument("-t", "--topic", required=True, help="ntfy.sh topic name")
    parser.add_argument("--start-from-beginning", action="store_true", help="Start reading from the beginning of the file instead of the end")
    args = parser.parse_args()

    filepath = os.path.abspath(args.log)
    topic = args.topic

    print(f"Monitoring log: {filepath}")
    print(f"Sending notifications to ntfy.sh/{topic}")

    send_ntfy(topic, f"Started monitoring log file: {os.path.basename(filepath)}", title="Monitor Started", tags="loudspeaker", priority=3)

    # Patterns to match
    running_pattern = re.compile(r"\[(\d+/\d+)\] Running:\s*(\S+)")
    testing_pattern = re.compile(r"\[(\d+/\d+)\] Testing\s+(.*)")
    skipping_pattern = re.compile(r"\[(\d+/\d+)\] Skipping completed run:\s*(\S+)")
    fitting_pattern = re.compile(r"Fitting\s+(\w+)")
    status_pattern = re.compile(r"Status:\s*(Success|Failed).*")

    # Marginal estimation batch patterns
    batches_info_pattern = re.compile(r"Divided\s+(\d+)\s+uncomputed sessions into\s+(\d+)\s+batches")
    batch_start_pattern = re.compile(r"Estimating marginals for batch\s+(\d+)/(\d+)")

    # Progress bar with details
    progress_bar_pattern = re.compile(
        r"Applying model:\s*(\d+)%.*?(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)(?:<(\d+:\d+(?::\d+)?))?,\s*([\d\.]+)(?:s/it|it/s)\]"
    )
    # Fallback standard progress pattern
    progress_pattern = re.compile(r"Applying model:\s*(\d+)%")

    # State tracking to avoid notification spam
    last_running_run = None
    last_progress = -1
    last_progress_time = 0
    total_batches = 0
    current_batch = 0
    last_notified_iteration = -1

    # Open file and monitor
    # Wait until file exists
    while not os.path.exists(filepath):
        print(f"Waiting for log file to be created: {filepath}...")
        time.sleep(2)

    with open(filepath, "r", errors="ignore") as f:
        if not args.start_from_beginning:
            # Go to end of file
            f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if not line:
                time.sleep(1.0)
                continue

            line = line.strip()
            if not line:
                continue

            # Match rules
            # 1. Divided uncomputed sessions
            m = batches_info_pattern.search(line)
            if m:
                total_batches = int(m.group(2))
                msg = f"Started Gibbs sampling for {total_batches} sessions."
                print(msg)
                send_ntfy(topic, msg, title="Estimation Started", tags="hourglass", priority=3)
                continue

            # 2. Estimating marginals batch start
            m = batch_start_pattern.search(line)
            if m:
                current_batch = int(m.group(1))
                total_batches = int(m.group(2))
                last_notified_iteration = -1
                msg = f"Running Gibbs sampling on session {current_batch}/{total_batches}"
                print(msg)
                send_ntfy(topic, msg, title="Session Start", tags="arrows_counterclockwise", priority=3)
                continue

            # 3. Running next configuration
            m = running_pattern.search(line)
            if m:
                step, run_name = m.groups()
                if run_name != last_running_run:
                    last_running_run = run_name
                    last_progress = -1 # Reset progress bar tracking
                    msg = f"Step {step}: Running {run_name}"
                    print(msg)
                    send_ntfy(topic, msg, title="MoSeq Sweep Running", tags="runner", priority=3)
                continue

            m = testing_pattern.search(line)
            if m:
                step, run_details = m.groups()
                run_name = f"Testing {run_details.strip()}"
                if run_name != last_running_run:
                    last_running_run = run_name
                    last_progress = -1 # Reset progress bar tracking
                    msg = f"Step {step}: {run_name}"
                    print(msg)
                    send_ntfy(topic, msg, title="MoSeq Benchmark Running", tags="runner", priority=3)
                continue

            # 4. Skipping run
            m = skipping_pattern.search(line)
            if m:
                step, run_name = m.groups()
                msg = f"Step {step}: Skipping completed {run_name}"
                print(msg)
                send_ntfy(topic, msg, title="MoSeq Sweep Skip", priority=2)
                continue

            # 5. Fitting
            m = fitting_pattern.search(line)
            if m:
                model_name = m.group(1)
                msg = f"Fitting model: {model_name}"
                print(msg)
                send_ntfy(topic, msg, title="MoSeq Fitting", tags="tools", priority=3)
                continue

            # 6. Detailed progress bar (notifies every 50 epochs / iterations)
            m = progress_bar_pattern.search(line)
            if m:
                percent, current_it, total_it, elapsed_str, remaining_str, speed_str = m.groups()
                current_it = int(current_it)
                total_it = int(total_it)
                speed = float(speed_str)

                # Check if multiple of 50
                if current_it > 0 and current_it % 50 == 0 and current_it != last_notified_iteration:
                    last_notified_iteration = current_it
                    remaining_sec = parse_time_str(remaining_str)

                    if total_batches > 0 and current_batch > 0:
                        remaining_batches = total_batches - current_batch
                        time_per_batch = total_it * speed
                        remaining_all_sec = remaining_batches * time_per_batch + remaining_sec
                        all_eta_str = format_seconds(remaining_all_sec)
                    else:
                        all_eta_str = "Unknown"

                    session_eta_str = format_seconds(remaining_sec)

                    msg = (
                        f"Progress: {current_it}/{total_it} epochs ({percent}%)\n"
                        f"Session ETA: {session_eta_str}\n"
                        f"Total Sweep ETA: {all_eta_str} ({total_batches - current_batch} sessions left)"
                    )
                    print(msg)
                    send_ntfy(
                        topic,
                        msg,
                        title=f"Progress: Session {current_batch}/{total_batches}",
                        tags="chart_with_upwards_trend",
                        priority=3
                    )
                continue

            # 7. Fallback progress bar percent
            m = progress_pattern.search(line)
            if m:
                percent = int(m.group(1))
                current_time = time.time()
                if (percent in [25, 50, 75, 100] and percent != last_progress) or (current_time - last_progress_time > 60 and percent != last_progress):
                    last_progress = percent
                    last_progress_time = current_time
                    msg = f"Run {last_running_run or 'Active'}: {percent}% complete"
                    print(msg)
                    send_ntfy(topic, msg, title="Fitting Progress", tags="chart_with_upwards_trend", priority=2)
                continue

            # 5. Status line (Success/Failure)
            m = status_pattern.search(line)
            if m:
                msg = line
                print(msg)
                status_type = m.group(1)
                tags = "white_check_mark" if status_type == "Success" else "x"
                priority = 3 if status_type == "Success" else 4
                send_ntfy(topic, msg, title=f"Run {status_type}", tags=tags, priority=priority)
                continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting monitor.")
        sys.exit(0)
