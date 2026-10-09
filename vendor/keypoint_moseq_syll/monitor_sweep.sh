#!/bin/bash
NTFY_URL="https://ntfy.sh/woody_videos_seb"

# Wait 30 seconds initially to allow the sweep script to fully launch
sleep 30

echo "Monitor started. Checking process status every 20 minutes..."

while true; do
    if ! ps aux | grep -iE "sweep_batch_sizes|_oom_trial" | grep -v grep > /dev/null; then
        msg="The Keypoint MoSeq batch size sweep is no longer running. The process has finished."
        echo "$(date): Sweep finished. Sending notification."
        curl -H "Title: [Sweep Monitor] Finished" -d "$msg" "$NTFY_URL"
        break
    fi
    echo "$(date): Sweep is still active. Sleeping for 20 minutes..."
    sleep 1200
done
