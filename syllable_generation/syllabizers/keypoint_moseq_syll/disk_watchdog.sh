#!/bin/bash
# Disk space watchdog — fires ntfy alerts at < 100 GB and < 50 GB free
# Runs every 30 minutes via a sleep loop; kill with pkill -f disk_watchdog

MOUNT="/"
NTFY_URL="https://ntfy.sh/woody_videos_seb"
ALERTED_100=false
ALERTED_50=false

echo "[watchdog] Disk watchdog started. Monitoring $MOUNT every 30 min."

while true; do
    FREE_GB=$(df -BG "$MOUNT" | awk 'NR==2 {gsub("G",""); print $4}')
    echo "[watchdog] $(date '+%H:%M:%S') — Free: ${FREE_GB}G"

    if [ "$FREE_GB" -lt 50 ] && [ "$ALERTED_50" = false ]; then
        MSG="CRITICAL: Only ${FREE_GB}G free on /. checkpoint.h5 writes may fail. Kill training or free space NOW."
        curl -s -d "$MSG" -H "Title: MoSeq Disk CRITICAL (<50GB)" "$NTFY_URL" > /dev/null
        echo "[watchdog] SENT CRITICAL alert (<50GB)"
        ALERTED_50=true

    elif [ "$FREE_GB" -lt 100 ] && [ "$ALERTED_100" = false ]; then
        MSG="WARNING: Only ${FREE_GB}G free on /. MoSeq training is running — monitor disk space."
        curl -s -d "$MSG" -H "Title: MoSeq Disk Warning (<100GB)" "$NTFY_URL" > /dev/null
        echo "[watchdog] SENT WARNING alert (<100GB)"
        ALERTED_100=true
    fi

    sleep 1800  # 30 minutes
done
