#!/bin/bash

# Configuration
LOG_FILE="logs/launch_optimized_flip.log"
NTFY_TOPIC="woody_videos_seb"

echo "Starting log monitor for $LOG_FILE"

# Make sure the file exists before trailing
touch $LOG_FILE

# Tail the log file and check for specific strings
tail -f -n +1 $LOG_FILE | while read -r line; do
    
    # ---------------------------------------------
    # PHASE MILESTONES
    # ---------------------------------------------
    
    if [[ "$line" == *"[init] Fresh start. Initializing model..."* ]]; then
        curl -d "Phase: SLDS Initialization started" "ntfy.sh/$NTFY_TOPIC" || true
    fi
    
    if [[ "$line" == *"[phase 1] Current iter"* ]]; then
        curl -d "Phase 1: AR Fit started" "ntfy.sh/$NTFY_TOPIC" || true
    fi

    if [[ "$line" == *"[phase 1] AR fit complete"* ]]; then
        curl -d "Phase 1: AR Fit complete" "ntfy.sh/$NTFY_TOPIC" || true
    fi

    if [[ "$line" == *"[phase 2] Starting SLDS fit"* ]]; then
        curl -d "Phase 2: SLDS Fit started" "ntfy.sh/$NTFY_TOPIC" || true
    fi
    
    # ---------------------------------------------
    # ITERATION PROGRESS MILESTONES
    # Note: Regex extracts the matches like 10/51 or 50/201 from tqdm bars
    # ---------------------------------------------
    
    # Example Phase 1 matches: 10/51, 20/51, etc.
    if [[ "$line" =~ ([0-9]+)/51\ \[ ]]; then
        ITER="${BASH_REMATCH[1]}"
        if [[ "$ITER" == "10" || "$ITER" == "20" || "$ITER" == "30" || "$ITER" == "40" || "$ITER" == "50" ]]; then
            curl -d "Phase 1 AR Fit Progress: $ITER / 51" "ntfy.sh/$NTFY_TOPIC" || true
        fi
    fi

    # Example Phase 2 matches: 10/201, 20/201, etc.
    if [[ "$line" =~ ([0-9]+)/201\ \[ ]]; then
        ITER="${BASH_REMATCH[1]}"
        # Notify every 20 iterations for the longer 200 iter process to avoid spamming
        if [[ "$ITER" == "20" || "$ITER" == "40" || "$ITER" == "60" || "$ITER" == "80" || "$ITER" == "100" || "$ITER" == "120" || "$ITER" == "140" || "$ITER" == "160" || "$ITER" == "180" || "$ITER" == "200" ]]; then
            curl -d "Phase 2 SLDS Fit Progress: $ITER / 200" "ntfy.sh/$NTFY_TOPIC" || true
        fi
    fi

    if [[ "$line" == *"FAILED: optimized_flip_activity_k100k"* || "$line" == *"SUCCESS: optimized_flip_activity_k100k"* ]]; then
        echo "Training Finished. Exiting monitor."
        exit 0
    fi

    if [[ "$line" == *"Saved model summary to:"* ]]; then
        curl -H "Priority: high" -H "Tags: tada,partying_face" -d "PROJECT FINISHED! Model completely saved." "ntfy.sh/$NTFY_TOPIC" || true
    fi
    
done
