#!/usr/bin/env bash
# [FORWARDER] Forwarder to pipelines/train.py
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/training/cohort_static_21kp_v2_seed101.yaml"
exec python "${REPO_ROOT}/pipelines/train.py" --config "$CONFIG" "$@"
