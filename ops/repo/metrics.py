#!/usr/bin/env python3
"""
ops/repo/metrics.py
===================
Calculates repository architecture and code health metrics, comparing the
current HEAD against the Phase-0 baseline (`pre-scaffolding-20261009`).

Metrics tracked:
- Total Python files (.py)
- Redundant / duplicate file copies
- Files with hardcoded /home/fsp585 paths
- Python files using sys.path hacks (append / insert)
- Number of duplicate send_ntfy definitions
- Production runners not forwarding to pipelines/
- Active backward-compatibility shims

Outputs results to: registry/METRICS.tsv
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_TAG = "pre-scaffolding-20261009"
METRICS_TSV = REPO_ROOT / "registry" / "METRICS.tsv"
SHIMS_TSV = REPO_ROOT / "registry" / "SHIMS.tsv"


def run_git(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, cwd=str(REPO_ROOT), text=True).strip()
    except Exception as e:
        return ""


def get_phase0_metrics() -> Dict[str, int]:
    """Computes or retrieves Phase-0 baseline metrics from git tag."""
    p0_files = [
        f for f in run_git(f"git ls-tree -r --name-only {BASELINE_TAG}").splitlines()
        if f.endswith(".py")
    ]
    if not p0_files:
        # Fallback to recorded baseline values if tag is inaccessible
        return {
            "python_files": 1609,
            "redundant_copies": 764,
            "files_with_home_path": 841,
            "sys_path_hacks": 792,
            "send_ntfy_defs": 86,
            "runners_not_forwarding": 12,
            "active_shims": 0,
        }

    p0_ntfy = 0
    p0_home = 0
    p0_syspath = 0
    hashes_p0 = defaultdict(list)

    for f in p0_files:
        try:
            content = run_git(f"git show {BASELINE_TAG}:{f}")
        except Exception:
            continue
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hashes_p0[h].append(f)
        if re.search(r"def send_ntfy\(", content):
            p0_ntfy += 1
        if "/home/fsp585" in content:
            p0_home += 1
        if "sys.path" in content:
            p0_syspath += 1

    redundant_p0 = sum(len(flist) - 1 for flist in hashes_p0.values() if len(flist) > 1)

    return {
        "python_files": len(p0_files),
        "redundant_copies": redundant_p0,
        "files_with_home_path": p0_home,
        "sys_path_hacks": p0_syspath,
        "send_ntfy_defs": p0_ntfy,
        "runners_not_forwarding": 12,
        "active_shims": 0,
    }


def get_current_metrics() -> Dict[str, int]:
    """Computes current metrics across the tracked repository."""
    curr_files = [f for f in run_git("git ls-files").splitlines() if f.endswith(".py")]

    curr_ntfy = 0
    curr_home = 0
    curr_syspath = 0
    hashes_curr = defaultdict(list)

    for f in curr_files:
        p = REPO_ROOT / f
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hashes_curr[h].append(f)
        if re.search(r"def send_ntfy\(", content):
            curr_ntfy += 1
        if "/home/fsp585" in content:
            curr_home += 1
        if "sys.path" in content:
            curr_syspath += 1

    redundant_curr = sum(len(flist) - 1 for flist in hashes_curr.values() if len(flist) > 1)

    # Active shims count
    active_shims = 0
    if SHIMS_TSV.exists():
        with open(SHIMS_TSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            active_shims = sum(1 for row in reader if row.get("status", "").upper() == "ACTIVE")

    # Check historical runners that are not forwarding
    runners = [
        "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_static_21kp_v2_production.py",
        "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_tandem_cohort_production_50ar_200slds.py",
        "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py",
        "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_pilot_17kp_vs_21kp_comparison.py",
        "syllable_generation/multiview_top_bot_experiments/02_training_runners/run_pilot_static_comparison.py",
        "syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/fuse_cohort_static_v2.py",
        "syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/fuse_all_tandem_cohort.py",
        "syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/fuse_pilot_static.py",
    ]
    unforwarded = 0
    for r in runners:
        p = REPO_ROOT / r
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "pipelines." not in txt and "fpcode." not in txt:
                unforwarded += 1

    return {
        "python_files": len(curr_files),
        "redundant_copies": redundant_curr,
        "files_with_home_path": curr_home,
        "sys_path_hacks": curr_syspath,
        "send_ntfy_defs": curr_ntfy,
        "runners_not_forwarding": unforwarded,
        "active_shims": active_shims,
    }


def compute_metrics() -> None:
    print("Computing repository architecture metrics against Phase-0 baseline...")
    p0 = get_phase0_metrics()
    cur = get_current_metrics()

    METRICS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_TSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["metric", "phase0_baseline", "current_head", "delta", "pct_change"])

        metrics_def = [
            ("python_files", "Python Files (.py)"),
            ("redundant_copies", "Redundant / Duplicate Copies"),
            ("files_with_home_path", "Files with /home/fsp585"),
            ("sys_path_hacks", "sys.path Hacks"),
            ("send_ntfy_defs", "send_ntfy Definitions"),
            ("runners_not_forwarding", "Runners Not Forwarding"),
            ("active_shims", "Active Shims Hit"),
        ]

        print("=" * 90)
        print("          FIBERPHOTOMETRYCODE ARCHITECTURE HEALTH & REFACTORING METRICS")
        print("=" * 90)
        print(f"{'Metric':<35} {'Phase-0':<12} {'Current':<12} {'Delta':<12} {'Change (%)'}")
        print("-" * 90)

        for key, label in metrics_def:
            b = p0.get(key, 0)
            c = cur.get(key, 0)
            delta = c - b
            pct = ((delta / b) * 100) if b != 0 else (100.0 if c != 0 else 0.0)
            pct_str = f"{pct:+.1f}%" if b != 0 else "N/A"
            writer.writerow([key, b, c, delta, f"{pct:.2f}%"])
            print(f"{label:<35} {b:<12} {c:<12} {delta:<12} {pct_str}")

        print("=" * 90)
        print(f"Metrics written to: {METRICS_TSV}")


if __name__ == "__main__":
    compute_metrics()
