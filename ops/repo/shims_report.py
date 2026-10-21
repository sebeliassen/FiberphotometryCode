#!/usr/bin/env python3
"""
ops/repo/shims_report.py
========================
Generates a structured report of all backward-compatibility shims,
their relocation destinations, and active status.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TSV_PATH = REPO_ROOT / "registry/SHIMS.tsv"


def print_shims_report():
    if not TSV_PATH.exists():
        print("registry/SHIMS.tsv not found. Run ops/repo/update_shims.py first.")
        return 1

    lines = TSV_PATH.read_text().strip().split("\n")
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line.strip()]

    print("=" * 88)
    print("                 BACKWARD-COMPATIBILITY SHIMS & DEPRECATION REGISTRY")
    print("=" * 88)
    print(f"{'Legacy Path':<50} {'Target Module':<30} {'Status'}")
    print("-" * 88)
    for r in rows:
        path, target, _, status, _ = r
        print(f"{path:<50} {target:<30} {status}")
    print("=" * 88)
    print(f"Total Cataloged Shims: {len(rows)}")
    print(f"Registry source: {TSV_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(print_shims_report())
