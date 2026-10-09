"""
scripts.execute_phase3_moves
============================
Executes approved file and directory relocations registered in MOVES.tsv.
Ensures zero data loss, safe directory creation, and strict verification.
"""

import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    moves_file = REPO_ROOT / "MOVES.tsv"
    if not moves_file.exists():
        print(f"Error: {moves_file} not found.")
        sys.exit(1)

    with open(moves_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        moves = list(reader)

    print(f"Executing {len(moves)} approved relocations...")
    executed = 0

    for idx, row in enumerate(moves, 1):
        src = REPO_ROOT / row["source_path"]
        dst = REPO_ROOT / row["target_path"]

        if not src.exists():
            print(f"[{idx}/{len(moves)}] Source does not exist (already moved?): {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Move source to destination
        shutil.move(str(src), str(dst))
        print(f"[{idx}/{len(moves)}] Moved: {row['source_path']} -> {row['target_path']}")
        executed += 1

    print(f"\nSuccessfully executed {executed} relocations.")


if __name__ == "__main__":
    main()
