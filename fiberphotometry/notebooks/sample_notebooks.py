#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

def find_notebooks(base_dir):
    """Recursively find all .ipynb files under base_dir."""
    return [p for p in Path(base_dir).rglob("*.ipynb")]

def get_mtime(path):
    return path.stat().st_mtime

def extract_code_cells(nb_path):
    """Load the notebook JSON and return a flat list of code-cell source strings."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") == "code":
            # join the source lines into one block
            src = "".join(cell.get("source", []))
            if src.strip():
                header = f"# --- Cell {idx} ---\n"
                code_cells.append(header + src + "\n")
    return code_cells

def main(folders, n, out_path):
    with open(out_path, 'w', encoding='utf-8') as outf:
        for folder in folders:
            nb_paths = find_notebooks(folder)
            nb_paths.sort(key=get_mtime, reverse=True)
            selected = nb_paths[:n]
            if not selected:
                continue
            for nb in selected:
                # get a clean relative path
                rel = os.path.relpath(str(nb), start=os.getcwd())
                outf.write(f"\n\n========== {rel} ==========\n\n")
                for block in extract_code_cells(nb):
                    outf.write(block)
    print(f"Done. Appended up to {n} notebooks per folder into {out_path!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample the n newest .ipynb files per folder and extract code cells."
    )
    parser.add_argument(
        "-n", type=int, required=True,
        help="Number of newest notebooks to take from each folder"
    )
    parser.add_argument(
        "-o", "--output", default="sampled_notebooks.txt",
        help="Output text file"
    )
    parser.add_argument(
        "folders", nargs="+",
        help="List of folders to scan (e.g. leonie sam sebastian)"
    )
    args = parser.parse_args()
    main(args.folders, args.n, args.output)
