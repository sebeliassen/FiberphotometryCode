#!/usr/bin/env bash
# bundle_py.sh — Create a single TXT with names + contents of .py files.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [-r] [-o OUTPUT.txt] [DIR]

  DIR            Directory to scan (default: current directory).
  -r             Recurse into subdirectories.
  -o OUTPUT.txt  Output filename (default: python_bundle_<dir>_<timestamp>.txt)

Notes:
- Wraps each file in a '===== FILE: <path> =====' header and a \`\`\`python code fence.
- Skips common junk folders: .git, .venv, venv, __pycache__, .mypy_cache, .pytest_cache.
EOF
}

RECURSIVE=0
OUTFILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -r) RECURSIVE=1; shift ;;
    -o) OUTFILE="${2:-}"; [[ -n "$OUTFILE" ]] || { echo "Missing value for -o" >&2; exit 1; }; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *)  DIR="$1"; shift ;;
  esac
done

DIR="${DIR:-.}"

# Resolve output filename
if [[ -z "$OUTFILE" ]]; then
  # Use basename of target dir, even if DIR is "."
  dirbase="$(basename "$(cd "$DIR" && pwd)")"
  OUTFILE="python_bundle_${dirbase}_$(date '+%Y%m%d-%H%M%S').txt"
fi

# Work from inside DIR to get nice relative paths
START_DIR="$(pwd)"
cd "$DIR"

# Depth handling
depth_args=()
if [[ $RECURSIVE -eq 0 ]]; then
  depth_args=(-maxdepth 1)
fi

# Collect files, skipping common folders; produce NUL-delimited list
# Compatible with macOS/BSD find.
mapfile -d '' files < <(
  find . "${depth_args[@]}" \
    \( -type d \( -name .git -o -name .venv -o -name venv -o -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache \) -prune \) -o \
    \( -type f -name '*.py' -print0 \)
)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .py files found in '$DIR'." >&2
  exit 1
fi

# Sort paths for stable order
sorted=()
if command -v sort >/dev/null 2>&1; then
  while IFS= read -r -d '' f; do sorted+=("$f"); done < <(printf '%s\0' "${files[@]}" | LC_ALL=C sort -z)
else
  sorted=("${files[@]}")
fi

# Write output in the original working directory
OUTPATH="$START_DIR/$OUTFILE"
{
  echo "# Python bundle generated on $(date)"
  echo "# Source directory: $(pwd)"
  echo
  for f in "${sorted[@]}"; do
    # Trim leading './'
    rel="${f#./}"
    echo "===== FILE: ${rel} ====="
    echo '```python'
    cat -- "$rel"
    echo
    echo '```'
    echo
  done
} > "$OUTPATH"

echo "Wrote: $OUTPATH"
