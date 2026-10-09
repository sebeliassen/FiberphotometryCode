#!/usr/bin/env sh
set -e

out="all_py_concat.txt"
: > "$out"  # truncate/create

# Expand to the list of .py files; handle "no matches" safely
set -- ./*.py
if [ ! -e "$1" ]; then
  echo "No .py files found."
  exit 0
fi

for f do
  [ -f "$f" ] || continue
  printf '## %s\n' "$(basename -- "$f")" >> "$out"
  cat -- "$f" >> "$out"
  printf '\n' >> "$out"
done

echo "Wrote $out"
