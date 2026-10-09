#!/bin/bash
#
# This script finds all .csv files in the current folder
# and prints the number of lines for each one.

echo "--- Analyzing CSV files ---"

# A counter for the number of files found
file_count=0

# Loop through all files ending in .csv
for file in *.csv; do
  # Check if a matching file actually exists to avoid errors
  if [ -f "$file" ]; then
    # Count the lines in the file.
    # Using `< "$file"` is a clean way to get just the number from wc.
    line_count=$(wc -l < "$file")
    
    # Print the result, stripping any leading whitespace from the count
    echo "File: $file, Lines: ${line_count// /}"
    
    file_count=$((file_count + 1))
  fi
done

# If no files were found, print a message
if [ "$file_count" -eq 0 ]; then
  echo "No .csv files found in this directory."
fi

echo "--- Done ---"
