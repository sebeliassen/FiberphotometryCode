#!/bin/bash

# Define the output file name
OUTPUT_FILE="all_project_code.txt"

# Clear the output file if it exists, or create it
> "$OUTPUT_FILE"

# Find all .py and .ipynb files recursively
find . -type f \( -name "*.py" -o -name "*.ipynb" \) | while read -r file; do
    # Print a separator and the filename to the output file
    echo "=========================================" >> "$OUTPUT_FILE"
    echo "FILE: $file" >> "$OUTPUT_FILE"
    echo "=========================================" >> "$OUTPUT_FILE"
    
    # Check file extension
    if [[ "$file" == *.py ]]; then
        # For .py files, just cat the content
        cat "$file" >> "$OUTPUT_FILE"
    elif [[ "$file" == *.ipynb ]]; then
        # For .ipynb files, use nbconvert
        # --to python: converts to script
        # --stdout: pipes to our file instead of creating a new one
        # --TemplateExporter.exclude_markdown=True: removes text cells
        jupyter nbconvert --to python --stdout --TemplateExporter.exclude_markdown=True "$file" >> "$OUTPUT_FILE"
    fi
    
    # Add a newline at the end for separation
    echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "All code files have been concatenated into $OUTPUT_FILE"