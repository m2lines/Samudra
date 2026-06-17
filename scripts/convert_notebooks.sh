#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Convert all Jupyter notebooks in a directory to Python scripts

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 <directory> [output_directory]"
    echo ""
    echo "Convert all .ipynb files in a directory to .py files"
    echo ""
    echo "Arguments:"
    echo "  directory         Directory containing .ipynb files"
    echo "  output_directory  Optional output directory for .py files"
    echo "                    (defaults to same as input directory)"
    echo ""
    echo "Examples:"
    echo "  $0 notebooks/"
    echo "  $0 notebooks/ scripts/"
}

# Check if directory argument is provided
if [ $# -lt 1 ]; then
    echo "Error: Directory argument required"
    show_usage
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-$INPUT_DIR}"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Find all .ipynb files
NOTEBOOKS=("$INPUT_DIR"/*.ipynb)

# Check if any notebooks were found
if [ ! -e "${NOTEBOOKS[0]}" ]; then
    echo "No .ipynb files found in '$INPUT_DIR'"
    exit 0
fi

echo "Found ${#NOTEBOOKS[@]} notebook(s) to convert:"
for notebook in "${NOTEBOOKS[@]}"; do
    echo "  - $(basename "$notebook")"
done

echo ""

# Convert each notebook
for notebook in "${NOTEBOOKS[@]}"; do
    if [ -f "$notebook" ]; then
        echo "Converting $(basename "$notebook")..."

        if uvx -q jupyter nbconvert --to script "$notebook" --output-dir "$OUTPUT_DIR"; then
            # Get the output filename (notebook name with .py extension)
            basename_no_ext=$(basename "$notebook" .ipynb)
            output_file="$OUTPUT_DIR/${basename_no_ext}.py"
            echo "  ✅  Created $output_file"
        else
            echo "  ❌  Failed to convert $(basename "$notebook")"
        fi
    fi
done

echo ""
echo "Conversion complete!"