#!/bin/bash

show_help() {
    echo "Usage: $0 <origin directory> <result directory>"
    echo "Check if the provided directories exist."
    exit 1
}

# Check for the -h option
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
fi

if [ $# -ne 2 ]; then
    echo "Wrong number of arguments entered. Please enter two directory paths."
    show_help
fi

originDir=$1
resultDir=$2

# Check if both directories exist
if [ ! -d "$originDir" ] || [ ! -d "$resultDir" ]; then
    echo "One or both directories do not exist."
    show_help
fi

echo "Both directories exist."
# Loop through each subfolder and its subfolders in originDir
find "$originDir" -mindepth 1 -maxdepth 1  -type d -print0 | while IFS= read -r -d '' subfolder; do
    echo "Processing subfolder: $subfolder"
    # Loop through subsubfolders within the current subfolder
    find "$subfolder" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' subsubfolder; do
        echo "Processing subsubfolder: $subsubfolder"

        # Create a new directory in resultDir with the specified naming convention
        new_dir="$resultDir/$(basename "$subfolder")_$(basename "$subsubfolder")"
        # Check if the directory already exists
        if [ -d "$new_dir" ]; then
            echo "Directory '$new_dir' already exists. Skipping."
        else
            mkdir -p "$new_dir"
            echo "mkdir $new_dir"
        # Execute the Python command for each subfolder and subsubfolder
        fi
        echo "creating tfds from $subsubfolder"
        python dataset_tool.py \
            create_from_images \
            "$new_dir" \
            "$subsubfolder" \
            --resolution 256
    done
done
