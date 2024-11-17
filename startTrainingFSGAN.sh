#!/bin/bash

show_help() {
    echo "Usage: $0 <origin directory> <result directory> <kimg>"
    echo "Example: bash startTraining.sh ./tfds/train/ ./results/  100"
    echo "Check if the provided directories exist."
    exit 1
}

# Check for the -h option
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
fi

if [ $# -ne 3 ]; then
    echo "Wrong number of arguments entered. Please enter four directory paths."    
    show_help
fi

originDir=$(realpath "$1")
resultDir=$(realpath "$2")
kImg=$3

# Check if both directories exist
if [ ! -d "$originDir" ] || [ ! -d "$resultDir" ] ; then
    echo "One or both directories do not exist."
    show_help
fi
echo "All directories exist."
# Loop through each subfolder and its subfolders in dir1
for subfolder in "$originDir"/*; do
    if [ -d "$subfolder" ]; then
        echo "Processing subfolder: $subfolder"
        baseNameSubFolder=$(basename "$subfolder")
        new_dir="$resultDir/$baseNameSubFolder"
        echo "$new_dir"
        # Check if the directory already exists
        if [ -d "$new_dir" ]; then
            echo "Directory '$new_dir' already exists. Skipping."
        else
            mkdir -p "$new_dir"
            echo "mkdir $new_dir"
        # Execute the Python command for each subfolder and subsubfolder
        fi
        # Execute the Python command for each subfolder
        python run_training.py \
            --config=config-ada-sv-flat \
            --result-dir="$new_dir/" \
            --data-dir="$originDir/" \
            --dataset-train="$baseNameSubFolder/" \
            --dataset-eval="$baseNameSubFolder/" \
	        --mirror-augment=False \
            --resume-pkl="stylegan2-coin-config-f-2800kimg.pkl" \
            --total-kimg="$kImg" \
            --metrics=None \
            --img-ticks=5 \
            --net-ticks=40 \
            2>&1 
    fi
done
