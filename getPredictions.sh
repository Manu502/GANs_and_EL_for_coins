#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <models_directory> <data_directory>"
    exit 1
fi

models_dir="$1"
data_dir="$2"
script_path="ModelEvaluator.py"

# Validate if directories exist
if [ ! -d "$models_dir" ]; then
    echo "Error: Models directory '$models_dir' not found."
    exit 1
fi

if [ ! -d "$data_dir" ]; then
    echo "Error: Data directory '$data_dir' not found."
    exit 1
fi

# Get list of model files
model_files=$(ls "$models_dir")

# Get list of data files
data_files=$(ls "$data_dir")

# Iterate over model and data combinations
for model_file in $model_files; do
    for data_file in $data_files; do
        model_path="$models_dir$model_file"
        
        # echo $model_path
        # echo $data_path
         # # Execute the script for each combination
        data_path="$data_dir$data_file"
        python3 "$script_path" "$model_path" "$data_path" 64
        
        
        # if you only want the evalutation data uncomment line 43&44 and comment line 38&39 
        # data_path="$data_dir$data_file/"
        # python3 "$script_path" "$model_path" "$data_path" 64 --eval
    done
done
