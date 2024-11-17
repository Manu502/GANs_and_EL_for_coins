#!/bin/bash
# Define the list of configurations
configurations=(
    # Base Models freeze until conv 4
    #Res50v2 Example
    "Both /home/Both/Train/ /home/Both/Validation/ /home/Both/Test/ 128 ResNet50V2 Res50_Both_Base_NoAug_FreezeConv4  conv4 /home/Both/Test13/"
    #Res101v2 Example
    "Both /home/Both/Train/ /home/Both/Validation/ /home/Both/Test/ 128 ResNet101V2 Res101_Both_Base_NoAug_FreezeConv4  conv4 /home/Both/Test13/"
    #Res50v2  Example
    "Both /home/Both/Train/ /home/Both/Validation/ /home/Both/Test/ 128 ResNet152V2 Res101_Both_Base_NoAug_FreezeConv4  conv4 /home/Both/Test13/"
    )
# Set default value for number
numOfRuns=1

# Check if an argument is provided
if [ $# -eq 1 ]; then
    # Check if the argument is a number
    if [[ $1 =~ ^[0-9]+$ ]]; then
        numOfRuns=$1
    else
        echo "Error: Argument is not a valid number"
        exit 1
    fi
fi

# Loop through the configurations and execute cnnTraining.py
for config in "${configurations[@]}"; do
    # Split configuration string into array
    IFS=' ' read -r -a config_array <<< "$config"
    
    # Extract variables from config_array
    type="${config_array[0]}"
    train_path="${config_array[1]}"
    test_path="${config_array[2]}"
    eval_path="${config_array[3]}"
    batch_size="${config_array[4]}"
    model="${config_array[5]}"
    output_file="${config_array[6]}"
    layer="${config_array[7]}"
    eval13_path="${config_array[8]}"
    augmentation="${config_array[9]:-}" # Optional augmentation flag
    
    # Run it numOfRuns times.
    for i in $(seq $numOfRuns); do
        echo "Iteration: $i"
        cnn_cmd="python3 cnnTraining.py $type $train_path $test_path $eval_path $batch_size $model $output_file $layer $eval13_path $augmentation"
        echo "Executing command: $cnn_cmd"
        $cnn_cmd
    done
    
    # Execute cnnTraining.py command
    echo "Executing: $cnn_cmd"
    eval "$cnn_cmd"
    
done
