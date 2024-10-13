#!/bin/bash

# Directory setup
input_folder="generated_graphs"
feature_folder="feats"
output_folder="features"

# Variables for number of graphs and selected features
num_graphs=$1
features=("${@:2}")

# Check if num_graphs and features are provided
if [ -z "$num_graphs" ] || [ ${#features[@]} -eq 0 ]; then
    echo "Usage: $0 <number_of_graphs> <features (a-f)>"
    echo "Example: $0 5 a b c"
    exit 1
fi

# 1. Clear output directories before adding new data
for feature in "${features[@]}"; do
    output_dir="$output_folder/f$feature"
    if [ -d "$output_dir" ]; then
        echo "Clearing directory: $output_dir"
        rm -rf "$output_dir"/*
    fi
done

# 2. Compile feature extraction codes
for feature in "${features[@]}"; do
    echo "Compiling feature extractor f$feature.cpp..."
    g++ "$feature_folder/f$feature.cpp" -o "f$feature.out"
    if [ $? -ne 0 ]; then
        echo "Compilation of f$feature.cpp failed!"
        exit 1
    fi
done

# 3. Process each graph and extract features
for ((i=1; i<=num_graphs; i++)); do
    input_file="$input_folder/input_graph_$i.txt"

    if [ ! -f "$input_file" ]; then
        echo "Input file $input_file does not exist, skipping..."
        continue
    fi

    for feature in "${features[@]}"; do
        # Create output directory if it doesn't exist
        output_dir="$output_folder/f$feature"
        mkdir -p "$output_dir"

        # Define output file
        output_file="$output_dir/${feature}_$i.txt"

        # Run feature extractor on the graph and direct output to the file
        echo "Extracting feature $feature from graph $i..."
        ./f$feature.out  "$input_file" > "$output_file"
    done
done

echo "Feature extraction completed."

