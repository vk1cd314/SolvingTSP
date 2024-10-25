#!/bin/bash

# Constants
GRAPH_SIZE=100  
GRAPH_NUMBR=1000  
GRAPH_DIR="generated_graphs"  
SOLVED_DIR="solved_graphs"  

# Create directories if they do not exist
mkdir -p $GRAPH_DIR
mkdir -p $SOLVED_DIR

# Cleanup old graph files before generating new ones
echo "Cleaning up existing graph files..."
rm -f ${GRAPH_DIR}/*.txt
rm -f ${SOLVED_DIR}/*.sol

# Generate random graphs
for i in $(seq 1 $GRAPH_NUMBR); do
    RANDOM_SEED=$RANDOM  # Generate a random seed
    INPUT_GRAPH_FILE="${GRAPH_DIR}/input_graph_${i}.txt"
    
    # Generate the graph and store it in a file
    ./gen_random_graph $GRAPH_SIZE $RANDOM_SEED > $INPUT_GRAPH_FILE
    echo "Generated $INPUT_GRAPH_FILE"
done

# Solve each graph using concorde-bin
for i in $(seq 1 $GRAPH_NUMBR); do
    INPUT_GRAPH_FILE="${GRAPH_DIR}/input_graph_${i}.txt"
    OUTPUT_FILE="${SOLVED_DIR}/graph_${i}.sol"

    # Solve the graph and pipe the output
    ./concorde-bin -o $OUTPUT_FILE -N 10 $INPUT_GRAPH_FILE
    echo "Solved graph $INPUT_GRAPH_FILE and stored result in $OUTPUT_FILE"
done

echo "All graphs generated and solved!"

echo "cleaning up directory"


rm -f *.mas *.sol *.pul *.sav

echo "Cleanup complete!"
