#!/bin/bash

# Constants
GRAPH_SIZE=300               # Size of each graph
GRAPH_NUMBR=1             # Number of graphs to generate
GRAPH_DIR="test-generated_graphs" # Directory for generated graphs
SPARSE_DIR="test-sparsified_graphs" # Directory for sparsified graphs
SOLVED_ORIG_DIR="test-solved_original" # Directory for original graph solutions
SOLVED_SPARSE_DIR="test-solved_sparsified" # Directory for sparsified graph solutions
NUMBER_OF_TURNS=10
# Clear directories
echo "Clearing directories..."
rm -rf $GRAPH_DIR $SPARSE_DIR $SOLVED_ORIG_DIR $SOLVED_SPARSE_DIR
mkdir -p $GRAPH_DIR $SPARSE_DIR $SOLVED_ORIG_DIR $SOLVED_SPARSE_DIR
echo "Directories cleared and recreated."

# Generate random graphs
echo "Generating random graphs..."
for i in $(seq 1 $GRAPH_NUMBR); do
    RANDOM_SEED=$RANDOM  # Generate a random seed
    INPUT_GRAPH_FILE="${GRAPH_DIR}/input_graph_${i}.txt"
    
    # Generate the graph and store it in a file
    ./gen_random_graph $GRAPH_SIZE $i > $INPUT_GRAPH_FILE
    echo "Generated $INPUT_GRAPH_FILE"
done

# Sparsify each graph and solve the original and sparsified versions
g++ frequency-quads/freqquad.cpp -o freqquad
echo "Sparsifying and solving graphs..."
for i in $(seq 1 $GRAPH_NUMBR); do
    INPUT_GRAPH_FILE="${GRAPH_DIR}/input_graph_${i}.txt"
    SPARSE_GRAPH_FILE="${SPARSE_DIR}/sparse_graph_${i}.txt"
    SOLVED_ORIG_FILE="${SOLVED_ORIG_DIR}/graph_${i}.sol"
    SOLVED_SPARSE_FILE="${SOLVED_SPARSE_DIR}/sparse_graph_${i}.sol"
    
    # Sparsify the graph and store it
    # ./freqquad $INPUT_GRAPH_FILE $NUMBER_OF_TURNS $i > $SPARSE_GRAPH_FILE
    # echo "Sparsified graph saved to $SPARSE_GRAPH_FILE"
    
    # Solve the original graph and save the result
    ./concorde-bin -o $SOLVED_ORIG_FILE -N 10 $INPUT_GRAPH_FILE
    echo "Solved original graph $INPUT_GRAPH_FILE and stored result in $SOLVED_ORIG_FILE"
    
    # Solve the sparsified graph and save the result
    # ./concorde-bin -o $SOLVED_SPARSE_FILE -N 10 $SPARSE_GRAPH_FILE
    # echo "Solved sparsified graph $SPARSE_GRAPH_FILE and stored result in $SOLVED_SPARSE_FILE"
done

echo "All graphs generated, sparsified, and solved!"

# Clean up intermediate files
echo "Cleaning up..."
rm -f *.mas *.sol *.pul *.sav
echo "Cleanup complete!"

