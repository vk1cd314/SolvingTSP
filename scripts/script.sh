#!/bin/bash
numbers=(20 50 100 200 300 400 500)

for number in "${numbers[@]}"
do
    echo "Started Batch $number"
    for ((i=0; i<10; i++))
    do
        python sparse.py $number
        echo "Running Solver for Batch $number, iteration $i";
        ./concorde-bin -o output.sol graphs/original_graph.tsp | grep "Optimal Solution:" >> tsp$number.res
        echo "Finished Solver for Batch $number, iteration $i";
    done
    echo "Finished Batch $number"
done
