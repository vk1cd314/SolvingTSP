#!/bin/bash
g++ gen_random_graph.cpp && ./a.out $1 $2 > input_graph.txt
g++ freqquad.cpp && ./a.out input_graph.txt > output_graph.txt
./concorde-bin -x -N 10 input_graph.txt | grep 'Exact lower bound'
./concorde-bin -x -N 10 output_graph.txt | grep 'Exact lower bound'
