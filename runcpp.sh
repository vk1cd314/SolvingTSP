#!/bin/bash
g++ gen_random_graph.cpp && ./a.out $1 $2 > input_graph.txt
g++ freqquad.cpp && ./a.out input_graph.txt > output_graph.txt

RED='\033[31m'
NC='\033[0m' 

echo -e "${RED}Results for Original Graph:${NC}"
time ./concorde-bin -x -N 10 input_graph.txt | grep 'Optimal Solution'

echo -e "${RED}Results for Sparsified Graph:${NC}"
time ./concorde-bin -x -N 10 output_graph.txt | grep 'Optimal Solution'

