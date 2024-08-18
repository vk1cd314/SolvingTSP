#!/bin/bash
g++ gen_random_graph.cpp -o gen_random_graph && ./gen_random_graph $1 $2 > input_graph.txt
g++ freqquad.cpp -o freqquad && ./freqquad input_graph.txt > output_graph.txt

RED='\033[31m'
YELLOW='\033[33m'
NC='\033[0m' 

echo -e "${RED}Results for ${YELLOW}Original${RED} Graph:${NC}"
time ./concorde-bin -x -N 10 input_graph.txt | grep 'Optimal Solution'

echo -e "${RED}Results for ${YELLOW}Sparsified${RED} Graph:${NC}"
time ./concorde-bin -x -N 10 output_graph.txt | grep 'Optimal Solution'

rm *input_graph*
rm *output_graph*
rm gen_random_graph freqquad

