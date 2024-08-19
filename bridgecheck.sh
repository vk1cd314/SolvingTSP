#!/bin/bash
g++ -O3 gen_random_graph.cpp -o gen_random_graph && ./gen_random_graph $1 $2 > input_graph.txt
g++ -O3 freq_tmp.cpp -o freqquad && ./freqquad input_graph.txt $3 $2 > output_graph.txt

RED='\033[31m'
YELLOW='\033[33m'
NC='\033[0m' 

echo -e "${RED}Results:${NC}"
g++ GK-2E-S.cpp -o edgecut && ./edgecut output_graph.txt res
cat res

