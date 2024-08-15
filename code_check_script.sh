#!/bin/bash
python sparse.py $1 $2
g++ freqquad.cpp && ./a.out graph.txt > out
./concorde-bin -x -N 10 top_two_thirds_graph.txt | grep 'Exact lower bound'
./concorde-bin -x -N 10 out | grep 'Exact lower bound'

