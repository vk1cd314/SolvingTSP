from utils import gen_random_conn_graph, plot_graph, graph_to_file, draw_tsp_solution
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from solvis import vis_res
import argparse
from tspformat import format_euclidean_to_tsplib, format_to_tsplib_edge_list, format_for_testing
from freqquad import do_one_iter
import time
import subprocess
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

parser = argparse.ArgumentParser(description="Script that takes a single integer argument N.")

parser.add_argument("N", type=int, help="An integer argument N")
parser.add_argument("seed", type=int, help="An integer argument seed")

args = parser.parse_args()

random_graph = gen_random_conn_graph(args.N, args.seed)
plot_graph(random_graph)
index_to_node = graph_to_file(random_graph)
subprocess.call(['./concorde-bin', '-x' , '-N' , '10', 'graphs/original_graph'])
with open('original_graph.sol', 'r') as file:
    file.readline()
    second_line = file.readline()

tour = second_line.strip().split(' ')
for i in range(len(tour)):
    tour[i] = int(tour[i])
tour.append(tour[0])

draw_tsp_solution(index_to_node, tour)

# plot_graph(random_graph)

# index_info_full = format_euclidean_to_tsplib(random_graph.get_edges())
# tmp = format_for_testing(random_graph.get_edges())

# N = args.N

# start_time = time.time()
# top_two_thirds = do_one_iter(random_graph, N)
# end_time = time.time()

# index_to_node = format_to_tsplib_edge_list(top_two_thirds)

# ans_correct = vis_res('graphs/original_graph.tsp', index_info_full)
# ans_sparse = vis_res('graphs/sparsified_graph.tsp', index_to_node)

# filename = "tsp" + str(N) + ".res"
# with open(filename, 'a') as f:
#     f.write(f"\n{ans_correct} {ans_sparse} {end_time - start_time}")