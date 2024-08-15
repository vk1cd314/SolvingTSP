from utils import gen_random_conn_graph, plot_graph, graph_to_file, draw_tsp_solution
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from solvis import vis_res, euclidean_distance
import argparse
from tspformat import format_euclidean_to_tsplib, format_to_tsplib_edge_list, format_for_testing
from freqquad import do_one_iter
import time
import subprocess
import copy
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

parser = argparse.ArgumentParser(
    description="Script that takes a single integer argument N.")

parser.add_argument("N", type=int, help="An integer argument N")
parser.add_argument("seed", type=int, help="An integer argument seed")

args = parser.parse_args()

random_graph = gen_random_conn_graph(args.N, seed=args.seed)


def write_graph_to_file(graph):
    with open('graph.txt', 'w') as file:
        vertices = graph.get_vertices()
        edges = graph.get_edges()
        file.write(f"{len(vertices)} {len(edges)}\n")
        for edge in edges:
            start_node_index = vertices.index(edge[0])
            end_node_index = vertices.index(edge[1])
            file.write(f"{start_node_index} {end_node_index} {
                       int(euclidean_distance(edge[0], edge[1]))}\n")


write_graph_to_file(random_graph)

# plot_graph(random_graph)
# index_to_node = graph_to_file(random_graph)
# subprocess.call(['./concorde-bin', '-x' , '-N' , '10', 'graphs/original_graph'])
# with open('original_graph.sol', 'r') as file:
#     file.readline()
#     second_line = file.readline()

# tour = second_line.strip().split(' ')
# for i in range(len(tour)):
#     tour[i] = int(tour[i])
# tour.append(tour[0])

# draw_tsp_solution(index_to_node, tour)

# plot_graph(random_graph)

# index_info_full = format_euclidean_to_tsplib(random_graph.get_edges())
# tmp = format_for_testing(random_graph.get_edges())

# N = args.N

# start_time = time.time()
# This has edges and a freq count. Print graph to file, similar to before but with top two-thirds of the edges based on frequency.
top_two_thirds = do_one_iter(random_graph, args.N)
cng = []
for edge, _ in top_two_thirds:
    cng.append(edge)

print(len(cng))

with open('top_two_thirds_graph.txt', 'w') as file:
    vertices = random_graph.get_vertices()
    vertices = random_graph.get_vertices()
    edges = random_graph.get_edges()
    file.write(f"{len(vertices)} {len(top_two_thirds)}\n")
    to_write = []
    for edge in cng:
        start_node_index = vertices.index(edge[0])
        end_node_index = vertices.index(edge[1])
        distance = euclidean_distance(edge[0], edge[1])
        to_write.append((min(start_node_index, end_node_index),
                         max(start_node_index, end_node_index), distance))
        # file.write(f"{start_node_index} {end_node_index} {round(round(distance, 2) * 100)}\n")
    to_write = sorted(to_write)
    for u, v, w in to_write:
        file.write(f"{u} {v} {int(w)}\n")

# Write the top two-thirds graph to a file
# def write_top_two_thirds_graph_to_file(graph):
#     with open('top_two_thirds_graph.txt', 'w') as file:
#         vertices = graph.get_vertices()
#         edges = graph.get_edges()
#         file.write(f"{len(vertices)} {len(edges)}\n")
#         for edge in edges:
#             start_node_index = vertices.index(edge[0])
#             end_node_index = vertices.index(edge[1])
#             distance = euclidean_distance(edge[0], edge[1])
#             file.write(f"{start_node_index} {end_node_index} {round(round(distance, 2) * 100)}\n")

# write_top_two_thirds_graph_to_file(top_two_thirds)


# end_time = time.time()

# index_to_node = format_to_tsplib_edge_list(top_two_thirds)

# ans_correct = vis_res('graphs/original_graph.tsp', index_info_full)
# ans_sparse = vis_res('graphs/sparsified_graph.tsp', index_to_node)

# filename = "tsp" + str(N) + ".res"
# with open(filename, 'a') as f:
#     f.write(f"\n{ans_correct} {ans_sparse} {end_time - start_time}")
