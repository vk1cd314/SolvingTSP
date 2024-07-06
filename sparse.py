from utils import gen_random_conn_graph, plot_graph
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from solvis import vis_res
import argparse
from tspformat import format_euclidean_to_tsplib, format_to_tsplib_edge_list, format_to_qs
from freqquad import do_one_iter
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

parser = argparse.ArgumentParser(description="Script that takes a single integer argument N.")

parser.add_argument("N", type=int, help="An integer argument N")

args = parser.parse_args()

random_graph = gen_random_conn_graph(args.N)
# plot_graph(random_graph, add_edges=True)

index_info_full = format_euclidean_to_tsplib(random_graph.get_edges())

N = args.N
print(len(random_graph.get_edges()))

top_two_thirds = do_one_iter(random_graph, N)

# plt.figure(figsize=(2.2, 2.2))
# for edge, value in top_two_thirds:
#     (x1, y1), (x2, y2) = edge
#     plt.plot([x1, x2], [y1, y2], color='blue')

# for nodes in random_graph.get_vertices():
#     x, y = nodes
#     plt.plot(x, y, marker='o', color='black')

# plt.axis('off')
# plt.title('Sparse Graph')
# plt.savefig("images/sparsegraph.pgf")

index_to_node = format_to_tsplib_edge_list(top_two_thirds)
format_to_qs(top_two_thirds)

ans_correct = vis_res('graphs/original_graph.tsp', index_info_full)
ans_sparse = vis_res('graphs/sparsified_graph.tsp', index_to_node)
print(ans_correct, ans_sparse)

assert abs(ans_correct - ans_sparse) < 1e-7