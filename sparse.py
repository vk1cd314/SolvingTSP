from utils import gen_random_conn_graph, plot_graph
import random
import math
import matplotlib.pyplot as plt
from solvis import vis_res
import argparse
from tspformat import format_to_tsplib, format_to_tsplib_edge_list
from freqquad import getFreq

parser = argparse.ArgumentParser(description="Script that takes a single integer argument N.")

parser.add_argument("N", type=int, help="An integer argument N")

args = parser.parse_args()

random_graph = gen_random_conn_graph(args.N)
plot_graph(random_graph, add_edges=True)

format_to_tsplib(random_graph.get_edges())

N = args.N
print(len(random_graph.get_edges()))

edge_freq = {}   
edge_freq_avg = {}

for edge in random_graph.get_edges():
    # 2 points fixed 
    u, v = edge
    have_pts = set()
    for vertex in random_graph.get_vertices():
        if u != vertex and v != vertex:
            have_pts.add(vertex)
    
    freq_quads = set()
    edge_freq[edge] = 0
    
    while len(freq_quads) < N and len(have_pts) >= 2:
        pt1, pt2 = random.sample(sorted(have_pts), 2)
        freq_dict = getFreq(u, v, pt1, pt2)
        if (u, v, min(pt1, pt2), max(pt1, pt2)) not in freq_quads:
            edge_freq[edge] += freq_dict[f'{u, v}']
        freq_quads.add((u, v, min(pt1, pt2), max(pt1, pt2)))
    edge_freq_avg[edge] = edge_freq[edge] / N

sorted_freq = sorted(edge_freq_avg.items(), key=lambda item: item[1])
 
sorted_data = sorted(sorted_freq, key=lambda x: x[1], reverse=True)

# Step 2: Calculate the number of elements corresponding to the top 2/3
two_thirds_length = int(len(sorted_data) * (2/3))

# Step 3: Select the top 2/3 elements
top_two_thirds = sorted_data[:two_thirds_length]
# print(top_two_thirds)

plt.figure(figsize=(10, 10))
for edge, value in top_two_thirds:
    (x1, y1), (x2, y2) = edge
    plt.plot([x1, x2], [y1, y2], marker='o')

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Graph of Edges')
plt.grid(True)
plt.savefig("images/sparsegraph")
# plt.show()

format_to_tsplib_edge_list(top_two_thirds)

ans_correct = vis_res('graphs/original_graph.tsp')
ans_sparse = vis_res('graphs/sparsified_graph.tsp')
print(ans_correct, ans_sparse)

assert abs(ans_correct - ans_sparse) < 1e-7