from utils import gen_random_conn_graph, plot_graph
import random
import math
import matplotlib.pyplot as plt
from solvis import vis_res

def format_to_tsplib(data, filename='graph.tsp'):
    # Extract all unique nodes
    nodes = set()
    for edge in data:
        nodes.update(edge)

    nodes = sorted(nodes)

    # Create a mapping from node coordinates to node index
    node_index = {node: i + 1 for i, node in enumerate(nodes)}

    # Write the TSPLIB format
    with open(filename, 'w') as f:
        f.write('NAME: Graph\n')
        f.write('TYPE: TSP\n')
        f.write(f'DIMENSION: {len(nodes)}\n')
        f.write('EDGE_WEIGHT_TYPE: EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')

        for node, index in node_index.items():
            f.write(f'{index} {node[0]} {node[1]}\n')

        f.write('EOF\n')

    print(f"TSPLIB file '{filename}' has been created.")

random_graph = gen_random_conn_graph(20)
plot_graph(random_graph, add_edges=True)

format_to_tsplib(random_graph.get_edges())

N = 20
print(len(random_graph.get_edges()))
def getFreq(a, b, c, d):
    ab = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) + random.uniform(1e-7, 1e-3)
    bc = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2) + random.uniform(1e-7, 1e-3)
    cd = math.sqrt((c[0] - d[0])**2 + (c[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    ad = math.sqrt((a[0] - d[0])**2 + (a[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    ac = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2) + random.uniform(1e-7, 1e-3)
    bd = math.sqrt((b[0] - d[0])**2 + (b[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    
    freq_dict = {}
    
    if ab + cd < ac + bd < ad + bc:
        freq_dict[f'{a, b}'] = 5
        freq_dict[f'{b, c}'] = 1
        freq_dict[f'{c, d}'] = 5
        freq_dict[f'{a, d}'] = 1
        freq_dict[f'{a, c}'] = 3
        freq_dict[f'{b, d}'] = 3
        
    elif ab + cd < ad + bc < ac + bd:
        freq_dict[f'{a, b}'] = 5
        freq_dict[f'{b, c}'] = 3
        freq_dict[f'{c, d}'] = 5
        freq_dict[f'{a, d}'] = 3
        freq_dict[f'{a, c}'] = 1
        freq_dict[f'{b, d}'] = 1
        
    elif ac + bd < ab + cd < ad + bc:
        freq_dict[f'{a, b}'] = 3
        freq_dict[f'{b, c}'] = 1
        freq_dict[f'{c, d}'] = 3
        freq_dict[f'{a, d}'] = 1
        freq_dict[f'{a, c}'] = 5
        freq_dict[f'{b, d}'] = 5
        
    elif ac + bd < ad + bc < ab + cd:
        freq_dict[f'{a, b}'] = 1
        freq_dict[f'{b, c}'] = 3
        freq_dict[f'{c, d}'] = 1
        freq_dict[f'{a, d}'] = 3
        freq_dict[f'{a, c}'] = 5
        freq_dict[f'{b, d}'] = 5
        
    elif ad + bc < ab + cd < ac + bd:
        freq_dict[f'{a, b}'] = 3
        freq_dict[f'{b, c}'] = 5
        freq_dict[f'{c, d}'] = 3
        freq_dict[f'{a, d}'] = 5
        freq_dict[f'{a, c}'] = 1
        freq_dict[f'{b, d}'] = 1
        
    elif ad + bc < ac + bd < ab + cd:
        freq_dict[f'{a, b}'] = 1
        freq_dict[f'{b, c}'] = 5
        freq_dict[f'{c, d}'] = 1
        freq_dict[f'{a, d}'] = 5
        freq_dict[f'{a, c}'] = 3
        freq_dict[f'{b, d}'] = 3
        
    return freq_dict


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
        # random_graph.remove_vertex(pt1)
        # random_graph.remove_vertex(pt2)
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
print(top_two_thirds)

plt.figure(figsize=(10, 10))
for edge, value in top_two_thirds:
    (x1, y1), (x2, y2) = edge
    plt.plot([x1, x2], [y1, y2], marker='o')

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Graph of Edges')
plt.grid(True)
plt.savefig("sparsegraph")
plt.show()

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def format_to_tsplib_edge_list(data, filename='graph_sparse.tsp'):
    nodes = set()
    for edge, _ in data:
        nodes.update(edge)

    nodes = sorted(nodes)
    node_index = {node: i + 1 for i, node in enumerate(nodes)}
    matrix = {}
    n = len(nodes)
    INF = 1e7
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            matrix[(i, j)] = INF
    # for i, node in enumerate(nodes):
    #     matrix[(node_index[node[0]], node_index[node[1]])] = euclidean_distance(node[0], node[1])
    
    for edge, _ in data:
        matrix[(node_index[edge[0]], node_index[edge[1]])] = euclidean_distance(edge[0], edge[1])
        matrix[(node_index[edge[1]], node_index[edge[0]])] = euclidean_distance(edge[0], edge[1])
        

    with open(filename, 'w') as f:
        f.write('NAME: Graph\n')
        f.write('TYPE: TSP\n')
        f.write(f'DIMENSION: {len(nodes)}\n')
        f.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        f.write('NODE_COORD_TYPE: TWOD_COORDS\n')
        f.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
        f.write('NODE_COORD_SECTION\n')
        for node, index in node_index.items():
            f.write(f'{index} {float(node[0])} {float(node[1])}\n')
        f.write('EDGE_WEIGHT_SECTION\n')
        for i in range(1, n + 1):
            s = ''
            for j in range(1, n + 1):
                s += str(matrix[(i, j)]) + ' '
            f.write(s + '\n')
        f.write('EOF\n')

    print(f"TSPLIB file '{filename}' has been created.")

format_to_tsplib_edge_list(top_two_thirds)

vis_res('graph.tsp')
vis_res('graph_sparse.tsp')