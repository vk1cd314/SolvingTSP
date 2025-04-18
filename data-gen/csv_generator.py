import csv
import os
import random
import shutil


def generate_random_features():
    """Generates a random feature string in the format 'feat1,feat2,feat3'."""
    return f"{random.random()},{random.random()},{random.random()}"

def read_graph(graph_file):
    """Reads the graph file and returns a list of edges (src, dst, weight)."""
    edges = []
    with open(graph_file, 'r') as f:
        num_nodes, num_edges = map(int, f.readline().strip().split())
        for line in f:
            src, dst, weight = line.strip().split()
            edges.append((int(src), int(dst), int(weight)))
    return edges, num_nodes

def read_tour(tour_file):
    """Reads the tour file and returns a set of edges (src, dst) in the tour."""
    tour_edges = set()
    with open(tour_file, 'r') as f:
        num_nodes = int(f.readline().strip())
        tour = list(map(int, f.read().split()))
        for i in range(len(tour) - 1):
            src, dst = tour[i], tour[i + 1]
            tour_edges.add((min(src, dst), max(src, dst)))
        src, dst = tour[-1], tour[0]
        tour_edges.add((min(src, dst), max(src, dst)))
    return tour_edges

def append_edges_csv(edges, tour_edges, output_csv, graph_id):
    """Appends the edges to edges.csv file."""
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for src, dst, weight in edges:
            label = 0
            if (src, dst) in tour_edges:
                label = 1
            writer.writerow([graph_id, src, dst, label, f'{weight}'])

def append_nodes_csv(num_nodes, output_csv, graph_id):
    """Appends the nodes to nodes.csv file."""
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for node_id in range(num_nodes):
            feat = 0
            writer.writerow([graph_id, node_id, f'{feat}'])

def append_graphs_csv(output_csv, graph_id):
    """Appends to the graphs.csv file."""
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([graph_id, '0', 0])

def process_multiple_graphs(graph_files, tour_files, output_dir):
    """Processes multiple graph and tour files and appends data to edges.csv, nodes.csv, and graphs.csv."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    edges_csv = os.path.join(output_dir, 'edges.csv')
    nodes_csv = os.path.join(output_dir, 'nodes.csv')
    graphs_csv = os.path.join(output_dir, 'graphs.csv')

    if not os.path.exists(edges_csv):
        with open(edges_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['graph_id', 'src_id', 'dst_id', 'label', 'feat'])

    if not os.path.exists(nodes_csv):
        with open(nodes_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['graph_id', 'node_id', 'feat'])

    if not os.path.exists(graphs_csv):
        with open(graphs_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['graph_id', 'feat', 'label'])

    for graph_id, (graph_file, tour_file) in enumerate(zip(graph_files, tour_files)):
        edges, num_nodes = read_graph(graph_file)
        tour_edges = read_tour(tour_file)

        append_edges_csv(edges, tour_edges, edges_csv, graph_id)
        append_nodes_csv(num_nodes, nodes_csv, graph_id)
        append_graphs_csv(graphs_csv, graph_id)

graph_path = './test-generated_graphs/'
sol_path = './test-solved_original/'

graph_files = [] 
for filename in os.listdir(graph_path):
    if os.path.isfile(os.path.join(graph_path, filename)):
        graph_files.append(graph_path+filename)


tour_files = [] 
for filename in os.listdir(sol_path):
    if os.path.isfile(os.path.join(sol_path, filename)):
        tour_files.append(sol_path+filename)

graph_files=sorted(graph_files)
tour_files=sorted(tour_files)

output_dir = './test-data/'
dataname = 'tspdata/'
data_dir = os.path.join(output_dir, dataname)
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    print("deleting cache")
    shutil.rmtree(data_dir)

edges_csv = os.path.join(output_dir, 'edges.csv')
nodes_csv = os.path.join(output_dir, 'nodes.csv')
graphs_csv = os.path.join(output_dir, 'graphs.csv')

for file_path in [edges_csv, nodes_csv, graphs_csv]:
    if os.path.exists(file_path):
        os.remove(file_path)

process_multiple_graphs(graph_files, tour_files, output_dir)
