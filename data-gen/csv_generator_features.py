import csv
import os
import random
import shutil

def read_features(feature_file):
    """Reads a feature file and returns a dictionary of features {(src, dst): feature_value}."""
    features = {}
    with open(feature_file, 'r') as f:
        num_nodes, num_edges = map(int, f.readline().strip().split())
        for line in f:
            src, dst, feat_value = line.strip().split()
            features[(int(src), int(dst))] = float(feat_value)
    return features
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

def append_edges_csv(edges, tour_edges, output_csv, graph_id, feature_dicts=None):
    """Appends the edges to edges.csv file, including multiple features."""
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for src, dst, weight in edges:
            label = 0
            if (src, dst) in tour_edges:
                label = 1
            # Prepare feature string: include weight and optional features
            feature_values = [weight]  # Start with weight
            if feature_dicts:
                for feature_dict in feature_dicts:
                    feature_values.append(feature_dict.get((src, dst), 0))  # Add feature, or 0 if missing
            # Convert all features to string
            feature_str = ','.join(map(str, feature_values))
            writer.writerow([graph_id, src, dst, label, feature_str])

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


def process_multiple_graphs(graph_files, tour_files, feature_files_dict, output_dir):
    """Processes multiple graph and tour files, along with optional feature files."""
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

        # Read feature files for this graph if they exist
        feature_dicts = []
        for feature_name, feature_files in feature_files_dict.items():
            feature_file = feature_files[graph_id] if graph_id < len(feature_files) else None
            if feature_file and os.path.exists(feature_file):
                feature_dicts.append(read_features(feature_file))

        append_edges_csv(edges, tour_edges, edges_csv, graph_id, feature_dicts)
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
features_path = './features/'
output_dir = './test-data/'
dataname = 'tspdata/'
feature_files_dict = {
    'fa': [os.path.join(features_path, 'fa', f) for f in sorted(os.listdir(os.path.join(features_path, 'fa')))],
    'fb': [os.path.join(features_path, 'fb', f) for f in sorted(os.listdir(os.path.join(features_path, 'fb')))],
    'fc': [os.path.join(features_path, 'fc', f) for f in sorted(os.listdir(os.path.join(features_path, 'fc')))],
    'fd': [os.path.join(features_path, 'fc', f) for f in sorted(os.listdir(os.path.join(features_path, 'fc')))],
    'fe': [os.path.join(features_path, 'fc', f) for f in sorted(os.listdir(os.path.join(features_path, 'fc')))],
    'ff': [os.path.join(features_path, 'fc', f) for f in sorted(os.listdir(os.path.join(features_path, 'fc')))],
}
selected_features = ["fa", 'fb']
filtered_dict = {feature: files for feature, files in feature_files_dict.items() if feature in selected_features}

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

process_multiple_graphs(graph_files, tour_files, filtered_dict, output_dir)
