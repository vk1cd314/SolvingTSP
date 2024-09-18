import csv
import os
import random
import shutil

# Constants for splitting the dataset
VAL_SPLIT = 0.1  # 10% validation
TEST_SPLIT = 0.2  # 20% test
# number of sparsified graphs
NUM_SPRS_GRPH = 1000
# test graphs number
TEST_THRESH = 200

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

def assign_split():
    """Assigns a split (train, val, test) based on predefined percentages."""
    rand = random.random()
    #if rand < TEST_SPLIT:
    #    return False, False, True  # test_mask is True
    if rand < TEST_SPLIT + VAL_SPLIT:
        return False, True, False  # val_mask is True
    else:
        return True, False, False  # train_mask is True

def append_edges_csv(edges, tour_edges, output_csv, graph_id, is_test):
    """Appends the edges to edges.csv file with train/val/test split."""
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for src, dst, weight in edges:
            label = 0
            if (src, dst) in tour_edges:
                label = 1
            train_mask, val_mask, test_mask = (False, False, True) if is_test else assign_split()
            writer.writerow([graph_id, src, dst, label, train_mask, val_mask, test_mask, f'{weight}'])

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

def process_multiple_graphs(graph_files, tour_files, output_dir, is_test):
    """Processes multiple graph and tour files and appends data to edges.csv, nodes.csv, and graphs.csv."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    edges_csv = os.path.join(output_dir, 'edges.csv')
    nodes_csv = os.path.join(output_dir, 'nodes.csv')
    graphs_csv = os.path.join(output_dir, 'graphs.csv')

    # Initialize the CSV files with headers if they don't exist
    if not os.path.exists(edges_csv):
        with open(edges_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['graph_id', 'src_id', 'dst_id', 'label', 'train_mask', 'val_mask', 'test_mask', 'feat'])

    if not os.path.exists(nodes_csv):
        with open(nodes_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['graph_id', 'node_id', 'feat'])

    if not os.path.exists(graphs_csv):
        with open(graphs_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['graph_id', 'feat', 'label'])

    # Process each graph and tour file pair
    for graph_id, (graph_file, tour_file) in enumerate(zip(graph_files, tour_files)):
        if is_test:
            graph_id += NUM_SPRS_GRPH
            if graph_id > TEST_THRESH + NUM_SPRS_GRPH:
                break

        edges, num_nodes = read_graph(graph_file)
        tour_edges = read_tour(tour_file)

        append_edges_csv(edges, tour_edges, edges_csv, graph_id, is_test)
        append_nodes_csv(num_nodes, nodes_csv, graph_id)
        append_graphs_csv(graphs_csv, graph_id)

# Directories for the graphs and tours
graph_path = './sparsified_graphs/'
sol_path = './solved_sparsified/'

graph_files = sorted([os.path.join(graph_path, f) for f in os.listdir(graph_path) if os.path.isfile(os.path.join(graph_path, f))])
tour_files = sorted([os.path.join(sol_path, f) for f in os.listdir(sol_path) if os.path.isfile(os.path.join(sol_path, f))])

# Output directory
output_dir = './generated-data/'
dataname = 'tspdata/'
data_dir = os.path.join(output_dir, dataname)

# Remove previous data
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    print("Deleting cache")
    shutil.rmtree(data_dir)

# Remove previous CSV files
for file_path in ['edges.csv', 'nodes.csv', 'graphs.csv']:
    file_full_path = os.path.join(output_dir, file_path)
    if os.path.exists(file_full_path):
        os.remove(file_full_path)

# Process the graphs and tours
process_multiple_graphs(graph_files, tour_files, output_dir, False)

# Adding test cases from the generated_graphs and solved_original directories
test_graph_path = './generated_graphs/'
test_sol_path = './solved_original/'

test_graph_files = sorted([os.path.join(test_graph_path, f) for f in os.listdir(test_graph_path) if os.path.isfile(os.path.join(test_graph_path, f))])
test_tour_files = sorted([os.path.join(test_sol_path, f) for f in os.listdir(test_sol_path) if os.path.isfile(os.path.join(test_sol_path, f))])

# Process the test cases (we assume they are for test set only)
process_multiple_graphs(test_graph_files, test_tour_files, output_dir, True)

