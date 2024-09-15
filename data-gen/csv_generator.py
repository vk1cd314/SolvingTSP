import csv

def read_graph(graph_file):
    """Reads the input graph file and returns a list of edges (src, dst, weight)."""
    edges = []
    with open(graph_file, 'r') as f:
        # Read the number of nodes and edges (not used directly in processing)
        num_nodes, num_edges = map(int, f.readline().strip().split())
        
        # Read all the edges
        for line in f:
            src, dst, weight = line.strip().split()
            edges.append((int(src), int(dst), int(weight)))
    return edges

def read_tour(tour_file):
    """Reads the tour file and returns a set of edges (src, dst) in the tour."""
    tour_edges = set()
    with open(tour_file, 'r') as f:
        # Read the number of nodes in the tour (not used directly)
        num_nodes = int(f.readline().strip())
        
        # Read the node sequence of the tour
        tour = list(map(int, f.read().split()))
        
        # Generate edges from the tour sequence
        for i in range(len(tour) - 1):
            src = tour[i]
            dst = tour[i + 1]
            # Add the edge to the set (smallest node first for consistency)
            tour_edges.add((min(src, dst), max(src, dst)))
            
        # Also connect the last node to the first to form a cycle
        src = tour[-1]
        dst = tour[0]
        tour_edges.add((min(src, dst), max(src, dst)))
    
    return tour_edges

def generate_edge_csv(graph_file, tour_file, output_csv, graph_id):
    """Generates the edge.csv file from the graph and tour files."""
    edges = read_graph(graph_file)
    print(edges[0])
    tour_edges = read_tour(tour_file)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        # Write the header
        writer.writerow(['graph_id', 'src_id', 'dst_id',  'feat'])
        
        # Write the edges
        for src, dst, weight in edges:
            # Sort the src, dst pair to maintain consistency
            edge = (min(src, dst), max(src, dst))
            # Check if this edge is in the tour
            label = 1 if edge in tour_edges else 0
            # Write the row: graph_id, src_id, dst_id, label, feat
            writer.writerow([graph_id, src, dst,  f'{weight}'])


def generate_nodes_csv(graph_file, output_csv, graph_id):
    """Generates the nodes.csv file from the graph file."""
    with open(graph_file, 'r') as f:
        # Read the number of nodes from the first line of the graph file
        num_nodes, _ = map(int, f.readline().strip().split())
    
    # Create the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        # Write the header
        writer.writerow(['graph_id', 'node_id', 'feat'])
        
        # Write each node with label and feat set to 0
        for node_id in range(num_nodes):
            writer.writerow([graph_id, node_id,  f'0'])

# Example usage
graph_number = 0  # Replace with the actual graph number
<<<<<<< HEAD
input_graph_file = f'../graph_out/input_graph.txt'
tour_file = f'../graph_out/input_graph.sol'
output_csv_file = '../dglgraph/edges.csv'
output_csv_node_file = '../dglgraph/nodes.csv'
=======
input_graph_file = f'input_graph.txt'
tour_file = f'input_graph.sol'
output_csv_file = './dglgraph/edges.csv'
output_csv_node_file = './dglgraph/nodes.csv'
>>>>>>> main

# Generate the CSV file
generate_edge_csv(input_graph_file, tour_file, output_csv_file, graph_number)
generate_nodes_csv(input_graph_file, output_csv_node_file, graph_number)

