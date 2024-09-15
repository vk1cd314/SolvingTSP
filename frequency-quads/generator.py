import sys
import random

def generate_fully_connected_graph(num_nodes):
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = random.randint(50, 500)  # Random weight between 50 and 500
            edges.append((i, j, weight))
    
    return edges

def write_graph_to_file(num_nodes, num_edges, edges, filename):
    with open(filename, 'w') as f:
        f.write(f"{num_nodes} {num_edges}\n")
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_graph.py <num_nodes> <filename>")
        sys.exit(1)

    try:
        num_nodes = int(sys.argv[1])
        filename = sys.argv[2]
    except ValueError:
        print("Error: Number of nodes must be an integer.")
        sys.exit(1)

    edges = generate_fully_connected_graph(num_nodes)
    num_edges = len(edges)

    write_graph_to_file(num_nodes, num_edges, edges, filename)
    print(f"Graph with {num_nodes} nodes and {num_edges} edges generated in '{filename}'.")
