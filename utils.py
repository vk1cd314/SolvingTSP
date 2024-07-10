from graph import Graph
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from solvis import euclidean_distance
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def gen_random_conn_graph(n, seed=69420, width=300, height=100):
    graph = Graph()
    
    max_points = width * height
    if n > max_points:
        raise ValueError(f"Cannot select {n} points from a {width}x{height} grid, as it only contains {max_points} points.")
    
    all_points = [(x, y) for x in range(width) for y in range(height)]
    
    random.seed(seed)
    selected_points = random.sample(all_points, n)
    
    for point in selected_points:
        graph.add_vertex(point)
    
    for i in range(len(selected_points)):
        for j in range(i + 1, len(selected_points)):
            graph.add_edge(selected_points[i], selected_points[j])

    return graph

def plot_graph(graph, add_edges=False):
    G = nx.Graph()
    
    for vertex in graph.get_vertices():
        G.add_node(vertex)

    if add_edges:
        for vertex in graph.get_vertices():
            for adjacent_vertex in graph.adjacency_list[vertex]:
                G.add_edge(vertex, adjacent_vertex)
    
    pos = {vertex: vertex for vertex in graph.get_vertices()}
    
    plt.figure(figsize=(2.2, 2.2))
    
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        plt.plot([x1, x2], [y1, y2], color='blue')
    
    for vertex in G.nodes():
        x, y = pos[vertex]
        plt.plot(x, y, marker='o', color='black')

    plt.axis('off')
    plt.title('Dense Graph')
    plt.savefig("images/densegraph.pgf")

def graph_to_file(graph):
    file = open("graphs/original_graph", "w")
    file.write(f"{len(graph.get_vertices())} {len(graph.get_edges())}\n")
    node_to_index = {node: i for i, node in enumerate(graph.get_vertices())}
    index_to_node = {i: node for i, node in enumerate(graph.get_vertices())}
    for edge in graph.get_edges():
        file.write(f"{node_to_index[edge[0]]} {node_to_index[edge[1]]} {round(round(euclidean_distance(edge[0], edge[1]), 2) * 100)}\n")
    file.close()
    return index_to_node

def draw_tsp_solution(index_to_node, tsp_solution, filename_prefix='results/tsp_step'):
    """
    Draws the TSP solution step by step and saves each step as an image.
    
    :param index_to_node: Dictionary mapping node index to (x, y) coordinates.
    :param tsp_solution: List of node indexes representing the TSP solution.
    :param filename_prefix: Prefix for the saved image files.
    """
    coords = [index_to_node[i] for i in tsp_solution]
    N = len(coords) - 1
    
    plt.figure(figsize=(30, 10))
    
    x, y = zip(*coords)
    node_size = 200
    plt.scatter(x, y, color='blue', s=node_size)  # Increase size of the nodes with 's' parameter
    
    for i, (xi, yi) in enumerate(coords):
        plt.annotate(i % N, (xi, yi), textcoords="offset points", xytext=(0,10), ha='center', fontsize=24)  # Increase fontsize
    
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], 'lightgrey', linewidth=3)
    
    plt.axis('off')
    plt.savefig(f'{filename_prefix}_0.pgf')
    
    for step in range(1, len(coords)):
        x_segment = [coords[step-1][0], coords[step][0]]
        y_segment = [coords[step-1][1], coords[step][1]]
        plt.plot(x_segment, y_segment, 'ro-', markersize=node_size ** 0.5, linewidth=3)
        
        plt.arrow(x_segment[0], y_segment[0], x_segment[1] - x_segment[0], y_segment[1] - y_segment[0],
                  head_width=0.05, head_length=0.1, fc='red', ec='red', linewidth=3)
        
        plt.axis('off')
        plt.savefig(f'{filename_prefix}_{step}.pgf')
    
    plt.close()