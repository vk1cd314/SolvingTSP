from graph import Graph
import random

def generate_random_fully_connected_graph(n, width=100, height=100):
    graph = Graph()
    
    max_points = width * height
    if n > max_points:
        raise ValueError(f"Cannot select {n} points from a {width}x{height} grid, as it only contains {max_points} points.")
    
    all_points = [(x, y) for x in range(width) for y in range(height)]
    
    selected_points = random.sample(all_points, n)
    
    for point in selected_points:
        graph.add_vertex(point)
    
    for i in range(len(selected_points)):
        for j in range(i + 1, len(selected_points)):
            graph.add_edge(selected_points[i], selected_points[j])

    return graph

random_graph = generate_random_fully_connected_graph(100)
random_graph.display()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(graph, filename):
    G = nx.Graph()
    
    for vertex in graph.get_vertices():
        G.add_node(vertex)
    
    # for vertex in graph.get_vertices():
    #     for adjacent_vertex in graph.adjacency_list[vertex]:
    #         G.add_edge(vertex, adjacent_vertex)
    
    pos = {vertex: vertex for vertex in graph.get_vertices()}
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=8, font_color='black')
    
    plt.savefig(filename, format='PNG')
    plt.close()

plot_graph(random_graph, 'fully_connected_graph.png')