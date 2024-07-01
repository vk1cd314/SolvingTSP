from graph import Graph
import random
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def gen_random_conn_graph(n, width=100, height=100):
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

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Graph of Edges')
    plt.grid(True)
    plt.savefig("images/densegraph.pgf")
    