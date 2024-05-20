from utils import gen_random_conn_graph, plot_graph
import random

random_graph = gen_random_conn_graph(20)
plot_graph(random_graph, add_edges=True)

N = 20
print(len(random_graph.get_edges()))
for edge in random_graph.get_edges():
    # 2 points fixed 
    u, v = edge
    have_pts = set()
    for vertex in random_graph.get_vertices():
        if u != vertex and v != vertex:
            have_pts.add(vertex)
    
    freq_quads = []
    
    while len(freq_quads) < N and len(have_pts) >= 2:
        pt1, pt2 = random.sample(have_pts, 2)
        random_graph.remove_vertex(pt1)
        random_graph.remove_vertex(pt2)
        freq_quads.append((u, v, pt1, pt2))

    