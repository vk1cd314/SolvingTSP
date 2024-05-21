from utils import gen_random_conn_graph, plot_graph
import random
import math

random_graph = gen_random_conn_graph(20)
plot_graph(random_graph, add_edges=True)

N = 20
print(len(random_graph.get_edges()))
def getFreq(a, b, c, d):
    ab = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    bc = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2)
    cd = math.sqrt((c[0] - d[0])**2 + (c[1] - d[1])**2)
    ad = math.sqrt((a[0] - d[0])**2 + (a[1] - d[1])**2)
    ac = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)
    bd = math.sqrt((b[0] - d[0])**2 + (b[1] - d[1])**2)
    
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

for edge in random_graph.get_edges():
    # 2 points fixed 
    u, v = edge
    have_pts = set()
    for vertex in random_graph.get_vertices():
        if u != vertex and v != vertex:
            have_pts.add(vertex)
    
    freq_quads = []
    edge_freq[edge] = 0
    
    while len(freq_quads) < N and len(have_pts) >= 2:
        pt1, pt2 = random.sample(have_pts, 2)
        random_graph.remove_vertex(pt1)
        random_graph.remove_vertex(pt2)
        freq_dict = getFreq(u, v, pt1, pt2)
        edge_freq[edge] += freq_dict[f'{u, v}']
        freq_quads.append((u, v, pt1, pt2))



import math

