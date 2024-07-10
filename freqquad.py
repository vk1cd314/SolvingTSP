import math
import random

def getFreq(a, b, c, d):
    ab = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) + random.uniform(1e-7, 1e-3)
    bc = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2) + random.uniform(1e-7, 1e-3)
    cd = math.sqrt((c[0] - d[0])**2 + (c[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    ad = math.sqrt((a[0] - d[0])**2 + (a[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    ac = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2) + random.uniform(1e-7, 1e-3)
    bd = math.sqrt((b[0] - d[0])**2 + (b[1] - d[1])**2) + random.uniform(1e-7, 1e-3)
    
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

def do_one_iter(random_graph, N):
    edge_freq = {}   
    edge_freq_avg = {}

    for edge in random_graph.get_edges():
        u, v = edge
        have_pts = set()
        for vertex in random_graph.get_vertices():
            if u != vertex and v != vertex:
                have_pts.add(vertex)
        
        freq_quads = set()
        edge_freq[edge] = 0

        have_pts = list(have_pts)
        f_quads = []
        for i in range(N):
            f_quads.append(random.sample(have_pts, 2))
        
        i = 0
        while i < len(f_quads):
            pt1, pt2 = f_quads[i]
            freq_dict = getFreq(u, v, pt1, pt2)
            if (u, v, min(pt1, pt2), max(pt1, pt2)) not in freq_quads:
                edge_freq[edge] += freq_dict[f'{u, v}']
            freq_quads.add((u, v, min(pt1, pt2), max(pt1, pt2)))
            i += 1
        edge_freq_avg[edge] = edge_freq[edge] / N

    sorted_freq = sorted(edge_freq_avg.items(), key=lambda item: item[1])
    
    sorted_data = sorted(sorted_freq, key=lambda x: x[1], reverse=True)

    two_thirds_length = int(len(sorted_data) * (2/3))

    top_two_thirds = sorted_data[:two_thirds_length]

    return top_two_thirds
