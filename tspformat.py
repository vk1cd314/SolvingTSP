from solvis import euclidean_distance

INF = 1e6

def format_for_testing(data, filename='graphs/test_graph.tsp'):
    nodes = set()
    for edge in data:
        nodes.update(edge)

    nodes = sorted(nodes)

    node_index = {node: i + 1 for i, node in enumerate(nodes)}

    with open(filename, 'w') as f:
        f.write('NAME: Graph\n')
        f.write('TYPE: TSP\n')
        f.write(f'DIMENSION: {len(nodes)}\n')
        f.write('EDGE_WEIGHT_TYPE: EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')

        for node, index in node_index.items():
            f.write(f'{index} {node[0]} {node[1]}\n')

        f.write('EOF\n')

    index_node = {i + 1: node for i, node in enumerate(nodes)}
    return index_node

def format_euclidean_to_tsplib(data, filename='graphs/original_graph.tsp'):
    nodes = set()
    for edge in data:
        nodes.update(edge)

    nodes = sorted(nodes)

    node_index = {node: i + 1 for i, node in enumerate(nodes)}

    index_node = {i + 1: node for i, node in enumerate(nodes)}
    matrix = {}
    n = len(nodes)
    global INF
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            matrix[(i, j)] = INF 
    
    for edge in data:
        mn = min(node_index[edge[0]], node_index[edge[1]])
        mx = max(node_index[edge[0]], node_index[edge[1]])
        matrix[(mn, mx)] = euclidean_distance(edge[0], edge[1]) 
        matrix[(mx, mn)] = euclidean_distance(edge[0], edge[1]) 

    with open(filename, 'w') as f:
        f.write('NAME: Graph\n')
        f.write('TYPE: TSP\n')
        f.write(f'DIMENSION: {len(nodes)}\n')
        f.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        f.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
        f.write('EDGE_WEIGHT_SECTION\n')
        val = 0
        for i in range(1, n + 1):
            s = ''
            for j in range(1, n + 1):
                s += str(round(matrix[(i, j)], 2)) + ' '
            val += n - (i + 1) + 1
            if len(s) != 0:
                f.write(s + '\n')
        f.write('EOF\n')

    return index_node

# def format_euclidean_to_qs(data, filename='graphs/original_graph.qs'):
#     nodes = set()
#     for edge in data:
#         nodes.update(edge)

#     nodes = sorted(nodes)
#     node_index = {node: i + 1 for i, node in enumerate(nodes)}
#     index_node = {i + 1: node for i, node in enumerate(nodes)}

#     with open(filename, 'w') as f:
#         # Write node count and edge count
#         f.write(f"{len(nodes)} {len(edges)}\n")

#         # Write node coordinates
#         for node in nodes:
#             f.write(f"{node[0]} {node[1]}\n")

#         # Write edge information
#         for edge in edges:
#             f.write(f"{node_index[edge[0]]} {node_index[edge[1]]} {edge[2]}\n")

#     print(f"QS format file '{filename}' has been created.")
#     return index_node

def format_to_tsplib_edge_list(data, filename='graphs/sparsified_graph.tsp'):
    nodes = set()
    edges = set()
    for edge, _ in data:
        nodes.update(edge)
        edges.add((edge[0], edge[1]))
        edges.add((edge[1], edge[0]))
    print(edges)

    nodes = sorted(nodes)
    node_index = {node: i + 1 for i, node in enumerate(nodes)}
    index_node = {i + 1: node for i, node in enumerate(nodes)}
    matrix = {}
    n = len(nodes)
    global INF
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            matrix[(i, j)] = INF 
    
    for edge, _ in data:
        mn = min(node_index[edge[0]], node_index[edge[1]])
        mx = max(node_index[edge[0]], node_index[edge[1]])
        matrix[(mn, mx)] = euclidean_distance(edge[0], edge[1]) 

    with open(filename, 'w') as f:
        f.write('NAME: Graph\n')
        f.write('TYPE: TSP\n')
        f.write(f'DIMENSION: {len(nodes)}\n')
        f.write('EDGE_WEIGHT_TYPE: EUC_2D\n')
        f.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        f.write('NODE_COORD_TYPE: TWOD_COORDS\n')
        f.write('NODE_COORD_SECTION\n')
        for node in nodes:
            f.write(f"{node_index[node]} {node[0]} {node[1]}\n")
        f.write('EDGE_DATA_SECTION\n')
        for edge in edges:
            f.write(f"{node_index[edge[0]]} {node_index[edge[1]]}\n")
        f.write('-1\n')
        f.write('EOF\n')

    return index_node

def format_to_qs(data, filename='graphs/sparsified_graph.qs'):
    nodes = set()
    edges = []
    for edge, info in data:
        nodes.update(edge)
        edges.append((edge[0], edge[1], euclidean_distance(edge[0], edge[1])))

    nodes = sorted(nodes)
    node_index = {node: i + 1 for i, node in enumerate(nodes)}
    index_node = {i + 1: node for i, node in enumerate(nodes)}

    with open(filename, 'w') as f:
        f.write(f"{len(nodes)} {len(edges)}\n")

        for node in nodes:
            f.write(f"{node[0]} {node[1]}\n")

        for edge in edges:
            f.write(f"{node_index[edge[0]]} {node_index[edge[1]]} {edge[2]}\n")

    return index_node
