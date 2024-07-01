from solvis import euclidean_distance

def format_to_tsplib(data, filename='graphs/original_graph.tsp'):
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

    print(f"TSPLIB file '{filename}' has been created.")

def format_to_tsplib_edge_list(data, filename='graphs/sparsified_graph.tsp'):
    nodes = set()
    for edge, _ in data:
        nodes.update(edge)

    nodes = sorted(nodes)
    node_index = {node: i + 1 for i, node in enumerate(nodes)}
    matrix = {}
    n = len(nodes)
    INF = 1e7
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            matrix[(i, j)] = INF
    # for i, node in enumerate(nodes):
    #     matrix[(node_index[node[0]], node_index[node[1]])] = euclidean_distance(node[0], node[1])
    
    for edge, _ in data:
        matrix[(node_index[edge[0]], node_index[edge[1]])] = euclidean_distance(edge[0], edge[1])
        matrix[(node_index[edge[1]], node_index[edge[0]])] = euclidean_distance(edge[0], edge[1])
        

    with open(filename, 'w') as f:
        f.write('NAME: Graph\n')
        f.write('TYPE: TSP\n')
        f.write(f'DIMENSION: {len(nodes)}\n')
        f.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        f.write('NODE_COORD_TYPE: TWOD_COORDS\n')
        f.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
        f.write('NODE_COORD_SECTION\n')
        for node, index in node_index.items():
            f.write(f'{index} {float(node[0])} {float(node[1])}\n')
        f.write('EDGE_WEIGHT_SECTION\n')
        for i in range(1, n + 1):
            s = ''
            for j in range(1, n + 1):
                s += str(matrix[(i, j)]) + ' '
            f.write(s + '\n')
        f.write('EOF\n')

    print(f"TSPLIB file '{filename}' has been created.")
