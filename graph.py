class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            self.adjacency_list[vertex1].append(vertex2)
            self.adjacency_list[vertex2].append(vertex1)

    def remove_vertex(self, vertex):
        if vertex in self.adjacency_list:
            for adjacent_vertex in self.adjacency_list[vertex]:
                self.adjacency_list[adjacent_vertex].remove(vertex)
            del self.adjacency_list[vertex]

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            if vertex2 in self.adjacency_list[vertex1]:
                self.adjacency_list[vertex1].remove(vertex2)
            if vertex1 in self.adjacency_list[vertex2]:
                self.adjacency_list[vertex2].remove(vertex1)

    def get_vertices(self):
        return list(self.adjacency_list.keys())

    def get_edges(self):
        edges = []
        for vertex in self.adjacency_list:
            for adjacent_vertex in self.adjacency_list[vertex]:
                if {vertex, adjacent_vertex} not in edges:
                    edges.append({vertex, adjacent_vertex})
        return edges

    def display(self):
        for vertex in self.adjacency_list:
            print(f"{vertex}: {self.adjacency_list[vertex]}")