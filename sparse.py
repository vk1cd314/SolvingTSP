from graph import Graph
graph = Graph()
graph.add_vertex('A')
graph.add_vertex('B')
graph.add_vertex('C')
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('B', 'C')

graph.display()

print("Vertices:", graph.get_vertices())
print("Edges:", graph.get_edges())

graph.remove_edge('A', 'B')
graph.display()

graph.remove_vertex('C')
graph.display()
