import os
import time
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Smaller graph

NODES = ["A", "B", "C", "D", "E"]
EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "E")]

G_small = nx.Graph()
G_small.add_nodes_from(NODES)
G_small.add_edges_from(EDGES)

print("Small graph created with: ")
print(f"\tNodes: {G_small.nodes()}")
print(f"\tEdges: {G_small.edges()}")

small_graph_path = os.path.join(BASE_DIR, "small_graph.gml")
nx.write_gml(G_small, small_graph_path)
print(f"Graph saved to {small_graph_path}")

print("\nGraph Properties")
print(f"\t# Nodes: {G_small.number_of_nodes()}")
print(f"\t# Edges: {G_small.number_of_edges()}")
print(f"\tDegree of A: {G_small.degree('A')}")

print("\nBasic Algorithms on Small Graph")
print(f"\tShortest path A->D: {nx.shortest_path(G_small, 'A', 'D')}")
print(f"\tConnected components: {list(nx.connected_components(G_small))}")

plt.figure(figsize=(6,4))
nx.draw(G_small, with_labels=True, node_color="lightblue", node_size=800, font_size=12)
plt.title("Small graph example")
plt.show()

# Bigger Graph

G_big = nx.karate_club_graph()

print("\nLarge graph loaded (Karate Club)")
print(f"\tNodes: {G_big.number_of_nodes()}")
print(f"\tEdges: {G_big.number_of_edges()}")

print("\nTiming algorithms")
start = time.time()
degrees = dict(G_big.degree())
end = time.time()
print(f"\tDegree calcluation took {end - start:.3f} seconds")

start = time.time()
clustering = nx.clustering(G_big)
end = time.time()
print(f"\tClustering coefficient calculation took {end - start:.3f} seconds")

start = time.time()
shortest_paths = dict(nx.shortest_path_length(G_big))
end = time.time()
print(f"\tAll pairs shortest paths calculation took {end - start:.3f} seconds")

plt.figure(figsize=(7,5))
nx.draw(G_big, with_labels=True, node_color="lightgreen", node_size=600, font_size=10)
plt.title("Karate Club Graph")
plt.show()

# Other stuff

G_directed = nx.DiGraph()
G_directed.add_edges_from([("X", "Y"), ("Y", "Z"), ("Z", "X")])
print("\nDirected Graph Example:")
print(f"\tEdges: {G_directed.edges()}")

G_weighted = nx.Graph()
G_weighted.add_weighted_edges_from([("A", "B", 3.0), ("B", "C", 1.5)])
print("\nWeighted Graph Example:")
for u, v, d in G_weighted.edges(data=True):
    print(f"\tEdge {u}-{v} weight: {d['weight']}")

weighted_graph_path = os.path.join(BASE_DIR, "weighted_graph.gml")
nx.write_gml(G_weighted, weighted_graph_path)
print(f"Weighted graph saved to {weighted_graph_path}")

# done
print("\nfinished practicing")