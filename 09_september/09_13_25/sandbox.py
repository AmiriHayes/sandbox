"""
NetworkX Daily Practice
-----------------------
Practice includes:
- Creating small graphs from nodes/edges
- Saving graphs to disk with os
- Basic graph operations & algorithms
- Visualization with nx.draw
- Importing large graphs with requests
- Timing algorithms with time
"""

# ============================
# 📦 Imports & Setup
# ============================
import os
import time
import requests
import networkx as nx
import matplotlib.pyplot as plt

# Define working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Example node and edge lists for small graph
NODES = ["A", "B", "C", "D", "E"]
EDGES = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "E")]

# URL for larger graph dataset (e.g. Karate Club from networkx repo)
GRAPH_URL = "https://raw.githubusercontent.com/networkx/networkx/main/examples/graph/karate.gml"

# ============================
# 🔹 Small Graph Practice
# ============================

# Create empty graph
G_small = nx.Graph()

# Add nodes & edges
G_small.add_nodes_from(NODES)
G_small.add_edges_from(EDGES)

print("Small graph created with:")
print(f"  Nodes: {G_small.nodes()}")
print(f"  Edges: {G_small.edges()}")

# Save graph to same directory
small_graph_path = os.path.join(BASE_DIR, "small_graph.gml")
nx.write_gml(G_small, small_graph_path)
print(f"Graph saved to {small_graph_path}")

# Check graph properties
print("\nGraph Properties:")
print(f"  Number of nodes: {G_small.number_of_nodes()}")
print(f"  Number of edges: {G_small.number_of_edges()}")
print(f"  Degree of A: {G_small.degree('A')}")

# Basic algorithms
print("\nBasic Algorithms on Small Graph:")
print(f"  Shortest path A → D: {nx.shortest_path(G_small, 'A', 'D')}")
print(f"  Connected components: {list(nx.connected_components(G_small))}")

# Visualization
plt.figure(figsize=(6, 4))
nx.draw(G_small, with_labels=True, node_color="lightblue", node_size=800, font_size=12)
plt.title("Small Graph Example")
plt.show()

# ============================
# 🔹 Larger Graph Practice
# ============================

# Download GML file for larger graph
# response = requests.get(GRAPH_URL)
# graph_file = os.path.join(BASE_DIR, "karate.gml")
# with open(graph_file, "wb") as f:
#     f.write(response.content)

# # Read the graph
# G_big = nx.read_gml(graph_file, label="id")

G_big = nx.karate_club_graph()

print("\nLarge graph loaded (Karate Club):")
print(f"  Nodes: {G_big.number_of_nodes()}")
print(f"  Edges: {G_big.number_of_edges()}")

# Timing algorithms
print("\nTiming algorithms on big graph:")

start = time.time()
degrees = dict(G_big.degree())
end = time.time()
print(f"  Degree calculation took {end - start:.6f} seconds")

start = time.time()
clustering = nx.clustering(G_big)
end = time.time()
print(f"  Clustering coefficient calculation took {end - start:.6f} seconds")

start = time.time()
shortest_paths = dict(nx.shortest_path_length(G_big))
end = time.time()
print(f"  All-pairs shortest paths took {end - start:.6f} seconds")

# Visualization of larger graph
plt.figure(figsize=(7, 5))
nx.draw(G_big, with_labels=True, node_color="lightgreen", node_size=600, font_size=10)
plt.title("Karate Club Graph")
plt.show()

# ============================
# 🔹 Extra: Graph Variants
# ============================

# Directed graph
G_directed = nx.DiGraph()
G_directed.add_edges_from([("X", "Y"), ("Y", "Z"), ("Z", "X")])
print("\nDirected Graph Example:")
print(f"  Edges: {G_directed.edges()}")

# Weighted graph
G_weighted = nx.Graph()
G_weighted.add_weighted_edges_from([("A", "B", 3.0), ("B", "C", 1.5)])
print("\nWeighted Graph Example:")
for u, v, d in G_weighted.edges(data=True):
    print(f"  Edge {u}-{v} weight: {d['weight']}")

# Save weighted graph
weighted_graph_path = os.path.join(BASE_DIR, "weighted_graph.gml")
nx.write_gml(G_weighted, weighted_graph_path)
print(f"Weighted graph saved to {weighted_graph_path}")

# ============================
# ✅ End of Practice
# ============================
print("\nPractice complete ✅")
