import networkx as nx
import numpy as np
import random

# Parameters
n_nodes = 100
n_classes = 3
p_same = 0.8   # Probability of connecting to same label
p_diff = 0.2   # Probability of connecting to different label

# Assign labels
labels = np.random.randint(0, n_classes, size=n_nodes)

# Build graph
G = nx.Graph()
G.add_nodes_from(range(n_nodes))

for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        if labels[i] == labels[j]:
            if random.random() < p_same:
                G.add_edge(i, j)
        else:
            if random.random() < p_diff:
                G.add_edge(i, j)

# Compute homophily values
def compute_node_homophily(G, labels):
    homophily = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            homophily[node] = 0
        else:
            same = sum(labels[n] == labels[node] for n in neighbors)
            homophily[node] = same / len(neighbors)
    return homophily

homophily_values = compute_node_homophily(G, labels)