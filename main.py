import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_nodes_from([1, 2, 3])
print(G.nodes())
# >>> [1, 2, 3]
G.add_edge(1, 2)
print(G.edges())
# >>> [(1, 2)]


nx.draw(G)
plt.savefig("graphs/main_graph.png")
