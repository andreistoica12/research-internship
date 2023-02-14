import networkx as nx
import matplotlib.pyplot as plt
import os

if not os.path.exists('graphs'):
    os.mkdir('graphs')


G = nx.Graph()
G.add_nodes_from([1, 2, 3])
print(G.nodes())
# >>> [1, 2, 3]
a = 3
b = 5
c = a + b
G.add_edge(1, 2)
print(G.edges())
# >>> [(1, 2)]


nx.draw(G)
plt.savefig("graphs/main_graph.png")
