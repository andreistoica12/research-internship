import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import nxviz as nv
import os

print(f'NetworkX version: {nx.__version__}')
print(f'Matplotlib version: {mpl.__version__}')


# if there is no graphs/ folder within the current directory, create it
if not os.path.exists('graphs'):
    os.mkdir('graphs')


G = nx.Graph()
G.add_nodes_from([1, 2, 3])
print(G.nodes())
# >>> [1, 2, 3]
G.add_edge(1, 2)
print(G.edges())
# >>> [(1, 2)]


nx.draw(G)
plt.savefig("graphs/main_graph.png")
