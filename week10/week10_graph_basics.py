# week10_graph_basics.py
# 完全依照 PPT 的示例，修正了 OCR 的语法问题，输出与 PPT 一致的图与打印信息
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ---------- Undirected Graph ----------
G = nx.Graph()
G.add_edges_from([
    ('A','B'), ('A','C'),
    ('B','D'), ('B','E'),
    ('C','F'), ('C','G')
])
plt.figure(figsize=(5,5))
plt.axis('off')
nx.draw_networkx(
    G,
    pos=nx.spring_layout(G, seed=0),
    node_size=600,
    cmap='coolwarm',
    font_size=14,
    font_color='white'
)
plt.tight_layout()
plt.savefig('undirected_graph.png', dpi=150)

# ---------- Directed Graph (DiGraph) ----------
DG = nx.DiGraph()
DG.add_edges_from([
    ('A','B'), ('A','C'),
    ('B','D'), ('B','E'),
    ('C','F'), ('C','G')
])
plt.figure(figsize=(5,5))
plt.axis('off')
nx.draw_networkx(
    DG,
    pos=nx.spring_layout(G, seed=0),
    node_size=600,
    cmap='coolwarm',
    font_size=14,
    font_color='white',
    arrows=True
)
plt.tight_layout()
plt.savefig('directed_graph.png', dpi=150)

# ---------- Weighted Graph ----------
WG = nx.Graph()
WG.add_edges_from([
    ('A','B', {'weight':10}),
    ('A','C', {'weight':20}),
    ('B','D', {'weight':30}),
    ('B','E', {'weight':40})
])
pos_w = nx.spring_layout(WG, seed=0)
labels = nx.get_edge_attributes(WG, 'weight')
plt.figure(figsize=(5,5))
plt.axis('off')
nx.draw_networkx(
    WG, pos=pos_w,
    node_size=600, cmap='coolwarm',
    font_size=14, font_color='white'
)
nx.draw_networkx_edge_labels(WG, pos=pos_w, edge_labels=labels)
plt.tight_layout()
plt.savefig('weighted_graph.png', dpi=150)

# ---------- Connected Graph ----------
G1 = nx.Graph()
G1.add_edges_from([(1,2),(2,3),(3,1),(4,5)])
G2 = nx.Graph()
G2.add_edges_from([(1,2),(2,3),(3,1),(1,4)])
print(f"Is graph 1 connected? {nx.is_connected(G1)}")
print(f"Is graph 2 connected? {nx.is_connected(G2)}")
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.axis('off')
nx.draw_networkx(G1, pos=nx.spring_layout(G1, seed=0),
                 node_size=600, cmap='coolwarm', font_size=14, font_color='white')
plt.title('Not connected')
plt.subplot(1,2,2); plt.axis('off')
nx.draw_networkx(G2, pos=nx.spring_layout(G2, seed=0),
                 node_size=600, cmap='coolwarm', font_size=14, font_color='white')
plt.title('Connected')
plt.tight_layout()
plt.savefig('connected_graphs.png', dpi=150)

# ---------- Degree / In-degree / Out-degree ----------
print(f"deg(A) = {G.degree['A']}")
print(f"deg^-(A) = {DG.in_degree['A']}")
print(f"deg^+(A) = {DG.out_degree['A']}")

# ---------- Centrality ----------
print("Degree centrality =", nx.degree_centrality(G))
print("Closeness centrality =", nx.closeness_centrality(G))
print("Betweenness centrality =", nx.betweenness_centrality(G))

# ---------- Adjacency matrix / list 示例 ----------
adj_matrix = np.array([
    [0,1,1,0,0,0,0],
    [1,0,0,1,1,0,0],
    [1,0,0,0,0,1,1],
    [0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
], dtype=int)
edge_list = [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
adj_list = {
    0:[1,2], 1:[0,3,4], 2:[0,5,6],
    3:[1], 4:[1], 5:[2], 6:[2]
}

# ---------- BFS ----------
def bfs(graph, start):
    visited, queue = [start], [start]
    while queue:
        node = queue.pop(0)
        for nb in graph[node]:
            if nb not in visited:
                visited.append(nb)
                queue.append(nb)
    return visited

# ---------- DFS ----------
def dfs(visited, graph, node):
    if node not in visited:
        visited.append(node)
        for nb in graph[node]:
            dfs(visited, graph, nb)
    return visited

print("BFS from 0:", bfs(adj_list, 0))
print("DFS from 0:", dfs([], adj_list, 0))