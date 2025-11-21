# week10_deepwalk_karate.py
# 严格按 PPT 参数与流程：Random walk → Word2Vec(skip-gram, hs=1) → 相似度 → TSNE → RandomForest
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(1)
np.random.seed(1)

# ----- Load dataset -----
G = nx.karate_club_graph()

# Process labels (Mr. Hi = 0, Officer = 1)
labels = []
for node in G.nodes:
    club = G.nodes[node]['club']
    labels.append(1 if club == 'Officer' else 0)

# Plot graph (optional image like PPT)
plt.figure(figsize=(8,8))
plt.axis('off')
nx.draw_networkx(
    G, pos=nx.spring_layout(G, seed=0),
    node_color=labels, cmap='coolwarm',
    node_size=800, font_size=12, font_color='white'
)
plt.tight_layout()
plt.savefig('karate_graph.png', dpi=150)

# ----- Random Walks -----
def random_walk(G, start, length):
    walk = [str(start)]
    current = start
    for _ in range(length):
        neighbors = list(G.neighbors(current))
        next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        current = next_node
    return walk

# 每个节点生成 80 条、长度 10 的随机游走（与 PPT 叙述一致）
num_walks_per_node = 80
walk_length = 10
walks = []
for node in G.nodes():
    for _ in range(num_walks_per_node):
        walks.append(random_walk(G, node, walk_length))

# ----- Word2Vec (Skip-gram, Hierarchical Softmax) -----
# PPT 参数：sg=1, hs=1, vector_size=100, window=10, workers=1, seed=1, epochs=30
model = Word2Vec(
    vector_size=100,
    window=10,
    min_count=0,
    sg=1,
    hs=1,
    workers=1,
    seed=1
)
model.build_vocab(walks)
# 注：部分 gensim 版本不支持 report_delay 参数，如报错请去掉该参数
model.train(walks, total_examples=model.corpus_count, epochs=30)

print(f"Shape of embedding matrix: {model.wv.vectors.shape}")

# ----- Similarity checks (Network homophily) -----
print("Nodes that are the most similar to node 0:")
for w, sim in model.wv.most_similar(positive=['0'], topn=10):
    print((w, float(sim)))
print(f"Similarity between node 0 and 4: {float(model.wv.similarity('0','4'))}")

# ----- TSNE visualization -----
num_nodes = G.number_of_nodes()
nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(num_nodes)])
labels_arr = np.array(labels)

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0)
tsne_emb = tsne.fit_transform(nodes_wv)

plt.figure(figsize=(6,6))
plt.scatter(tsne_emb[:,0], tsne_emb[:,1], s=100, c=labels_arr, cmap='coolwarm')
plt.tight_layout()
plt.savefig('tsne_karate.png', dpi=150)

# ----- Classification with RandomForest (PPT 的划分) -----
train_mask = [2,4,6,8,10,12,14,16,18,20,22,24]
test_mask = [0,1,3,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29,30,31,32,33]

clf = RandomForestClassifier(random_state=0)
clf.fit(nodes_wv[train_mask], labels_arr[train_mask])
y_pred = clf.predict(nodes_wv[test_mask])
acc = accuracy_score(y_pred, labels_arr[test_mask])
print(f"Accuracy={acc*100:.2f}%")