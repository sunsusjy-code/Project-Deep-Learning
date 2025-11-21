# week11/deepwalk_karate.py   # PPT源码直接复现
import random, numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)
G = nx.karate_club_graph()
labels = [1 if G.nodes[n]['club']=='Officer' else 0 for n in G.nodes]

# 随机游走
def random_walk(G, start, length):
    walk = [str(start)]
    cur = start
    for _ in range(length):
        nb = list(G.neighbors(cur))
        nxt = np.random.choice(nb)
        walk.append(str(nxt))
        cur = nxt
    return walk

walks = []
for node in G.nodes:
    for _ in range(80):
        walks.append(random_walk(G, node, 10))

model = Word2Vec(walks, hs=1, sg=1, vector_size=100, window=10, workers=1, seed=1, min_count=0)
model.train(walks, total_examples=model.corpus_count, epochs=30)

nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(G.number_of_nodes())])
labels_arr = np.array(labels)
print(f"Shape of embedding matrix: {model.wv.vectors.shape}")

print("Nodes that are the most similar to node 0:")
for w, sim in model.wv.most_similar(positive=['0'], topn=10):
    print((w, float(sim)))
print(f"Similarity between node 0 and 4: {float(model.wv.similarity('0','4'))}")

tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0)
tsne_emb = tsne.fit_transform(nodes_wv)
plt.figure(figsize=(6,6))
plt.scatter(tsne_emb[:,0], tsne_emb[:,1], s=100, c=labels_arr, cmap='coolwarm')
plt.tight_layout()
plt.savefig('tsne_karate.png', dpi=150)

train_mask = [2,4,6,8,10,12,14,16,18,20,22,24]
test_mask = [0,1,3,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29,30,31,32,33]
clf = RandomForestClassifier(random_state=0)
clf.fit(nodes_wv[train_mask], labels_arr[train_mask])
y_pred = clf.predict(nodes_wv[test_mask])
acc = accuracy_score(y_pred, labels_arr[test_mask])
print(f"Accuracy={acc*100:.2f}%")