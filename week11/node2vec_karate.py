# week11/node2vec_karate.py   # PPT源码完整复现
import random, numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

random.seed(0)
np.random.seed(0)
G = nx.karate_club_graph()
labels = [1 if G.nodes[n]['club']=='Officer' else 0 for n in G.nodes]

def node2vec_walk(G, start, walk_length, p, q):
    walk = [str(start)]
    for _ in range(walk_length):
        cur = int(walk[-1])
        neighbors = list(G.neighbors(cur))
        if len(walk) == 1:
            probs = np.ones(len(neighbors)) / len(neighbors)
        else:
            prev = int(walk[-2])
            probs = []
            for neighbor in neighbors:
                if neighbor == prev:
                    probs.append(1/p)
                elif G.has_edge(neighbor, prev):
                    probs.append(1)
                else:
                    probs.append(1/q)
            probs = np.array(probs)
            probs /= probs.sum()
        next_node = np.random.choice(neighbors, p=probs)
        walk.append(str(next_node))
    return walk

walks = []
for node in G.nodes:
    for _ in range(80):
        walks.append(node2vec_walk(G, node, 10, p=10, q=3))

model = Word2Vec(walks, hs=1, sg=1, vector_size=100, window=10, workers=2, seed=0, min_count=1)
model.train(walks, total_examples=model.corpus_count, epochs=30)
nodes_wv = np.array([model.wv.get_vector(str(i)) for i in range(G.number_of_nodes())])
labels_arr = np.array(labels)
train_mask=[2,4,6,8,10,12,14,16,18,20,22,24]
test_mask=[0,1,3,5,7,9,11,13,15,17,19,21,23,25,26,27,28,29,30,31,32,33]
clf=RandomForestClassifier(random_state=0)
clf.fit(nodes_wv[train_mask],labels_arr[train_mask])
y_pred=clf.predict(nodes_wv[test_mask])
acc=accuracy_score(y_pred,labels_arr[test_mask])
print(f'Node2Vec accuracy={acc*100:.2f}%')