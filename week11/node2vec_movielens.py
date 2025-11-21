# week11/node2vec_movielens.py
import pandas as pd
import networkx as nx
from collections import defaultdict
from node2vec import Node2Vec

# 下载/读取 MovieLens 100k
import os, zipfile
from urllib.request import urlretrieve

if not os.path.exists('ml-100k/u.data'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    urlretrieve(url, 'ml-100k.zip')
    with zipfile.ZipFile('ml-100k.zip') as zf:
        zf.extractall('.')

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id','movie_id','rating','unix_timestamp'])
ratings = ratings[ratings.rating >= 4]

pairs = defaultdict(int)
for _, group in ratings.groupby('user_id'):
    movies = list(group['movie_id'])
    for i in range(len(movies)):
        for j in range(i+1, len(movies)):
            pair = tuple(sorted((movies[i], movies[j])))
            pairs[pair] += 1

G = nx.Graph()
for (m1, m2), weight in pairs.items():
    if weight >= 20:
        G.add_edge(str(m1), str(m2), weight=weight)
print("Total number of graph nodes:", G.number_of_nodes())
print("Total number of graph edges:", G.number_of_edges())

node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 载入电影名映射
item_path = 'ml-100k/u.item'
movies = pd.read_csv(item_path, sep='|', header=None, encoding='latin-1', usecols=[0,1], names=['movie_id','title'])
def recommend(title):
    movie_id = str(movies[movies.title == title].movie_id.values[0])
    print(f"Top 5 similar to: {title}")
    for item in model.wv.most_similar(movie_id)[:5]:
        name = movies[movies.movie_id == int(item[0])].title.values[0]
        print(f"{name}: {item[1]:.2f}")

recommend('Star Wars (1977)')