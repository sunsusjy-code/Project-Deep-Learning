import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 创建一个简单的无向图
G = nx.Graph()
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
G.add_edges_from(edges)

# WL测试的迭代过程
iterations = 3
color_map = np.zeros(G.number_of_nodes())  # 初始化节点颜色

# 画出初始图
plt.figure(figsize=(15, 5))
for i in range(iterations):
    plt.subplot(1, iterations + 1, i + 1)
    
    # 更新节点颜色，模拟WL测试的过程
    color_map = (color_map + 1) % (i + 2)  # 简单循环颜色
    
    # 使用颜色绘制图
    nx.draw(G, node_color=color_map, with_labels=True, cmap='viridis', vmin=0, vmax=iterations)
    plt.title(f'Iteration {i + 1}')
    plt.axis('off')

# 保存图像
plt.subplot(1, iterations + 1, iterations + 1)
plt.axis('off')
plt.savefig('wl_testing_process.png', dpi=300, bbox_inches='tight')
plt.show()