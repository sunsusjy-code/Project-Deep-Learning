# week11/vanilla_gnn_cora.py（PPT邻接矩阵GNN复现）
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root=".", name="Cora") #数据集：cora
data = dataset[0]

# 创建归一化邻接矩阵（加自环），转为稀疏
edge_index = data.edge_index
num_nodes = data.num_nodes #拿到“图”里所有的边的连接信息，以及总共有多少个节点。
adj = torch.zeros((num_nodes, num_nodes)) #创建一个全零的矩阵，尺寸是 节点数 × 节点数 这是一个空白的表格，用来表示节点之间有没有连接。
adj += torch.eye(num_nodes) #加上自环（每个点自己和自己连）把对角线都变成1，意思是每个节点都和自己有一条边（自环）。
deg = adj.sum(1) #算每个节点的度（有多少连接）每一行加起来，知道每个节点总共连了多少条边（包括自己）
deg_inv = deg.pow(-1) #计算每个节点的“度的倒数” 就是度分之一，后面用于归一化
deg_inv[deg_inv == float('inf')] = 0 #把无穷大，譬如度是0导致的都变为0 避免出错，虽然加了自环一般不会有这种情况
D_inv = torch.diag(deg_inv) #把这些倒数塞进对角矩阵（用一条对角线存这些数，别的地方都是0）
adj_normalized = torch.mm(D_inv, adj) #做归一化：把邻接矩阵的每一行都除以这个节点有多少边
adj_sparse = adj_normalized.to_sparse() #变成稀疏格式（节省空间）

class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
        #初始化参数和内部结构（init）
        #in_dim 和 out_dim: 输入、输出的特征维数。
        #self.linear: 一个线性变换（全连接），常见于神经网络，用来把每个节点的特征从 in_dim 变成 out_dim。
        #bias=False 是说没有偏置项，让层更基础。
    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        return x

class VanillaGNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(in_dim, hid_dim)
        self.gnn2 = VanillaGNNLayer(hid_dim, out_dim)
        #初始化：init（模型结构搭建） 一层gnn1输入到隐藏层维度，一层gnn2输出到类别数。
    def forward(self, x, adj):
        h = self.gnn1(x, adj)
        h = torch.relu(h)
        h = self.gnn2(h, adj)
        return F.log_softmax(h, dim=1)
    #前向传播：forward（输入到输出变换流程）
    #节点特征x先过第一层GNN并聚合邻居信息。
    #然后过ReLU激活（加非线性）。
    #然后再过第二层GNN。
    #最后输出经过log-softmax（用于分类任务，且是对数形式以供交叉熵损失用）。
    def fit(self, data, adj, epochs=100):  #开始多轮循环（epochs次）
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4) #创建优化器（Adam），设定学习率和正则项。
        criterion = torch.nn.CrossEntropyLoss() #创建损失函数（交叉熵损失）。
        for epoch in range(epochs+1):
            self.train()#切换到训练模式
            optimizer.zero_grad()#梯度清零。
            out = self(data.x, adj)#得到输出
            loss = criterion(out[data.train_mask], data.y[data.train_mask])#计算loss，只在训练集mask范围。
            acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()#计算训练准确率。
            loss.backward()#反向传播
            optimizer.step()#参数更新
         #每20轮打印一次验证集和训练集指标
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])#out[data.val_mask]是预测值 data.y[data.val_mask]是真实标签 这个函数是损失函数（如交叉熵），计算“预测”与“真实标签”之间的不一致程度（数字越低越好）。
                val_pred = out[data.val_mask].argmax(dim=1)#得到模型对验证集上每个节点的最终预测标签。.argmax(dim=1) 沿着类别维度，找到得分最高的那个类别的索引，就是模型对于每个节点“认为最可能”的类别
                val_acc = (val_pred == data.y[data.val_mask]).float().mean()#计算验证集的准确率（猜对的比例）。
                #val_pred == data.y[data.val_mask] 把预测和真实标签逐个比对，能得到一串布尔值（True=猜对，False=猜错）。
                #.float() 把True/False变成1.0/0.0。
                #.mean() 求平均值，就是分类正确的比例（比如0.87代表87%的验证集节点分类正确）。
                print(f"Epoch {epoch:3d}|Train Loss:{loss:.3f}|Train Acc:{acc*100:5.2f}%|ValLoss:{val_loss:.2f}|ValAcc:{val_acc*100:.2f}%")
                #打印这一个周期的训练与验证集的各种指标。
        print(f"\nGNN test accuracy: {(out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).float().mean()*100:.2f}%")
        #在训练循环结束后，打印最终的测试集准确率。
gnn = VanillaGNN(dataset.num_features, 16, dataset.num_classes)
print(gnn)
gnn.fit(data, adj_sparse, 100)