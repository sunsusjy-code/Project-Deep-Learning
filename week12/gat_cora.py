# week12/gat_cora.py
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv

def accuracy(y_pred, y_true):#写了一个准确率计算函数
    return (y_pred == y_true).sum().item() / len(y_true)#y_pred == y_true 这一步产生的就是布尔向量 直接 .sum() 就得到“True”的数量，即预测对了几个。最后除以总数得到比例

dataset = Planetoid(root=".", name="Cora")#加载Cora数据集：节点=论文，边=引用关系，特征=词袋描述，标签=论文种类。
data = dataset[0]#Cora只有一个大图，dataset[0]就是整个图的数据。就是把这个唯一的大图取出来并赋值给 data。

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)#表示多头注意力，每个节点会用8种“关注”方式聚合邻居信息。
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)#输入维度要乘以头数，因为多头拼接特征输出。heads=1，不再多头，直接输出到类别数。

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)#在输入和第一层输出前后都加Dropout，训练时随机丢掉60%特征，有助防止过拟合。
        h = self.gat1(h, edge_index)#会对每个节点，把自己的特征和所有邻居的特征用注意力加权平均（每个头会有不同的关注方式），拼接成更丰富的新特征。
        h = F.elu(h)#增强非线性表达能力
        h = F.dropout(h, p=0.6, training=self.training)#在输入和第一层输出前后都加Dropout，训练时随机丢掉60%特征，有助防止过拟合。
        h = self.gat2(h, edge_index)#heads=1，最终输出类别数的logits。
        return F.log_softmax(h, dim=1)#针对每个节点做归一化，输出可用于交叉熵损失

    def fit(self, data, epochs=100):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.01)
        for epoch in range(epochs+1):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f"Epoch {epoch:3d}|TrainLoss:{loss:.3f}|Train Acc:{acc*100:5.2f}%|Val Loss:{val_loss:.2f}|ValAcc:{val_acc*100:.2f}%")
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc

gat = GAT(dataset.num_features, 32, dataset.num_classes, heads=8)
print(gat)
gat.fit(data, epochs=100)
acc = gat.test(data)
print(f"GAT test accuracy: {acc*100:.2f}%")