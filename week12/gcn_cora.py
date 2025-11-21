# week12/gcn_cora.py
import torch #PyTorch基础包，做深度学习用。
import torch.nn.functional as F #包含激活函数和损失函数。
from torch_geometric.datasets import Planetoid # PyTorch Geometric自带的引文网络类数据集，包括Cora、CiteSeer、PubMed。
from torch_geometric.nn import GCNConv #图卷积层，是GCN的核心。

def accuracy(y_pred, y_true):#写了一个准确率计算函数
    return (y_pred == y_true).sum().item() / len(y_true)#y_pred == y_true 这一步产生的就是布尔向量 直接 .sum() 就得到“True”的数量，即预测对了几个。最后除以总数得到比例

torch.manual_seed(1)#设置随机种子，保证每次跑出来的结果一致。
dataset = Planetoid(root=".", name="Cora")#加载Cora数据集：节点=论文，边=引用关系，特征=词袋描述，标签=论文种类。
data = dataset[0] #Cora只有一个大图，dataset[0]就是整个图的数据。就是把这个唯一的大图取出来并赋值给 data。

class GCN(torch.nn.Module):#GCN模型结构
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)#搭建两层图卷积网络

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index) #第一层卷积
        h = torch.relu(h) #激活函数
        h = self.gcn2(h, edge_index) #第二层卷积
        return F.log_softmax(h, dim=1) #对每个结点做类别概率输出

    def fit(self, data, epochs=100):
        criterion = torch.nn.CrossEntropyLoss()#交叉熵损失函数，适合做分类问题。
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4) #用Adam优化器，学习率0.01，权重衰减5e-4 self.parameters()指当前模型的全部可学习参数。

        for epoch in range(epochs+1):#一共训练epochs轮，每一轮叫一个epoch。
            self.train() #模型切换到训练模式（会启用Dropout、BatchNorm等）。
            optimizer.zero_grad() #清空梯度，防止上次反向传播的梯度影响本次。
            out = self(data.x, data.edge_index) #用输入特征x和图的连接关系edge_index，跑一遍前向传播。out形状是[节点数, 类别数]，表示对每个节点的所有类别logit。 
            loss = criterion(out[data.train_mask], data.y[data.train_mask]) #算损失，只在训练集上求交叉熵损失。
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])#把训练集节点的输出做argmax(dim=1)（每行最大值的索引，相当于预测类别），与真实标签用accuracy函数算准确率。
            loss.backward()#反向传播，计算所有参数的梯度
            optimizer.step()#根据梯度更新参数，使模型更适合训练数据。
            if epoch % 20 == 0:#输出一次训练和验证的效果。
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])#用同样的方法挑出验证集节点的输出和标签，算验证损失。
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])#验证集上的准确率。
                print(f"Epoch {epoch:3d}|Train Loss:{loss:.3f}|Train Acc:{acc*100:5.2f}%|ValLoss:{val_loss:.2f}|ValAcc:{val_acc*100:.2f}%")
    @torch.no_grad()#测试时不需要计算和保存梯度，节约内存和运算量。
    def test(self, data):
        self.eval()#模型切换到评估模式（禁用Dropout/BatchNorm等训练特性）。
        out = self(data.x, data.edge_index)#跑一次前向传播，获取所有节点输出。只关心测试集节点的预测和标签。
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])#计算测试集准确率。
        return acc #输出准确率。

gcn = GCN(dataset.num_features, 16, dataset.num_classes) #dataset.num_features： Cora数据集每个节点的特征有多少维。16 隐藏层节点数，可以理解为网络“中间大脑”的宽度。dataset.num_classes： 最终要分几类（比如Cora有7类论文）。
print(gcn)
gcn.fit(data, epochs=100)#**开始训练模型！**让gcn在数据data上用fit方法，训练100轮（epoch）
acc = gcn.test(data)#试模型的准确率。
print(f"\nGCN test accuracy: {acc*100:.2f}%\n")