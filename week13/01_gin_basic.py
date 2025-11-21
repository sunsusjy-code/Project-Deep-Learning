import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
print(f'数据集: {dataset}')
print(f'图数量: {len(dataset)}')
print(f'节点特征数: {dataset.num_features}')
print(f'类别数: {dataset.num_classes}')

# 数据统计（用于PPT）
print(f'平均节点数: {dataset.data.num_nodes / len(dataset):.2f}')
print(f'平均边数: {dataset.data.num_edges / len(dataset):.2f}')

# 2. 定义GIN模型
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GIN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # 输入层
        self.convs.append(GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        ))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # 隐藏层
        for i in range(num_layers - 2):
            self.convs.append(GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                )
            ))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        
        # 输出层
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout层
        self.dropout = torch.nn.Dropout(p=0.5)  # 这里设置 dropout 率
        
    def forward(self, x, edge_index, batch):
        # 节点嵌入
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        x = self.dropout(x)  # 在每层后加上 dropout

        # 图级池化：求和（GIN的关键！）
        x = global_add_pool(x, batch)
        
        # 分类
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# 3. 训练和测试函数
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # 计算训练准确率
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(loader)
    avg_accuracy = correct / len(loader.dataset)
    
    return avg_loss, avg_accuracy

def test(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            total_loss += F.nll_loss(out, data.y, reduction='sum').item()  # sum up batch loss
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    
    avg_loss = total_loss / len(loader.dataset)  # average loss per sample
    avg_accuracy = correct / len(loader.dataset)
    
    return avg_loss, avg_accuracy

# 4. 主训练循环
def main():
    # 确保 dataset 已经在外部定义并可被使用
    dataset.shuffle()
    train_dataset = dataset[:800]
    test_dataset = dataset[800:]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 模型初始化
    model = GIN(input_dim=dataset.num_features, 
                hidden_dim=128, 
                output_dim=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练过程记录（用于PPT图表）
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(200):
        train_loss, train_accuracy = train(model, train_loader, optimizer)
        test_loss, test_accuracy = test(model, test_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # 打印训练和测试结果
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
    
    # 5. 可视化结果
    plt.figure(figsize=(12, 4))
    
    # 训练损失可视化
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 测试准确率可视化
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/gin_training_proteins.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()