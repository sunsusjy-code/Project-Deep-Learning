import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, global_mean_pool, global_add_pool
import matplotlib.pyplot as plt

# 定义输出文件路径
RESULTS_PATH = 'results/gin_gcn_sage_comparison.png'

# GIN模型
class GINModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)  # GIN使用求和池化
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# GCN模型
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # GCN通常使用均值池化
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# GraphSAGE模型
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # GraphSAGE也使用均值池化
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

def compare_models():
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    dataset = dataset.shuffle()
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    models = {
        'GIN': GINModel(dataset.num_features, 64, dataset.num_classes),
        'GCN': GCNModel(dataset.num_features, 64, dataset.num_classes),
        'GraphSAGE': GraphSAGEModel(dataset.num_features, 64, dataset.num_classes)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f'Training {name}...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        best_acc = 0
        accuracies = []
        
        for epoch in range(50):
            # 训练
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
            
            # 测试
            model.eval()
            correct = 0
            for data in test_loader:
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
            
            acc = correct / len(test_dataset)
            accuracies.append(acc)
            best_acc = max(best_acc, acc)
        
        results[name] = {
            'best_accuracy': best_acc,
            'accuracies': accuracies
        }
        print(f'{name} Best Accuracy: {best_acc:.4f}')
    
    # 可视化对比结果
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result['accuracies'], label=name, linewidth=2)
    
    plt.title('GIN vs GCN vs GraphSAGE on PROTEINS Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_PATH, dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == '__main__':
    results = compare_models()