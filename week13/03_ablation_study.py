import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
import matplotlib.pyplot as plt

# 测试不同池化方式的GIN变体
class GINWithPooling(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling_type='sum'):
        super(GINWithPooling, self).__init__()
        self.pooling_type = pooling_type
        
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
        
        # 不同的池化方式
        if self.pooling_type == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            x = global_max_pool(x, batch)
        
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

def pooling_ablation_study():
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    dataset = dataset.shuffle()
    
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    pooling_types = ['sum', 'mean', 'max']
    results = {}
    
    for pooling in pooling_types:
        print(f'Testing {pooling} pooling...')
        model = GINWithPooling(
            dataset.num_features, 64, dataset.num_classes, pooling_type=pooling
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        best_acc = 0
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
            best_acc = max(best_acc, acc)
        
        results[pooling] = best_acc
        print(f'{pooling} pooling - Best Accuracy: {best_acc:.4f}')
    
    # 可视化池化方式对比
    plt.figure(figsize=(8, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = plt.bar(results.keys(), results.values(), color=colors, alpha=0.7)
    
    plt.title('Effect of Different Pooling Methods in GIN', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    
    # 在柱子上添加数值
    for bar, acc in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('results/pooling_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == '__main__':
    results = pooling_ablation_study()