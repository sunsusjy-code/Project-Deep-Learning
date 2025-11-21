from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
print(f'数据集: {dataset}')
print(f'图数量: {len(dataset)}')