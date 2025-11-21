import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root=".", name="CiteSeer")
data = dataset[0]

edge_index = data.edge_index
num_nodes = data.num_nodes
adj = torch.zeros((num_nodes, num_nodes))
adj[edge_index[0], edge_index[1]] = 1
adj += torch.eye(num_nodes)
deg = adj.sum(1)
deg_inv = deg.pow(-1)
deg_inv[deg_inv == float('inf')] = 0
D_inv = torch.diag(deg_inv)
adj_normalized = torch.mm(D_inv, adj)
adj_sparse = adj_normalized.to_sparse()

class VanillaGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        return x

class VanillaGNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gnn1 = VanillaGNNLayer(in_dim, hid_dim)
        self.gnn2 = VanillaGNNLayer(hid_dim, out_dim)
    def forward(self, x, adj):
        h = self.gnn1(x, adj)
        h = torch.relu(h)
        h = self.gnn2(h, adj)
        return F.log_softmax(h, dim=1)
    def fit(self, data, adj, epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs+1):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, adj)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_pred = out[data.val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean()
                print(f"Epoch {epoch:3d}|Train Loss:{loss:.3f}|Train Acc:{acc*100:5.2f}%|ValLoss:{val_loss:.2f}|ValAcc:{val_acc*100:.2f}%")
        print(f"\nGNN test accuracy: {(out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).float().mean()*100:.2f}%")

gnn = VanillaGNN(dataset.num_features, 16, dataset.num_classes)
print(gnn)
gnn.fit(data, adj_sparse, 100)