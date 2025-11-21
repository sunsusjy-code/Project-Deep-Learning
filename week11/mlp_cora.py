# week11/mlp_cora.py (PPT逻辑完整还原)
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import pandas as pd

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim_in, dim_h)
        self.linear2 = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

    def fit(self, data, epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs+1):
            self.train()
            optimizer.zero_grad()
            out = self(data.x)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_pred = out[data.val_mask].argmax(dim=1)
                val_acc = (val_pred == data.y[data.val_mask]).float().mean()
                print(f"Epoch {epoch:3d}|Train Loss:{loss:.3f}|Train Acc:{acc*100:5.2f}%|ValLoss:{val_loss:.2f}|ValAcc:{val_acc*100:.2f}%")
        print(f"\nMLP test accuracy: {(out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).float().mean()*100:.2f}%")

mlp = MLP(dataset.num_features, 16, dataset.num_classes)
print(mlp)
mlp.fit(data, 100)