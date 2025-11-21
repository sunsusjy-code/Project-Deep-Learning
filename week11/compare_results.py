# week12/compare_results.py
import pandas as pd

data = {
    "Model": ["MLP", "Vanilla GNN", "GCN", "GAT"],
    "Test Accuracy (%)": [53.00, 76.60, 79.70, 81.10]
}
df = pd.DataFrame(data)
print(df.to_markdown(index=False))