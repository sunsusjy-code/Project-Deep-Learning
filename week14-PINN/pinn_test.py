import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义神经网络 (这部分没变)
class PhysicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t):
        output = self.linear_relu_stack(t)
        return output

# 2. 定义 PINN 的损失函数 (核心部分!) 
def compute_loss(model, t):
    # --- A. 物理损失 (Physics Loss) ---
    # 我们需要 t 的导数，所以必须开启 requires_grad
    y = model(t)
    
    # 自动求导：计算 dy/dt
    # create_graph=True 是为了让导数也能参与反向传播训练
    dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    
    # 物理方程的残差：Residual = dy/dt + 2y
    # 我们希望这个 Residual 越接近 0 越好
    physics_loss = torch.mean((dy_dt + 2*y)**2)
    
    # --- B. 初始条件损失 (IC Loss) ---
    # 我们希望 t=0 时，y=1
    t_0 = torch.zeros(1, 1).to(t.device)
    y_0 = model(t_0)
    ic_loss = (y_0 - 1)**2
    
    # 总 Loss = 物理 Loss + IC Loss
    return physics_loss + ic_loss

# 3. 开始训练
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PhysicsNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练数据：在 0 到 1 秒之间随机采样 100 个点
t_train = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True).to(device)

print("🚀 开始训练...")
for i in range(2000):
    optimizer.zero_grad()
    loss = compute_loss(model, t_train)
    loss.backward()
    optimizer.step()
    
    if i % 200 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}")

print("✅ 训练完成！")

# 4. 画图验证
with torch.no_grad(): # 预测时不需要求导
    t_test = torch.linspace(0, 1, 100).view(-1, 1).to(device)
    y_pred = model(t_test).cpu().numpy()
    y_true = np.exp(-2 * t_test.cpu().numpy()) # 真实解析解

plt.figure(figsize=(8,5))
plt.plot(t_test.cpu(), y_true, 'k--', label='Exact Solution (e^-2t)')
plt.plot(t_test.cpu(), y_pred, 'r-', label='PINN Prediction')
plt.legend()
plt.title("PINN for dy/dt = -2y")
plt.savefig("pinn_result.png") # 保存图片
print("📊 结果图已保存为 pinn_result.png")

print("Liang shuxuan is handsome")