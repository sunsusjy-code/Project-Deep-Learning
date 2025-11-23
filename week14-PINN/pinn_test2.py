import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 0. 设置
# 如果有显卡就用 CUDA，否则用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 定义神经网络
class PhysicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: 2个变量 (x, t), 输出: 1个变量 (u)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        # 把 x 和 t 拼起来作为输入
        inputs = torch.cat([x, t], dim=1)
        output = self.linear_relu_stack(inputs)
        return output

# 2. 物理损失函数 (Boss 关卡)
def compute_loss(model, x, t):
    u = model(x, t)
    
    # 计算梯度
    # create_graph=True 是为了后面能计算二阶导数
    grads = torch.autograd.grad(outputs=u, inputs=[x, t], 
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)
    u_x = grads[0] # 对 x 的导数
    u_t = grads[1] # 对 t 的导数
    
    # 计算二阶导数 (u_xx)
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x, 
                               grad_outputs=torch.ones_like(u_x), 
                               create_graph=True)[0]
    
    # Burgers' Equation (物理方程)
    nu = 0.01 / np.pi
    f = u_t + u * u_x - nu * u_xx
    
    # 返回物理残差的均方误差
    return torch.mean(f**2)

# --- 数据准备 ---

# A. 初始条件 (IC): t=0 时, u = -sin(pi*x)
t_ic = torch.zeros(100, 1).to(device)
x_ic = (torch.rand(100, 1) * 2 - 1).to(device)
u_ic = -torch.sin(np.pi * x_ic)

# B. 边界条件 (BC): x=-1 或 1 时, u = 0
x_bc = torch.vstack([torch.ones(50, 1), -torch.ones(50, 1)]).to(device)
t_bc = torch.rand(100, 1).to(device)
u_bc = torch.zeros(100, 1).to(device)

# C. PDE 配点 (物理约束): 在时空区域内随机撒点
t_pde = torch.rand(2000, 1).to(device).requires_grad_(True)
x_pde = (torch.rand(2000, 1) * 2 - 1).to(device).requires_grad_(True)

# --- 训练循环 ---

model = PhysicsNetwork().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🚀 开始训练 Burgers' Equation...")

for i in range(3000): # 训练 3000 次
    optimizer.zero_grad()
    
    # 1. 物理 Loss (让方程成立)
    loss_f = compute_loss(model, x_pde, t_pde)
    
    # 2. IC Loss (初始条件)
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic)**2)
    
    # 3. BC Loss (边界条件)
    u_bc_pred = model(x_bc, t_bc)
    loss_bc = torch.mean((u_bc_pred - u_bc)**2)
    
    # 4. 总 Loss
    total_loss = loss_f + loss_ic + loss_bc
    total_loss.backward()
    optimizer.step()
    
    if i % 200 == 0:
        print(f"Iter {i}, Total: {total_loss.item():.5f}, PDE: {loss_f.item():.5f}, IC: {loss_ic.item():.5f}, BC: {loss_bc.item():.5f}")

print("✅ 训练完成!")

# --- 5. 可视化结果 (画出漂亮的彩色图) ---
print("🎨 正在绘制结果...")

# 1. 生成网格 (Grid)
# 我们把时空切成 100x100 的小方格来画图
x_vals = np.linspace(-1, 1, 100)
t_vals = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_vals, t_vals)

# 2. 准备输入数据
# 需要把网格拉平，变成 (10000, 2) 的形状喂给模型
X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
T_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)

# 3. 模型预测
with torch.no_grad(): # 预测时不需要求导
    u_pred = model(X_flat, T_flat).cpu().numpy()

# 4. 把预测结果变回网格形状 (100, 100)
U_pred = u_pred.reshape(100, 100)

# 5. 画图
plt.figure(figsize=(10, 6))
# 使用 pcolormesh 画热力图
plt.pcolormesh(T, X, U_pred, cmap='jet', shading='auto')
plt.colorbar(label='Velocity u') # 颜色条
plt.xlabel('Time t')
plt.ylabel('Position x')
plt.title("Burgers' Equation Solution (PINN)")
plt.savefig("burgers_solution.png")
print("🖼️ 结果已保存为 burgers_solution.png，快去打开看看！")