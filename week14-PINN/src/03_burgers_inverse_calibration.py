import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🕵️ 使用设备: {device}")

# ==========================================
# 第一阶段：上帝模式 (生成高质量的“激波”数据)
# ==========================================
print("\n🤖 阶段一：上帝模式 - 生成观测数据...")
# ⚠️ 修正点：我们需要一个高质量的 Teacher，否则侦探会被误导！

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Teacher 网络可以稍微宽一点，确保能拟合激波
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(), 
            nn.Linear(20, 20), nn.Tanh(), 
            nn.Linear(20, 20), nn.Tanh(), 
            nn.Linear(20, 1)
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

teacher = TeacherNet().to(device)
# 学习率可以先大后小，这里简单处理
optim_teacher = torch.optim.Adam(teacher.parameters(), lr=0.005)
real_nu = 0.01 / np.pi

print("   -> 正在严格训练 Teacher (这需要一点时间)...")

# ⚠️ 修正点：增加训练次数到 5000，确保激波出现！
for i in range(5001): 
    x = (torch.rand(2000, 1)*2-1).to(device).requires_grad_(True)
    t = torch.rand(2000, 1).to(device).requires_grad_(True)
    
    # 物理 Loss
    u = teacher(x, t)
    grads = torch.autograd.grad(u, [x, t], torch.ones_like(u), create_graph=True)
    u_x, u_t = grads[0], grads[1]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u*u_x - real_nu*u_xx
    loss_f = torch.mean(f**2)
    
    # IC Loss (t=0)
    x_ic = (torch.rand(500, 1)*2-1).to(device)
    t_ic = torch.zeros(500, 1).to(device)
    u_ic = -torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((teacher(x_ic, t_ic) - u_ic)**2)
    
    # BC Loss (x=-1, 1)
    x_bc = torch.vstack([torch.ones(200, 1), -torch.ones(200, 1)]).to(device)
    t_bc = torch.rand(400, 1).to(device)
    loss_bc = torch.mean((teacher(x_bc, t_bc))**2)

    # ⚠️ 修正点：给 IC/BC 极大的权重，强迫 Teacher 学会对
    loss = loss_f + 20.0 * loss_ic + 20.0 * loss_bc
    
    optim_teacher.zero_grad(); loss.backward(); optim_teacher.step()
    
    if i % 1000 == 0:
        print(f"      Teacher Iter {i}, Loss: {loss.item():.4f}")

# --- 生成观测数据 ---
num_obs = 1000 # 增加观测点数量
x_obs = (torch.rand(num_obs, 1) * 2 - 1).to(device)
t_obs = torch.rand(num_obs, 1).to(device)
with torch.no_grad():
    u_obs = teacher(x_obs, t_obs) 

print(f"   -> 生成了 {num_obs} 个高质量观测点！")


# ==========================================
# 第二阶段：侦探模式 (Project B 核心)
# ==========================================
print("\n🕵️ 阶段二：侦探模式 - 开始反推参数...")

class InversePhysicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 1)
        )
        # 初始猜测 nu = exp(-1.0) ≈ 0.36 (故意猜错，猜得很大)
        self.nu_log = nn.Parameter(torch.tensor(-1.0)) 

    def forward(self, x, t):
        return self.linear_relu_stack(torch.cat([x, t], dim=1))
    
    def get_nu(self):
        return torch.exp(self.nu_log)

def compute_inverse_loss(model, x, t, x_obs, t_obs, u_obs):
    # 1. 物理 Loss (和之前一样)
    u = model(x, t)
    grads = torch.autograd.grad(u, [x, t], torch.ones_like(u), create_graph=True)
    u_x, u_t = grads[0], grads[1]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    current_nu = model.get_nu()
    f = u_t + u * u_x - current_nu * u_xx
    loss_f = torch.mean(f**2)
    
    # 2. 数据 Loss (关键！新增的！) 🚨
    # 网络预测的观测点数据，必须和我们给的“线索”一致
    u_pred_obs = model(x_obs, t_obs)
    loss_data = torch.mean((u_pred_obs - u_obs)**2)
    
    return loss_f, loss_data

# 准备 PDE 配点
t_pde = torch.rand(2000, 1).to(device).requires_grad_(True)
x_pde = (torch.rand(2000, 1) * 2 - 1).to(device).requires_grad_(True)

model = InversePhysicsNetwork().to(device)

# === 修改前 ===
#optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # 稍微调大一点学习率
# === 修改后：给 Nu 开个“VIP 加速通道” ===
# 1. 把参数分成两组：一组是 nu，一组是网络权重
nu_params = [model.nu_log]
net_params = [p for p in model.parameters() if p is not model.nu_log]

# 2. 给 nu 设置 10 倍的学习率 (0.05)，网络权重保持 0.005
optimizer = torch.optim.Adam([
    {'params': net_params, 'lr': 0.005},
    {'params': nu_params, 'lr': 0.05}  # 🔥 猛踩油门！
])

history_nu = []
real_target = 0.01 / np.pi

print(f"🎯 真实 Nu: {real_target:.6f}")
print(f"🎬 初始 Nu: {model.get_nu().item():.6f}")

for i in range(8000):#3. 顺便把训练次数加到 5000 或 8000
    optimizer.zero_grad()
    
    loss_f, loss_data = compute_inverse_loss(model, x_pde, t_pde, x_obs, t_obs, u_obs)
    
    # 总 Loss = 物理误差 + 数据误差 (给数据误差加个权重，让它重视线索)
    total_loss = loss_f + 100 * loss_data
    
    total_loss.backward()
    optimizer.step()
    
    current_nu = model.get_nu().item()
    history_nu.append(current_nu)
    
    if i % 200 == 0:
        err = abs(current_nu - real_target)/real_target * 100
        print(f"Iter {i}, Loss: {total_loss.item():.4f}, DataLoss: {loss_data.item():.4f}, 🕵️ Nu: {current_nu:.6f} (Err: {err:.1f}%)")

# 画图
plt.figure()
plt.plot(history_nu)
plt.axhline(y=real_target, color='r', linestyle='--', label='True Nu')
plt.title("Detective Progress: Finding Nu")
plt.xlabel("Iter")
plt.ylabel("Nu Value")
plt.legend()
plt.savefig("nu_fixed.png")
print("✅ 完成！结果已保存为 nu_fixed.png")