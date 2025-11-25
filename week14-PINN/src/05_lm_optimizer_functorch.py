import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import functools

# 检查是否可以使用新的 torch.func (PyTorch 2.0+)
# 教授的 PPT 用的是旧版 functorch [cite: 417]，我们尽量兼容
try:
    from torch.func import vmap, grad, jacrev, functional_call
    print("✅ 使用 PyTorch 2.0+ 原生 torch.func")
except ImportError:
    from functorch import vmap, grad, jacrev, make_functional
    print("⚠️ 使用 functorch (旧版)，如果报错请升级 PyTorch")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用设备: {device}")

# --- 1. 定义物理问题 (Problem Setup) ---
# 对应 PPT [cite: 648-650]
# 我们要解方程: -Laplacian(u) = f
# 设定真实解 u = x^2 + y^2，那么 f 必须等于 -4

def exact_u(x, y):
    k = 1.0
    return (x**2 + y**2) / k

def get_f_value(x, y):
    return -4.0

print("✅ 模块一：物理问题定义完成")
# --- 2. 定义神经网络与函数化 (Model & Functionalization) ---
# 对应 PPT [cite: 194-198]

class Plain(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        # 教授 PPT 用的是 double 精度，我们也保持一致以提高 LM 稳定性
        self.ln1 = nn.Linear(in_dim, h_dim).double()
        self.act1 = nn.Tanh()
        self.ln2 = nn.Linear(h_dim, h_dim).double()
        self.act2 = nn.Tanh()
        self.ln3 = nn.Linear(h_dim, out_dim).double()

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.ln3(out)
        return out

# 初始化模型
# 2个输入(x,y) -> 20个隐藏神经元 -> 1个输出(u)
num_neuron = 20
model = Plain(2, num_neuron, 1).to(device)

# --- 核心黑科技：函数化 (Functionalization) ---
# 我们需要把参数(params)和模型结构(func_model)分离开
# 这样才能对 params 求导计算 Jacobian

# 提取参数 (params)
params = dict(model.named_parameters())

# 定义一个"纯函数"版本的模型 forward
# 输入: params, x
# 输出: u
# --- 修正后的 fnet_single ---
def fnet_single(params, x):
    # 不需要手动 unsqueeze，因为 nn.Linear 可以处理单个向量输入
    # 输入 x: [2] -> 输出: [1]
    out = functional_call(model, params, (x,))
    return out.squeeze() # 确保返回的是标量(scalar)，这对 grad 很重要

print("✅ 模块二：神经网络定义 & 函数化完成")
print(f"   - 参数数量: {sum(p.numel() for p in params.values())}")

# --- 3. 数据生成 & 残差计算 (Data & Residuals) ---
# [cite_start]对应 PPT [cite: 673-675, 697-698]

# 3.1 生成采样点 (Collocation Points)
# 简单的均匀网格 (简单起见，暂不用 Chebyshev)
# --- 修正后的数据生成 ---
cnt = 10
x_range = torch.linspace(-1, 1, cnt)
y_range = torch.linspace(-1, 1, cnt)
X, Y = torch.meshgrid(x_range, y_range, indexing='ij')

# ⚠️ 关键修改：加上 .double()
x_pde = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device).double()

# 3.2 定义单个点的残差函数 (Single Point Residual)
# 注意：这里输入是单个点 x (shape: [2])，输出是一个标量残差
def compute_pde_residual_single(params, x):
    # 计算 u 对 x 的一阶导 (Gradient) -> [du/dx, du/dy]
    # argnums=1 表示对第二个参数 x 求导
    grads = grad(fnet_single, argnums=1)(params, x)
    
    # 计算 u 对 x 的二阶导 (Hessian) -> [[u_xx, u_xy], [u_yx, u_yy]]
    # 也就是对 grads 再求一次导
    hess = jacrev(grad(fnet_single, argnums=1), argnums=1)(params, x)
    
    # 提取 Laplacian: u_xx + u_yy
    # hess[0,0] 是 u_xx, hess[1,1] 是 u_yy
    u_xx = hess[0, 0]
    u_yy = hess[1, 1]
    laplacian = u_xx + u_yy
    
    # PDE: -Delta u = f  =>  -Delta u - f = 0
    # 我们设定的 f 是 -4 (get_f_value)
    # 所以残差 = -laplacian - (-4)
    target_f = -4.0
    res = -laplacian - target_f
    return res

print("✅ 模块三：残差函数定义完成")

# --- 验证一下 vmap 是否工作 ---
# vmap 允许我们将"单点函数"自动变成"批量函数"
# in_dims=(None, 0) 表示: params 不变(None), x 按第0维批处理
batch_residual_fn = vmap(compute_pde_residual_single, in_dims=(None, 0))

# 试着算一次残差向量
r_vector = batch_residual_fn(params, x_pde)
print(f"   - 采样点数量: {x_pde.shape[0]}")
print(f"   - 残差向量形状: {r_vector.shape} (应该等于采样点数量)")
print(f"   - 初始 MSE Loss: {torch.mean(r_vector**2).item():.6f}")

# --- 4. LM 优化器实现 (Levenberg-Marquardt Optimization) ---
# [cite_start]对应 PPT [cite: 563, 574-577, 701-706]

# 4.1 辅助函数：计算残差向量 r 和 雅可比矩阵 J
def get_r_and_J(params, x):
    # 计算残差向量 r (shape: [N])
    r = batch_residual_fn(params, x)
    
    # 计算雅可比字典 J_dict
    # jacrev 会返回一个和 params 结构一样的字典
    # 字典里每个值的 shape 是 (N, param_shape)
    J_dict = jacrev(batch_residual_fn)(params, x)
    
    # --- 核心：把字典拍扁成矩阵 J (shape: [N, P]) ---
    J_list = []
    for name, val in J_dict.items():
        # val.shape: (N, d1, d2...) -> view -> (N, d1*d2...)
        # 例如 (100, 20, 20) -> (100, 400)
        N = val.shape[0]
        J_list.append(val.view(N, -1))
    
    # 在列维度拼接，形成巨大的 J 矩阵
    J = torch.cat(J_list, dim=1)
    return r, J

# 4.2 辅助函数：把扁平的更新向量加回参数字典
def update_params(params, delta_theta_flat):
    new_params = {}
    idx = 0
    for name, val in params.items():
        numel = val.numel() # 参数里的元素个数
        # 从扁平向量里切出一块
        delta_slice = delta_theta_flat[idx : idx + numel]
        # 恢复形状并相加
        new_params[name] = val + delta_slice.view(val.shape)
        idx += numel
    return new_params

# --- 4.3 主循环 (Main Loop) ---
print("\n🚀 开始 LM 优化 (Phase 2)...")

# 超参数设置 (参考 PPT 经验值)
mu = 1e-1          # 初始阻尼因子 (Damping Factor)
div_factor = 3.0   # 成功时 mu 减小的比例
mul_factor = 2.0   # 失败时 mu 增大的比例
max_iter = 100     # 迭代次数 (LM 收敛很快，通常不需要几千次)

# 记录 Loss
loss_history = []

for i in range(max_iter):
    # 1. 计算当前的 r 和 J
    r, J = get_r_and_J(params, x_pde)
    
    # 计算当前 Loss (MSE = mean(r^2))
    mse_loss = torch.mean(r**2)
    loss_history.append(mse_loss.item())
    
    # 2. 构建线性方程系统: (J.T @ J + mu * I) @ delta_theta = -J.T @ r
    # H_approx = J.T @ J (高斯牛顿近似海森矩阵)
    H = J.T @ J 
    # g = J.T @ r (梯度)
    g = J.T @ r
    
    # 3. 尝试更新 (Trial Step)
    # 这是一个循环，如果 Loss 变大，就要增大 mu 重试，直到 Loss 变小
    step_success = False
    current_try = 0
    
    while not step_success and current_try < 5:
        # A = H + mu * I (加阻尼)
        # P 是参数总数 (501)
        P = H.shape[0]
        I = torch.eye(P).to(device).double()
        A = H + mu * I
        
        # 解线性方程 A * delta = -g
        # 使用 torch.linalg.solve 比求逆更准更稳
        delta_theta = torch.linalg.solve(A, -g)
        
        # 得到试探性的新参数
        trial_params = update_params(params, delta_theta)
        
        # 计算新 Loss
        r_new = batch_residual_fn(trial_params, x_pde)
        mse_loss_new = torch.mean(r_new**2)
        
        # 4. 判断是否接受 (Accept/Reject)
        if mse_loss_new < mse_loss:
            # 成功！接受参数
            params = trial_params
            # 调小 mu (胆子大一点，以此逼近高斯牛顿)
            mu = mu / div_factor
            step_success = True
            print(f"Iter {i:3d} | ✅ Loss: {mse_loss.item():.8f} -> {mse_loss_new.item():.8f} | mu: {mu:.1e}")
        else:
            # 失败！拒绝参数
            # 调大 mu (胆子小一点，回归梯度下降)
            mu = mu * mul_factor
            current_try += 1
            # print(f"      | ⚠️ 尝试失败，增加阻尼 mu -> {mu:.1e}")

    if mse_loss.item() < 1e-8:
        print("🎉 达到极高精度，提前停止！")
        break

print("\n✅ LM 优化完成！")
print(f"   - 最终 Loss: {loss_history[-1]:.10f}")

# 画个图看看
plt.figure()
plt.semilogy(loss_history)
plt.title("LM Optimization Convergence")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss (Log Scale)")
plt.savefig("lm_convergence.png")
print("📊 收敛图已保存: lm_convergence.png")