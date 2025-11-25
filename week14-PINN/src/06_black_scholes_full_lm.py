import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import functools

# 检查环境
try:
    from torch.func import vmap, grad, jacrev, functional_call
    print("✅ 使用 PyTorch 2.0+ 原生 torch.func")
except ImportError:
    from functorch import vmap, grad, jacrev, make_functional
    print("⚠️ 使用 functorch (旧版)")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64) # LM 对精度要求极高，全程使用 double
print(f"🚀 使用设备: {device}")

# ==========================================
# 1. 金融环境与数据生成 (God Mode)
# ==========================================
REAL_SIGMA = 0.20  # 真实波动率 20%
RISK_FREE_RATE = 0.05

def black_scholes_formula(S, K, T, r, sigma):
    # 标准欧式看涨期权定价公式
    import scipy.stats as si
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call_val = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return torch.tensor(call_val, dtype=torch.float64, device=device)

# 生成“市场数据” (Market Data)
# 我们生成 100 个观测点
N_obs = 100
K_strike = 100.0

S_obs = (torch.rand(N_obs, 1, device=device) * 50.0 + 75.0) # 股价 75~125
t_obs = (torch.rand(N_obs, 1, device=device) * 0.9 + 0.1)   # 时间 0.1~1.0
X_obs = torch.cat([S_obs, t_obs], dim=1)

# 计算真实价格
V_market = black_scholes_formula(S_obs.cpu().numpy(), K_strike, t_obs.cpu().numpy(), RISK_FREE_RATE, REAL_SIGMA).squeeze()

# 生成 PDE 配点 (Collocation Points)
# 我们生成 200 个 PDE 点
N_pde = 200
S_pde = (torch.rand(N_pde, 1, device=device) * 50.0 + 75.0).requires_grad_(True)
t_pde = (torch.rand(N_pde, 1, device=device) * 0.9 + 0.1).requires_grad_(True)
X_pde = torch.cat([S_pde, t_pde], dim=1)

print(f"✅ 数据准备完毕: {N_obs} 市场点, {N_pde} PDE 点")

# ==========================================
# 2. 定义微型网络 (Tiny Model)
# ==========================================
# 为了让 LM 跑得动，我们用极简网络
class TinyPricingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.net(x)

model = TinyPricingModel().to(device)

# --- 参数扁平化 (Flattening) ---
# 我们需要把 (网络权重 + sigma) 拼成一个长向量 theta
# 1. 提取网络权重
params_dict = dict(model.named_parameters())
param_names = list(params_dict.keys())
param_shapes = [p.shape for p in params_dict.values()]
param_numels = [p.numel() for p in params_dict.values()]

# 2. 初始化 Sigma (瞎猜一个值)
init_sigma = 0.5 # 猜 50% (真实是 20%)
sigma_param = torch.tensor([np.log(init_sigma)], device=device, requires_grad=True) # 使用 log 保证正数

# 3. 拼合所有参数到一个向量
# 这是一个 helper 函数，把 list 转成 vector
def params_to_vector(params_dict, sigma_tensor):
    vecs = [p.flatten() for p in params_dict.values()]
    vecs.append(sigma_tensor.flatten())
    return torch.cat(vecs)

# 4. 反向 helper：把 vector 拆回 (params_dict, sigma)
def vector_to_params(vec):
    # 拆解网络权重
    new_params = {}
    idx = 0
    for i, name in enumerate(param_names):
        count = param_numels[i]
        new_params[name] = vec[idx : idx+count].view(param_shapes[i])
        idx += count
    # 拆解 sigma
    new_sigma = vec[idx:]
    return new_params, new_sigma

# 初始参数向量 theta_0
theta_init = params_to_vector(params_dict, sigma_param).detach().requires_grad_(True)
print(f"📦 参数总数量 (Weights + Sigma): {theta_init.numel()}")

# ==========================================
# 3. 定义残差向量函数 (The Big Residual)
# ==========================================
# 这个函数输入一个巨大的 theta 向量，输出一个巨大的残差向量 [r_data, r_pde]

def get_all_residuals(theta, x_obs_batch, v_obs_batch, x_pde_batch):
    # A. 拆包参数
    curr_params, curr_sigma_log = vector_to_params(theta)
    sigma = torch.exp(curr_sigma_log) # 还原 sigma
    
    # --- B. 计算 Data Residuals (拟合市场价格) ---
    # 定义临时的 forward 函数
    def forward_func(p, x):
        return functional_call(model, p, (x,)).squeeze()
    
    v_pred = forward_func(curr_params, x_obs_batch)
    res_data = v_pred - v_obs_batch
    
    # --- C. 计算 PDE Residuals (Black-Scholes) ---
    # 定义单点 PDE 计算函数
    def pde_step(p, x_single):
        # 一阶导: dV/dS, dV/dt (x[0]=S, x[1]=t)
        grads = grad(forward_func, argnums=1)(p, x_single)
        V_S, V_t = grads[0], grads[1]
        
        # 二阶导: d2V/dS2
        hess = jacrev(grad(forward_func, argnums=1), argnums=1)(p, x_single)
        V_SS = hess[0, 0]
        
        S_val = x_single[0]
        V_val = forward_func(p, x_single)
        
        # Black-Scholes 残差
        # V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0
        f = V_t + 0.5 * (sigma**2) * (S_val**2) * V_SS + RISK_FREE_RATE * S_val * V_S - RISK_FREE_RATE * V_val
        return f

    # 批量计算 PDE 残差
    res_pde = vmap(pde_step, in_dims=(None, 0))(curr_params, x_pde_batch)
    
# ============ ⚠️ 修改这里 ============
    # 强制把它们都变成 1维向量 (flatten)
    # 这样能防止 [100] 和 [200, 1] 这种维度打架的情况
    res_data = res_data.reshape(-1)
    res_pde = res_pde.reshape(-1)
    # ====================================

    # D. 拼接所有残差
    # 我们给 Data Residual 加点权重 (比如 x10)，因为它更重要
    return torch.cat([res_data * 10.0, res_pde])

# ==========================================
# 4. LM 优化主循环 (Full LM)
# ==========================================
print("\n🚀 开始全量 LM 优化 (同时优化 Weights 和 Sigma)...")

theta = theta_init.clone()
mu = 1.0 # 初始阻尼
loss_history = []
sigma_history = []

t0 = time.time()

for i in range(50): # LM 收敛极快，50次足够
    # 1. 计算雅可比矩阵 J 和残差 r
    # J 的形状: [N_samples, N_params] -> [300, 141] 左右
    
    # 计算 r
    r = get_all_residuals(theta, X_obs, V_market, X_pde)
    mse = torch.mean(r**2).item()
    
    # 记录当前 sigma
    _, curr_sig_log = vector_to_params(theta)
    curr_sig = torch.exp(curr_sig_log).item()
    sigma_history.append(curr_sig)
    loss_history.append(mse)
    
    print(f"Iter {i:2d} | Loss: {mse:.6f} | 🕵️ Sigma: {curr_sig:.4f} (Target: {REAL_SIGMA}) | mu: {mu:.1e}")
    
    if mse < 1e-6:
        print("🎉 收敛达成！")
        break
        
    # 计算 J (这是最耗时的一步)
    # jacrev 对第一个参数(theta)求导
    J = jacrev(get_all_residuals, argnums=0)(theta, X_obs, V_market, X_pde)
    
    # --- LM 更新步 ---
    H = J.T @ J
    g = J.T @ r
    P = theta.shape[0]
    
    # 简单的 LM 更新逻辑 (省略了回溯步，简化演示)
    # theta_new = theta - (H + mu*I)^-1 @ g
    try:
        delta = torch.linalg.solve(H + mu * torch.eye(P, device=device), -g)
        theta = theta + delta
        
        # 激进策略：每次成功都减小 mu (逼近高斯牛顿)
        mu = max(1e-7, mu / 2.0)
        
    except RuntimeError:
        # 如果矩阵奇异，增大 mu (回退到梯度下降)
        print("⚠️ 矩阵奇异，增加阻尼...")
        mu = mu * 10.0

t1 = time.time()
print(f"\n✅ 训练结束！耗时: {t1-t0:.2f}秒")
print(f"最终 Sigma 预测: {sigma_history[-1]:.5f}")
print(f"真实 Sigma: {REAL_SIGMA}")
print(f"误差: {abs(sigma_history[-1] - REAL_SIGMA)/REAL_SIGMA*100:.2f}%")

# 画图
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title('Loss Convergence (Full LM)')
plt.yscale('log')

plt.subplot(1,2,2)
plt.plot(sigma_history)
plt.axhline(REAL_SIGMA, color='r', linestyle='--')
plt.title('Sigma Calibration')
plt.savefig('full_lm_result.png')