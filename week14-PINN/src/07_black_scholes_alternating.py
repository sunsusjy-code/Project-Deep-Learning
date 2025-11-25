import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.func import vmap, grad, jacrev, functional_call

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
print(f"🚀 使用设备: {device} | 策略: 暴力过拟合(Overfit) + 辅助轮(Clamp)")

# ==========================================
# 1. 数据增强 (加大数据量)
# ==========================================
REAL_SIGMA = 0.20
RISK_FREE_RATE = 0.05
K_strike = 100.0
# 🟢 修改点 1: 增加数据点，让网络没法“偷懒”
N_obs = 300  
N_pde = 1000

def black_scholes_formula(S, K, T, r, sigma):
    import scipy.stats as si
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call_val = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return torch.tensor(call_val, dtype=torch.float64, device=device)

# 数据生成
S_obs = (torch.rand(N_obs, 1, device=device) * 50.0 + 75.0)
t_obs = (torch.rand(N_obs, 1, device=device) * 0.9 + 0.1)
X_obs = torch.cat([S_obs, t_obs], dim=1)
V_market = black_scholes_formula(S_obs.cpu().numpy(), K_strike, t_obs.cpu().numpy(), RISK_FREE_RATE, REAL_SIGMA).squeeze()

S_pde = (torch.rand(N_pde, 1, device=device) * 50.0 + 75.0).requires_grad_(True)
t_pde = (torch.rand(N_pde, 1, device=device) * 0.9 + 0.1).requires_grad_(True)
X_pde = torch.cat([S_pde, t_pde], dim=1)

# ==========================================
# 2. 模型定义
# ==========================================
class OptionPricingModel_ALT(nn.Module):
    def __init__(self):
        super().__init__()
        # 稍微加深一点网络，保证拟合能力
        self.net = nn.Sequential(
            nn.Linear(2, 32, dtype=torch.float64), nn.Tanh(),
            nn.Linear(32, 32, dtype=torch.float64), nn.Tanh(),
            nn.Linear(32, 1, dtype=torch.float64)
        )
        self.act_out = nn.Softplus()
        # 初始盲猜 0.5
        self.sigma_log = nn.Parameter(torch.tensor([np.log(0.5)], dtype=torch.float64))

    def forward(self, x):
        return self.act_out(self.net(x)).squeeze()
    
    def get_sigma(self):
        return torch.exp(self.sigma_log)

model = OptionPricingModel_ALT().to(device)

# 参数管理
all_params_list = list(model.parameters())
param_numels = [p.numel() for p in all_params_list]
param_shapes = [p.shape for p in all_params_list]
param_names = [name for name, p in model.named_parameters()]

def vector_to_params_dict(vec):
    new_params_list = []
    idx = 0
    for i, count in enumerate(param_numels):
        new_params_list.append(vec[idx : idx+count].view(param_shapes[i]))
        idx += count
    return dict(zip(param_names, new_params_list))

# ==========================================
# 3. 核心函数 (Jacobian等保持标准版)
# ==========================================
def compute_jacobian_chunked(theta, x_o, v_m, x_p, chunk_size=50):
    def func_data(t):
        p = vector_to_params_dict(t)
        return (functional_call(model, p, (x_o,)).squeeze() - v_m).view(-1) * 10.0
    J_data = jacrev(func_data)(theta)
    
    J_pde_list = []
    N = x_p.shape[0]
    for i in range(0, N, chunk_size):
        x_chunk = x_p[i:i+chunk_size]
        def func_pde(t):
            p = vector_to_params_dict(t)
            s_val = torch.exp(p['sigma_log'])
            r_rate = RISK_FREE_RATE
            def step(x_s):
                def inner(x): return functional_call(model, p, (x.unsqueeze(0),)).squeeze()
                g = grad(inner)(x_s)
                h = jacrev(grad(inner))(x_s)
                return g[1] + r_rate*x_s[0]*g[0] + 0.5*(s_val**2)*(x_s[0]**2)*h[0,0] - r_rate*inner(x_s)
            return vmap(step)(x_chunk).view(-1)
        torch.cuda.empty_cache()
        J_chunk = jacrev(func_pde)(theta) 
        J_pde_list.append(J_chunk)
    return torch.cat([J_data, torch.cat(J_pde_list, dim=0)], dim=0)

def get_all_residuals(theta_vector, x_obs, v_market, x_pde):
    curr_params = vector_to_params_dict(theta_vector)
    sigma_val = torch.exp(curr_params['sigma_log'])
    r_rate = RISK_FREE_RATE
    v_pred = functional_call(model, curr_params, (x_obs,)).squeeze()
    res_data = (v_pred - v_market) * 10.0
    def pde_step(x_s):
        def inner(x): return functional_call(model, curr_params, (x.unsqueeze(0),)).squeeze()
        g = grad(inner)(x_s)
        h = jacrev(grad(inner))(x_s)
        V_S, V_t = g[0], g[1]
        V_SS = h[0,0]
        return g[1] + r_rate*x_s[0]*g[0] + 0.5*(sigma_val**2)*(x_s[0]**2)*h[0,0] - r_rate*inner(x_s)
    res_pde = vmap(pde_step)(x_pde)
    return torch.cat([res_data.view(-1), res_pde.view(-1)])


# ==========================================
# 4. 优化流程 (修正版)
# ==========================================

# --- Phase 1: 暴力过拟合 Data ---
# 目标：必须把 Data Loss 压到 1e-3 以下，否则形状就是错的
print("\n🔥 Phase 1: 暴力拟合 Data (Lock Sigma)...")
model.sigma_log.requires_grad = False
optimizer_net = torch.optim.Adam(model.net.parameters(), lr=0.01)

for i in range(3000): # 🟢 增加步数
    optimizer_net.zero_grad()
    
    v_pred = model(X_obs)
    loss_data = torch.mean((v_pred - V_market)**2)
    
    loss_data.backward()
    optimizer_net.step()
    
    if i % 500 == 0:
        print(f"Iter {i}: Data Loss {loss_data.item():.6f} | Sigma {model.get_sigma().item():.4f}")

print(f"✅ Phase 1 结束. 最终 Data Loss: {loss_data.item():.6f}")
if loss_data.item() > 0.01:
    print("⚠️ 警告：数据拟合依然很差，后续反演可能会失败！")


# --- Phase 2: 联合训练 (带辅助轮) ---
# 我们不再单独训练 Sigma，而是联合训练，但是给 Sigma 加上“辅助轮” (Clamp)
print("\n🔥 Phase 2: 联合训练 (Joint with Clamp)...")
model.sigma_log.requires_grad = True
optimizer_all = torch.optim.Adam(model.parameters(), lr=0.005)

for i in range(2000):
    optimizer_all.zero_grad()
    
    # 1. 计算联合 Loss
    # Data 部分
    v_pred = model(X_obs)
    loss_data = torch.mean((v_pred - V_market)**2) * 100.0 # 保持高权重
    
    # PDE 部分
    sigma = model.get_sigma()
    r_rate = RISK_FREE_RATE
    v_pde = model(X_pde)
    grads = torch.autograd.grad(v_pde, X_pde, torch.ones_like(v_pde), create_graph=True)[0]
    V_S, V_t = grads[:, 0:1], grads[:, 1:2]
    grads_2 = torch.autograd.grad(V_S, X_pde, torch.ones_like(V_S), create_graph=True)[0]
    V_SS = grads_2[:, 0:1]
    S_val = X_pde[:, 0:1]
    f = V_t + r_rate * S_val * V_S + 0.5 * (sigma**2) * (S_val**2) * V_SS - r_rate * v_pde
    loss_pde = torch.mean(f**2)
    
    loss = loss_data + loss_pde
    loss.backward()
    optimizer_all.step()
    
    # 🟢 关键点：每次更新完，强制把 Sigma 拉回合理区间
    # 防止 Adam 偷懒跑到 0 去。我们假设波动率至少是 0.1
    with torch.no_grad():
        model.sigma_log.data.clamp_(min=math.log(0.1))

    if i % 200 == 0:
        print(f"Iter {i}: Loss {loss.item():.4e} | Sigma {model.get_sigma().item():.4f}")

print("✅ Phase 2 结束，进入 LM...")

# --- Phase 3: LM 终极微调 ---
print("\n🚀 Phase 3: LM 终极微调...")
mu = 1e-1
max_lm_steps = 20

for i in range(max_lm_steps):
    theta_lm = nn.utils.parameters_to_vector(model.parameters())
    try:
        J = compute_jacobian_chunked(theta_lm, X_obs, V_market, X_pde, chunk_size=100)
        with torch.no_grad():
            r = get_all_residuals(theta_lm, X_obs, V_market, X_pde)
            loss_val = torch.mean(r**2)
        H = J.T @ J
        g = J.T @ r
        A = H + mu * torch.eye(theta_lm.shape[0], device=device).double()
        delta = torch.linalg.solve(A, -g)
        
        if torch.norm(delta) > 1.0: delta = delta / torch.norm(delta)
        
        with torch.no_grad():
            ptr = 0
            for p in model.parameters():
                num = p.numel()
                p.data += delta[ptr:ptr+num].view(p.shape)
                ptr += num
        
        # LM 阶段也可以加个简单的 Clamp，防止飞出宇宙
        with torch.no_grad():
            model.sigma_log.data.clamp_(min=math.log(0.05), max=math.log(2.0))
            
        sig = model.get_sigma().item()
        mu = max(1e-6, mu / 3.0)
        print(f"LM {i}: Loss {loss_val:.4e} | Sigma {sig:.5f}")
            
    except Exception as e:
        print(f"Error: {e}")
        break

print(f"\n最终结果: {model.get_sigma().item():.5f} (真实: {REAL_SIGMA})")