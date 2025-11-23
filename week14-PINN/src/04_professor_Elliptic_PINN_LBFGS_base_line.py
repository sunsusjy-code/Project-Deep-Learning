# Import libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from collections import OrderedDict

# Device
device = torch.device('cpu')


# Model
class Plain(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.ln1 = nn.Linear(in_dim, h_dim).double()
        self.act1 = nn.Sigmoid()
        self.ln = nn.Linear(h_dim, out_dim).double()

    def forward(self, x):
        out = self.ln1(x)
        out = self.act1(out)
        out = self.ln(out)
        return out

# Experiment
class experiment(Plain):
    def __init__(self, N_inner, episode, num_neuron, device):
        super().__init__(in_dim=2, h_dim=num_neuron, out_dim=1)
        # initial setting
        self.N_inner = N_inner
        self.episode = episode
        self.model = Plain(2, num_neuron, 1).to(device)
        self.optimizer = self.optimizer_LBFGS(self.model)


    # Problem Setting
    def exact_u(self, x, y):

        return np.cos(y)

    def exact_du(self, x, y):

        return -np.sin(y)

    # f define
    def rF(self, x, y):

        return np.cos(y)

    # Count parameters
    @staticmethod
    def count_parameters(model, requires_grad=True):
        """Count trainable parameters for a nn.Module."""
        if requires_grad:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    # Define loss functions
    def loss(self, model, X_inner, Rf_inner, X_bd, U_bd):
        # loss_bd: boundary condition 4
        bd_pred = model(X_bd)
        loss_bd = torch.mean((bd_pred - U_bd) ** 2)


        # loss_res: system residual 1
        inner_pred = model(X_inner)
        dudX = torch.autograd.grad(
            inner_pred, X_inner,
            grad_outputs=torch.ones_like(inner_pred),
            retain_graph=True,
            create_graph=True
        )[0]  # u_x u_y
        dudX_xX = torch.autograd.grad(
            dudX[:, 0], X_inner,
            grad_outputs=torch.ones_like(dudX[:, 0]),
            retain_graph=True,
            create_graph=True
        )[0]  # u_xx u_xy
        dudX_yX = torch.autograd.grad(
            dudX[:, 1], X_inner,
            grad_outputs=torch.ones_like(dudX[:, 1]),
            retain_graph=True,
            create_graph=True
        )[0]  # u_yx u_yy


        laplace = (dudX_xX[:, 0:1] + dudX_yX[:, 1:2])  # u_xx + u_yy

        loss_res = torch.mean((-laplace - Rf_inner) ** 2)

        loss = loss_bd + loss_res

        return loss

    # Chebyshev_first_kind
    @staticmethod
    def chebyshev_first_kind(dim, n):
        X = []
        x = []
        X = (np.mgrid[[slice(None, n), ] * dim])
        XX = np.cos(np.pi * (X + 0.5) / n)
        for i in range(len(X)):
            x.append(np.array(XX[i].tolist()).reshape(n ** dim, 1))
        return np.hstack(np.array(x))

    # Collect training points
    def collect_training_points(self):
        ## X_inner: points inside the domain, totally (N_inner)**2 points
        self.X_inner = self.chebyshev_first_kind(2, self.N_inner)
        x = self.X_inner[:, 0:1]
        y = self.X_inner[:, 1:2]
        self.Rf_inner = self.rF(x, y)

        ## X_bd: points on the boundary, totally 4*N_inner points
        cheby_point = self.chebyshev_first_kind(1, self.N_inner)
        dumy_one = np.ones((self.N_inner, 1))
        xx1 = np.hstack((cheby_point, -1.0 * dumy_one ))
        xx2 = np.hstack((-1.0 * dumy_one, cheby_point ))
        xx3 = np.hstack((dumy_one, cheby_point ))
        xx4 = np.hstack((cheby_point, dumy_one ))
        self.X_bd = np.vstack([xx1, xx2, xx3, xx4])


        ## U_bd: function values on the boundary, totally 4*N_inner points
        x = self.X_bd[:, 0:1]
        y = self.X_bd[:, 1:2]
        self.U_bd = self.exact_u(x, y)

        # Switch variables to torch
        self.X_bd_torch = torch.tensor(self.X_bd, requires_grad=True, device=device, dtype=torch.float64)
        self.U_bd_torch = torch.tensor(self.U_bd, device=device, dtype=torch.float64)
        self.X_inner_torch = torch.tensor(self.X_inner, requires_grad=True, device=device, dtype=torch.float64)
        self.Rf_inner_torch = torch.tensor(self.Rf_inner, device=device, dtype=torch.float64)

    # Validation points plot
    def Validation_points(self):
        ## X_inner: points inside the domain, totally (N_inner)**2 points
        self.X_valid_inner = 2.0 * lhs(2, (self.N_inner) ** 2) - 1.0
        x = self.X_valid_inner[:, 0:1]
        y = self.X_valid_inner[:, 1:2]
        #self.X_valid_inner = np.hstack((self.X_valid_inner, z))
        self.Rf_valid_inner = self.rF(x, y)

        self.X_valid_inner_torch = torch.tensor(self.X_valid_inner, requires_grad=(True), device=device).double()
        self.Rf_valid_inner_torch = torch.tensor(self.Rf_valid_inner, device=device).double()

    def optimizer_LBFGS(self, model):
        optimizerLBFGS = torch.optim.LBFGS(
            model.parameters(),
            lr=0.1,
            max_iter=self.episode,
            max_eval=self.episode,
            history_size=50,
            tolerance_grad=np.finfo(float).eps,
            tolerance_change=np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        return optimizerLBFGS

    def loss_func_lbfgs(self):

        self.optimizer.zero_grad()
        self.itera += 1

        lossLBFGS = self.loss(self.model, self.X_inner_torch, self.Rf_inner_torch, self.X_bd_torch, self.U_bd_torch)
        lossLBFGS_valid = self.loss(self.model, self.X_valid_inner_torch, self.Rf_valid_inner_torch, self.X_bd_torch, self.U_bd_torch)

        if self.itera % (self.episode / 10) == 0:
            print('Iter %d, LossLBFGS: %.5e' % (self.itera, lossLBFGS.item()))
            print('Iter %d, LossLBFGS_valid: %.5e' % (self.itera, lossLBFGS_valid.item()))

        self.savedloss.append(lossLBFGS.item())
        self.savedloss_valid.append(lossLBFGS_valid.item())

        lossLBFGS.backward(retain_graph=True)

        return lossLBFGS

    # Train model
    def Train(self):
        self.itera = 0

        self.savedloss = []
        self.savedloss_valid = []
        self.model.train()

        lossLBFGS = self.loss(self.model, self.X_inner_torch, self.Rf_inner_torch, self.X_bd_torch, self.U_bd_torch)
        lossLBFGS_valid = self.loss(self.model, self.X_valid_inner_torch, self.Rf_valid_inner_torch, self.X_bd_torch, self.U_bd_torch)

        print('Iter %d, LossLBFGS: %.5e' % (self.itera, lossLBFGS.item()))
        print('Iter %d, LossLBFGS_valid: %.5e' % (self.itera, lossLBFGS_valid.item()))
        self.savedloss.append(lossLBFGS.item())
        self.savedloss_valid.append(lossLBFGS_valid.item())

        # Backward and optimize
        self.optimizer.step(self.loss_func_lbfgs)

    # Error detacting
    def Error_detacting(self):
        # number of test points
        N_test = 12800

        # Error on the interior points
        X_inn = 2.0 * lhs(2, N_test) - 1.0
        self.xx = X_inn[:, 0:1]
        self.yy = X_inn[:, 1:2]
        self.u_test = self.exact_u(self.xx, self.yy)
        X_inn_torch = torch.tensor(X_inn, device=device).double()
        self.u_pred = self.model(X_inn_torch).detach().cpu().numpy()

        error = np.absolute(self.u_pred - self.u_test)
        self.error = error
        error_u_inf = np.linalg.norm(error, np.inf)
        # print('Error u (absolute inf-norm): %e' % (error_u_inf))
        error_u_2 = np.linalg.norm(error, 2) / np.sqrt(N_test)
        # print('Error u (absolute 2-norm): %e' % (error_u_2))
        return error_u_inf, error_u_2

    # Training points plot
    def Training_points_plot(self):
        plt.figure(figsize=(5, 5))

        plt.scatter(self.X_inner[:, 0], self.X_inner[:, 1],
                    c="b", s=1, marker=".")
        plt.scatter(self.X_bd[:, 0], self.X_bd[:, 1],
                    c="r", s=5, marker=".")
        plt.scatter(self.X_interface[:, 0], self.X_interface[:, 1],
                    c="k", s=5, marker="o")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.title('training points')

        plt.show()

    # Vaildation points plot
    def Validation_points_plot(self):
        plt.scatter(self.X_valid_inner[:, 0], self.X_valid_inner[:, 1], c="b", s=5, marker=".")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.title('validation points')
        plt.show()

    # Model 3D plot
    def Model_3D_plot(self):
        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.3))

        # ===============
        #  First subplot
        # ===============
        # set up the axes for the first plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.scatter(self.xx, self.yy, self.u_pred, c=self.u_pred)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # ===============
        # Second subplot
        # ===============
        # set up the axes for the second plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(self.xx, self.yy, self.u_test, c=self.u_test, cmap='plasma')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # 将原本第 269 行的 plt.show() 替换为：
        plt.savefig('elliptic_3d_comparison.png', dpi=300) # 保存为高清图
        # plt.show() # 这一行注释掉，或者直接删掉
        print("3D 对比图已保存为 elliptic_3d_comparison.png")
        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.3))

    # Error 3D plot
    def Error_3D_plot(self):
        fig = plt.figure(figsize=plt.figaspect(0.3))
        # ===============
        #  First subplot
        # ===============
        # set up the axes for the first plot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.scatter(self.xx, self.yy, self.error, c=self.error, cmap='viridis', marker=".", s=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(elev=60, azim=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Abs. error')

        # ===============
        # Second subplot
        # ===============
        # set up the axes for the second plot
        ax = fig.add_subplot(1, 2, 2)
        surf = ax.scatter(self.xx, self.yy, c=self.error, cmap='OrRd', s=10)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Abs. error')
        ax.axis('equal')
        plt.show()

    # Training plot
    def Training_plot(self):
        start = 0
        end = self.itera
        idx = list(range(start, end, 1))

        fig = plt.figure(figsize=(10, 10))
        plt.ylim(10 ** (-13), 10 ** (2))
        plt.yscale("log")
        plt.xscale("log")
        plt.plot(idx, self.savedloss[start:end], label="loss")
        plt.plot(idx, self.savedloss_valid[start:end], label="loss_valid")
        plt.legend()
        plt.show()


N_inner_list = [8]
num_neuron_list = [20]
import pandas as pd

episode = 1000
results_df = pd.DataFrame()
for num_neuron in num_neuron_list:
    for N_inner in N_inner_list:
        name = f"exp_{N_inner}"


        globals()[name] = experiment(N_inner, episode, num_neuron, device)
        globals()[name].collect_training_points()
        globals()[name].Validation_points()
        globals()[name].Train()
        globals()[name].Error_detacting
        error_L_inf, error_L_2 = globals()[name].Error_detacting()
        data = {
            'L_inf': [error_L_inf],
            'L_2': [error_L_2]
        }
        indexName = [name]
        df = pd.DataFrame(data, index=indexName)
        results_df = pd.concat([results_df, df])

        globals()[name].Model_3D_plot()
        globals()[name].Error_3D_plot()

print(results_df)
results_df.to_csv('experiment_LBFGS.csv')