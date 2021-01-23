from collections import deque
import numpy as np
from sir import SIR
import functools
from numpy import linalg
import scipy
from utils import khatrirao, mtkrprod
from utils import rmse

class STELAR:

    def __init__(self, rank, mu=0, nu=0, max_iter=100, inner_max_itr=10):
        # Model hyperparameters
        self.rank = rank
        self.mu = mu
        self.nu = nu

        # Maximum number of outer and inner iterations
        self.max_iter, self.inner_max_itr = max_iter, inner_max_itr

        # Early stopping
        self.max_iter_no_impr = 5

        # Input tensor
        self.x_norm, self.ndim, self.shape = [None] * 3

        # Factor matrices
        self.U, self.U_d, self.UtU = [], [], []

        # SIR model parameters
        self.beta, self.gamma = None, None
        self.s0, self.i0 = None, None
        self.sir_model = SIR()

        # Track RMSE error for validation set
        self.rmse_valid = deque([float('Inf')], maxlen=self.max_iter_no_impr)

        # Track cost
        self.cost_fit_hist, self.cost_l2_reg_hist, self.cost_sir_reg_hist = [], [], []

    # Predict future rows of C
    def predict_s_i_c(self, time_steps):
        s_est = np.zeros((time_steps, self.rank))
        i_est = np.zeros((time_steps, self.rank))
        s_est[0], i_est[0] = self.s0, self.i0
        for t in range(1, time_steps):
            s_est[t] = s_est[t - 1] - self.beta * s_est[t - 1] * i_est[t - 1]
            i_est[t] = i_est[t - 1] + self.beta * s_est[t - 1] * i_est[t - 1] - self.gamma * i_est[t - 1]
        c_est = s_est * i_est * self.beta
        return s_est, i_est, c_est

    # Inner product for efficient cost computation
    def inner_prod(self, x):
        res = np.zeros(self.rank)
        for r in range(self.rank):
            tmp = x
            for n in range(self.ndim):
                tmp = np.tensordot(tmp, self.U[n][:, r], axes=(0, 0))
            res[r] = tmp
        return res.sum()

    # Norm of the low-rank tensor
    def norm(self):
        return np.sqrt(functools.reduce(np.multiply, ([self.U[n].T @ self.U[n] for n in range(self.ndim)])).sum())

    # Compute cost function value
    def cost(self, x, c_est):
        cost_fit = self.x_norm ** 2 - 2 * self.inner_prod(x) + self.norm() ** 2
        cost_l2_reg = self.mu * sum([linalg.norm(self.U[n]) ** 2 for n in range(self.ndim)])
        cost_sir_reg = self.nu * linalg.norm(self.U[2] - c_est) ** 2
        return cost_fit, cost_l2_reg, cost_sir_reg

    # ADMM subproblem
    def ao_admm_sub(self, WtW, WtY, U, U_d):
        rho = np.trace(WtW) / self.rank
        cholesky_l = np.linalg.cholesky(WtW + rho * np.eye(self.rank))

        for itr in range(self.inner_max_itr):
            # primal updates
            U_t = scipy.linalg.solve_triangular(cholesky_l, WtY + rho * (U + U_d).T, lower=True)
            U_t = scipy.linalg.solve_triangular(cholesky_l.T, U_t)
            U = (U_t.T - U_d).clip(min=0)
            # dual update
            U_d = U_d + U - U_t.T
        return U, U_d

    # Predict num_days ahead after training the model
    def predict(self, num_days):
        _, _, U_time_est = self.predict_s_i_c(num_days)
        U_est = [self.U[0], self.U[1], U_time_est]
        x_est = np.reshape((U_est[0]) @ khatrirao(U_est, 0).T, [self.shape[0], self.shape[1], U_time_est.shape[0]])
        return x_est

    # Fit the model
    def fit(self, x, x_valid):
        self.x_norm = linalg.norm(x)
        self.ndim = x.ndim
        self.shape = x.shape

        # STELAR model initialization
        self.U = [np.random.rand(self.shape[n], self.rank) for n in range(self.ndim)]
        self.U_d = [np.zeros((self.shape[n], self.rank)) for n in range(self.ndim)]
        self.UtU = [self.U[n].T @ self.U[n] for n in range(self.ndim)]

        # SIR model for each column of C
        self.beta = [1e-3] * self.rank
        self.gamma = [1e-1] * self.rank
        self.i0 = [10] * self.rank
        self.s0 = [50] * self.rank

        for itr in range(self.max_iter):

            # Update A
            UtU_mult = np.ones((self.rank, self.rank))
            for k in range(self.ndim):
                if k != 0:
                    UtU_mult = UtU_mult * self.UtU[k]
            WtW = UtU_mult + self.mu * np.eye(self.rank)
            WtY = mtkrprod(x, self.U, 0).T
            self.U[0], self.U_d[0] = self.ao_admm_sub(WtW, WtY, self.U[0], self.U_d[0])
            self.UtU[0] = self.U[0].T @ self.U[0]

            # Update B
            UtU_mult = np.ones((self.rank, self.rank))
            for k in range(self.ndim):
                if k != 1:
                    UtU_mult = UtU_mult * self.UtU[k]
            WtW = UtU_mult + self.mu * np.eye(self.rank)
            WtY = mtkrprod(x, self.U, 1).T
            self.U[1], self.U_d[1] = self.ao_admm_sub(WtW, WtY, self.U[1], self.U_d[1])
            self.UtU[1] = self.U[1].T @ self.U[1]

            # Update C
            s_est, i_est, c_est = self.predict_s_i_c(self.U[2].shape[0])
            UtU_mult = np.ones((self.rank, self.rank))
            for k in range(self.ndim):
                if k != 2:
                    UtU_mult = UtU_mult * self.UtU[k]
            WtW = UtU_mult + self.nu * np.eye(self.rank) + self.mu * np.eye(self.rank)
            WtY = mtkrprod(x, self.U, 2).T + self.nu * c_est.T
            self.U[2], self.U_d[2] = self.ao_admm_sub(WtW, WtY, self.U[2], self.U_d[2])
            self.UtU[2] = self.U[2].T @ self.U[2]

            # Update SIR model of the C factor
            for r in range(self.rank):
                init_sir = [self.s0[r], self.i0[r], self.beta[r], self.gamma[r]]
                self.s0[r], self.i0[r], self.beta[r], self.gamma[r] = self.sir_model.fit(self.U[2][:, r], init_sir)
            s_est, i_est, c_est = self.predict_s_i_c(self.U[2].shape[0])

            # Update cost
            cost_fit, cost_l2_reg, cost_sir_reg = self.cost(x, c_est)
            self.cost_fit_hist.append(cost_fit)
            self.cost_l2_reg_hist.append(cost_l2_reg)
            self.cost_sir_reg_hist.append(cost_sir_reg)

            # Prediction
            stelar_val_est = self.predict(x.shape[2] + x_valid.shape[2])
            stelar_val_est = stelar_val_est[:, 0, x.shape[2]: x.shape[2] + x_valid.shape[2]]

            if rmse(x_valid[:, 0, :], stelar_val_est) > max(self.rmse_valid):
                return self.rmse_valid[-1]

            self.rmse_valid.append(rmse(x_valid[:, 0, :], stelar_val_est))
            print(f'Iteration {itr}: val rmse: {self.rmse_valid[-1]}')

        return self.rmse_valid[-1]
