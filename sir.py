from scipy import optimize
import numpy as np


class SIR:
    # Lower and upper bounds for s0, i0, \beta and \gamma
    def __init__(self, bounds=([0, 0, 1e-10, 1e-10], [np.inf, np.inf, 1, 1])):
        self.bounds = bounds
        self.data = None

    # Predicts the daily infected cases C(t)
    def predict(self, time_full, s0, i0, beta, gamma):
        s = np.zeros(len(time_full))
        i = np.zeros(len(time_full))
        s[0], i[0] = s0, i0
        for t in time_full[1:]:
            s[t] = max(s[t - 1] - beta * s[t - 1] * i[t - 1], 0)
            i[t] = i[t - 1] + beta * s[t - 1] * i[t - 1] - gamma * i[t - 1]
        return beta * s * i

    # Predicts susceptible, infected and recovered populations
    def predict_all(self, time_full, s0, i0, beta, gamma):
        s = np.zeros(len(time_full))
        i = np.zeros(len(time_full))
        r = np.zeros(len(time_full))
        s[0], i[0] = s0, i0
        for t in time_full[1:]:
            s[t] = s[t - 1] - beta * s[t - 1] * i[t - 1]
            i[t] = i[t - 1] + beta * s[t - 1] * i[t - 1] - gamma * i[t - 1]
            r[t] = r[t - 1] + gamma * i[t - 1]
        return s, i, r, beta * s * i

    # Curve fitting to find parameters s0, i0, \beta and \gamma based on observed daily new infections
    def fit(self, data, init):
        self.data = data
        time_full = range(data.shape[0])
        popt, pcov = optimize.curve_fit(self.predict, time_full, self.data, bounds=self.bounds, p0=init, maxfev=150000)
        return popt
