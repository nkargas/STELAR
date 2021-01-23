import numpy as np


def rmse(x, y):
    return np.sqrt(np.sum(np.linalg.norm(x - y) ** 2) / x.size)


def mae(x, y):
    return np.sum(np.abs(x - y)) / x.size


def khatrirao(u, skip=None):
    if len(u) == 1:
        return u[0]

    n_cols = u[0].shape[1]
    if skip is not None:
        u = [u[n] for n in range(len(u)) if n != skip]

    n_rows = np.prod([u[n].shape[0] for n in range(len(u))])
    x = np.zeros((n_rows, n_cols))
    for r in range(n_cols):
        v = u[0][:, r]
        for n in range(1, len(u)):
            v = np.kron(v, u[n][:, r])
        x[:, r] = v
    return x


def unfold(x, mode):
    return np.reshape(np.moveaxis(x.data, mode, 0), (x.shape[mode], -1))


def mtkrprod(x, u, mode):
    return unfold(x, mode) @ khatrirao(u, skip=mode)
