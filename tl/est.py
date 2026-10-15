import math

import numpy as np
import torch


def sg_kernel(m, p, deriv=2):
    s = np.arange(-m, m + 1, dtype=float)
    A = np.vander(s, p + 1, increasing=True)
    e = np.zeros(p + 1)
    e[deriv] = 1.0
    return (A @ (np.linalg.pinv(A.T @ A) @ e)) * math.factorial(deriv)


def _corr(X, h):
    T_, N, d = X.shape
    k = len(h)
    w = torch.as_tensor(h, dtype=X.dtype, device=X.device).view(1, 1, k).repeat(N * d, 1, 1)
    y = torch.nn.functional.conv1d(X.permute(1, 2, 0).reshape(1, N * d, T_), w, groups=N * d)
    return y.reshape(N, d, T_ - k + 1).permute(2, 0, 1)


def est_fd(X, dt):
    out = X * float("nan")
    out[1:-1] = (X[2:] - 2 * X[1:-1] + X[:-2]) / dt ** 2
    return out


def est_sg(X, dt, m, p, deriv=2, smooth=False):
    h = sg_kernel(m, p, 0 if smooth else deriv)
    o = _corr(X, h) / (1.0 if smooth else dt ** deriv)
    out = X * float("nan")
    out[m:X.shape[0] - m] = o
    return out


def est_smooth_fd(X, dt, ms, ps=2):
    Xs = est_sg(X, dt, ms, ps, smooth=True)
    T_ = X.shape[0]
    out = X * float("nan")
    sl = slice(ms + 1, T_ - ms - 1)
    out[sl] = (Xs[ms + 2:T_ - ms] - 2 * Xs[ms + 1:T_ - ms - 1] + Xs[ms:T_ - ms - 2]) / dt ** 2
    return out


def nmse_on(a, b, lo, hi):
    A = a[lo:hi].reshape(-1)
    B = b[lo:hi].reshape(-1)
    return float((((A - B) ** 2).mean()) / (B ** 2).mean())
