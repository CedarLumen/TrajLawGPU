import numpy as np
import torch

from .bases import frame
from .sim import offdiag


def lstsq(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def stlsq(A, y, iters=12, tau=1e-4):
    n = A.shape[1]
    c = lstsq(A, y)
    an = np.linalg.norm(A, axis=0)
    yn = max(np.linalg.norm(y), 1e-30)
    for _ in range(iters):
        m = np.abs(c) * an > tau * yn
        if m.sum() == 0:
            break
        c = np.zeros(n)
        c[m] = lstsq(A[:, m], y)
    return c


def kmeans(X, K, iters=10, seed=1, cap=30000):
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(cap, len(X)), replace=False)]
    cen = sub[rng.choice(len(sub), K, replace=False)].copy()
    for _ in range(iters):
        a = ((sub[:, None, :] - cen[None, :, :]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            m = a == k
            if m.sum():
                cen[k] = sub[m].mean(0)
    return cen


def gram_y(A, y, sel=None, chunk=400000):
    n = A.shape[1]
    G = torch.zeros((n, n), dtype=torch.float64, device=A.device)
    b = torch.zeros(n, dtype=torch.float64, device=A.device)
    idx = torch.nonzero(sel).squeeze(1) if sel is not None else torch.arange(len(A), device=A.device)
    for i in range(0, len(idx), chunk):
        s = idx[i:i + chunk]
        X = A[s].to(torch.float64)
        G = G + X.transpose(0, 1) @ X
        b = b + X.transpose(0, 1) @ y[s].to(torch.float64)
    return G, b


def ridge(G, b, lam):
    n = G.shape[0]
    L = lam * torch.trace(G) / n
    eye = torch.eye(n, dtype=G.dtype, device=G.device)
    return torch.linalg.solve(G + L * eye, b)


class KRR:
    def __init__(self, cen, h2, mu, sd):
        self.cen = torch.as_tensor(cen, dtype=torch.float64)
        self.h2 = float(h2)
        self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.sd = torch.as_tensor(sd, dtype=torch.float64)
        self.K = len(self.cen)

    def _U(self, R, V):
        r, rh, n, vp, vq = frame(R, V)
        return torch.stack([r, vp, vq], -1), rh, n

    def _phi(self, U, a, b):
        d = ((U.unsqueeze(-2) - self.cen[a:b].to(U.device)) ** 2).sum(-1)
        return torch.exp(-d / (2 * self.h2)), a

    def design(self, R, V, chunk=24, dtype=torch.float32):
        U, rh, n = self._U(R, V)
        ts, N, _, dd = R.shape
        m = offdiag(N, R.dtype, R.device).unsqueeze(-1)
        out = torch.empty((ts, N, dd, 2 * self.K), dtype=dtype, device=R.device)
        for a in range(0, self.K, chunk):
            P, a0 = self._phi(U, a, min(a + chunk, self.K))
            for k in range(a0, min(a + chunk, self.K)):
                g = P[..., k - a0].unsqueeze(-1)
                out[..., 2 * k] = ((g * rh) * m).sum(2)
                out[..., 2 * k + 1] = ((g * n) * m).sum(2)
        return out.reshape(-1, 2 * self.K)

    def predict(self, R, V, c, chunk=24):
        U, rh, n = self._U(R, V)
        out = R * 0
        c = np.asarray(c)
        for a in range(0, self.K, chunk):
            P, a0 = self._phi(U, a, min(a + chunk, self.K))
            for k in range(a0, min(a + chunk, self.K)):
                g = P[..., k - a0].unsqueeze(-1)
                out = out + float(c[2 * k]) * g * rh + float(c[2 * k + 1]) * g * n
        return out
