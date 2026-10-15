import torch

from .sim import offdiag

NAMES = ["R̂", "rR̂", "r²R̂", "r³R̂", "v∥R̂", "v∥²R̂", "v⊥R̂", "v∥n̂", "v⊥n̂", "v⊥²R̂"]
M = len(NAMES)
IDX = dict((n, i) for i, n in enumerate(NAMES))

POS = [0, 1, 2, 3]
VEL = list(range(10))
VPERP = [IDX["v⊥R̂"], IDX["v⊥n̂"], IDX["v⊥²R̂"]]

BASES = [
    lambda r, rh, n, vp, vq: rh,
    lambda r, rh, n, vp, vq: r.unsqueeze(-1) * rh,
    lambda r, rh, n, vp, vq: (r ** 2).unsqueeze(-1) * rh,
    lambda r, rh, n, vp, vq: (r ** 3).unsqueeze(-1) * rh,
    lambda r, rh, n, vp, vq: vp.unsqueeze(-1) * rh,
    lambda r, rh, n, vp, vq: (vp ** 2).unsqueeze(-1) * rh,
    lambda r, rh, n, vp, vq: vq.unsqueeze(-1) * rh,
    lambda r, rh, n, vp, vq: vp.unsqueeze(-1) * n,
    lambda r, rh, n, vp, vq: vq.unsqueeze(-1) * n,
    lambda r, rh, n, vp, vq: (vq ** 2).unsqueeze(-1) * rh,
]


def frame(R, V):
    r = R.norm(dim=-1)
    rh = R / r.clamp_min(1e-300).unsqueeze(-1)
    n = torch.stack([-rh[..., 1], rh[..., 0]], dim=-1)
    return r, rh, n, (V * rh).sum(-1), (V * n).sum(-1)


def pair_sum_design(R, V, cols, mask=None, dtype=None):
    ts, N, _, d = R.shape
    dtype = dtype or R.dtype
    r, rh, n, vp, vq = frame(R, V)
    m = offdiag(N, R.dtype, R.device).unsqueeze(-1)
    out = torch.empty((ts, N, d, len(cols)), dtype=dtype, device=R.device)
    for j, i in enumerate(cols):
        g = BASES[i](r, rh, n, vp, vq)
        if mask is not None:
            g = g * mask.unsqueeze(-1)
        out[:, :, :, j] = (g * m).sum(2)
    return out.reshape(-1, len(cols))


def eval_law(R, V, c, cols=None, mask=None):
    cols = list(range(M)) if cols is None else list(cols)
    r, rh, n, vp, vq = frame(R, V)
    out = R * 0
    for j, i in enumerate(cols):
        w = float(c[j])
        if abs(w) > 1e-14:
            g = BASES[i](r, rh, n, vp, vq)
            if mask is not None:
                g = g * mask.unsqueeze(-1)
            out = out + w * g
    return out


def self_design(X, V):
    ts, N, d = X.shape
    cols = []
    for k in range(d):
        for src in range(2 * d + 1):
            c = X[..., 0] * 0
            if src == 0:
                c[..., k] = 1.0
            elif src <= d:
                c[..., k] = X[..., src - 1]
            else:
                c[..., k] = V[..., src - 1 - d]
            cols.append(c.reshape(-1))
    return torch.stack(cols, dim=-1)
