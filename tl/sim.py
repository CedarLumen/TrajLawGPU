import torch

_OD = {}


def offdiag(n, dtype, device):
    key = (n, dtype, str(device))
    m = _OD.get(key)
    if m is None:
        m = (~torch.eye(n, dtype=torch.bool, device=device)).to(dtype)
        _OD[key] = m
    return m


def diffs(X, V, box=None):
    R = X.unsqueeze(-2) - X.unsqueeze(-3)
    Vv = V.unsqueeze(-2) - V.unsqueeze(-3)
    if box is not None:
        R = R - box * torch.round(R / box)
    m = offdiag(X.shape[-2], X.dtype, X.device).unsqueeze(-1)
    return R * m, Vv * m


def pair_accel(terms, X, V, box=None):
    R, Vv = diffs(X, V, box)
    a = R * 0
    for x in terms:
        a = a + x.f(R, Vv)
    return a.sum(-2)


def accel(terms, ext, X, V, box=None):
    a = pair_accel(terms, X, V, box)
    for e in ext:
        a = a + e.f(X)
    return a


def total_potential(terms, ext, X, box=None):
    R, _ = diffs(X, X, box)
    U = X[..., 0] * 0
    m = offdiag(X.shape[-2], R.dtype, R.device)
    for x in terms:
        U = U + (x.u(R) * m).sum(-1) * 0.5
    for e in ext:
        U = U + e.u(X)
    return U


def energy(terms, ext, Xs, Vs, box=None):
    return 0.5 * (Vs ** 2).sum((-1, -2)) + total_potential(terms, ext, Xs, box).sum(-1)


def _step(terms, ext, X, V, Xn, Vn, dt, box=None):
    k1x, k1v = V, accel(terms, ext, X, V, box)
    k2x = V + 0.5 * dt * k1v
    k2v = accel(terms, ext, X + 0.5 * dt * k1x, V + 0.5 * dt * k1v, box)
    k3x = V + 0.5 * dt * k2v
    k3v = accel(terms, ext, X + 0.5 * dt * k2x, V + 0.5 * dt * k2v, box)
    k4x = V + dt * k3v
    k4v = accel(terms, ext, X + dt * k3x, V + dt * k3v, box)
    Xn.copy_(X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x))
    Vn.copy_(V + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def _rk4_graph(terms, ext, X0, V0, n, dt, every, box=None):
    """Record the whole run into a CUDA Graph and replay it.

    For a small system like N=40 the GPU is all launch overhead (measured: eager 1.87 ms/step,
    16% utilization). Recorded, the kernels run back to back at 0.21 ms/step, faster than numpy.
    It is not a question of the speedup ratio: without the graph the GPU is unusable at this size.
    """
    frames = n // every + 1
    m = max(1, -(-32 // every))
    step = _step

    X, V = X0.clone(), V0.clone()
    Xn, Vn = torch.empty_like(X), torch.empty_like(V)
    Sx = torch.empty((m,) + tuple(X.shape), dtype=X.dtype, device=X.device)
    Sv = torch.empty_like(Sx)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step(terms, ext, X, V, Xn, Vn, dt, box)
            X.copy_(Xn)
            V.copy_(Vn)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    X.copy_(X0)
    V.copy_(V0)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for k in range(m):
            for _ in range(every):
                step(terms, ext, X, V, Xn, Vn, dt, box)
                X.copy_(Xn)
                V.copy_(Vn)
            Sx[k].copy_(X)
            Sv[k].copy_(V)

    Xs = torch.empty((frames,) + tuple(X.shape), dtype=X.dtype, device=X.device)
    Vs = torch.empty_like(Xs)
    Xs[0].copy_(X)
    Vs[0].copy_(V)
    nch = -(-(frames - 1) // m)
    for j in range(nch):
        g.replay()
        take = min(m, frames - 1 - j * m)
        Xs[1 + j * m:1 + j * m + take] = Sx[:take]
        Vs[1 + j * m:1 + j * m + take] = Sv[:take]
    return Xs, Vs


def rk4(terms, ext, X0, V0, T, dt, every=1, steps=None, graph=None, box=None):
    n = int(round(T / dt)) if steps is None else int(steps)
    if graph is None:
        graph = str(X0.device).startswith("cuda") and n >= 4 * every
    if graph and str(X0.device).startswith("cuda"):
        try:
            return _rk4_graph(terms, ext, X0, V0, n, dt, every, box)
        except Exception:
            pass
    X, V = X0.clone(), V0.clone()
    Xs = torch.empty((n // every + 1,) + tuple(X.shape), dtype=X.dtype, device=X.device)
    Vs = torch.empty_like(Xs)
    j = 0
    for i in range(n + 1):
        if i % every == 0:
            Xs[j], Vs[j] = X, V
            j += 1
        if i == n:
            break
        k1x, k1v = V, accel(terms, ext, X, V, box)
        k2x = V + 0.5 * dt * k1v
        k2v = accel(terms, ext, X + 0.5 * dt * k1x, V + 0.5 * dt * k1v, box)
        k3x = V + 0.5 * dt * k2v
        k3v = accel(terms, ext, X + 0.5 * dt * k2x, V + 0.5 * dt * k2v, box)
        k4x = V + dt * k3v
        k4v = accel(terms, ext, X + dt * k3x, V + dt * k3v, box)
        X = X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        V = V + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return Xs, Vs
