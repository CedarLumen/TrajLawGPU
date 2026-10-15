import numpy as np

from . import rt


def observe(Xs, cfg, rng):
    X = np.asarray(Xs.detach().cpu() if hasattr(Xs, "detach") else Xs, dtype=np.float64)
    T1, N, _ = X.shape
    vis = np.ones((T1, N), bool)
    mode = cfg["obs_mode"]
    if mode == "random":
        keep = rng.choice(N, size=cfg["n_obs"], replace=False)
        vis[:] = False
        vis[:, keep] = True
    elif mode == "window":
        lo = np.asarray(cfg["win_lo"], float)
        hi = np.asarray(cfg["win_hi"], float)
        vis &= np.all((X >= lo) & (X <= hi), axis=-1)
    elif mode != "full":
        raise ValueError(mode)
    Xo = X.copy()
    if cfg["sigma_x"] > 0:
        Xo = Xo + rng.normal(0.0, cfg["sigma_x"], size=Xo.shape)
    return rt.t(np.where(vis[..., None], Xo, np.nan)), vis


def np_(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


def pair_configs(Xs, Vs, vis):
    Xs, Vs = np_(Xs), np_(Vs)
    T1, N, d = Xs.shape
    ii, jj = np.triu_indices(N, 1)
    both = vis[:, ii] & vis[:, jj]
    R = Xs[:, ii, :] - Xs[:, jj, :]
    V = Vs[:, ii, :] - Vs[:, jj, :]
    r = np.linalg.norm(R, axis=-1)
    m = both & (r > 1e-9)
    Rh = R / np.maximum(r, 1e-300)[..., None]
    vpar = np.sum(V * Rh, axis=-1)
    nv = np.stack([-Rh[..., 1], Rh[..., 0]], axis=-1)
    vperp = np.sum(V * nv, axis=-1)
    sp = np.linalg.norm(V, axis=-1)
    return dict(r=r[m], vpar=vpar[m], vperp=vperp[m], speed=sp[m],
                cos=(vpar / np.maximum(sp, 1e-300))[m], n=int(m.sum()))


def make_edges(pc, nb=12):
    def e(x):
        lo, hi = float(np.min(x)), float(np.percentile(x, 99.5))
        return np.linspace(lo, hi if hi > lo else lo + 1e-6, nb + 1)
    return [e(pc["r"]), e(pc["vpar"]), e(pc["vperp"])]


def coverage(pc, edges, nb=12):
    X = np.stack([pc["r"], pc["vpar"], pc["vperp"]], axis=-1)
    H, _ = np.histogramdd(X, bins=edges)
    cov = float((H > 0).mean())
    c = np.abs(pc["cos"])
    return dict(grid_covered=cov, grid_outside=1.0 - cov,
                cabs_median=float(np.median(c)),
                frac_head_on=float((c > 0.98).mean()),
                frac_oblique=float((c < 0.5).mean()),
                n_pairs=pc["n"])


def sample_ratio(known, unknown, Xs, Vs, stride=10):
    from .sim import diffs
    R, V = diffs(Xs[::stride], Vs[::stride])
    fk = known.f(R, V).norm(dim=-1)
    fu = unknown.f(R, V).norm(dim=-1)
    nz = fk[fk > 0]
    if nz.numel() == 0:
        return np.array([np.nan])
    m = fk > 0.05 * nz.median()
    return np_(fu[m] / fk[m])


def frac_unknown(known, unknown, Xs, Vs, stride=10):
    from .sim import diffs
    R, V = diffs(Xs[::stride], Vs[::stride])
    fu = unknown.f(R, V).norm(dim=-1)
    ft = (known.f(R, V) + unknown.f(R, V)).norm(dim=-1)
    return float((fu ** 2).mean().sqrt() / max(float((ft ** 2).mean().sqrt()), 1e-300))
